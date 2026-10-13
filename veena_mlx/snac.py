"""The SNAC 24 kHz decoder: audio codes in, waveform out.

repo/snac/ (snac 1.2.1): `SNAC.decode` = `ResidualVectorQuantize.from_codes` (vq.py 88-95) then `Decoder`
(layers.py 37-76), statement for statement. Arrays keep PyTorch's shapes, (batch, channels, time); MLX
convolutions want (batch, time, channels) and kernels as (out, k, in), and that swap is done inside the
two conv layers below, in one place.
"""
import json
import math
from pathlib import Path

import mlx.core as mx


def _weight(w, name):
    """torch.nn.utils.parametrizations.weight_norm stores g as `original0` and v as `original1`:
    weight = v * (g / ||v||), the norm over every axis but the first, no epsilon. For a ConvTranspose1d the
    first axis is the INPUT channel; that is what torch does and what is done here."""
    g, v = w[f"{name}.parametrizations.weight.original0"], w[f"{name}.parametrizations.weight.original1"]
    norm = mx.sqrt((v * v).sum(axis=tuple(range(1, v.ndim)), keepdims=True))
    return v * (g / norm)


class WNConv1d:
    """weight_norm(nn.Conv1d). PyTorch weight (out, in/groups, k) -> MLX (out, k, in/groups)."""

    def __init__(self, w, name, stride=1, padding=0, dilation=1, groups=1):
        self.weight = _weight(w, name).transpose(0, 2, 1)
        self.bias = w.get(f"{name}.bias")
        self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups

    def __call__(self, x):
        y = mx.conv1d(x.transpose(0, 2, 1), self.weight, stride=self.stride, padding=self.padding,
                      dilation=self.dilation, groups=self.groups)
        if self.bias is not None:
            y = y + self.bias
        return y.transpose(0, 2, 1)


class WNConvTranspose1d:
    """weight_norm(nn.ConvTranspose1d), groups 1. PyTorch weight (in, out, k) -> MLX (out, k, in)."""

    def __init__(self, w, name, stride, padding, output_padding):
        self.weight = _weight(w, name).transpose(1, 2, 0)
        self.bias = w[f"{name}.bias"]
        self.stride, self.padding, self.output_padding = stride, padding, output_padding

    def __call__(self, x):
        y = mx.conv_transpose1d(x.transpose(0, 2, 1), self.weight, stride=self.stride, padding=self.padding,
                                output_padding=self.output_padding)
        return (y + self.bias).transpose(0, 2, 1)


def snake(x, alpha):
    """layers.py 170-176: x + (alpha + 1e-9)^-1 * sin(alpha * x)^2, alpha [1, C, 1]."""
    return x + (1.0 / (alpha + 1e-9)) * mx.square(mx.sin(alpha * x))     # mx.square, not ** 2: bit for bit with torch's pow(2)


class ResidualUnit:
    """layers.py 79-96: Snake, depthwise conv k7 with dilation, Snake, 1x1 conv, plus the input."""

    def __init__(self, w, name, dim, dilation, groups, kernel=7):
        self.w, self.name = w, name
        self.conv1 = WNConv1d(w, f"{name}.block.1", dilation=dilation, padding=((kernel - 1) * dilation) // 2, groups=groups)
        self.conv2 = WNConv1d(w, f"{name}.block.3")

    def __call__(self, x):
        y = snake(x, self.w[f"{self.name}.block.0.alpha"])
        y = self.conv1(y)
        y = snake(y, self.w[f"{self.name}.block.2.alpha"])
        y = self.conv2(y)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class DecoderBlock:
    """layers.py 133-159: Snake, transposed conv (upsample by `stride`), NoiseBlock, three ResidualUnits."""

    def __init__(self, w, name, input_dim, output_dim, stride, groups):
        self.w, self.name = w, name
        self.up = WNConvTranspose1d(w, f"{name}.block.1", stride=stride, padding=math.ceil(stride / 2), output_padding=stride % 2)
        self.noise_linear = WNConv1d(w, f"{name}.block.2.linear")
        self.res = [ResidualUnit(w, f"{name}.block.{3 + i}", output_dim, dilation=d, groups=groups) for i, d in enumerate((1, 3, 9))]

    def __call__(self, x, noise):
        x = snake(x, self.w[f"{self.name}.block.0.alpha"])
        x = self.up(x)
        # NoiseBlock (layers.py 119-130): x + noise * linear(x), noise standard normal [B, 1, T]
        x = x + noise * self.noise_linear(x)
        for r in self.res:
            x = r(x)
        return x


class SNACDecoder:
    def __init__(self, models_dir):
        models_dir = Path(models_dir)
        cfg = json.loads((models_dir / "snac_config.json").read_text())
        assert cfg["depthwise"] and cfg["noise"] and cfg["attn_window_size"] is None, "written for snac_24khz's settings"
        w = self.w = {k: v for k, v in mx.load(str(models_dir / "snac_24khz.safetensors")).items()
                      if k.startswith(("decoder.", "quantizer."))}
        self.sampling_rate = cfg["sampling_rate"]
        self.vq_strides = cfg["vq_strides"]
        self.codebook_size = cfg["codebook_size"]
        self.rates = cfg["decoder_rates"]
        self.hop_length = math.prod(cfg["encoder_rates"])
        latent = cfg.get("latent_dim") or cfg["encoder_dim"] * 2 ** len(cfg["encoder_rates"])      # snac.py 36-37
        ch = cfg["decoder_dim"]
        self.out_proj = [WNConv1d(w, f"quantizer.quantizers.{i}.out_proj") for i in range(len(self.vq_strides))]
        # Decoder with depthwise=True and no attention (layers.py 49-55)
        self.conv_in1 = WNConv1d(w, "decoder.model.0", padding=3, groups=latent)
        self.conv_in2 = WNConv1d(w, "decoder.model.1")
        self.blocks = [DecoderBlock(w, f"decoder.model.{2 + i}", ch // 2 ** i, ch // 2 ** (i + 1), s, groups=ch // 2 ** (i + 1))
                       for i, s in enumerate(self.rates)]
        n = 2 + len(self.rates)
        self.final_alpha = w[f"decoder.model.{n}.alpha"]
        self.conv_out = WNConv1d(w, f"decoder.model.{n + 1}", padding=3)

    def from_codes(self, codes):
        """vq.py 88-95. codes: one integer array [B, T_i] per level, coarse to fine. Returns [B, latent, T]."""
        z_q = 0.0
        for i, c in enumerate(codes):
            # torch's embedding raises on an index outside the codebook; MLX would quietly return zeros
            lo, hi = int(c.min()), int(c.max())
            if lo < 0 or hi >= self.codebook_size:
                raise IndexError(f"SNAC code out of range at level {i}: {lo}..{hi}, codebook has {self.codebook_size} entries")
            z_p_i = self.w[f"quantizer.quantizers.{i}.codebook.weight"][c].transpose(0, 2, 1)     # decode_code
            z_q_i = self.out_proj[i](z_p_i)
            z_q_i = mx.repeat(z_q_i, self.vq_strides[i], axis=-1)                                 # repeat_interleave
            z_q = z_q + z_q_i
        return z_q

    def noise_shapes(self, n_fine, batch=1):
        """The shape of each NoiseBlock's random array for `n_fine` finest-level codes."""
        shapes, t = [], n_fine
        for s in self.rates:
            t *= s
            shapes.append((batch, 1, t))
        return shapes

    def decode(self, codes, noise=None, key=None):
        """SNAC.decode (snac.py 85-88). Returns audio [B, 1, T_fine * hop]. The decoder adds random noise in
        every block: by default it is drawn here, from `key` (an `mx.random.key`) when one is given, so
        that a seeded call gives the same waveform again. `noise` replays recorded draws (tests)."""
        codes = [mx.array(c) for c in codes]
        assert len(codes) == len(self.vq_strides) and self.vq_strides[-1] == 1
        if noise is None:
            shapes = self.noise_shapes(codes[-1].shape[1], codes[-1].shape[0])
            keys = mx.random.split(key, len(shapes)) if key is not None else [None] * len(shapes)
            noise = [mx.random.normal(shape=s, key=k) for s, k in zip(shapes, keys)]
        assert len(noise) == len(self.blocks), "one noise array per decoder block"
        x = self.from_codes(codes)
        x = self.conv_in2(self.conv_in1(x))
        for block, n in zip(self.blocks, noise):
            x = block(x, n)
        x = snake(x, self.final_alpha)
        return mx.tanh(self.conv_out(x))
