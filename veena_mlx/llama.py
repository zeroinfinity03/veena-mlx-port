"""Veena's language model: a Llama decoder, as Hugging Face `transformers` runs it.

repo/transformers/modeling_llama.py (5.19.0), statement for statement: LlamaRMSNorm 51-62,
LlamaRotaryEmbedding 65-106 (its frequencies from modeling_rope_utils.py), rotate_half / apply_rotary_pos_emb 109-134, LlamaMLP 137-150, repeat_kv
153-163, attention 166-240, LlamaDecoderLayer 243-277, the model and head below them.

The functions take the weights as a dict keyed by the official tensor names, so the same code runs the
whole model in bf16 (real use) and one layer at a time in float32 (the strict test).
"""
import math

import mlx.core as mx
import numpy as np


def rms_norm(x, weight, eps):
    """LlamaRMSNorm: worked out in float32 whatever the model's type, cast back, then scaled."""
    dtype = x.dtype
    h = x.astype(mx.float32)
    variance = (h * h).mean(axis=-1, keepdims=True)
    h = h * mx.rsqrt(variance + eps)
    return weight * h.astype(dtype)


def rope_inv_freq(cfg):
    """The rotation frequencies, one per pair of head dimensions, in float32.

    Veena's config has `rope_scaling` with `rope_type: "llama3"` (Llama 3.1's variant), so the plain
    formula is only the start: transformers' `_compute_llama3_parameters`
    (repo/transformers/modeling_rope_utils.py 580-661) divides the slow frequencies by `factor` and blends
    the ones in between. A config without `rope_scaling` gets the plain frequencies
    (`LlamaRotaryEmbedding.compute_default_rope_parameters`)."""
    base, dim = cfg["rope_theta"], cfg["head_dim"]
    exponent = np.arange(0, dim, 2).astype(np.float32) / np.float32(dim)
    # torch's float32 power is correctly rounded here; numpy's float32 power is 1 to 6 float32 steps off for
    # two of Veena's 64 frequencies. The power is therefore taken in float64 and rounded once
    # (tests/gates_code.py checks the result against the official buffer, bit for bit).
    power = (np.float64(base) ** exponent.astype(np.float64)).astype(np.float32)
    inv_freq = (np.float32(1.0) / power).astype(np.float32)
    scaling = cfg.get("rope_scaling") or {}
    kind = scaling.get("rope_type", "default")
    if kind == "default":
        return mx.array(inv_freq)
    if kind != "llama3":
        raise NotImplementedError(f"rope_type {kind!r}")
    factor = scaling["factor"]
    low_freq_factor = scaling["low_freq_factor"]
    high_freq_factor = scaling["high_freq_factor"]
    old_context_len = scaling["original_max_position_embeddings"]
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    f32 = np.float32
    wavelen = (f32(2 * math.pi) / inv_freq).astype(f32)
    # wavelen < high_freq_wavelen: do nothing; wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = np.where(wavelen > f32(low_freq_wavelen), inv_freq / f32(factor), inv_freq).astype(f32)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = ((f32(old_context_len) / wavelen - f32(low_freq_factor)) / f32(high_freq_factor - low_freq_factor)).astype(f32)
    smoothed_inv_freq = ((f32(1) - smooth_factor) * inv_freq_llama / f32(factor) + smooth_factor * inv_freq_llama).astype(f32)
    is_medium_freq = ~(wavelen < f32(high_freq_wavelen)) & ~(wavelen > f32(low_freq_wavelen))
    return mx.array(np.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama).astype(f32))


def rope_tables(positions, inv_freq, dtype):
    """LlamaRotaryEmbedding.forward: cos and sin [T, head_dim] for the given integer positions, made in
    float32 and cast to the model's type."""
    freqs = positions.astype(mx.float32)[:, None] * inv_freq[None, :]
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def linear(x, weight):
    """nn.Linear without bias: weight is (out, in)."""
    return x @ weight.T


class KVCache:
    """One layer's keys and values so far, [B, kv_heads, T, head_dim], after the rotation and before the
    repeat, as transformers' cache holds them. Kept in float32 (see `attention`).

    The buffers are allocated once, for the whole utterance (`capacity` = prompt + the most new tokens),
    and each step writes its one position into them: the way mlx-lm's cache and our YuE and ACE-Step
    ports do it. Joining a new array at every step gives the same numbers but copies the whole cache again
    for every token. Each layer has its own buffers and its own length; if a caller goes past `capacity`,
    that layer's own buffers are enlarged."""

    def __init__(self, capacity):
        self.capacity, self.n = capacity, 0
        self.k = self.v = None

    def append(self, k, v):
        B, KV, T, D = k.shape
        if self.k is None:
            size = max(self.capacity, T)
            self.k, self.v = mx.zeros((B, KV, size, D), dtype=k.dtype), mx.zeros((B, KV, size, D), dtype=v.dtype)
        elif self.n + T > self.k.shape[2]:
            size = self.n + T + 256
            nk, nv = mx.zeros((B, KV, size, D), dtype=k.dtype), mx.zeros((B, KV, size, D), dtype=v.dtype)
            nk[:, :, :self.n], nv[:, :, :self.n] = self.k[:, :, :self.n], self.v[:, :, :self.n]
            self.k, self.v = nk, nv
        self.k[:, :, self.n:self.n + T] = k
        self.v[:, :, self.n:self.n + T] = v
        self.n += T
        return self.k[:, :, :self.n], self.v[:, :, :self.n]


def attention(w, p, x, cos, sin, mask, cfg, cache):
    """LlamaAttention. x [B, T, hidden]; `cache` is None (nothing is kept) or this layer's KVCache, which
    receives the new keys and values."""
    B, T, _ = x.shape
    H, KV, D = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    q = linear(x, w[f"{p}.q_proj.weight"]).reshape(B, T, H, D).transpose(0, 2, 1, 3)
    k = linear(x, w[f"{p}.k_proj.weight"]).reshape(B, T, KV, D).transpose(0, 2, 1, 3)
    v = linear(x, w[f"{p}.v_proj.weight"]).reshape(B, T, KV, D).transpose(0, 2, 1, 3)
    # apply_rotary_pos_emb: queries and keys only
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    # The attention core, softmax(q k^T * head_dim^-0.5 + mask) v, is worked out in float32 and rounded to
    # the model's type once, at the end. That is how the official default (scaled_dot_product_attention)
    # behaves on bf16 inputs, and it keeps a Python number from being rounded to bf16 before it is used.
    # So keys and values are converted to float32 once, when they are made, and cached that way.
    dtype = q.dtype
    k, v = k.astype(mx.float32), v.astype(mx.float32)
    if cache is not None:
        k, v = cache.append(k, v)
    # repeat_kv without the copy: key/value head j serves query heads j*n_rep .. j*n_rep + n_rep - 1, so
    # the queries are grouped as [B, KV, n_rep, T, D] and each group meets its one key/value head.
    n_rep = H // KV
    qg = q.astype(mx.float32).reshape(B, KV, n_rep, T, D)
    scores = (qg @ k[:, :, None].transpose(0, 1, 2, 4, 3)) * (D ** -0.5)        # [B, KV, n_rep, T, T_all]
    if mask is not None:
        scores = scores + mask
    out = (mx.softmax(scores, axis=-1) @ v[:, :, None]).astype(dtype)          # [B, KV, n_rep, T, D]
    out = out.reshape(B, H, T, D).transpose(0, 2, 1, 3).reshape(B, T, H * D)
    return linear(out, w[f"{p}.o_proj.weight"])


def mlp(w, p, x):
    """LlamaMLP: down(silu(gate(x)) * up(x))."""
    gate = linear(x, w[f"{p}.gate_proj.weight"])
    # F.silu rounds once. In bf16, `gate * sigmoid(gate)` would round the sigmoid first and then the product.
    g = gate.astype(mx.float32)
    act = (g * mx.sigmoid(g)).astype(gate.dtype)
    return linear(act * linear(x, w[f"{p}.up_proj.weight"]), w[f"{p}.down_proj.weight"])


def decoder_layer(w, i, x, cos, sin, mask, cfg, cache=None):
    """LlamaDecoderLayer: pre-norm attention and MLP, each added back to its input."""
    p = f"model.layers.{i}"
    eps = cfg["rms_norm_eps"]
    x = x + attention(w, f"{p}.self_attn", rms_norm(x, w[f"{p}.input_layernorm.weight"], eps), cos, sin, mask, cfg, cache)
    x = x + mlp(w, f"{p}.mlp", rms_norm(x, w[f"{p}.post_attention_layernorm.weight"], eps))
    return x


def causal_mask(t_new, t_total, dtype):
    """Additive mask [t_new, t_total] for `t_new` new positions at the end of a sequence of `t_total`:
    a position does not see the ones after it. None when a single new position sees everything."""
    if t_new == 1:
        return None
    rows = mx.arange(t_total - t_new, t_total)[:, None]
    cols = mx.arange(t_total)[None, :]
    return mx.where(cols > rows, mx.array(-mx.inf, dtype=dtype), mx.array(0, dtype=dtype))


class Llama:
    def __init__(self, weights, cfg):
        self.w, self.cfg = weights, cfg
        self.layers = cfg["num_hidden_layers"]
        self.dtype = weights["model.embed_tokens.weight"].dtype
        self.inv_freq = rope_inv_freq(cfg)

    def new_cache(self, capacity):
        """An empty cache for a sequence of at most `capacity` positions: one KVCache per layer."""
        return [KVCache(capacity) for _ in range(self.layers)]

    @staticmethod
    def cache_arrays(cache):
        """The arrays a cache holds, to hand to mx.eval after a step."""
        return [a for c in cache for a in (c.k, c.v)]

    def forward(self, input_ids, cache=None):
        """input_ids [B, T_new], the next positions of the sequence. With `cache` (from `new_cache`) the
        positions already in it are attended to and the new ones are added to it; without, the ids are a
        whole sequence. Returns the hidden states [B, T_new, hidden] after the final norm."""
        w, cfg = self.w, self.cfg
        t_new = input_ids.shape[1]
        t_past = 0 if cache is None else cache[0].n
        x = w["model.embed_tokens.weight"][input_ids]
        cos, sin = rope_tables(mx.arange(t_past, t_past + t_new), self.inv_freq, self.dtype)
        mask = causal_mask(t_new, t_past + t_new, mx.float32)
        for i in range(self.layers):
            x = decoder_layer(w, i, x, cos, sin, mask, cfg, None if cache is None else cache[i])
        return rms_norm(x, w["model.norm.weight"], cfg["rms_norm_eps"])

    def logits(self, hidden):
        """lm_head, read from the file as stored. Returned in float32, as `generate` hands them to the sampler."""
        return linear(hidden, self.w["lm_head.weight"]).astype(mx.float32)
