"""Veena: text and a voice name in, 24 kHz speech out.

The official route is the snippet on the model card (repo/model-card/README.md 62-186); the constants
and each step below are taken from it, with the line it comes from.
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer

from . import sampling
from .llama import Llama
from .snac import SNACDecoder

MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "veena"      # <repo>/models/veena
SAMPLE_RATE = 24000

# control token ids, fixed for Veena (model card 90-97)
START_OF_SPEECH_TOKEN = 128257
END_OF_SPEECH_TOKEN = 128258
START_OF_HUMAN_TOKEN = 128259
END_OF_HUMAN_TOKEN = 128260
START_OF_AI_TOKEN = 128261
END_OF_AI_TOKEN = 128262
AUDIO_CODE_BASE_OFFSET = 128266
PAD_TOKEN = 128263                                # tokenizer.pad_token_id, <custom_token_7>
CODEBOOK = 4096
FRAME = 7                                         # audio tokens per frame
SPEAKERS = ["kavya", "agastya", "maitri", "vinaya"]   # the four the model card documents


@dataclass
class Speech:
    audio: np.ndarray                 # float32 in [-1, 1], 24,000 Hz
    tokens: list                      # every id the model wrote, the stop token included
    frames: int                       # whole frames of 7 audio tokens that were decoded
    stopped: bool                     # True if the model wrote a stop token; False if it ran into the limit
    seconds: dict = field(default_factory=dict)   # time spent: "tokens", "audio"


class Veena:
    def __init__(self, models_dir=None):
        """Reads the official files from <repo>/models/veena (see export_weights.py)."""
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.cfg = json.loads((self.models_dir / "config.json").read_text())
        self.tokenizer = Tokenizer.from_file(str(self.models_dir / "tokenizer.json"))
        weights = {}
        for shard in sorted(self.models_dir.glob("model-*-of-*.safetensors")):
            weights.update(mx.load(str(shard)))
        mx.eval(list(weights.values()))      # MLX loads lazily; read the 7.6 GB now, once, not inside the first sentence
        self.lm = Llama(weights, self.cfg)
        self.snac = SNACDecoder(self.models_dir)

    def prompt_ids(self, text, speaker="kavya"):
        """model card 105-116: [HUMAN] <spk_speaker> text [/HUMAN] [AI] [SPEECH]."""
        if self.tokenizer.token_to_id(f"<spk_{speaker}>") is None:
            raise ValueError(f"No voice named {speaker!r}. Documented voices: {', '.join(SPEAKERS)}")
        prompt = f"<spk_{speaker}> {text}"
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        # `generate` builds an attention mask from the pad token id, so a pad token inside the prompt would be
        # hidden from the official model and shift the positions after it. It can only get there if the text
        # spells out "<custom_token_7>"; that is refused here rather than handled differently.
        if PAD_TOKEN in prompt_tokens:
            raise ValueError("The text contains the pad token <custom_token_7>")
        return [START_OF_HUMAN_TOKEN, *prompt_tokens, END_OF_HUMAN_TOKEN, START_OF_AI_TOKEN, START_OF_SPEECH_TOKEN]

    @staticmethod
    def max_new_tokens(text):
        """model card 121."""
        return min(int(len(text) * 1.3) * 7 + 21, 700)

    def generate_tokens(self, input_ids, max_new_tokens, temperature=0.4, top_p=0.9, top_k=50,
                        repetition_penalty=1.05, seed=None, do_sample=True, on_logits=None):
        """model card 124-134: write tokens until 128258 or 128262, or `max_new_tokens`. Returns the new ids
        (a stop token, if written, is the last one). `on_logits` receives each step's float32 logits (tests)."""
        rng = np.random.default_rng(seed)
        ids = list(input_ids)
        stop = (END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN)
        cache = self.lm.new_cache(len(ids) + max_new_tokens)
        logits = self.lm.logits(self.lm.forward(mx.array([ids]), cache)[:, -1, :])
        new = []
        for _ in range(max_new_tokens):
            scores = np.array(logits[0])
            if on_logits is not None:
                on_logits(scores)
            if do_sample:
                token = sampling.sample(scores, ids, rng, penalty=repetition_penalty, temperature=temperature, top_k=top_k, top_p=top_p)
            else:
                token = sampling.greedy(scores, ids, repetition_penalty)
            ids.append(token)
            new.append(token)
            if token in stop:
                break
            logits = self.lm.logits(self.lm.forward(mx.array([[token]]), cache)[:, -1, :])
            mx.eval(logits, self.lm.cache_arrays(cache))
        return new

    @staticmethod
    def codes_from_tokens(generated_ids):
        """model card 137-180: keep the audio tokens, take whole frames of 7, split each frame over SNAC's
        three levels. Returns ([level 0, level 1, level 2] as lists, number of frames)."""
        snac_tokens = [t for t in generated_ids if AUDIO_CODE_BASE_OFFSET <= t < AUDIO_CODE_BASE_OFFSET + FRAME * CODEBOOK]
        if not snac_tokens:
            raise ValueError("No audio tokens generated")
        # The official snippet gives up when the count is not a multiple of 7. An unfinished last frame can
        # only come from the token limit or a stop token arriving mid-frame; it is dropped here instead.
        frames = len(snac_tokens) // FRAME
        if frames == 0:
            raise ValueError("Fewer than 7 audio tokens generated")
        offsets = [AUDIO_CODE_BASE_OFFSET + i * CODEBOOK for i in range(FRAME)]
        lvl = [[], [], []]
        for f in range(frames):
            t = snac_tokens[f * FRAME:(f + 1) * FRAME]
            lvl[0].append(t[0] - offsets[0])
            lvl[1] += [t[1] - offsets[1], t[4] - offsets[4]]
            lvl[2] += [t[2] - offsets[2], t[3] - offsets[3], t[5] - offsets[5], t[6] - offsets[6]]
        for codes in lvl:
            if min(codes) < 0 or max(codes) >= CODEBOOK:
                raise ValueError("Invalid SNAC token values")
        return lvl, frames

    def decode(self, generated_ids, noise=None, key=None):
        """Audio tokens -> waveform, float32 in [-1, 1] (model card 183-186: decode, squeeze, clamp)."""
        lvl, frames = self.codes_from_tokens(generated_ids)
        audio = np.array(self.snac.decode([np.array([c], dtype=np.int32) for c in lvl], noise, key)[0, 0])
        if not np.isfinite(audio).all():      # checked before the clamp, which would hide an infinity
            raise ValueError("The decoder produced values that are not numbers")
        return np.clip(audio, -1, 1), frames

    def speak(self, text, speaker="kavya", temperature=0.4, top_p=0.9, top_k=50, repetition_penalty=1.05, seed=None):
        """One utterance, as `generate_speech` on the model card. One call makes at most 100 frames (8.5 s);
        longer text has to be split by the caller, as with the official snippet."""
        t0 = time.time()
        new = self.generate_tokens(self.prompt_ids(text, speaker), self.max_new_tokens(text), temperature, top_p, top_k,
                                   repetition_penalty, seed)
        t1 = time.time()
        # the codec adds random noise too: with a seed, that is fixed as well, so the waveform repeats exactly
        audio, frames = self.decode(new, key=None if seed is None else mx.random.key(seed))
        mx.clear_cache()        # hand MLX's spare buffers back; the weights stay loaded
        return Speech(audio=audio, tokens=new, frames=frames, stopped=new[-1] in (END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN),
                      seconds={"tokens": t1 - t0, "audio": time.time() - t1})
