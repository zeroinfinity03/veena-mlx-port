# Veena → MLX (Python)

Goal: run Maya Research's Veena (Hindi and English text to speech) on Apple MLX with our own code, and
get what the official route gets. Same method as our other ports: read the reference line by line
(`NOTES.md`), keep the official weights as they are, record fixtures from the real PyTorch code, then
port one piece at a time, each behind a measured gate.

The official route is the snippet on the model card: Hugging Face `transformers` generates audio codes
with the Veena language model, and the `snac` package turns the codes into sound. Our port replaces both
with code on Apple's `mlx`; the only other package it needs is `tokenizers`, the tokenizer library
`transformers` itself uses. No PyTorch at run time.

## Layout

```
Veena/
├── CLAUDE.md          the rules for working in this folder
├── inputs.md          what the model takes, from the official route, and what the port does with each input
├── speech/            generated speech: <name>.wav
├── repo/              the reference, never edited: model card, transformers' Llama and sampling files, the snac package
├── pytorch/           official weights: Veena/ (7.6 GB) and snac_24khz/ (80 MB), copied from Guru's voice/models
├── mlx-port/
│   ├── veena_mlx/     our code
│   ├── export/        official files -> models/ (only SNAC is converted; Veena's weights are safetensors already)
│   ├── models/        what our code reads
│   ├── NOTES.md       the reference read line by line, what has to match, lessons applied, results
│   └── README.md
└── tests/             the gates (official code and ours side by side), fixtures/, listening/
```

## Milestones (easiest first)

| # | Piece | Gate |
|---|---|---|
| 0 | weights in place | every tensor present, same name, same numbers as the official files |
| 1 | tokenizer and prompt | token ids identical to `transformers` on Hindi, English and mixed text, every voice |
| 2 | SNAC decoder | each layer against PyTorch; whole decoder with replayed noise: waveform match |
| 3 | Llama, the code | a small random Llama in float32 against `transformers`: logits, KV cache, greedy streams identical |
| 4 | Llama, the real weights | strict float32, layer by layer: logits within 1e-4 and the same best token at every position |
| 5 | sampling | repetition penalty, temperature, top-k, top-p on recorded logits: same surviving tokens, same probabilities; the loop against the real `generate` |
| 6 | end to end | text → 24 kHz WAV in bf16; the WAV file against soundfile's; listening A/B against the official route; speed and memory |
| 7 | independent review | reviewers read our code against the reference; every finding checked by measurement; then our own side-by-side read of every file |

Seeds do not carry between PyTorch and MLX, so sampled speech is never sample-identical to an official
sampled run. That is why the gates compare float32 logits, greedy streams and recorded distributions.

## Use

```bash
cd Veena/mlx-port
.venv/bin/python -m veena_mlx "नमस्ते, आप कैसे हैं?" --speaker kavya --name hello      # -> Veena/speech/hello.wav
```

```python
from veena_mlx import Veena

veena = Veena()                                   # loads the model once (about 10 GB while it runs)
speech = veena.speak("Binary search halves the list at every step.", speaker="agastya", seed=7)
speech.audio      # numpy float32 in [-1, 1], 24,000 Hz
speech.stopped    # False means the token limit cut the sentence: give shorter text
```

One call speaks one utterance of at most 8.5 s (the official limit of 700 audio tokens). Longer text
has to be given a sentence at a time, as with the official snippet. Voices: `kavya`, `agastya`, `maitri`,
`vinaya`.

Vik's Mac has 24 GB: run one generation at a time, and not while another heavy job is running.

## Status (2026-10-12, Apple M5, MLX 0.32.3)

- **M0 weights: done.** Veena's two safetensors files, `config.json` and `tokenizer.json` are in
  `models/` as they are (sha256 equal to the official files on Hugging Face). SNAC: 269 tensors,
  19,842,914 numbers, float32, identical after conversion from the pickle.
- **M1 tokenizer and prompt: PASS.** 132 prompts (12 texts in Hindi, English and mixed x 11 voices),
  4,125 ids, all identical to `transformers`' AutoTokenizer. A text that spells out the pad token is refused.
- **M2 SNAC decoder: PASS.** Against the `snac` package on 1, 5, 24 and 100 frames with the same noise:
  codes to latent within 1.6e-7, audio within 1e-5, SNR 112 to 117 dB.
- **M3 Llama, the code: PASS.** Five small random models in float32, two of them with Veena's own rotary
  settings: logits within 3.5e-6 of `transformers`, the same through the KV cache one token at a time
  (3.6e-6), 10 of 10 greedy streams with repetition penalty identical to `generate`. The rotation
  frequencies for Veena's real config equal the official buffer bit for bit.
- **M4 Llama, the real weights: PASS.** Strict float32, layer by layer, a short utterance (172 ids) and a
  full-length one (721 ids): hidden states after each of the 28 layers within 7.0e-7, logits within
  4.4e-6, the best token identical at all 893 positions. The model as it runs (bf16) against those float32
  logits: within 1.9e-2, best token different at 26 of 893 positions, each time a near-tie (score gap at
  most 0.146).
- **M5 sampling: PASS.** 60 score vectors of 156,951 tokens through `transformers`' own processors and
  ours: the same surviving tokens every time, probabilities within 5e-8, the same greedy choice. 60 more
  rounded to bf16 as the real model's are (34 with equal scores among the survivors): the same number of
  survivors with the same scores, and the same tokens as the official chain with a stable sort. 18
  unusual argument cases (None, 0, negative, above 1): the same steps built or the same refusal as a real
  `generate` call. The generation loop against the real `generate` on 7 scripted runs (stop tokens, the
  limit): identical.
- **M6 end to end: PASS.** Three short sentences (Hindi, English, mixed; 1.9 to 2.7 s): whole frames,
  every token in the band of its position, the model stops by itself, the same seed gives the same
  tokens and the same waveform. The WAV file equals what soundfile (the model card's writer) writes, 0 of
  440,005 samples different. The official route (the model card's functions on PyTorch, bf16, Apple GPU)
  on the same three sentences: the same lengths within a frame or two (155, 204, 204 tokens against our
  155, 197, 218) and the same loudness (rms 0.073, 0.067, 0.082 against 0.075, 0.068, 0.083); the words
  differ in detail, as two official runs do, because the draws are random. Both in `../tests/listening/`
  as `<name>_mlx.wav` and `<name>_official.wav`.
- **Listening: PASS.** Whisper, without seeing the text, hears 22 of 24 words in ours and 23 of 24 in the
  official samples; both Hindi sentences complete in both, one slurred word each in the English one
  (`NOTES.md` section 7).
- **Speed and memory, idle machine.** Ours 17.0 to 17.7 tokens/s, the official route 18.1: level. 82
  tokens make one second of speech, so either one needs a little under 5 s of work per second of speech
  on this Mac (a 5 s sentence takes about 24 s; a full 8.5 s call 39.8 s). Both read 6.6 GB of weights
  for every token, which sets the pace: 57 ms per token against a floor of about 43 ms. The audio
  decoder takes 0.1 s per sentence. Peak memory 9.4 GB. The gain of this port is no PyTorch at run time,
  not speed. The model card's own example runs the model 4-bit for speed; that needs CUDA and is a
  different, lossy model.
- **M7 review: done.** Three independent reviewers (Llama, sampling, SNAC), then our own side-by-side read
  of every file in its final state against its official counterpart. 16 findings, each confirmed by
  measurement first: one real error (the rotary-position variant, also caught by gate 4), one slowdown
  of our own making (the cache copied every token), the rest rounding-level or guards. The read of the
  final code found no difference on anything the model card's call uses. Findings and what was done
  about each: `NOTES.md` section 6; deliberate differences: section 8; what the gates do not cover:
  section 10.
