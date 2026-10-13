# Port spec: read line by line, then measured

Reference, all in `../repo/` and `../pytorch/` (never edited):
- `maya-research/Veena` (Apache 2.0): the weights, tokenizer and model card. There is no official
  inference code beyond the snippet on the model card (`repo/model-card/README.md`).
- The snippet runs Hugging Face `transformers` (`LlamaForCausalLM.generate`; here 5.19.0,
  `repo/transformers/modeling_llama.py`, `logits_process.py`) and the `snac` package 1.2.1
  (`repo/snac/`, MIT) with `hubertsiuzdak/snac_24khz`.
- The weight files in `../pytorch/` are copies of the ones Guru already had; their sha256 equals the
  official ones on Hugging Face (checked 2026-10-12).

## 1. What Veena is

Text and a voice name in, 24 kHz speech out, in two models:

| Model | Size | What it does |
|---|---|---|
| Veena, a Llama-style LM | 255 tensors, 3.783 B numbers, bf16, 7.6 GB | reads the prompt and writes audio codes, 7 tokens per frame |
| SNAC 24 kHz decoder | 142 of the codec's 269 tensors, float32 (130 decoder, 3 codebooks, 9 for the three out_proj) | turns the codes into a waveform, 2,048 samples (85.3 ms) per frame |

LM: 28 layers, width 3072, 24 heads of 128 with 8 key/value heads (each shared by 3 heads), MLP 8192
with SiLU, RMSNorm eps 1e-5, RoPE theta 500,000, vocabulary 156,951. `lm_head.weight` is stored as its
own tensor.

Tokens: Llama 3's tokenizer plus 28,683 `<custom_token_N>` (id 128256 + N) and 11 `<spk_NAME>` tokens.
The model card documents 4 voices (kavya, agastya, maitri, vinaya); the tokenizer also has apsara, charu,
ishana, kyra, mohini, varun, soumya.

## 2. The official route, step by step (model card lines 102-186)

1. `prompt = f"<spk_{speaker}> {text}"`, tokenized without special tokens.
2. Input ids: `[128259] + prompt ids + [128260, 128261, 128257]` (start of human, end of human, start
   of AI, start of speech).
3. `max_new_tokens = min(int(len(text) * 1.3) * 7 + 21, 700)`. So one call makes at most 100 frames,
   8.5 s of speech; longer text has to be split by the caller.
4. `generate(do_sample=True, temperature=0.4, top_p=0.9, repetition_penalty=1.05,
   eos_token_id=[128258, 128262])`.
5. Keep the generated ids in `[128266, 128266 + 7 * 4096)`. If there are none, or their count is not a
   multiple of 7, the snippet fails.
6. De-interleave each frame of 7 into SNAC's three levels: level 0 gets token 0; level 1 tokens 1 and 4;
   level 2 tokens 2, 3, 5, 6; each minus `128266 + position * 4096`. A code outside 0..4095 is an error.
7. `snac.decode(codes)`, squeeze, clamp to [-1, 1].

## 3. What has to be exactly right

Sampling (measured by running the official `generate` with the card's arguments on a small random
Llama and reading the processor list, 2026-10-12):

1. **Four steps, in this order, on float32 scores**: repetition penalty 1.05, temperature 0.4,
   **top-k 50**, top-p 0.9. Top-k is not in the card's call nor in `generation_config.json`; it is a
   library default that the official route applies all the same.
2. **Repetition penalty** covers every id so far, the prompt included: `score < 0 ? score * p : score / p`.
3. **Top-k** removes scores strictly below the 50th largest (ties with it stay).
4. **Top-p** as transformers writes it: sort ascending, softmax, running sum, remove where the sum is
   `<= 1 - top_p`, always keep the largest.
5. Then softmax and one multinomial draw. Stop when 128258 or 128262 is drawn.

LM:

6. RMSNorm is computed in float32 and cast back: `x * rsqrt(mean(x^2) + eps)`, then `* weight`.
7. RoPE is the **Llama 3.1 variant**: `config.json` has `rope_scaling` with `rope_type: "llama3"`
   (factor 32, low 1, high 4, original length 8192). Start from `inv_freq = 1 / theta^(arange(0, 128, 2) / 128)`
   in float32; frequencies whose wavelength `2 pi / inv_freq` is above 8192 are divided by 32, those
   between 2048 and 8192 are blended (`_compute_llama3_parameters`, `repo/transformers/modeling_rope_utils.py`
   580-661). 35 of the 64 frequencies change. Then angles = position * inv_freq, `cos` and `sin` of
   `[angles, angles]` cast to the model dtype; `q*cos + rotate_half(q)*sin` with
   `rotate_half(x) = [-x2, x1]`; applied to q and k only. Positions continue from the cache length.
8. Attention: q `[B, 24, T, 128]`, k and v `[B, 8, T, 128]` each repeated 3 times in a row
   (`repeat_interleave`), scale `128^-0.5`, causal, softmax, `o_proj`. No biases anywhere.
9. MLP: `down(silu(gate(x)) * up(x))`. Pre-norm residuals. Final `norm`, then `lm_head`.
10. The KV cache must give the same logits as recomputing the whole sequence.

SNAC decoder (`repo/snac/`):

11. `from_codes`: per level, codebook lookup `[B, T_i, 8]` → `out_proj` (weight-normed 1x1 conv to 768) →
    repeated 4, 2, 1 times along time; the three are summed.
12. Decoder: depthwise conv k7 (768 groups) → 1x1 conv to 1024 → four blocks (1024→512 stride 8,
    512→256 stride 8, 256→128 stride 4, 128→64 stride 2) → Snake → conv k7 to 1 channel → tanh.
13. Block: Snake, transposed conv (kernel 2·stride, padding ceil(stride/2), output_padding stride%2),
    NoiseBlock, three ResidualUnits with dilation 1, 3, 9 (Snake, depthwise conv k7, Snake, 1x1 conv, skip).
14. Snake: `x + (alpha + 1e-9)^-1 * sin(alpha * x)^2`.
15. NoiseBlock: `x + randn(B, 1, T) * linear(x)`, the linear a weight-normed 1x1 conv without bias. The
    decoder is therefore random: four draws per call.
16. weight_norm here is `torch.nn.utils.parametrizations.weight_norm`: tensors `…original0` (g) and
    `…original1` (v), `w = v * (g / ||v||)`, the norm over every axis but the first, no epsilon.

## 4. Lessons from our earlier ports, applied here

The three earlier ports' notes were read end to end for this (`musicgen/YuE/mlx-port/NOTES.md`,
`ACE-Step/mlx-port/NOTES.md`, `StableAudio/mlx-port/NOTES.md`), on 2026-10-12, after a first pass that had
only read parts of them. Several items below were first rediscovered here by experiment or by the
reviewers and would have been right from the start had the notes been read first.

| Lesson | Where it was learnt | What it means for Veena |
|---|---|---|
| A library or config default is part of the model | Kokoro (ALBERT's `gelu_new`) | top-k 50 (section 3 item 1), the Llama 3.1 rotary variant (item 7): found by running the official code and reading the real `config.json` whole |
| Use the reference's own tokenizer code | ACE-Step fix 14 (another tokenizer gave other ids for Hindi nukta + vowel sign) | the `tokenizers` library on the official `tokenizer.json`; gate on Hindi, English, mixed; 20,085 prompts fuzzed by the reviewer |
| Load stored tensors, never recompute them | YuE problem 2 (a position table stored in bf16) | `lm_head.weight` is read from the file, not assumed equal to the embedding |
| Mirror the reference's precision step by step | YuE "precision profile", ACE-Step fix 12 (softmax in fp32) | RMSNorm, rotary tables, the attention core and SiLU in float32, rounded once; logits to float32 before sampling |
| A bf16 array times a Python number rounds differently in torch and MLX | YuE problem 7 ("opmath") | no bf16-times-scalar anywhere: the attention scale is applied in float32 |
| Constants must be the same float the reference uses | Stable Audio C1 (`Float.pi` one step low) | `2 pi` and the rotary constants are formed as torch forms them; the frequencies are compared with the official buffer bit for bit |
| The reference's own choices at a tie are not stable | YuE problems 9, 11, 12 (bf16 greedy streams differ across torch's backends; torch's sort is unstable; a "tie swap" is not a port difference) | gate 5 on bf16-valued scores accepts only swaps between equal scores; gate 4's bf16 check accepts a different best token only on a near-tie |
| Running sums in the reference's accumulator type | YuE problem 8 (bf16 cumsum) | top-p's running sum is kept the way torch keeps it (float64, each element rounded to float32) |
| The cache is allocated once and written in place | YuE (the reference's `StaticKVCache`, preallocated to prefix + max tokens), ACE-Step "productionization: KV-cache preallocation" | `KVCache` (section 9). The first version here joined a new array per token, and a later one re-copied the whole cache per token: 20% slower on a full utterance |
| A cache that grows must grow per layer | ACE-Step fix 13 (layers 1-27 would have written past their buffers) | each `KVCache` owns its buffers and length; gate 3 also runs with a cache that is too small |
| Seeds do not carry between PyTorch and MLX | all three | gates use greedy streams, teacher forcing, recorded scores and replayed noise |
| Two bf16 backends disagree more than two fp32 ones | YuE problem 5 | correctness is proven in float32; bf16 is checked with near-tie rules |
| The M5 runs float32 products at reduced precision by default | YuE problem 4, ACE-Step "noise floors" (corrected), Stable Audio section 11 | `MLX_ENABLE_TF32=0` in the package (section 5) |
| Weight norm, transposed-conv layout, lengths | ACE-Step and YuE VAE sections, Kokoro | section 3 items 13 and 16; the SNAC gate; every layer checked by the reviewer |
| Fail loudly, not silently; check for bad numbers BEFORE the clip | ACE-Step fix 8, Stable Audio C4 | no audio tokens, a code out of range, a pad token in the prompt, NaN or infinity in the audio: each is an error |
| Write the file the way the reference writes it | ACE-Step fix 10, Stable Audio section 15 item 5 | `audio.save_wav` writes soundfile's bytes exactly; the rule was measured, the first guess at it was wrong (problem 5 below) |
| Input bounds are refused with a message, not a crash | Stable Audio C6 (negative seed), ACE-Step fix 11 | the CLI refuses empty text and a negative seed; a temperature of 0 or less is refused as the official code refuses it |
| Read every weight at load; say where the time goes | Stable Audio E7 / fix 8, ACE-Step "efficiency" | `Veena()` reads the 7.6 GB once; `Speech.seconds` splits tokens and audio |
| Free MLX's spare buffers after a run | ACE-Step "efficiency" (`clearCache` never called) | `mx.clear_cache()` after every utterance |
| No two heavy models in memory together | Stable Audio A2, Vik's rule | one generation at a time; the speech gate checks free memory first |
| Gates must run the real functions, not copies | Stable Audio T2, Q2 | gate 5b drives `Veena.generate_tokens` itself; one WAV writer for the CLI and the gates |
| A skipped stage must not read as a pass | Stable Audio T4 | the bf16 part of gate 4 says "skipped" when it does not run |
| Do not loosen a gate after a failure without knowing why | Stable Audio T1 | the listening gate's rule was changed only after four more seeds showed the miss was one draw (section 7) |
| Listen with an independent model | YuE "listening test" (Whisper, line for line) | `tests/listening_check.py` |
| Know the memory-bandwidth floor | YuE timing notes | section 9 |
| Tests, then independent reviewers, then our own side-by-side read | ACE-Step and Stable Audio "how it was checked" | section 6 |
| Say what is not covered | ACE-Step "coverage gaps", Stable Audio "not covered by the gates" | section 10 |

## 5. Precision

- The language model runs in bf16, the type the weights are released in. RMSNorm, the rotary tables, the
  attention core (scores, softmax, weighted sum) and SiLU are worked out in float32 and rounded to bf16
  once; logits are handed to the sampler in float32. That is what the official code does in each of those
  places (the attention core as the default `scaled_dot_product_attention` behaves on bf16 inputs).
- Correctness is proven in float32 (gates 3 and 4). bf16 is then measured against float32: logits within
  1.9e-2, and where the best token changes (26 of 893 positions) the two were near-tied (score gap at most 0.146). Two bf16 backends
  never agree bit for bit, PyTorch's own included, so there is no single official bf16 answer to match.
- SNAC is float32. `veena_mlx/__init__.py` sets `MLX_ENABLE_TF32=0`, because on Apple's newest GPUs MLX
  otherwise runs float32 products at reduced precision. Measured on five real streams with the same noise:
  SNR against the official decoder 117 to 120 dB with it off, 44 dB with MLX's default (for scale, two
  official decodes with different noise are 33 dB apart, so even 44 dB is far below what the decoder's own
  randomness does). It costs SNAC about 4% in speed and does not touch the bf16 products of the language
  model. A program that uses MLX before importing `veena_mlx` has to set the variable itself.
- The model card's example loads the model 4-bit (bitsandbytes NF4), which needs CUDA and does not run on
  a Mac. This port runs the released bf16 weights, which is what a plain `from_pretrained` gives.

## 6. Review (2026-10-12)

Method, the same as for the ACE-Step and YuE ports: gates first, then independent reviewers who read our
code against the reference with instructions to find a mistake, then our own line-by-line read. Every
finding below was confirmed by measurement before anything was changed.

**Final side-by-side read (2026-10-12, on the final code, after every fix below).** Each file read in
full against its official counterpart, whole functions, not searches:

| Ours | Read against | Result |
|---|---|---|
| `llama.py` | `repo/transformers/modeling_llama.py` (LlamaRMSNorm, LlamaRotaryEmbedding, rotate_half, apply_rotary_pos_emb, LlamaMLP, repeat_kv, eager_attention_forward, LlamaAttention, LlamaDecoderLayer, the model's forward); `modeling_rope_utils.py` 580-661; transformers' cache update and `_sample` | matches: RMSNorm in float32 with eps inside the rsqrt, cast, then the weight; the Llama 3.1 frequencies, statement for statement; cos/sin of `[angles, angles]`; rotation of q and k only; keys cached after the rotation and before the repeat; key/value head j serving query heads 3j to 3j+2; scale `128^-0.5`; causal mask; softmax; `o_proj`; SiLU MLP; pre-norm residuals; final norm; stored `lm_head`; positions continuing from the cache length; logits to float32 |
| `sampling.py` | `repo/transformers/logits_process.py`: RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper; `_get_logits_processor` (1402-1635) and `_sample` (3097-3285) in the installed `transformers/generation/utils.py`; `DynamicLayer.update` in `cache_utils.py` | matches: the order; the penalty over every id so far with `score < 0 ? score * p : score / p`; `scores / temperature`; top-k cutting strictly below the k-th largest; top-p's ascending sort, softmax, running sum, `<= 1 - top_p`, largest always kept; softmax and one multinomial draw; argmax when not sampling; a step left out exactly when the official code leaves it out |
| `snac.py` | `repo/snac/vq.py` 88-95 (`from_codes`, `decode_code`), `layers.py` (Decoder, DecoderBlock, NoiseBlock, ResidualUnit, `snake`, the two weight-normed conv factories), `snac.py` 85-88 | matches: codebook lookup, `out_proj`, repeat by 4, 2, 1, sum; depthwise k7 then 1x1; four blocks of Snake, transposed conv, noise, three residual units with dilation 1, 3, 9; final Snake, k7 conv, tanh; weight norm over every axis but the first |
| `model.py` | `repo/model-card/README.md` 62-186, the whole snippet | matches: the token ids, the prompt string and its wrapping, the token limit, the stop tokens, the audio-token filter, which of a frame's 7 tokens go to which level and in which order, the offsets, the range check, decode then clamp |
| `audio.py` | soundfile's `sf.write` as the model card calls it (measured, 440,005 samples) | the same bytes |

On everything the model card's call uses, the read found no difference other than the deliberate ones in
section 8. It found two things outside that call, both made exact (finding 16). After it, every gate was
run again on the final code: all pass, and the three test sentences come out token for token as before.

Before the read, three independent reviewers (Llama, sampling, SNAC), each told to assume there was a
mistake and find it; and before them, the gates.

Findings:

| # | Finding | Found by | What was done |
|---|---|---|---|
| 1 | **Rotary positions used the plain formula; Veena's config asks for the Llama 3.1 variant** (`rope_scaling`). 35 of 64 frequencies wrong; logits off by 2 to 3% and the best token different at 14 of 484 positions. The small random models did not have the setting, so gate 3 passed | gate 4 on the real weights, and independently the Llama and sampling reviewers | `rope_inv_freq` ports `_compute_llama3_parameters`; gate 3 now has two models with Veena's settings and compares the frequencies with the official buffer bit for bit; gate 4 covers a full-length sequence (721 ids) |
| 2 | Two of the 64 frequencies were 1 and 6 float32 steps from the official buffer (numpy's float32 power) | Llama reviewer | the power is taken in float64 and rounded once: 0 of 64 differ |
| 3 | bf16 only: `gate * sigmoid(gate)` rounded twice where `F.silu` rounds once (32% of values one step off; no measurable effect on logits) | Llama reviewer | SiLU worked out in float32, rounded once |
| 4 | bf16 only: attention scores were rounded to bf16 before the softmax | Llama reviewer | the attention core is worked out in float32 and rounded once at the end |
| 5 | `generate` masks pad tokens (id 128263); a prompt containing one would be treated differently. Only possible if the text spells out `<custom_token_7>` | Llama reviewer | such text is refused |
| 6 | Real use ran SNAC with MLX's default reduced precision; only the gates were strict | SNAC reviewer | `MLX_ENABLE_TF32=0` set in the package (section 5) |
| 7 | Snake squared with `** 2` (MLX's pow, up to 2 float32 steps from torch); out-of-range codes given straight to the decoder returned zeros instead of failing | SNAC reviewer | `mx.square`; `from_codes` raises on a code outside the codebook |
| 8 | Deliberate: the official snippet returns nothing when the audio-token count is not a multiple of 7; ours drops the unfinished last frame. No input was found where that gives audio from misaligned frames | written that way; checked by the SNAC reviewer on 15 edge cases | kept |
| 9 | Top-p ties: bf16-valued logits always have equal scores among the top 50, and which of several equal tokens survives the top-p cut depends on the sort. Ours is stable. The official code keeps exactly our sets on Apple's GPU (500 of 500 steps) and with `stable=True`; on the CPU `torch.sort` is not stable and picks other equal-score tokens in up to 40% of open steps (the kept scores are identical every time). So the official code has no single answer here | Llama and sampling reviewers | kept (stable sort, equal to the official route on this Mac's GPU); gate 5 now also runs bf16-valued scores: survivors' count and scores against the official chain as it is, the tokens themselves against it with a stable sort |
| 10 | Top-p arithmetic differed from torch in the last bits (torch multiplies by the reciprocal of the sum and keeps the running sum in float64, rounding each element); on vectors built to sit on the threshold, 40 of 3,240 kept one token more or less | sampling reviewer | mirrored torch's recipe; what is left is a 1-step difference between torch's and numpy's `exp`, which torch's own CPU and GPU builds also have between them |
| 11 | Arguments outside the model card's call: a negative temperature was accepted (the official code refuses it); `top_k` 0 or None raised (the official code leaves the step out) | sampling reviewer | both now behave as the official code (and see finding 16) |
| 12 | `Veena.generate_tokens` itself (stop tokens, the limit) was not tested against `generate` | sampling reviewer | gate 5b: 7 scripted runs against the real `generate` (stop as 1st, 3rd, 5th token, at and past the limit, no stop, other special tokens), all identical |
| 13 | The cache was re-converted and re-copied on every token (introduced with the fix for finding 4): per-token time rose from 57 ms at the start of a full utterance to 75 ms at its end | the MLX read of the code, by measurement | `KVCache`, allocated once and written in place; 57 ms throughout; the output unchanged bit for bit |
| 14 | A seed fixed the tokens but not the decoder's noise, so the waveform differed from run to run | the MLX read | the decoder's noise is drawn from an MLX key made from the seed; the speech gate checks the waveform repeats |
| 15 | WAV files were written by truncating `x * 32767`; soundfile, which the model card uses, does something else | reading the earlier ports' notes (ACE-Step fix 10) | `audio.save_wav`, soundfile's bytes exactly (gate 2b) |
| 16 | Outside the model card's call: a repetition penalty, temperature or top-p of `None`, which `generate` takes as "leave the step out", raised a TypeError; a penalty of 0 or less, a negative top-k and a negative top-p were not refused. SNAC's optional `latent_dim` setting was not read (snac_24khz does not set it) | the final side-by-side read | each step now runs, and its argument is checked, exactly when `generate` builds it. Not assumed: gate 5 takes the step list from a real `generate` call for 18 argument cases (None, 0, 1.0, negative, above 1) and runs it against ours, 18 of 18 the same. `latent_dim` is honoured when present |

Checked by the sampling reviewer and found the same on both sides: the processor list and why top-k is
50 in both transformers 4.52.4 (the version Veena was saved with) and 5.19.0; the repetition penalty bit
for bit on 40 awkward vectors; temperature and top-k bit for bit; the multinomial draw (chi-square on
400,000 draws); `max_new_tokens`; and the tokenizer: 20,085 prompts fuzzed (nukta, ZWJ, NBSP, emoji, all
Unicode normal forms, literal special-token strings), 0 differ.

## 7. Listening (2026-10-12)

`tests/listening_check.py`: Whisper large-v3-turbo transcribes every sample without seeing the text.

| Text | Ours | Official |
|---|---|---|
| नमस्ते, आप कैसे हैं? | every word | every word |
| मैं तो पूरा presentation prepare कर चुका हूं! | every word (the English words written in Devanagari) | every word |
| Binary search halves the list at every step. | "FiniteSearch halves the list at every step." | "Binary search has the list at every step." |

One slurred word each on the English sentence, a different one. Four more seeds of ours on that
sentence: "Binary search" right in all four, "has" for "halves" in two. So it is the draw, and the
"halves" slip is the model's own. The gate is therefore on the total (ours 22 words, official 23) and
on no sample losing more than a quarter of its words.

## 8. Deliberate differences from the official snippet

- **An unfinished last frame is dropped**; the snippet returns nothing when the audio-token count is not
  a multiple of 7.
- **Refused with an error** where the snippet would go on: an unknown voice name (it would be read as
  text), a text that spells out the pad token (finding 5), audio holding NaN or infinity.
- **Randomness**: numpy's generator for the tokens, MLX's for the decoder's noise. The same seed gives the
  same speech here, not the same speech as PyTorch.
- **The weights run as released, in bf16.** The model card's own example runs them 4-bit (bitsandbytes,
  CUDA only). `transformers` 5.19.0 loads a bf16 checkpoint as bf16 by default (checked on a small model).
- **Equal scores at the top-p cut** are resolved by a stable sort (finding 9).
- **Not ported**: batches of more than one utterance, padding and attention masks, the rotary variants
  Veena does not use, SNAC's encoder.

## 9. Speed and memory (idle machine, 2026-10-12)

- 17.0 to 17.7 tokens/s on the three test sentences over three runs; the official route (PyTorch, bf16,
  Apple GPU) 18.1 on the same sentences. 82 tokens are one second of speech: 4.7 to 4.9 s of work per
  second of speech (official 4.55).
- A full-length utterance (700 new tokens, 8.5 s of speech): 39.8 s; 56.6 ms per token at the start,
  57.4 ms at the end.
- Where a token's time goes: 97.5% is the GPU's products with the weights; choosing the token (numpy)
  0.7 ms, building the graph 0.5 ms, copying the logits out 0.2 ms.
- The floor: each token reads 28 x 100.7 M layer weights and the 482 M `lm_head`, 3.30 B numbers in
  bf16 = 6.6 GB. At the M5's roughly 153 GB/s that is 43 ms; we measure 57 ms, about three quarters of
  the limit. `x @ W.T` on the stored weight, MLX's own `nn.Linear`, a pre-transposed weight and float16
  all take the same time within 5% (measured on one layer), so the layout is not where time is lost.
- MLX's fused attention kernel is not used: it gave different numbers (logits off by 0.13) and was no faster.
- SNAC decodes 2.6 s of audio in 0.1 s.
- Memory: 7.6 GB of weights, 7.8 GB peak for the language model, 9.4 GB peak with the decoder.
- Nothing short of smaller weights makes it much faster. A 4-bit version would be a different, lossy model.

## 10. Not covered by the gates

- Sampled speech against an official sampled run, token for token: impossible, the generators differ.
  What stands in for it: float32 logits on the real weights, greedy streams, recorded distributions,
  the listening check.
- The official model in bf16 against ours in bf16 on the same sequence. Ours is measured against exact
  float32 instead.
- Full-length real speech in the speech gate: it keeps to three short sentences on purpose. A full-length
  sequence is covered in gate 4 (frames repeated up to 700 tokens).
- The seven voices that are in the tokenizer but not on the model card: their prompts are gated, their
  speech is not.
- Sentences longer than one call (8.5 s): splitting text is left to the caller, as officially.
- `transformers` 4.52.4, the version Veena was saved with: read for the sampling defaults, not run.

## 11. Problems met

1. **A setting in the real config that the small test models did not have** (finding 1). The lesson, again:
   read the real `config.json` in full, dicts included (the first look at it printed only the plain
   values and so hid `rope_scaling`), and always run the strict gate on the real weights, not only on
   small random models. The wrong version still produced well-formed audio tokens and plausible speech,
   so listening alone would not have caught it.
2. **Holding the whole model in float32 needs 15 GB per side.** Gate 4 therefore runs one layer at a
   time on both sides, with the same functions the model uses in real use.
3. **Sampling adds top-k 50 although nobody asked for it** (section 3 item 1): found by running the
   official `generate` and printing its processor list.
4. **Speed first looked terrible** (4.6 to 7 tokens/s) because another Veena job shared the GPU. On an
   idle machine ours and the official PyTorch route are level (section 9). A 3.8 B-parameter model in
   bf16 reads its weights once per token; no port changes that. Only smaller weights (4-bit, as the
   model card's own example uses) would, and that is a different, lossy model.
5. **The first guess at how soundfile rounds was wrong.** `round(x * 32768)` differs from soundfile in
   39% of samples; so does truncation. Six candidate rules were tried against soundfile's own output:
   only `round(x * 2^31)` as a 32-bit integer, then its top 16 bits, matched all 640,006 samples.
6. **A fix made the port slower** (finding 13). Moving the attention core to float32 (finding 4) was done
   by converting the whole cache on every token. Nothing failed, because the numbers were right; it
   showed only as per-token time growing along a full utterance. Timing a full-length utterance start to
   end is now part of checking any change to the loop.
7. **Parts of the earlier ports' notes had been read, not all of them.** The cache, the WAV writer, the
   listening check and the bad-number check were all in those notes (section 4). Read them whole, first.
