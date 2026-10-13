# Model Inputs: Veena (MLX)

What the model takes, from the official route: the snippet on Maya Research's model card
(`repo/model-card/README.md` 62-204, function `generate_speech`) and the files released with the
weights (`config.json`, `generation_config.json`, `tokenizer.json`). The tables mark what our MLX port
(`mlx-port/veena_mlx/`, run with `python -m veena_mlx`) does with each one. Nothing here is a guess:
every value is in those files or was measured by running the official code (`mlx-port/NOTES.md`).

## Speak (text → speech)

| Input | Range / default | What it does | MLX port |
|---|---|---|---|
| `text` | required | what is said. Hindi, English, or both in one sentence | yes |
| `speaker` | `kavya` (default), `agastya`, `maitri`, `vinaya` | the voice. It goes into the prompt as one token, `<spk_NAME>` | yes; an unknown name is refused |
| `temperature` | 0.4 | how much the voice varies from the most likely reading. Must be above 0 | yes, same default |
| `top_p` | 0.9 | only the likeliest tokens that together make up 90% can be drawn | yes, same default |
| `seed` | random | the official snippet has none (PyTorch's global generator decides) | added: the same seed gives the same speech again, tokens and waveform |

## Fixed by the official snippet (not inputs)

| Setting | Value | Where it comes from |
|---|---|---|
| repetition penalty | 1.05, over the prompt and everything written so far | model card 131 |
| top-k | 50 | nowhere in the card or the config: a default inside `transformers` that the official call gets all the same (measured) |
| sampling | on (`do_sample=True`) | model card 128 |
| length limit | `min(int(len(text) * 1.3) * 7 + 21, 700)` new tokens | model card 121 |
| stop | token 128258 (end of speech) or 128262 (end of AI) | model card 133 |
| prompt | `[128259] <spk_NAME> text [128260, 128261, 128257]` | model card 106-116 |
| output | 24,000 Hz, one channel; the card writes it with `sf.write` (16-bit WAV) | model card 186, 193 |

`generation_config.json` in the weights says temperature 0.6. The snippet passes 0.4 itself, so 0.4 is
what the official route runs, and what the port defaults to.

## Length: one call is at most 8.5 seconds

- 7 tokens are one frame of sound, 85.3 ms. The limit of 700 tokens is 100 frames, 8.53 s.
- The limit grows with the text: 1.3 frames per character plus 3. A text of 75 characters or more
  gets the full 700.
- The model normally stops by itself well before the limit. Measured: 20 characters of Hindi, limit
  203 tokens, stopped at 155 (1.88 s); 44 characters of English, limit 420, stopped at 197 (2.39 s).
- If it runs into the limit the speech is cut mid-word. The port reports this (`speech.stopped` is
  False, and the command prints a warning). The official snippet says nothing.
- Longer text has to be given one sentence at a time and the pieces joined. Neither the official
  snippet nor the port splits text.

## Voices

The model card documents four: `kavya`, `agastya`, `maitri`, `vinaya` ("4 professional voice artists").
The tokenizer also holds `<spk_apsara>`, `<spk_charu>`, `<spk_ishana>`, `<spk_kyra>`, `<spk_mohini>`,
`<spk_varun>`, `<spk_soumya>`. The port accepts them (their prompts are tested); how they sound is not
documented by Maya Research and not checked by us.

## Text

- Languages: "Currently supports only Hindi and English. Performance on other Indian languages is not
  guaranteed" (model card, limitations). Mixed sentences are supported; the card's own example writes
  the Hindi in Devanagari and the English words in Latin letters in one sentence.
- The card's three examples, one per kind, are in `tests/gates_speech.py` and `tests/texts.py`.
- The card gives no further writing advice (no tags for emotion or pauses: "Emotion and prosody control
  tokens" are listed under future updates).
- Refused by the port: empty text (command line), and text that spells out `<custom_token_7>`, the
  model's padding token.

## Official features the port does not have

| Feature | Official | MLX port |
|---|---|---|
| 4-bit loading (`BitsAndBytesConfig`, NF4) | the card's example; needs an NVIDIA GPU | no. The port runs the weights as released (bf16), which is the model at full quality |
| several sentences in one batch | possible with `generate` | no, one sentence per call |
| other `generate` options (beams, min length, ...) | possible, not used by the snippet | no |

## Speed on this Mac (M5, 24 GB)

About 17 tokens a second, so a little under 5 seconds of work for each second of speech, the same as
the official PyTorch route here. Memory while it runs: 9.4 GB. One generation at a time.
