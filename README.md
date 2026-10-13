# veena-mlx-port

[Veena](https://huggingface.co/maya-research/Veena) (Maya Research's Hindi and English text to speech) on
Apple silicon with [MLX](https://github.com/ml-explore/mlx): the Llama-style model that writes audio codes and
the SNAC codec that turns them into 24 kHz speech, ported from the official route (Hugging Face
`transformers` plus the `snac` package) and measured against it. No PyTorch at run time.

- **Same output as the official route**, gate by gate: tokenizer ids identical on Hindi, English and mixed text;
  the real weights in float32 within 4.4e-6 of the official logits with the same best token at all 893
  positions tested; sampling with the same surviving tokens and probabilities; SNAC within 1e-5. Details:
  [`NOTES.md`](NOTES.md) and [`PORT_README.md`](PORT_README.md) (the port's own readme, with its status).
- **Speed:** about 17.5 tokens a second on an Apple M5, level with the official PyTorch route on the same GPU;
  both are limited by reading 6.6 GB of weights per token. About 5 s of work per second of speech.
- **Voices:** `kavya`, `agastya`, `maitri`, `vinaya`. One call speaks at most 8.5 s (the official limit).

```bash
uv sync
uv run python -m veena_mlx "यहाँ ten numbers की एक list है।" --speaker kavya --name hello   # -> samples/hello.wav
```

Weights are not in this repository. Put Veena's `config.json`, `tokenizer.json`, both
`model-0000x-of-00002.safetensors` files and the SNAC files made by `export_weights.py` in `models/veena/`.
What the model takes and its limits: [`INPUTS.md`](INPUTS.md).

`examples/hindi_narration.py` speaks a long script of short beats: one sentence per call, and a fixed voice
anchor (the voice's first sentence, text and audio codes, in front of every prompt), because on its own Veena
drifts to a different-sounding speaker from call to call.

The test suite that compared each stage with the official PyTorch code is not included here (it needs the
official weights and PyTorch); its results are recorded in `NOTES.md`.

Apache License 2.0; see `LICENSE` and `NOTICE`.
