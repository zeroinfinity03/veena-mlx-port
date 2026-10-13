"""Say one sentence:

    uv run python -m veena_mlx "नमस्ते, आप कैसे हैं?" --speaker kavya --name hello

Writes a 24 kHz 16-bit WAV to samples/<name>.wav (or to --out). One call speaks at most 8.5 seconds (the official limit of 700 audio tokens);
longer text has to be given one sentence at a time.
"""
import argparse
import sys
import time
from pathlib import Path

from . import SAMPLE_RATE, SPEAKERS, Veena, save_wav

SPEECH_DIR = Path(__file__).resolve().parents[1] / "samples"      # <repo>/samples, where generated speech goes


def main():
    p = argparse.ArgumentParser(prog="veena_mlx", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text")
    p.add_argument("--speaker", default="kavya", help=f"documented voices: {', '.join(SPEAKERS)}")
    p.add_argument("--name", default="out", help="written to samples/<name>.wav")
    p.add_argument("--out", help="write to this path instead")
    p.add_argument("--seed", type=int, help="the same seed gives the same speech again")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top-p", type=float, default=0.9)
    a = p.parse_args()
    if not a.text.strip():
        sys.exit("Nothing to say: the text is empty.")
    if a.seed is not None and a.seed < 0:
        sys.exit("The seed must be 0 or more.")

    out = Path(a.out) if a.out else SPEECH_DIR / f"{a.name}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    veena = Veena()
    speech = veena.speak(a.text, speaker=a.speaker, temperature=a.temperature, top_p=a.top_p, seed=a.seed)
    save_wav(out, speech.audio)
    seconds = len(speech.audio) / SAMPLE_RATE
    print(f"{out}: {seconds:.2f} s of speech, {len(speech.tokens)} tokens, made in {time.time() - t0:.1f} s")
    if not speech.stopped:
        print("The model reached the token limit before finishing: the audio is cut. Give it shorter text.")


if __name__ == "__main__":
    main()
