"""Long Hindi narration with Veena on veena_mlx: a script of short beats in, narration.wav and timings.json out.

    uv run python examples/hindi_narration.py script.md                         # writes next to the script
    uv run python examples/hindi_narration.py script.md --voice agastya --out samples/x.wav

The script is markdown with numbered beats (`N. **Say:** …`, `**Say (key):** …`), and a voice line
"Voice: `kavya`." (kavya, agastya, maitri or vinaya; `--voice` overrides it). The Say text is Hindi in
Devanagari with every English word, numbers included, in English letters: "यहाँ ten numbers की एक list है।"

Veena speaks at most 8.5 s per call, so every beat is spoken one sentence at a time and joined. On its
own it also drifts to a different-sounding speaker from call to call, so every call carries the same
anchor: the voice's first sentence (its text and its audio codes) in front of the prompt, as the voice's
previous line. Pauses: 0.3 s lead-in, 0.6 s after a beat, 1.2 s after a key beat, 0.8 s tail. Veena gives no word times: timings.json has each
beat's start and end, and an empty word list.

It needs about 10 GB while it runs and about 5 s of work per second of speech: run one at a time.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repo root, so veena_mlx imports
import mlx.core as mx
import numpy as np

from veena_mlx import SAMPLE_RATE, SPEAKERS, Veena, save_wav
from veena_mlx.model import AUDIO_CODE_BASE_OFFSET, CODEBOOK, END_OF_AI_TOKEN, END_OF_SPEECH_TOKEN, FRAME

LEAD_IN_S, BEAT_PAUSE_S, KEY_IDEA_PAUSE_S, TAIL_S = 0.3, 0.6, 1.2, 0.8
SENTENCE_GAP_S = 0.25                                                       # between sentences of one beat
SAY = re.compile(r"^\s*\d+\.\s+\*\*Say( \(key\))?:\*\*\s*(.+?)\s*$")
VOICE = re.compile(r"^Voice: `([a-z]+)`")


def read_script(path):
    voice, beats = None, []
    for line in path.read_text().splitlines():
        if m := VOICE.match(line):
            voice = m.group(1)
        elif m := SAY.match(line):
            beats.append({"say": m.group(2), "key": bool(m.group(1))})
    if not beats:
        sys.exit(f"hindi.py: no '**Say:**' lines in {path}")
    return voice, beats


def sentences(text):
    return [s for s in re.split(r"(?<=[।!?.])\s+", text.strip()) if s]


def silence(seconds):
    return np.zeros(int(round(SAMPLE_RATE * seconds)), dtype=np.float32)


def pitch_hz(audio):
    """Median pitch of the voiced parts, by autocorrelation: a quick check that the speaker stays the same."""
    n, vals = int(SAMPLE_RATE * 0.04), []
    for i in range(0, len(audio) - n, n):
        w = audio[i:i + n] - audio[i:i + n].mean()
        if np.sqrt(np.mean(w ** 2)) < 0.02:
            continue
        c = np.correlate(w, w, "full")[n - 1:]
        lo, hi = SAMPLE_RATE // 400, SAMPLE_RATE // 70
        k = lo + int(np.argmax(c[lo:hi]))
        if c[k] > 0.3 * c[0]:
            vals.append(SAMPLE_RATE / k)
    return float(np.median(vals)) if vals else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("script")
    p.add_argument("--voice", choices=SPEAKERS)
    p.add_argument("--out", help="the WAV to write (default: narration.wav next to the script); timings go beside it")
    a = p.parse_args()
    script = Path(a.script).resolve()
    voice, beats = read_script(script)
    voice = a.voice or voice or "kavya"
    if voice not in SPEAKERS:
        sys.exit(f"hindi.py: {voice} isn't a Veena voice ({', '.join(SPEAKERS)}).")
    out = Path(a.out).resolve() if a.out else script.with_name("narration.wav")
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    veena = Veena()

    def say(text, prefix, seed):
        new = veena.generate_tokens(prefix + veena.prompt_ids(text, voice), veena.max_new_tokens(text), seed=seed)
        audio, _ = veena.decode(new, key=mx.random.key(seed))
        mx.clear_cache()
        return audio, new, new[-1] in (END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN)

    def spoken(text, seed):
        audio, _, finished = say(text, anchor, seed)
        if finished:
            return audio, 0
        halves = re.split(r"(?<=,)\s+", text, maxsplit=1)          # ran into the 8.5 s limit: say it in two halves
        if len(halves) < 2:
            return audio, 1
        first, _, _ = say(halves[0], anchor, seed + 500)
        second, _, _ = say(halves[1], anchor, seed + 700)
        return np.concatenate([first, silence(0.15), second]), 1

    anchor_text = sentences(beats[0]["say"])[0]
    anchor_audio, anchor_new, _ = say(anchor_text, [], seed=1)
    codes = [t for t in anchor_new if AUDIO_CODE_BASE_OFFSET <= t < AUDIO_CODE_BASE_OFFSET + FRAME * CODEBOOK]
    anchor = veena.prompt_ids(anchor_text, voice) + codes + [END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN]

    parts, position, timed, cut, pitches = [silence(LEAD_IN_S)], LEAD_IN_S, [], 0, []
    for n, beat in enumerate(beats, 1):
        start, beat_parts = position, []
        for k, sentence in enumerate(sentences(beat["say"])):
            if k:
                beat_parts.append(silence(SENTENCE_GAP_S))
            if n == 1 and k == 0:
                audio = anchor_audio
            else:
                audio, c = spoken(sentence, 1000 * n + k)
                cut += c
            beat_parts.append(audio)
        beat_audio = np.concatenate(beat_parts)
        pitches.append(pitch_hz(beat_audio))
        parts.append(beat_audio)
        position += len(beat_audio) / SAMPLE_RATE
        timed.append({"index": n - 1, "say": beat["say"], "key": beat["key"], "start": round(start, 3), "end": round(position, 3), "words": []})
        pause = KEY_IDEA_PAUSE_S if beat["key"] else BEAT_PAUSE_S
        parts.append(silence(pause))
        position += pause
        if n % 10 == 0:
            print(f"{voice}: beat {n}/{len(beats)}, {time.time() - t0:.0f} s", flush=True)
    parts.append(silence(TAIL_S))
    audio = np.concatenate(parts)
    save_wav(out, audio)
    timings = {"voice": voice, "engine": "Veena (MLX)", "duration": round(len(audio) / SAMPLE_RATE, 3), "beats": timed}
    stem = out.with_suffix("")
    timings_path = script.with_name("timings.json") if not a.out else Path(f"{stem}.timings.json")
    timings_path.write_text(json.dumps(timings, indent=1, ensure_ascii=False) + "\n")
    if not a.out:
        script.with_name("timings.js").write_text("window.TIMINGS = " + json.dumps(timings, ensure_ascii=False) + ";\n")
    p_ok = [x for x in pitches if x == x]
    spread = max(p_ok) - min(p_ok) if p_ok else float("nan")
    print(f"{out.name}: {timings['duration']} s, {len(beats)} beats, made in {(time.time() - t0) / 60:.1f} min; "
          f"sentences that hit the 8.5 s limit: {cut}; pitch per beat {min(p_ok):.0f}-{max(p_ok):.0f} Hz (spread {spread:.0f} Hz)")
    if spread > 90:
        print("Warning: the pitch moves a lot between beats; listen for a change of speaker.")


if __name__ == "__main__":
    main()
