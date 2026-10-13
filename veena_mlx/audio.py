"""Writing the waveform to a file the way the official snippet does.

The model card ends with `sf.write("output.wav", audio, 24000)`. soundfile (libsndfile 1.2.2) turns a
float sample into 16-bit PCM in two steps: to a 32-bit integer, round(x * 2^31) clipped to the int32
range, and then the top 16 bits of that. This was found by experiment, not assumed (rounding x * 32768
is NOT it: that differs in 39% of samples), and tests/gates_code.py compares our file with soundfile's.
"""
import wave

import numpy as np


def pcm16(audio):
    """float samples -> int16, bit for bit as soundfile writes them."""
    x = np.asarray(audio, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("The audio holds values that are not numbers")
    i32 = np.clip(np.rint(x.astype(np.float64) * 2147483648.0), -2147483648, 2147483647).astype(np.int64)
    return (i32 >> 16).astype("<i2")


def save_wav(path, audio, sample_rate=24000):
    """One channel, 16-bit PCM."""
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm16(audio).tobytes())
