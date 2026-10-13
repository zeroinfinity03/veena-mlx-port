"""Veena (Maya Research) Hindi and English text to speech on Apple MLX: our own port.

    from veena_mlx import Veena
    veena = Veena()
    speech = veena.speak("नमस्ते, आप कैसे हैं?", speaker="kavya")
    speech.audio      # numpy float32, 24,000 Hz
"""
import os

# On Apple's newest GPUs MLX runs 32-bit matrix products at reduced (TF32) precision unless told
# otherwise. The attention sums and the audio decoder are 32-bit; this keeps them exact. It has to be set
# before MLX does its first product. (The bf16 products of the language model are not affected either way.)
os.environ.setdefault("MLX_ENABLE_TF32", "0")

from .audio import save_wav  # noqa: E402
from .model import SAMPLE_RATE, SPEAKERS, Speech, Veena  # noqa: E402

__all__ = ["Veena", "Speech", "SPEAKERS", "SAMPLE_RATE", "save_wav"]
