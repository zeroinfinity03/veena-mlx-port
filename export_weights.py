"""Official files -> mlx-port/models/, the folder our MLX code reads.

Run once, with the tests environment (it has PyTorch):
    cd Veena && tests/.venv/bin/python mlx-port/export/export_weights.py

Veena:  model-0000{1,2}-of-00002.safetensors, config.json and tokenizer.json are copied as they are
        (APFS clones: no extra disk space). They are safetensors already, with the official names.
SNAC:   pytorch_model.bin is a pickle, which MLX cannot read. Its tensors are written to
        snac_24khz.safetensors under the same names, with the same numbers; config -> snac_config.json.

Nothing is converted: layouts and weight norm are handled where the model loads (veena_mlx/snac.py).
"""
import hashlib
import shutil
import subprocess
from pathlib import Path

import torch
from safetensors.numpy import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "pytorch"
DST = ROOT / "mlx-port" / "models"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


DST.mkdir(parents=True, exist_ok=True)
for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "config.json", "tokenizer.json"):
    src, dst = SRC / "Veena" / name, DST / name
    if not dst.exists():
        if subprocess.run(["cp", "-c", str(src), str(dst)]).returncode != 0:     # clone; plain copy off APFS
            shutil.copyfile(src, dst)
    assert sha256(src) == sha256(dst), name
    print(f"Veena  {name}: in place, identical to the official file ({dst.stat().st_size / 1e6:.1f} MB)")

sd = torch.load(SRC / "snac_24khz" / "pytorch_model.bin", map_location="cpu", weights_only=True)
weights = {k: v.numpy() for k, v in sd.items()}
save_file(weights, str(DST / "snac_24khz.safetensors"))
shutil.copyfile(SRC / "snac_24khz" / "config.json", DST / "snac_config.json")
back = load_file(str(DST / "snac_24khz.safetensors"))
assert set(back) == set(weights) and all(back[k].dtype == weights[k].dtype and back[k].tobytes() == weights[k].tobytes() for k in weights)
print(f"SNAC   {len(weights)} tensors, {sum(v.size for v in weights.values()):,} numbers, "
      f"types {sorted({str(v.dtype) for v in weights.values()})}: identical after reading back")
