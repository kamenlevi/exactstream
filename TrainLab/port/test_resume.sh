#!/bin/bash
# PORT resume test (MTP Head, Wed 11-04 20:3x): 4 steps straight vs 2 steps, stop, a NEW process resumes from the
# checkpoint, 2 more steps. Synthetic model and data (port/dflash2_torch.py build_synthetic), CPU. PASS = the per-step
# losses and the final weights are bitwise identical. Usage: bash test_resume.sh [extra train_port.py flags]
set -e
cd "$(dirname "$0")"
S=${PORT_TMP:-$(mktemp -d)}; rm -rf "$S"; mkdir -p "$S"
PY=${PY:-$HOME/TrainLab/.venv-torch/bin/python}
ARGS=(--synthetic --S 2 --A 8 --bs 8 --micro 1 --pmask 0.5 --epochs 2 --no-val --threads 4 "$@")
$PY -I train_port.py "${ARGS[@]}" --max-steps 4 --ckpt-dir "$S/x1" --log-json "$S/l1.json" --out "$S/w1.safetensors" | tail -2
$PY -I train_port.py "${ARGS[@]}" --max-steps 2 --ckpt-dir "$S/x2" | tail -1
$PY -I train_port.py "${ARGS[@]}" --max-steps 2 --ckpt-dir "$S/x2" --log-json "$S/l2.json" --out "$S/w2.safetensors" | grep -E "RESUME|DONE"
$PY -I - "$S" <<'PYEOF'
import json, sys, torch
from safetensors.torch import load_file
S = sys.argv[1]
a, b = json.load(open(f"{S}/l1.json")), json.load(open(f"{S}/l2.json"))
print("losses straight", a["loss"]); print("losses resumed ", b["loss"])
wa, wb = load_file(f"{S}/w1.safetensors"), load_file(f"{S}/w2.safetensors")
ok_l, ok_w = a["loss"] == b["loss"], all(torch.equal(wa[k], wb[k]) for k in wa) and wa.keys() == wb.keys()
print(f"RESUME TEST: losses {'identical' if ok_l else 'DIFFER'}, weights {'identical' if ok_w else 'DIFFER'} "
      f"({len(wa)} tensors) -> {'PASS' if ok_l and ok_w else 'FAIL'}")
PYEOF
