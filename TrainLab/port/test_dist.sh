#!/bin/bash
# PORT multi-process test (MTP Head, Wed 11-04 20:4x): the same 3 steps (S 2: two micro-batches per step) in one
# process, in 2 processes with the gradient all-reduce (--dist ddp), and in 2 processes with Adam's moments sharded
# (--dist zero1, torch ZeroRedundancyOptimizer). torch.distributed with gloo on CPU (Kaggle: nccl on 2 x T4, same code).
# Synthetic model and data. PASS = per-step losses and final weights equal the single process's.
set -e
cd "$(dirname "$0")"
S=${PORT_TMP:-$(mktemp -d)}; rm -rf "$S"; mkdir -p "$S"
PY=${PY:-$HOME/TrainLab/.venv-torch/bin/python}
ARGS=(--synthetic --S 2 --A 8 --bs 8 --micro 1 --pmask 0.5 --epochs 2 --no-val --max-steps 3 "$@")
$PY -I train_port.py "${ARGS[@]}" --threads 4 --log-json "$S/l0.json" --out "$S/w0.safetensors" | tail -1
for d in ddp zero1; do
  $PY -m torch.distributed.run --nproc_per_node 2 --master_port 29531 train_port.py "${ARGS[@]}" --threads 2 --dist $d \
      --ckpt-dir "$S/ck_$d" --log-json "$S/l_$d.json" --out "$S/w_$d.safetensors" 2>&1 | grep -E "^\[port\] (DONE|step 3)" || true
done
$PY -I - "$S" <<'PYEOF'
import json, sys, torch
from safetensors.torch import load_file
S = sys.argv[1]
l0, w0 = json.load(open(f"{S}/l0.json"))["loss"], load_file(f"{S}/w0.safetensors")
ok = True
for d in ("ddp", "zero1"):
    l, w = json.load(open(f"{S}/l_{d}.json"))["loss"], load_file(f"{S}/w_{d}.safetensors")
    dl = max(abs(a - b) for a, b in zip(l0, l)); dw = max(float((w0[k].float() - w[k].float()).abs().max()) for k in w0)
    same = l == l0 and all(torch.equal(w0[k], w[k]) for k in w0)
    ok &= dl < 1e-6 and dw < 1e-2
    print(f"{d}: losses {l} | max |loss diff| {dl:.2e}, max |weight diff| {dw:.2e}, bitwise {'yes' if same else 'no'}")
print("DIST TEST:", "PASS" if ok else "FAIL", "(single", l0, ")")
PYEOF
