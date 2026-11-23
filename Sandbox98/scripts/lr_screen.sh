#!/bin/bash
# lr screen for the base drafter (written 18:4x, before it ran): native head, 80 TTT steps on the same base slices
# (seed 1114, cosine over 80), lr 1e-5 / 3e-5 / 1e-4. Each step's loss is on a slice new to the arm (prequential).
# result -> action: the arm with the lowest mean loss over steps 41-80 sets the base run's lr; if 1e-4 wins by > 5 %,
# the base uses 1e-4 (else the smaller winner, to stay near t30b's 1e-5 regime).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/BigMoE/.venv/bin/python}
for LR in 1e-5 3e-5 1e-4; do
  $PY sb/train.py --run lrs_$LR --start native --files base --steps 80 --rows 96 --lr $LR --seed 1114 > logs/lrs_$LR.log 2>&1
done
$PY - <<'EOF'
import json, numpy as np
r = {}
for lr in ("1e-5", "3e-5", "1e-4"):
    s = json.load(open(f"ckpt/lrs_{lr}.json"))
    r[lr] = (float(np.mean(s["loss"][:20])), float(np.mean(s["loss"][40:80])))
    print(f"lr {lr}: loss steps 1-20 {r[lr][0]:.4f}, steps 41-80 {r[lr][1]:.4f}")
json.dump(r, open("results/lr_screen.json", "w"), indent=1)
EOF
