#!/bin/bash
# Chain (no agent turn between steps): split -> lr screen -> base drafter s30 + the gate (gate.sh) at the screen's lr.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/BigMoE/.venv/bin/python}
$PY sb/mksplit.py
bash scripts/lr_screen.sh
LR=$($PY - <<'PYEOF'
import json
r = json.load(open("results/lr_screen.json"))
late = {lr: v[1] for lr, v in r.items()}
best = min(late, key=late.get)
# pre-registered: 1e-4 only if it wins by > 5 % over the best smaller lr
small = min((v, lr) for lr, v in late.items() if lr != "1e-4")
pick = "1e-4" if best == "1e-4" and late["1e-4"] < 0.95 * small[0] else small[1]
print(pick)
PYEOF
)
echo "lr screen -> base lr $LR" | tee logs/lr_pick.log
bash gate.sh $LR
