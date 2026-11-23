#!/bin/sh
# L1-24 screen + L1-23 leftovers (Leap, Sat 2026-10-24): routed-expert paths at 4 / 8 / 16 rows on real 125B weights
# (exp24.py, ~7 GB), then bf16 formulations for the 16-row check (bf23.py). ~5 min.
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
echo "== L1-24 b1 $(date +%H:%M:%S): $(sysctl -n vm.swapusage | cut -d'(' -f1)"
$PY scripts/exp24.py || echo "== exp24 FAILED"
$PY scripts/bf23.py || echo "== bf23 FAILED"
echo "== done $(date +%H:%M:%S): $(sysctl -n vm.swapusage | cut -d'(' -f1)"
