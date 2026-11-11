#!/bin/sh
# L1-21 batch 1 (Leap, Sat 2026-10-24; pre-registered in ~/Leap/TABLE_350.md): smoke test, then the picker over the
# L1-20 drafter's top-64 lattice at blocks 16 and 32, then the original drafter at block 16 as the clean control.
# Then the L1-22 screen (trees over the lattice, priced; scripts/tree22.py).
# Each pick21 run: lattices on L1-20's windows -> train a ~5 M picker (local) -> held-out HumanEval replay, priced. ~25 min, < 10 GB.
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
S=$HOME/Leap/scripts
echo "== L1-21 b1 $(date +%H:%M:%S): $(sysctl -n vm.swapusage | cut -d'(' -f1)"
P21_SMOKE=1 P21_BS=16,32 P21_WEIGHTS=data/l120_ft.safetensors P21_TAG=_smoke $PY $S/pick21.py || { echo "== smoke FAILED"; exit 1; }
echo "== L1-20 drafter $(date +%H:%M:%S)"
P21_BS=16,32 P21_WEIGHTS=data/l120_ft.safetensors P21_TAG=_ft $PY $S/pick21.py || { echo "== FT run FAILED"; exit 1; }
echo "== control: original drafter $(date +%H:%M:%S)"
P21_BS=16 P21_TAG=_orig $PY $S/pick21.py || echo "== control FAILED"
echo "== L1-22 screen: trees over the lattice, priced $(date +%H:%M:%S)"
R22_WEIGHTS=data/l120_ft.safetensors $PY $S/tree22.py || echo "== tree22 FAILED"
echo "== done $(date +%H:%M:%S): $(sysctl -n vm.swapusage | cut -d'(' -f1)"
