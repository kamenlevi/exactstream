#!/bin/sh
# Leap Sat 2026-10-24: once L1-20's training is done (state "done"), run the row-cost + lattice screen on the fine-tuned
# drafter inside the same lock window (batch 2 is still scoring; both are replays, no timing).
cd "$HOME/Leap" || exit 1
until grep -q '"status": "done"' data/l120_ft.state.json 2>/dev/null; do sleep 20; done
while pgrep -f "scripts/cand21.py" >/dev/null && [ -z "$R21_SKIPWAIT" ]; do sleep 10; done
echo "== cand21 FT $(date +%H:%M:%S): $(sysctl -n vm.swapusage | cut -d'(' -f1)"
R21_WEIGHTS=data/l120_ft.safetensors R21_TAG=_l120 $HOME/BigMoE/.venv/bin/python scripts/cand21.py
echo "== done $(date +%H:%M:%S): $(sysctl -n vm.swapusage | cut -d'(' -f1)"
