#!/bin/sh
# Leap Sat 2026-10-24: the pruned-125B miss estimate (prune26.py, CPU, ~1 min; orchestrator's question), then L1-25
# (test_qmmg25.py: experts on the tensor units, real weights, ~3 min, ~7 GB).
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
echo "== l125_b1 $(date +%H:%M:%S): $(sysctl -n vm.swapusage | cut -d'(' -f1)"
$PY scripts/prune26.py || echo "== prune26 FAILED"
$PY scripts/test_qmmg25.py || echo "== qmmg25 FAILED"
echo "== done $(date +%H:%M:%S): $(sysctl -n vm.swapusage | cut -d'(' -f1)"
