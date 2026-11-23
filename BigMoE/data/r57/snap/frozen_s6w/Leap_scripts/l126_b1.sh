#!/bin/sh
# L1-26 (fragment-native layout, test_qmm26.py, ~4 min, ~12 GB) + mend (c) overhead measurement (ovh27.py, ~1 min).
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
echo "== l126_b1 $(date +%H:%M:%S): $(sysctl -n vm.swapusage | cut -d'(' -f1)"
$PY scripts/test_qmm26.py || echo "== qmm26 FAILED"
$PY scripts/ovh27.py || echo "== ovh27 FAILED"
echo "== done $(date +%H:%M:%S): $(sysctl -n vm.swapusage | cut -d'(' -f1)"
