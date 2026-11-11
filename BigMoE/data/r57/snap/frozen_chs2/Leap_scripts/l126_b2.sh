#!/bin/sh
# L1-26 rerun after the contiguity fix (strided scales / biases were copied on every call: a fixed ~1.5 ms per layer).
cd "$HOME/Leap" || exit 1
echo "== l126_b2 $(date +%H:%M:%S)"
$HOME/BigMoE/.venv/bin/python scripts/test_qmm26.py || echo "== qmm26 FAILED"
echo "== done $(date +%H:%M:%S)"
