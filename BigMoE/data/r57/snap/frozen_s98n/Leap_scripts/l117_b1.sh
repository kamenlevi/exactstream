#!/bin/sh
# L1-17: fewer experts per token on the 125B (Fit12 Q2 inputs, real router + 4-bit experts). ~3 min.
cd "$HOME/Leap" || exit 1
$HOME/SpecDecode/.venv/bin/python scripts/topk17.py
