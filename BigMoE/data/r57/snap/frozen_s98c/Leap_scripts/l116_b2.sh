#!/bin/sh
# L1-16 batch 2 (Leap, Sat 2026-10-24): output error at equal bits on Fit12's Q2 inputs (layers 2/12/24/36/46): MSB-3,
# RTN-2, shared input / output subspace + 4- or 2-bit coefficients, per-expert activation-aware low rank. ~5 min.
cd "$HOME/Leap" || exit 1
$HOME/SpecDecode/.venv/bin/python scripts/act16.py
