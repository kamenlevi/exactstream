#!/bin/sh
# L1-15 batch 1 (Leap, Sat 2026-10-24): MLX's matmul kernel choice per GDN projection shape, then the fused exact
# pieces (prework35, gated_norm35, fused in_proj) vs the eager mixer, bitwise + time. ~3 min, 35B in RAM (~21 GB).
cd "$HOME/Leap" || exit 1
PY=$HOME/SpecDecode/.venv/bin/python
$PY scripts/qmm_bands35.py
$PY scripts/test_gdn35.py
