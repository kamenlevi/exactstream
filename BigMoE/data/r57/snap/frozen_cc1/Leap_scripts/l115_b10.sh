#!/bin/sh
# L1-15 batch 10 (Leap, Sat 2026-10-24): the code-MMA matmul (qmm16c: fp32-exact dequantization, accuracy = MLX's qmv)
# in the kernel program: spec == plain bitwise + timing; the loop's phase times with the drafter finished before the
# target is built (clean target timing), stock vs kernels; per-prompt acceptance. ~6 min, 35B in RAM (~21 GB).
cd "$HOME/Leap" || exit 1
PY=$HOME/SpecDecode/.venv/bin/python
MIX_CFGS='[{"ks_in": 2, "nt_in": 2, "ks_out": 4, "nt_out": 2, "rec": "gates", "r": 8, "mm": "c"},
           {"ks_in": 2, "nt_in": 2, "ks_out": 4, "nt_out": 2, "rec": "gates", "r": 8},
           {"ks_in": 2, "nt_in": 2, "ks_out": 8, "rec": "gates", "r": 8, "mm": "c"}]' $PY scripts/test_mixer16.py
PROF_MODE=2 $PY scripts/small35_prof.py
PROF_MODE=1 $PY scripts/small35_prof.py
$PY scripts/small35_acc.py
