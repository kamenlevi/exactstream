#!/bin/sh
# L1-15 batch 6 (Leap, Sat 2026-10-24): Python graph building vs evaluation for 30 GDN mixers, stock vs the kernel
# program (best config of batch 5), and both under mx.compile; then the 35B DFlash loop end to end with the GDN kernel
# program (12 code prompts x 6 runs). ~5 min, 35B in RAM (~21 GB).
cd "$HOME/Leap" || exit 1
PY=$HOME/SpecDecode/.venv/bin/python
MIX_SPLIT=1 MIX_CFGS='[{"ks_in": 2, "nt_in": 2, "ks_out": 4, "nt_out": 2, "rec": "gates", "r": 8}]' $PY scripts/test_mixer16.py
$PY scripts/small35_gdnk.py
