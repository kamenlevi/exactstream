#!/bin/sh
# L1-15 batch 7 (Leap, Sat 2026-10-24): prework / gated norm now take the row count at run time (one compiled kernel for
# every S): re-check spec == plain bitwise + timing, then the DFlash loop's phase timers, stock vs GDN kernels. ~5 min.
cd "$HOME/Leap" || exit 1
PY=$HOME/SpecDecode/.venv/bin/python
MIX_CFGS='[{"ks_in": 2, "nt_in": 2, "ks_out": 4, "nt_out": 2, "rec": "gates", "r": 8},
           {"ks_in": 2, "ks_out": 2},
           {"ks_in": 2, "nt_in": 2, "ks_out": 4, "nt_out": 2, "rec": "gates", "r": 8}]' $PY scripts/test_mixer16.py
$PY scripts/small35_prof.py
