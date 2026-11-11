#!/bin/sh
# L1-15 batch 5 (Leap, Sat 2026-10-24): the decisive test re-timed over valid configs (n tiles must divide N / 32: in_proj
# 386 tiles -> nt 1 or 2), bitwise spec == plain for the baseline and the candidate best, per-stage times. ~2 min.
cd "$HOME/Leap" || exit 1
PY=$HOME/SpecDecode/.venv/bin/python
MIX_CFGS='[{"ks_in": 1, "ks_out": 2},
           {"ks_in": 2, "ks_out": 2},
           {"ks_in": 2, "nt_in": 2, "ks_out": 4, "nt_out": 2, "rec": "gates", "r": 8},
           {"ks_in": 1, "nt_in": 2, "ks_out": 4, "nt_out": 2, "rec": "gates", "r": 8},
           {"ks_in": 2, "ks_out": 2, "rec": "gates", "r": 8},
           {"ks_in": 1, "nt_in": 2, "ks_out": 2, "rec": "gates", "r": 4},
           {"ks_in": 2, "nt_in": 2, "ks_out": 8, "rec": "gates", "r": 8}]' $PY scripts/test_mixer16.py
$PY scripts/test_qmm16c.py
