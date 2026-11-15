#!/bin/sh
# L1-15 batch 4 (Leap, Sat 2026-10-24): exhaustive bf16 op checks, qmm16 NT x KS sweep, the gate-fused recurrence,
# prework / mixer vs stock (sigmoid fix), then the decisive test with the new configs. ~4 min, 35B in RAM (~21 GB).
cd "$HOME/Leap" || exit 1
PY=$HOME/SpecDecode/.venv/bin/python
$PY scripts/exact_ops_probe.py
$PY scripts/test_qmm16_nt.py
$PY scripts/test_recg.py
$PY scripts/test_gdn35.py
$PY scripts/test_mixer16.py
