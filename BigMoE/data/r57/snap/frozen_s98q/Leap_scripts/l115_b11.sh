#!/bin/sh
# L1-15 batch 11: loop phases fully serialized (drafter, rollback, target each finished and timed alone). ~3 min.
cd "$HOME/Leap" || exit 1
PY=$HOME/SpecDecode/.venv/bin/python
PROF_MODE=3 $PY scripts/small35_prof.py
