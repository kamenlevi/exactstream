#!/bin/sh
cd "$HOME/Leap" || exit 1
PY=$HOME/SpecDecode/.venv/bin/python
$PY scripts/mlx_opcost.py
$PY scripts/small35_opcount.py
