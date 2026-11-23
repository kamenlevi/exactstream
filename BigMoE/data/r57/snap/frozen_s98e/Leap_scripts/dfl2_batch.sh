#!/bin/sh
# Leap L2 batch (one quiet lock): GPU smoke of the drafter replay, the 125B feature capture, the full replay.
PY=$HOME/BigMoE/.venv/bin/python
cd $HOME
FREE=$(memory_pressure -Q 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
echo "== $(date +%H:%M:%S) memory free ${FREE}%"
[ -n "$FREE" ] && [ "$FREE" -lt 45 ] && { echo "== not enough free memory (need >= 45%), stopping"; exit 3; }
echo "== smoke $(date +%H:%M:%S)"
$PY ~/Leap/scripts/dfl2_eval.py --smoke || { echo "== smoke FAILED"; exit 1; }
echo "== extract $(date +%H:%M:%S)"
$PY ~/Leap/scripts/dfl2_extract.py || { echo "== extract FAILED"; exit 1; }
echo "== eval $(date +%H:%M:%S)"
$PY ~/Leap/scripts/dfl2_eval.py || { echo "== eval FAILED"; exit 1; }
echo "== done $(date +%H:%M:%S)"
