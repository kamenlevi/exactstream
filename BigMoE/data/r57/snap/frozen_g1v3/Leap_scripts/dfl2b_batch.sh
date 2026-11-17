#!/bin/sh
# Leap L2b batch (one quiet lock): agentic DFlash2 with longer drafter context (DFL2_PREREG.md, L2b final design).
# Full-context features of the 12 agentic turns, then the window sweep, NTK-aware RoPE and the hybrid bound.
PY=$HOME/BigMoE/.venv/bin/python
E=$HOME/Leap/scripts/dfl2_eval.py
FULL=$HOME/Leap/data/dfl2full
cd $HOME
FREE=$(memory_pressure -Q 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
echo "== L2b $(date +%H:%M:%S) memory free ${FREE}%"
[ -n "$FREE" ] && [ "$FREE" -lt 45 ] && { echo "== not enough free memory (need >= 45%), stopping"; exit 3; }
echo "== smoke (window 16384, NTK x8) $(date +%H:%M:%S)"
$PY $E --smoke --window 16384 --ntk 8 || { echo "== smoke FAILED"; exit 1; }
echo "== extract, all context rows $(date +%H:%M:%S)"
DFL2_ONLY=agent DFL2_KEEP=100000 DFL2_OUT=$FULL $PY ~/Leap/scripts/dfl2_extract.py || { echo "== extract FAILED"; exit 1; }
for W in 2048 256 512 1024 4096 8192 16384; do
  echo "== eval window $W $(date +%H:%M:%S)"
  $PY $E --dir $FULL --window $W --tag _w$W || { echo "== eval window $W FAILED"; exit 1; }
done
echo "== eval window 16384 + NTK x8 $(date +%H:%M:%S)"
$PY $E --dir $FULL --window 16384 --ntk 8 --tag _w16384ntk8 || { echo "== eval NTK FAILED"; exit 1; }
for W in 2048 16384; do
  echo "== hybrid bound, window $W $(date +%H:%M:%S)"
  $PY $E --dir $FULL --window $W --every --tag _hyb$W || { echo "== hybrid $W FAILED"; exit 1; }
done
echo "== done $(date +%H:%M:%S)"
