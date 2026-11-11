#!/bin/sh
# Non-exact deferral quality on the 125B (NewIdeas' tf_defer125.py, read-only reuse of fork125bt; launch holding a quiet
# lock; unlocks at the end). frac 0.25 (C=128), all four tf sets: dh_w4, dh_w2, d1_w4 every 8; exact-streaming control
# dh_w10 every 16. Results append to ~/NewIdeas/data/tfdefer/results.txt; copies of the lines in data/stall/defer.txt.
OUT=$HOME/MoonShots/data/stall; PY=$HOME/BigMoE/.venv/bin/python; S=$HOME/NewIdeas/scripts/tf_defer125.py
SETS=tf_gsm,tf_code,tf_ifeval,tf_multi
sh $HOME/MoonShots/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
for v in "dh_w4 8" "dh_w2 8" "d1_w4 8" "dh_w10 16"; do
  set -- $v
  $PY $S $1 0.25 $SETS $2 > "$OUT/defer_$1.txt" 2>&1
  echo "$(date +%H:%M) $(grep -E 'KLD .* over|Traceback|[A-Za-z]*Error' "$OUT/defer_$1.txt" | tail -2 | cut -c1-330 | tr '\n' ' ')" >> "$OUT/defer.txt"
done
peer unlock > /dev/null 2>&1
echo "$(date +%H:%M) defer finished" >> "$OUT/defer.txt"
