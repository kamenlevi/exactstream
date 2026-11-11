#!/bin/sh
# v3 tiers + deferral quality (launch holding a quiet lock; unlocks at the end): dh_w4 with v3 tiers and the v3-only control
# (dh_w10 = every rank waited) in the same run, C=128, all four tf sets, every 8.
OUT=$HOME/MoonShots/data/stall; PY=$HOME/BigMoE/.venv/bin/python
SETS=tf_gsm,tf_code,tf_ifeval,tf_multi
sh $HOME/MoonShots/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
for V in dh_w4_v3 dh_w10_v3; do
  $PY $HOME/MoonShots/stall/tf_defer_v3.py $V 0.25 $SETS 8 > "$OUT/defer_$V.txt" 2>&1
  echo "$(date +%H:%M) $(grep -E 'tf_defer_v3\]|KLD .* over|Traceback|[A-Za-z]*Error' "$OUT/defer_$V.txt" | tail -3 | cut -c1-330 | tr '\n' ' ')" >> "$OUT/defer_v3.txt"
done
peer unlock > /dev/null 2>&1
echo "$(date +%H:%M) defer_v3 finished" >> "$OUT/defer_v3.txt"
