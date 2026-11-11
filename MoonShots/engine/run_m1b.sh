#!/bin/sh
# M1b speed: BigMoE's model vs the engine, 4-row verify passes, fresh process each (launch holding a quiet lock).
# REP=1: compute-only (timed passes repeat the last warm chunk); the engine also runs with MoE=shared (no routed experts,
# no gate: the gate-free floor of the dense path).
OUT=$HOME/MoonShots/data/day15; PY=$HOME/BigMoE/.venv/bin/python; E=$HOME/MoonShots/engine; R=${REP:-0}
sh $HOME/MoonShots/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
for v in "ref bigmoe" "eng bigmoe" "eng shared"; do
  set -- $v
  M1B_REPEAT=$R M1B_MOE=$2 $PY $E/test_m1b_speed.py $1 0.25 32 64 > "$OUT/m1b_$1_$2_rep$R.txt" 2>&1
  echo "$(date +%H:%M) done m1b $1/$2 rep$R: $(grep -E 'verify pass|Traceback|[A-Za-z]*Error' "$OUT/m1b_$1_$2_rep$R.txt" | cut -c1-230 | tr '\n' ' ')" >> "$OUT/summary.txt"
done
peer unlock > /dev/null 2>&1
