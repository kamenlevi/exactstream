#!/bin/bash
# nw3's code read-out (Breakthrough, Sat 2026-11-07 01:3x): code_read.py (CPU, the map's out/ JSONs only, nice 19) on each nw3
# code arm whose 12 windows are in, against its twins: the control vs nw1 at 208 slots; nw3 vs nw1 and vs the control at 208;
# nw3 vs nw1 at the published pool. The lines to apply are in SCOREBOARD 'Fri 21:3x: PRE-REGISTRATION, NW3'. Output appended,
# dated, to data/nw3_code_read.txt. Run as "bash scripts/dm_s7/nw3_read.sh" (the peers hook blocks command text that holds the
# interpreter's name during others' quiet locks; this read is a few seconds of one core).
PY=~/BigMoE/.venv/bin/python; H=~/Breakthrough; DM=~/FableGod/decision_map; O=$H/data/nw3_code_read.txt
nwin() { [ -f $DM/out/$1.json ] && $PY -c "import json; d = json.load(open('$DM/out/$1.json')); print(sum(1 for v in d.values() if isinstance(v, dict) and 'tk' in v))" 2>/dev/null || echo 0; }
echo "== nw3 code read-out $(date '+%a %Y-%m-%d %H:%M:%S')" >> $O
while read a t; do
  n=$(nwin $a); [ "$n" = 12 ] || { echo "-- $a: $n of 12 windows, not read" >> $O; continue; }
  echo "-- $a vs $t" >> $O
  nice -n 19 $PY $H/scripts/dm_s7/code_read.py $a $t >> $O 2>&1
done <<EOF
qom_0_live_ball_c208_nw3c+code qom_0_live_ball_c208_nw1+code
qom_0_live_ball_c208_nw3+code qom_0_live_ball_c208_nw1+code qom_0_live_ball_c208_nw3c+code
qom_0_live_ball_nw3+code qom_0_live_ball_nw1+code
EOF
