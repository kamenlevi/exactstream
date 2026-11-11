#!/bin/bash
# Wed 23:00-23:45 (the orchestrator 15:1x: if Kamen closes his apps at 23:00, RELAX moves to ~23:50), one quiet ticket,
# MEM ~44 GB: the all-in FAST code preset at C 224 first (Kamen's 300 ask; ~15 min), then lever 2 stage 1 at C 160 (~30 min).
# usage: NOTBEFORE=23:00 NOTAFTER=23:15 WAITMEM=44 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 40 50 "..." /bin/bash scripts/eve_2300.sh
cd "$HOME/BigMoE" || exit 1
ALLIN_DEADLINE=${ALLIN_DEADLINE:-2325} /bin/bash scripts/allin_wed.sh
L2_DEADLINE=${L2_DEADLINE:-2340} /bin/bash scripts/l2_wed.sh
