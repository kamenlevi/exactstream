#!/bin/bash
# Wed 20:0x (Kamen via the orchestrator: Chrome closed early, so the 45 GB run goes SOONER): the 23:00 ticket's arms
# (eve_2300.sh: allin_wed.sh = u3c / qall / qallf / u3c2 at C 224, rx0, qallr; then l2_wed.sh at C 160) in a 20:35-21:00
# window right after P-CAP. Deadlines count from the actual start, as 23:00 -> 23:25 / 23:40: all-in + 25 min, lever 2
# + 40 min (ticket 50 min). If this one runs, the 23:00 ticket is cancelled; if it never starts, 23:00 stays.
cd "$HOME/BigMoE" || exit 1
A=$(date -v+25M +%H%M); B=$(date -v+40M +%H%M)
echo "$(date '+%m-%d %H:%M:%S') eve_early start: all-in deadline $A, lever 2 deadline $B" >> data/allin/allin.log
ALLIN_DEADLINE=$A /bin/bash scripts/allin_wed.sh
L2_DEADLINE=$B /bin/bash scripts/l2_wed.sh
