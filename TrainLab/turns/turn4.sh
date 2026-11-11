#!/bin/sh
# TrainLab turn 4: continue the MTP fine-tune if turn 3's tuned tau is promising (turns/train4 = argument line for
# mtp35.py train; default: resume mtp_d6) and re-score; otherwise (turns/train4 starting with "train35") the charter's
# healing curve (frugal C13).
PY=$HOME/BigMoE/.venv/bin/python; T=$HOME/TrainLab
cd $T
F="grep --line-buffered -v Token.indices"
echo "== $(date '+%m-%d %H:%M:%S') turn 4 start (limit $TL_LIMIT s)"
START=$(date +%s)
# NewIdeas 19:40: the 125B top-2 estimator feasibility capture (~10 min, ~24 GB, BigMoE's streamed engine)
[ $(ls data/cap125est/*.npz 2>/dev/null | wc -l) -ge 25 ] || $PY cap125_est.py 1536 2>&1 | $F
USED=$(( $(date +%s) - START ))
set -- $(cat turns/train4)
if [ "$1" = "train35" ]; then shift; $PY train35.py "$@" --time-limit $(( TL_LIMIT - USED - 60 )) 2>&1 | $F
else $PY mtp35.py "$@" --time-limit $(( TL_LIMIT - USED - 480 )) 2>&1 | $F; $PY mtp35.py eval --run mtp_d6_lr3e6 --K 16 2>&1 | $F; fi
echo "== $(date '+%m-%d %H:%M:%S') turn 4 end"
