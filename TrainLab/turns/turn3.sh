#!/bin/sh
# TrainLab turn 3 (10-21, NewIdeas 19:40: pre-routed frugal does not transfer to the 125B -> its window/distance rows
# and healing are dropped; order = MTP kill test, 125B top-2 estimator, sticky routing). This turn: MTP smoke test,
# the sticky hard-cap zero-shot rows (exact baseline + cap1/2/4), then the MTP chained-drafting kill test
# (stock tau, depth-6 training-time-test fine-tune, tuned tau).
PY=$HOME/BigMoE/.venv/bin/python; T=$HOME/TrainLab
cd $T
F="grep --line-buffered -v Token.indices"
echo "== $(date '+%m-%d %H:%M:%S') turn 3 start (limit $TL_LIMIT s)"
START=$(date +%s)
$PY mtp35.py smoke 2>&1 | $F
$PY train35.py eval exact cap1 cap2 cap4 2>&1 | $F
[ -f data/mtp_stock_done ] || { $PY mtp35.py eval --K 8,16,32 2>&1 | $F; touch data/mtp_stock_done; }
USED=$(( $(date +%s) - START ))
$PY mtp35.py train --run mtp_d6 --depth 6 --lr 3e-5 --time-limit $(( TL_LIMIT - USED - 480 )) 2>&1 | $F
$PY mtp35.py eval --run mtp_d6 --K 8,16,32 2>&1 | $F
echo "== $(date '+%m-%d %H:%M:%S') turn 3 end"
