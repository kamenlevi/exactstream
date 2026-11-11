#!/bin/sh
# NewIdeas QUEUE B2 (Sun 04:12): CONFIDENCE-GATED PRECISION for cold reads. One tf screen on the Eval suite's engine:
# exact 4-bit prompt prefill, 4-row verify passes (row by row through the suite's decode tier branch), three precision
# arms = v3 (p4433333333), ranks 4-9 at MSB-2 (p4433222222), ranks 2-9 at MSB-2 (p4422222222; the worst case: every
# lower-rank expert 2-bit, not only cold ones). Per position: KLD vs the cached 4-bit ref, agree, own top-1, the MTP
# drafter's depth-1 top-1 log-prob (TL_MTP=1) -> data/budget125_tf_R4_plan4433333333_mtp_b2_pos.npz. NewIdeas gates on
# CPU (ugb_rule.py). PRE-REGISTERED kill: no gate passes the task-result bar with >= 70 % of code passes on 2-bit.
# Quiet-window safe (PAUSE wait per pass; a timed-out sequence is dropped). One-sequence smoke first.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
echo "== smoke $(date +%H:%M:%S)"
TL_TAG=b2 TL_MTP=1 TL_VARIANT=plan4433333333 $PY sticky/budget125_tf.py 4 p4433333333,p4422222222 tf_code 24 2>&1 | grep --line-buffered -v "Token indices" | tee /tmp/turn18_smoke.$$
grep -q "budget125_tf R=4 variant plan4433333333: 1 sequences" /tmp/turn18_smoke.$$ || { echo "== smoke FAILED"; rm -f /tmp/turn18_smoke.$$; exit 1; }
rm -f /tmp/turn18_smoke.$$
echo "== full $(date +%H:%M:%S)"
TL_TAG=b2 TL_MTP=1 TL_VARIANT=plan4433333333 $PY sticky/budget125_tf.py 4 p4433333333,p4433222222,p4422222222 tf_gsm,tf_ifeval,tf_code,tf_multi 4 2>&1 | grep --line-buffered -v "Token indices"
