#!/bin/sh
# R-31 (b) profile (NewIdeas Tue 21:0x): where do 16-row verify passes lose >= 3.5x per position in this harness, in time
# AND memory? A few positions only (TL_MAXPOS 64 of the first code sequence), per pass: time, the suite's tier calls
# (tier_decode_calls = single-row decode tiers; tier_prefill_calls / masked_scratch = multi-row calls tiering the full-E
# scratch), RSS and MLX memory; stops at swap + 1 GB. Runs: v3 tiers at 4 rows, v3 tiers at 16 rows, exact (no tiers)
# at 16 rows.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -P $$; exit 143' TERM                     # run_locked's stop reaches the running python too
for spec in "4 plan4433333333 p4433333333" "16 plan4433333333 p4433333333" "16 ref exact"; do
  set -- $spec
  echo "== PROF R $1 variant $2 arm $3 $(date +%H:%M:%S)"
  TL_PROF=1 TL_MAXPOS=64 TL_TAG=prof TL_VARIANT=$2 TL_LIMIT=300 \
    $PY sticky/budget125_tf.py $1 $3 tf_code 99 2>&1 | grep --line-buffered -E "^PROF|^\[1/|^plan variant|peak RSS|Traceback|Error"
done
echo "== PROF done $(date +%H:%M:%S)"
