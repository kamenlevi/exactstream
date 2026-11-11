#!/bin/sh
# D-05 MTPHYB + D-08 CALC walk (MTP Head, Wed 10-28 16:56; REPORT 'D-05 MTPHYB + D-08 CALC walk'). One quiet lock:
# DFlash2 L1-20 walk on code + math with every check's anchor saved (~1 min), then the native MTP chain (k = 3) in the
# shipped 24 GB v3 engine on 32 tf prompts, 300 tokens each (~8-9 min). Read-only on ~/BigMoE, ~/Leap, ~/EvalSuite.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -P $$; exit 143' TERM                     # run_locked's stop reaches the running python too
echo "== D-08 walk $(date +%H:%M:%S)"
HF_HUB_OFFLINE=1 $PY sticky/d02b_walk.py 2>&1 | grep -v "Token indices"
echo "== D-05 MTP depth $(date +%H:%M:%S)"
eval "$(grep -E '^(REF|X|V3|FULL)=' $HOME/BigMoE/scripts/mtp_depth.sh)"
env $X $FULL BIGMOE_JOINT=1 BIGMOE_MTP_PF_GROUP=48 $PY sticky/mtp_depth_ids.py data/d05_prompts.jsonl \
  data/d05_mtp_depth.json 3 300 2>&1 | grep --line-buffered -v "Token indices"
echo "== done $(date +%H:%M:%S)"
