#!/bin/sh
# D-02a (MTP Head, Thu 10-29; REPORT 'D-02a'). One quiet lock (15 min): the capture (sticky/d02a_extract.py: GSM8K p,
# IFEval / multi / agent features + p; resumable, skips finished keys), then the D-02b walk (L1-20) on the three new
# domains. Guards: >= 24 GB free to start, stop if swap grows > 2 GB. Read-only on ~/Leap, ~/BigMoE, ~/EvalSuite, ~/Chat1000.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -P $$; exit 143' TERM                     # run_locked's stop reaches the running python too
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
freegb() { vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages purgeable/{p=$3} END{printf "%d", (f+i+p)*16384/1e9}'; }
SW0=$(swap); F=$(freegb)
echo "== D-02a $(date +%H:%M:%S): ${F} GB free, swap ${SW0} MB"
[ "$F" -ge 24 ] || { echo "== < 24 GB free: not starting"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then
      echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d02a_extract.py; pkill -f d02b_walk.py; exit
    fi
  done ) & G=$!
echo "== capture $(date +%H:%M:%S)"
D02A_ONLY=tf_gsm,tf_ifeval,tf_multi,agent $PY sticky/d02a_extract.py 2>&1 | grep --line-buffered -v "Token indices"
n=$(ls data/d02a/*.npz 2>/dev/null | wc -l)
[ "$n" -ge 126 ] || { echo "== capture incomplete ($n of 126): walk skipped"; kill $G; exit 1; }
echo "== walk $(date +%H:%M:%S)"
TL_WALK_TAG=_d02a TL_WALK_SETS="ifeval=$HOME/TrainLab/data/d02a/tf_ifeval_*.npz,multi=$HOME/TrainLab/data/d02a/tf_multi_*.npz,agent=$HOME/TrainLab/data/d02a/agent_*.npz" \
  HF_HUB_OFFLINE=1 $PY sticky/d02b_walk.py 2>&1 | grep --line-buffered -v "Token indices"
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
