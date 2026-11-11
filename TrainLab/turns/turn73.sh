#!/bin/sh
# DLOG codemath capture (MTP Head, Mon 11-02; for Fable's E2 data slope and HPD-2): sticky/dlog_capture.py DLOG_SRC=codemath =
# the 125B's own answers to 92 new HumanEval-style stubs (sticky/code_tasks.py) + GSM8K test problems no set uses, rendered as
# L1-20's data. Same harness and preset as turn66 (D-23's preset, DFlash2 + cut, exact; a data run, no timing): C 160 if NO-SWAP
# >= 37 GB, else C 128 if >= 33, else not starting. Target DLOG_TARGET answer tokens per domain (default 30000: with L1-20's
# ~17K anchors, code ~20K (the stubs run out) + math 30K reach E2's second doubling, ~68K); answers capped at 512 tokens.
# Budget 3900 / 4100 s running time (one 75-min quiet ticket); swap guard +2 GB; output logs/dlog_n1102.out.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -f dlog_capture.py; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
sleep 30
SW0=$(swap); F=$(noswap)
if [ "$F" -ge 37 ]; then C=160; elif [ "$F" -ge 33 ]; then C=128; else
  echo "== DLOG codemath $(date +%H:%M:%S): ${F} GB NO-SWAP < 33: not starting (re-queue later)"; exit 1; fi
echo "== DLOG codemath capture $(date +%H:%M:%S): ${F} GB NO-SWAP -> C $C; swap ${SW0} MB, $(df -g $HOME | awk 'NR==2 {print $4}') GB free disk"
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f dlog_capture.py; exit; fi
  done ) & G=$!
B=${DLOG_BUDGET:-3900}; HD=${DLOG_HARD:-4100}
echo "== DLOG budget ${B} s, hard ${HD} s (running time), target ${DLOG_TARGET:-30000} per domain"
DLOG_SRC=codemath DLOG_NIGHT=n1102 DLOG_TARGET=${DLOG_TARGET:-30000} DLOG_MAXTOK=512 DLOG_BUDGET=$B DLOG_HARD=$HD DLOG_CAP_GB=8 \
  DLOG_MIN_FREE_GB=114 HF_HUB_OFFLINE=1 $PY sticky/dlog_capture.py $C 2>&1 | tee -a logs/dlog_n1102.out \
  | grep --line-buffered -v "Token indices\|moe-sorted-gather"
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
