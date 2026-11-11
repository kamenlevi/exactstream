#!/bin/sh
> (a quote from the project owner, kept private)
# fresh MGSM prompts from the 45 GB default's live loop (D-23's preset, DFlash2 + cut, C 224), every kept row's training data
# (DFlash2's h, the MTP head's widened stream, the 125B's top-32) -> data/dlog/n1101/ (sticky/dlog_capture.py). A data run: no
# timing claims. Quiet ticket (one 45 GB process at a time), resumable. Gate as D-23: settle 30 s, NO-SWAP >= 44 GB; stop if
# swap grows > 2 GB. Disk: stops under 114 GB free or at 8 GB written tonight (the orchestrator's caps).
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -f dlog_capture.py; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
sleep 30
SW0=$(swap); F=$(noswap)
echo "== DLOG capture $(date +%H:%M:%S): ${F} GB NO-SWAP, swap ${SW0} MB, $(df -g $HOME | awk 'NR==2 {print $4}') GB free disk"
[ "$F" -ge 44 ] || { echo "== needs >= 44 GB NO-SWAP (C 224): not starting (re-queue in 15 min)"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f dlog_capture.py; exit; fi
  done ) & G=$!
# the time budget (running s): the caller's DLOG_BUDGET / DLOG_HARD (default one 30-min ticket). No clock stop (Kamen 20:3x:
# runs may continue past 06:00 and through Monday)
B=${DLOG_BUDGET:-1500}; HD=${DLOG_HARD:-1580}
echo "== DLOG budget ${B} s, hard ${HD} s (running time)"
DLOG_NIGHT=n1101 DLOG_BUDGET=$B DLOG_HARD=$HD DLOG_CAP_GB=8 DLOG_MIN_FREE_GB=114 HF_HUB_OFFLINE=1 $PY sticky/dlog_capture.py 224 2>&1 \
  | tee -a logs/dlog_n1101.out | grep --line-buffered -v "Token indices\|moe-sorted-gather"
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
