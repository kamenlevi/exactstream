#!/bin/sh
# DLOG capture at a smaller cache (MTP Head, Sun 11-01 23:2x; orchestrator 23:1x: tonight's idle NO-SWAP is ~36-41 GB, so C 224's
# ~41 GB peak (gate 44) does not fit). Same harness as turn64 (sticky/dlog_capture.py, D-23's preset, DFlash2 + cut, exact: the
# tokens and every logged array are the same at any C, the cache only holds expert records), C picked after the settle:
# C 160 (peak ~32 GB) if NO-SWAP >= 37, else C 128 (~28 GB) if >= 33, else not starting. Target DLOG_TARGET answer tokens per
# language (default 20000, E1's set). A data run: no timing. Swap guard +2 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -f dlog_capture.py; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
sleep 30
SW0=$(swap); F=$(noswap)
if [ "$F" -ge 37 ]; then C=160; elif [ "$F" -ge 33 ]; then C=128; else
  echo "== DLOG $(date +%H:%M:%S): ${F} GB NO-SWAP < 33: not starting (re-queue later)"; exit 1; fi
echo "== DLOG capture $(date +%H:%M:%S): ${F} GB NO-SWAP -> C $C; swap ${SW0} MB, $(df -g $HOME | awk 'NR==2 {print $4}') GB free disk"
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f dlog_capture.py; exit; fi
  done ) & G=$!
B=${DLOG_BUDGET:-3900}; HD=${DLOG_HARD:-4100}
echo "== DLOG budget ${B} s, hard ${HD} s (running time), target ${DLOG_TARGET:-20000} per language"
DLOG_NIGHT=n1101 DLOG_TARGET=${DLOG_TARGET:-20000} DLOG_BUDGET=$B DLOG_HARD=$HD DLOG_CAP_GB=8 DLOG_MIN_FREE_GB=114 HF_HUB_OFFLINE=1 \
  $PY sticky/dlog_capture.py $C 2>&1 | tee -a logs/dlog_n1101.out | grep --line-buffered -v "Token indices\|moe-sorted-gather"
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
