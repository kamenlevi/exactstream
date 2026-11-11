#!/bin/bash
# MTP Head, Fri 11-06 23:3x: Fable GOAL's ask 3 (K wall; lowest priority). Every-position walks (TL_WALK_EVERY +
# TL_WALK_CF, the d23h form) of the retrain variants, so GOAL can re-rank them on CODE ROWS under CD16. They were judged
# on exact K, where 62-67 % of first misses are comment / docstring text. L1-20's own are data/d02b_walk_d23h_dfl_*.
#   seed1 (ext/seed/l120_s1, the noise reference), pcap (ext/pcap/pcap10, TL_EXPAND=1), poff (ext/poff/poff60):
#     HumanEval+ (24) + MBPP (100).
#   x4 (data/e2_x4): HumanEval+ only. The MBPP walk set is the first 100 of E2's MBPP answers in capture order, inside
#     x2 / x4's training data.
> (a quote from the project owner, kept private)
# Gate: no other quiet lock / ticket, NO-SWAP >= 12 GB, disk >= 101 GB, 3 clear minutes; start by Sat 21:00. Share lock
# MEM 6 GB. Swap guard +2 GB. ~25 min.
# Logs: data/ev_retrains.log (this); per variant data/d02b_walk_ev_<v>.txt (+ npz, gitignored).
cd ~/TrainLab || exit 1
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
swap_mb() { sysctl -n vm.swapusage | awk '{print $6}' | tr -d M; }
noswap_gb() { vm_stat | awk '/page size of/ {ps = $8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v = $NF; sub("[.]", "", v); s += v} END {printf "%d", s * ps / 2^30}'; }
disk_gb() { df -g / | awk 'NR == 2 {print $4}'; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
nquiet() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
at() { date -j -f "%Y-%m-%d %H:%M" "$1" +%s; }
NB=$(at "${NOTBEFORE:-$(date '+%Y-%m-%d %H:%M')}"); LAST=$(at "2026-11-07 21:00")
PY=~/BigMoE/.venv/bin/python3
CODE="code=$HOME/Leap/data/dfl2d_code/*.npz"; MBPP="mbpp=$HOME/TrainLab/data/taps_mbpp_walk/*.npz"
log "1. not before ${NOTBEFORE:-now}; then no other quiet lock / ticket, NO-SWAP >= 12 GB, disk >= 101 GB, 3 clear minutes"
while [ "$(date +%s)" -lt "$NB" ]; do sleep 30; done
clear=0
while :; do
  [ "$(date +%s)" -ge "$LAST" ] && { log "NO START by Sat 21:00"; exit 5; }
  q=$(nquiet); a=$(noswap_gb); d=$(disk_gb)
  if [ "$q" -eq 0 ] && [ "$a" -ge 12 ] && [ "$d" -ge 101 ]; then clear=$((clear + 1)); else clear=0; fi
  [ "$clear" -ge 6 ] && break
  sleep 30
done
out=$(peer lock share 40m --cpu 300 --gpu 60 "MEM 6 GB: MTP Head every-position walks of 4 drafter retrains for Fable GOAL (~25 min; NewIdeas' slot)" 2>&1)
log "lock: $(printf '%s' "$out" | head -1)"
printf '%s' "$out" | grep -q "reserved" || { log "NO LOCK"; exit 4; }
walk() {   # $1 tag, $2 weights, $3 sets, $4 extra env (or "")
  local s0; s0=$(swap_mb); log "walk $1 (swap $s0 MB, NO-SWAP $(noswap_gb) GB): $2"
  env $4 TL_DRAFT=$2 TL_WALK_EVERY=1 TL_WALK_CF=1 TL_WALK_TAG=_ev_$1 TL_WALK_SETS="$3" HF_HUB_OFFLINE=1 \
    /usr/bin/time -l $PY sticky/d02b_walk.py > data/ev_walk_$1.log 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 10
    if awk -v x="$(swap_mb)" -v y="$s0" 'BEGIN{exit !(x-y>2048)}'; then log "swap +2 GB: stopping"; pkill -TERM -f "d02b_wal[k].py"; wait $pid; return 9; fi
  done
  wait $pid; local rc=$?
  log "rc=$rc, footprint $(grep 'peak memory footprint' data/ev_walk_$1.log | awk '{printf "%.1f GB", $1/2^30}'): $(grep -E 'kept per check' data/ev_walk_$1.log | cut -c1-110 | tr '\n' ' ')"
  return $rc
}
walk seed1 ext/seed/l120_s1.safetensors "$CODE,$MBPP" "" && \
walk poff ext/poff/poff60.safetensors "$CODE,$MBPP" "" && \
walk pcap ext/pcap/pcap10.safetensors "$CODE,$MBPP" "TL_EXPAND=1" && \
walk x4 data/e2_x4.safetensors "$CODE" ""
rc=$?
peer unlock 2>&1 | tail -1
log "DONE rc=$rc"
