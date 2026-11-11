#!/bin/bash
# MTP Head, Wed 11-04 23:4x: re-measure the PORT walk's footprint after the head-buffer fix (REPORT 'PORT walk
# validated': the math walk peaked at 5.97 GB before the fix; my port-test rule is <= 4 GB). Pre-registered in REPORT
# 'PORT walk footprint re-measure'. The same command as the 21:0x validation (math, batch 30, --bf16-store, 4 threads,
# 154 s then), CPU only, under a share lock MEM 7 GB.
# Starts only in a real gap: no other session's quiet lock or queued quiet ticket, no other MEM >= 5 GB lock,
# >= 24 GB available, lid open, for 3 clear minutes. Gives up at $DEADLINE (default Thu 2026-11-05 06:45, before
# BigMoE's 07:05 ticket). Swap guard +2 GB (SIGTERM). Log: data/port/walk_foot.log.
cd ~/TrainLab || exit 1
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "${WF_DEADLINE:-2026-11-05 06:45}" +%s)
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
swap_mb() { sysctl -n vm.swapusage | awk '{print $6}' | tr -d M; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} /Pages purgeable/ {p=$3} END {printf "%d", (f+i+s+p)*ps/2^30}'; }
lid_closed() { ioreg -r -k AppleClamshellState -d 4 | grep -q '"AppleClamshellState" = Yes'; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
nquiet() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
nbig() { others | locks | grep -oE "MEM [0-9]+ GB" | awk '$2 >= 5' | wc -l | tr -d " "; }
clear=0
log "waiting: no other quiet lock / ticket, no other MEM >= 5 GB lock, >= 24 GB available, lid open, 3 clear minutes (gives up $(date -r "$DEADLINE" '+%a %H:%M'))"
while :; do
  q=$(nquiet); b=$(nbig); a=$(avail_gb); l=open; lid_closed && l=closed
  if [ "$q" -eq 0 ] && [ "$b" -eq 0 ] && [ "$a" -ge 24 ] && [ "$l" = open ]; then clear=$((clear + 1)); else clear=0; fi
  [ "$clear" -ge 6 ] && { log "clear: ${a} GB available"; break; }
  [ "$(date +%s)" -ge "$DEADLINE" ] && { log "GIVE UP: quiet=$q big=$b avail=${a} GB lid=$l"; exit 3; }
  sleep 30
done
out=$(peer lock share 20m --cpu 400 --gpu 0 "MEM 7 GB: MTP Head PORT walk footprint re-measure (CPU only, math walk, ~4 min)" 2>&1)
log "lock: $(printf '%s' "$out" | head -1)"
printf '%s' "$out" | grep -q "reserved" || { log "NO LOCK"; exit 4; }
s0=$(swap_mb); log "start (swap $s0 MB)"
/usr/bin/time -l ~/TrainLab/.venv-torch/bin/python -I port/walk_port.py --sets "math=$HOME/Leap/data/dfl2d_math" \
  --bf16-store --threads 4 --batch 30 > data/port/walk_foot.log 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  sleep 10
  s=$(swap_mb)
  if awk -v x="$s" -v y="$s0" 'BEGIN{exit !(x-y>2048)}'; then
    log "swap $s MB = +2 GB over the start: stopping"; pkill -TERM -f "port/walk_por[t].py"; break
  fi
done
wait $pid; rc=$?
peer unlock 2>&1 | tail -1
log "rc=$rc: $(grep -E '^  math' data/port/walk_foot.log | cut -c1-150)"
log "footprint $(grep 'peak memory footprint' data/port/walk_foot.log | awk '{printf "%.2f GB", $1/2^30}'), max RSS $(grep 'maximum resident set size' data/port/walk_foot.log | awk '{printf "%.2f GB", $1/2^30}') (swap $(swap_mb) MB)"
log "DONE"
