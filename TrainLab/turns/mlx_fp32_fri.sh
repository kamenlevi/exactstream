#!/bin/bash
# MTP Head, Thu 11-05 11:3x (the orchestrator 11:2x): the full-scale confirmation of REPORT 'PORT steps FAIL: ROOT CAUSE
# FOUND' (pre-registered: REPORT 'PORT full-scale confirmation'). Friday by day right after EvalSuite's runner 45 ends, if
# >= 10 min remain before 15:00 (start by 14:50); otherwise not on Friday by day (the orchestrator: first on the
# Fri -> Sat night after FAR-G1; re-armed then).
#  1. Wait: ~/EvalSuite/results/q45.pid written after this runner was armed and its process gone (runner 45's end;
#     EvalSuite's day runners leave the file behind), and not before Fri 05:00 (the night's queue goes first).
#  2. Gate: no other session's quiet lock / ticket, no other lock showing MEM > 5 GB (by day one big job), >= 40 GB
#     available, 1 clear minute; start by 14:50.
#  3. Share lock MEM 34 GB (L1-20's MLX peak 28.8 GB at step 1, 33.7 later): MLX's own trainer (port/ref_mlx_steps.py),
#     5 steps of L1-20's recipe with MLX_ENABLE_TF32=0 (vs the port, data/port/steps_thu.json); then, if >= 5 min remain
#     before 15:00, the same with TF32 on (vs L1-20's log: does MLX still reproduce it?). Swap guard +2 GB.
# Logs: data/port/fp32_fri.log (this), data/port/steps_mlx_{fp32,tf32}.{json,log}.
cd ~/TrainLab || exit 1
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
swap_mb() { sysctl -n vm.swapusage | awk '{print $6}' | tr -d M; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} /Pages purgeable/ {p=$3} END {printf "%d", (f+i+s+p)*ps/2^30}'; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
nquiet() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
nbig() { others | locks | grep -oE "MEM [0-9]+ GB" | awk '$2 > 5' | wc -l | tr -d " "; }
at() { date -j -f "%Y-%m-%d %H:%M" "$1" +%s; }
ARMED=$(date +%s); NOTBEFORE=$(at "2026-11-06 05:00"); LAST=$(at "2026-11-06 14:50"); END=$(at "2026-11-06 15:00")
PIDF=~/EvalSuite/results/q45.pid
PY=~/BigMoE/.venv/bin/python3; TP=~/TrainLab/.venv-torch/bin/python
log "1. waiting for runner 45's end (q45.pid newer than now, process gone), not before Fri 05:00; start by Fri 14:50"
while :; do
  now=$(date +%s)
  [ "$now" -ge "$LAST" ] && { log "NOT ON FRIDAY BY DAY: runner 45 had not ended by 14:50 -> the Fri -> Sat night (re-arm)"; exit 5; }
  if [ "$now" -ge "$NOTBEFORE" ] && [ -f "$PIDF" ] && [ "$(stat -f %m "$PIDF")" -ge "$ARMED" ]; then
    p=$(tr -dc 0-9 < "$PIDF")
    if [ -n "$p" ] && ! kill -0 "$p" 2>/dev/null; then log "runner 45 ended (pid $p gone)"; break; fi
  fi
  sleep 60
done
clear=0
log "2. gate: no other quiet lock / ticket, no other MEM > 5 GB lock, >= 40 GB available, 1 clear minute"
while :; do
  q=$(nquiet); b=$(nbig); a=$(avail_gb)
  if [ "$q" -eq 0 ] && [ "$b" -eq 0 ] && [ "$a" -ge 40 ]; then clear=$((clear + 1)); else clear=0; fi
  [ "$clear" -ge 2 ] && { log "clear: ${a} GB available"; break; }
  [ "$(date +%s)" -ge "$LAST" ] && { log "NOT ON FRIDAY BY DAY: no gap by 14:50 (quiet=$q big=$b avail=${a} GB) -> night (re-arm)"; exit 5; }
  sleep 30
done
out=$(peer lock share 15m --cpu 300 --gpu 100 "MEM 34 GB: MTP Head PORT full-scale confirmation, MLX trainer 5 steps TF32 off then on (~5 min; the orchestrator 11:2x)" 2>&1)
log "lock: $(printf '%s' "$out" | head -1)"
printf '%s' "$out" | grep -q "reserved" || { log "NO LOCK"; exit 4; }
run() {   # $1 tag, $2 TF32 value; output data/port/steps_mlx_$1.{json,log}
  local s0; s0=$(swap_mb); log "3. MLX trainer, 5 steps, MLX_ENABLE_TF32=$2 (swap $s0 MB, $(avail_gb) GB available)"
  MLX_ENABLE_TF32=$2 /usr/bin/time -l $PY -I port/ref_mlx_steps.py data/port/steps_mlx_$1.json --steps 5 > data/port/steps_mlx_$1.log 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 10
    local s; s=$(swap_mb)
    if awk -v x="$s" -v y="$s0" 'BEGIN{exit !(x-y>2048)}'; then log "swap $s MB = +2 GB: stopping"; pkill -TERM -f "port/ref_mlx_step[s].py"; break; fi
  done
  wait $pid; local rc=$?
  log "rc=$rc, footprint $(grep 'peak memory footprint' data/port/steps_mlx_$1.log | awk '{printf "%.1f GB", $1/2^30}'): $(grep -E '^\[mlx-steps\] step' data/port/steps_mlx_$1.log | awk '{printf "%s ", $5}')"
  return $rc
}
run fp32 0 && [ -f data/port/steps_mlx_fp32.json ] && $TP -I port/compare_l120.py data/port/steps_thu.json data/port/steps_mlx_fp32.json
if [ $(( END - $(date +%s) )) -ge 300 ]; then
  run tf32 1 && [ -f data/port/steps_mlx_tf32.json ] && $TP -I port/compare_l120.py data/port/steps_mlx_tf32.json
else
  log "the TF32-on rerun skipped: < 5 min before 15:00"
fi
peer unlock 2>&1 | tail -1
log "DONE"
