#!/bin/bash
# MTP Head, Thu 11-05 11:3x (the orchestrator 11:2x): the full-scale confirmation of REPORT 'PORT steps FAIL: ROOT CAUSE
# FOUND' (pre-registered: REPORT 'PORT full-scale confirmation'). Friday by day, start by 14:50, the run ends by 15:00;
# otherwise Saturday by day (the orchestrator 09:4x; or tell NewIdeas).
# Amendments: 1 (07:3x) after Leap's part B; 2 (08:0x) after BigMoE's FAST-NAX A/B (turns/mlx_fp32_fri{2,3}.sh, both
# stopped in their waits, no lock taken). fri3 never saw runner 45's end (08:00:13), because EvalSuite deleted q45.pid.
# Amendment 3, Fri 11-06 09:4x (the orchestrator 09:4x): EvalSuite's agent check (day_runner48.sh, shared lock MEM 39 GB,
# lease to 11:10, may run to ~12:10) took the GPU at 09:40. Swap hit 13.65 GB at 09:43. PORT at ~37 GB beside it would
# overflow 64 GB. So:
#  1. Wait for runner 48 to end: its pid (taken from q48.pid when this was armed) gone, and no EvalSuite lock left.
#  2. Gate: no other session's quiet lock / ticket, no other lock showing MEM > 5 GB (no exceptions now, memory is
#     tight), NO-SWAP >= 40 GB (PORT's ~37 GB peak + 3 GB margin; NO-SWAP = free + speculative + purgeable +
#     file-backed pages, Leap's noswap10), 3 clear minutes; start by 14:50.
#  3. Share lock MEM 37 GB: MLX's own trainer (port/ref_mlx_steps.py), 5 steps of L1-20's recipe with MLX_ENABLE_TF32=0
#     (vs the port, data/port/steps_thu.json); then, if >= 5 min remain before 15:00, the same with TF32 on (vs L1-20's
#     log). Swap guard +2 GB.
# Logs: data/port/fp32_fri4.log (this), data/port/steps_mlx_{fp32,tf32}.{json,log}.
cd ~/TrainLab || exit 1
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
swap_mb() { sysctl -n vm.swapusage | awk '{print $6}' | tr -d M; }
noswap_gb() { vm_stat | awk '/page size of/ {ps = $8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v = $NF; sub("[.]", "", v); s += v} END {printf "%d", s * ps / 2^30}'; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
nquiet() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
nbig() { others | locks | grep -oE "MEM [~0-9-]+ GB" | tr -d "~" | awk '{split($2, r, "-"); if (r[length(r)] + 0 > 5) n++} END {print n + 0}'; }
nevals() { others | grep -E "^- Eval suite" | locks | wc -l | tr -d " "; }
at() { date -j -f "%Y-%m-%d %H:%M" "$1" +%s; }
LAST=$(at "2026-11-06 14:50"); END=$(at "2026-11-06 15:00")
R48=$(tr -dc 0-9 < ~/EvalSuite/results/q48.pid 2>/dev/null)
PY=~/BigMoE/.venv/bin/python3; TP=~/TrainLab/.venv-torch/bin/python
log "1. waiting for EvalSuite's runner 48 (pid ${R48:-?}) to end and its lock to go; start by 14:50"
[ -n "$R48" ] || { log "no q48.pid at arming: stop (re-arm by hand)"; exit 8; }
while :; do
  if ! kill -0 "$R48" 2>/dev/null && [ "$(nevals)" -eq 0 ]; then log "runner 48 ended and EvalSuite holds no lock"; break; fi
  [ "$(date +%s)" -ge "$LAST" ] && { log "NOT ON FRIDAY BY DAY: runner 48 still on at 14:50 -> Saturday by day (re-arm)"; exit 5; }
  sleep 60
done
clear=0
log "2. gate: no other quiet lock / ticket, no other MEM > 5 GB lock, NO-SWAP >= 40 GB, 3 clear minutes"
while :; do
  q=$(nquiet); b=$(nbig); a=$(noswap_gb)
  if [ "$q" -eq 0 ] && [ "$b" -eq 0 ] && [ "$a" -ge 40 ]; then clear=$((clear + 1)); else clear=0; fi
  [ "$clear" -ge 6 ] && { log "clear: NO-SWAP ${a} GB, swap $(swap_mb) MB"; break; }
  [ "$(date +%s)" -ge "$LAST" ] && { log "NOT ON FRIDAY BY DAY: no gap by 14:50 (quiet=$q big=$b NO-SWAP=${a} GB) -> Saturday by day (re-arm)"; exit 5; }
  sleep 30
done
out=$(peer lock share 15m --cpu 300 --gpu 100 "MEM 37 GB: MTP Head PORT full-scale confirmation, MLX trainer 5 steps TF32 off then on (~6 min; after EvalSuite's agent check, the orchestrator 09:4x)" 2>&1)
log "lock: $(printf '%s' "$out" | head -1)"
printf '%s' "$out" | grep -q "reserved" || { log "NO LOCK"; exit 4; }
run() {   # $1 tag, $2 TF32 value; output data/port/steps_mlx_$1.{json,log}
  local s0; s0=$(swap_mb); log "3. MLX trainer, 5 steps, MLX_ENABLE_TF32=$2 (swap $s0 MB, NO-SWAP $(noswap_gb) GB)"
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
