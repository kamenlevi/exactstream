#!/bin/bash
# MTP Head, Thu 11-05 (the orchestrator 21:5x): PORT steps 1-5 at L1-20's full recipe on the CPU (fp32 weights + Adam
# + grads ~9 GB, ~11-12 GB in all) vs L1-20's logged losses (port/compare_l120.py; PASS = every step within 2e-3 rel).
# Rules: a share lock MEM 12 GB, once runner 42 (NAX6 Part 1) runs, only if NO-SWAP stays >= 20 GB beside it: start
# needs >= 32 GB available (12 + 20) for 3 clear minutes and no other session's quiet lock / queued quiet ticket.
# Gives up at $GIVE_UP (default 11:15; the conversion window ~11:20 is the fallback, ask the orchestrator).
# Swap guard +2 GB (SIGTERM). Logs: data/port/steps_thu.log, data/port/steps_thu.json.
# Added Thu 06:4x (REPORT 'PORT walk footprint re-measure'; turns/walk_foot.sh found no gap overnight): after the steps
# check has exited (one port process at a time), the math walk re-measure (~4 min, <= 6 GB) under the same lock, only
# if >= 26 GB is available and no other session holds or queues a quiet lock; same guards. Log data/port/walk_foot.log.
cd ~/TrainLab || exit 1
GIVE_UP=${GIVE_UP:-1115}
# The orchestrator 09:2x-09:3x: EvalSuite's runner 43 waits for $GO_FILE (touched when this script exits, on every
# path) or 10:50, and while any share lock shows MEM > 5 GB. By day this check and Breakthrough's split code arms go
# one at a time, in either order: the gate also waits while another session's lock shows MEM > 5 GB, and if two locks
# land together, this one backs off. Start by $START_BY (= the 11:30 give-up); the walk only by $WALK_BY; $CAP stops a
# runaway (the trainer logs every step's loss, so a stop leaves the steps done so far in data/port/steps_thu.log).
CAP=${CAP:-1200}; START_BY=${START_BY:-1130}; WALK_BY=${WALK_BY:-1140}
GO_FILE=${GO_FILE-$HOME/EvalSuite/results/q43/GO}
log() { echo "[$(date +%H:%M:%S)] $*"; }
swap_mb() { sysctl -n vm.swapusage | awk '{print $6}' | tr -d M; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} /Pages purgeable/ {p=$3} END {printf "%d", (f+i+s+p)*ps/2^30}'; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
nquiet() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
nbig() { others | locks | grep -oE "MEM [0-9]+ GB" | awk '$2 > 5' | wc -l | tr -d " "; }
trap '[ -n "$GO_FILE" ] && touch "$GO_FILE" && log "touched $GO_FILE (runner 43 may go)"' EXIT
while :; do
  clear=0
  log "waiting: no other quiet lock / ticket, no other lock with MEM > 5 GB, >= 32 GB available, 3 clear minutes (gives up $GIVE_UP / $START_BY)"
  while :; do
    q=$(nquiet); b=$(nbig); a=$(avail_gb)
    if [ "$q" -eq 0 ] && [ "$b" -eq 0 ] && [ "$a" -ge 32 ]; then clear=$((clear + 1)); else clear=0; fi
    [ "$clear" -ge 6 ] && { log "clear: ${a} GB available"; break; }
    [ "$(date +%H%M)" -ge "$GIVE_UP" ] && { log "GIVE UP at $GIVE_UP: quiet=$q big=$b avail=${a} GB"; exit 3; }
    [ "$(date +%H%M)" -ge "$START_BY" ] && { log "GIVE UP: not started by $START_BY: quiet=$q big=$b avail=${a} GB"; exit 3; }
    sleep 30
  done
  out=$(peer lock share 40m --cpu 600 --gpu 0 "MEM 12 GB: MTP Head PORT steps 1-5 vs L1-20, then a 4-min walk re-measure (CPU only, ~15 min; the orchestrator 21:5x / 09:2x)" 2>&1)
  log "lock: $(printf '%s' "$out" | head -1)"
  printf '%s' "$out" | grep -q "reserved" || { log "NO LOCK"; exit 4; }
  sleep 10
  q=$(nquiet); b=$(nbig)
  [ "$q" -eq 0 ] && [ "$b" -eq 0 ] && break
  log "another quiet lock / MEM > 5 GB lock landed with mine (quiet=$q big=$b): unlocking, waiting again"
  peer unlock 2>&1 | tail -1
done
s0=$(swap_mb); log "start (swap $s0 MB)"
/usr/bin/time -l ~/TrainLab/.venv-torch/bin/python -I port/train_port.py --data ~/Leap/data/l120 --epochs 2 --lr 2e-5 \
  --S 4 --A 64 --micro 1 --bs 32 --pmask 0.5 --no-val --max-steps 5 --ckpt-layers --attn-chunk 8 --rows 128 --threads 6 \
  --log-json data/port/steps_thu.json > data/port/steps_thu.log 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  sleep 15
  s=$(swap_mb); a=$(avail_gb)
  if awk -v x="$s" -v y="$s0" 'BEGIN{exit !(x-y>2048)}' || [ "$a" -lt 20 ]; then
    log "swap $s MB / avail ${a} GB: NO-SWAP margin gone, stopping"; stopped=1; pkill -TERM -f "port/train_por[t].py"; break
  fi
  if [ "$(date +%H%M)" -ge "$CAP" ]; then
    log "cap $CAP reached (runaway guard): stopping (the steps done so far are in the log)"; stopped=1; pkill -TERM -f "port/train_por[t].py"; break
  fi
done
wait $pid; log "rc=$? footprint $(grep 'peak memory footprint' data/port/steps_thu.log | awk '{printf "%.1f GB", $1/2^30}')"
q=$(nquiet); a=$(avail_gb)
if [ "${stopped:-0}" -eq 0 ] && [ "$q" -eq 0 ] && [ "$a" -ge 26 ] && [ "$(date +%H%M)" -lt "$WALK_BY" ]; then
  s0=$(swap_mb); log "walk re-measure: start (${a} GB available, swap $s0 MB)"
  /usr/bin/time -l ~/TrainLab/.venv-torch/bin/python -I port/walk_port.py --sets "math=$HOME/Leap/data/dfl2d_math" \
    --bf16-store --threads 4 --batch 30 > data/port/walk_foot.log 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 10
    s=$(swap_mb); a=$(avail_gb)
    if awk -v x="$s" -v y="$s0" 'BEGIN{exit !(x-y>2048)}' || [ "$a" -lt 20 ]; then
      log "swap $s MB / avail ${a} GB: NO-SWAP margin gone, stopping the walk"; pkill -TERM -f "port/walk_por[t].py"; break
    fi
    if [ "$(date +%H%M)" -ge "$CAP" ]; then log "cap $CAP reached: stopping the walk"; pkill -TERM -f "port/walk_por[t].py"; break; fi
  done
  wait $pid; log "walk rc=$?: $(grep -E '^  math' data/port/walk_foot.log | cut -c1-150)"
  log "walk footprint $(grep 'peak memory footprint' data/port/walk_foot.log | awk '{printf "%.2f GB", $1/2^30}')"
else
  log "walk re-measure skipped: steps stopped=${stopped:-0}, quiet=$q, ${a} GB available, $(date +%H:%M) (needs 0, 0, >= 26, before $WALK_BY)"
fi
peer unlock 2>&1 | tail -1
[ -f data/port/steps_thu.json ] && ~/TrainLab/.venv-torch/bin/python -I port/compare_l120.py data/port/steps_thu.json
log "DONE"
