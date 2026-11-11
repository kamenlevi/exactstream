#!/bin/bash
# MTP Head, Wed 11-04 22:2x (the orchestrator 22:2x): P-OFF's 60K training + code walk goes FIRST in the Fri night, ahead
# of Leap's NATIVE-FRI arms and BigMoE's FAR-G1 capture. REPORT 'P-OFF amendment' (60K lines: PASS >= +0.18 code,
# KILL <= +0.05) and 'P-OFF diagnostic RESULT' (alive, 0.86).
#  1. Not before $START (Thu 2026-11-05 23:45; was Fri 00:30, see the amendment below).
#  2. A QUEUED quiet ticket (peer lock quiet --queue 90m; PEER_QUEUE_TIMEOUT 3 h): the queue shows us #1 in line, and the
#     grant comes once T0 (EvalSuite's 14 GB share job, to ~00:15) has ended and the Mac has drained. No grant by ~03:30
#     -> stop (the orchestrator re-slots). Nothing of mine may `peer unlock` meanwhile (it would drop the ticket).
#  3. Under the quiet lock: sticky/poff_data.py (idempotent), sticky/poff_train.py from L1-20 (1 epoch, --minutes 55,
#     L1-20's recipe), the code walk (TL_WALK_TAG=_poff), then unlock. Swap guard +2 GB (SIGTERM: the trainer
#     checkpoints). This replaces turns/poff_t.sh, which takes a share lock that would overwrite the quiet one.
# Run attached (run_in_background) so the lock sees the process. Logs: data/poff_fri.log, data/poff_train.log,
# data/poff_walk.log.
> (a quote from the project owner, kept private)
# Thu 17:53; only Leap's FAST-NAX share check (~23:25-23:40) is ahead; BigMoE queues behind us at 23:47. Steps unchanged.
cd ~/TrainLab || exit 1
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
swap_mb() { sysctl -n vm.swapusage | awk '{print $6}' | tr -d M; }
PY=~/BigMoE/.venv/bin/python3
OUT=$HOME/TrainLab/ext/poff/poff60.safetensors
START=${START:-2026-11-05 23:45}
TARGET=$(date -j -f "%Y-%m-%d %H:%M" "$START" +%s)
log "1. waiting for $START ($(( (TARGET - $(date +%s)) / 60 )) min)"
while [ "$(date +%s)" -lt "$TARGET" ]; do sleep 60; done
log "2. queueing a quiet ticket (waits for T0's end and a drained Mac; leaves the queue after 3 h)"
> (a quote from the project owner, kept private)
rc=$?
log "ticket rc=$rc: $(tail -1 data/poff_fri_lock.log)"
if [ $rc -ne 0 ] || ! grep -q "quiet lock held" data/poff_fri_lock.log; then log "NO GRANT: stop (ask the orchestrator)"; exit 4; fi
guarded() {   # $1 = pkill pattern, rest = command; output to $LOGF
  local pat=$1; shift
  local s0; s0=$(swap_mb); log "start (swap $s0 MB): $*"
  "$@" > "$LOGF" 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 20
    local s; s=$(swap_mb)
    if awk -v a="$s" -v b="$s0" 'BEGIN{exit !(a-b>2048)}'; then
      log "swap $s MB = +2 GB over the start: SIGTERM"; pkill -TERM -f "$pat"; break
    fi
  done
  wait $pid; local rc=$?
  log "rc=$rc (swap $(swap_mb) MB)"
  return $rc
}
log "3. data folder"
$PY -I sticky/poff_data.py --tokens 60000 --out $HOME/TrainLab/ext/poff/train60 | tee -a data/poff_fri.data.log | tail -1
LOGF=data/poff_train.log guarded "poff_trai[n].py" /usr/bin/time -l $PY sticky/poff_train.py --init ~/Leap/data/l120_ft.safetensors \
  --epochs 1 --minutes 55 --lr 2e-5 --S 4 --A 64 --micro 1 --bs 32 --pmask 0.5 --data ~/TrainLab/ext/poff/train60 --out $OUT
rc=$?
if [ $rc -eq 0 ] && grep -q "DONE: saved" data/poff_train.log; then
  LOGF=data/poff_walk.log guarded "d02b_wal[k].py" env TL_WALK_SETS="code=$HOME/Leap/data/dfl2d_code/*.npz" TL_DRAFT=$OUT \
    TL_WALK_TAG=_poff /usr/bin/time -l $PY sticky/d02b_walk.py
  log "walk: $(grep -E 'kept per check' data/poff_walk.log | tail -1 | cut -c1-200)"
else
  log "training did not finish (rc=$rc): no walk"
fi
peer unlock 2>&1 | tail -1
log "DONE"
