#!/bin/bash
# MTP Head, Fri 11-06 01:4x (the orchestrator 01:2x): SEED probe = L1-20's recipe rerun with --seed 1, then the D-02b
# walk (code + math), to size retrain noise (REPORT 'SEED: L1-20 retrain noise', pre-registered). After Leap's
# NATIVE-FRI and FLAT-2; the chunk must END by 10:15 (q45 GO and runner 45 keep their slot).
#  1. Not before 02:10. Then wait for 3 clear minutes with no other session's quiet lock or queued ticket.
#  2. A queued quiet ticket (MEM 37 GB; it leaves the queue at 09:25). No start after 09:25.
#  3. Under the lock: a compile check, sticky/seed_train.py --seed 1 with L1-20's args (swap guard +2 GB; a stop at
#     10:12), then sticky/d02b_walk.py on the new weights (TL_WALK_TAG=_seed1), then unlock.
# Run attached (run_in_background) so the lock sees the process. Logs: data/seed_l120.log (this), data/seed_lock.log,
# data/seed_train.log, data/seed_walk.log, data/d02b_walk_seed1.txt.
cd ~/TrainLab || exit 1
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
swap_mb() { sysctl -n vm.swapusage | awk '{print $6}' | tr -d M; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
nquiet() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
at() { date -j -f "%Y-%m-%d %H:%M" "$1" +%s; }
PY=~/BigMoE/.venv/bin/python3
OUT=$HOME/TrainLab/ext/seed/l120_s1.safetensors
NOTBEFORE=$(at "2026-11-06 02:10"); LASTSTART=$(at "2026-11-06 09:25"); STOPAT=$(at "2026-11-06 10:12")
log "1. not before 02:10, then 3 clear minutes with no other quiet lock or ticket"
while [ "$(date +%s)" -lt "$NOTBEFORE" ]; do sleep 30; done
clear=0
while :; do
  [ "$(date +%s)" -ge "$LASTSTART" ] && { log "NO START: no clear gap by 09:25"; exit 6; }
  q=$(nquiet)
  if [ "$q" -eq 0 ]; then clear=$((clear + 1)); else clear=0; fi
  [ "$clear" -ge 6 ] && break
  sleep 30
done
TMO=$(( LASTSTART - $(date +%s) ))
log "2. clear for 3 min: queueing a quiet ticket (it leaves the queue in $((TMO / 60)) min, at 09:25)"
PEER_QUEUE_TIMEOUT=$TMO peer lock quiet --queue 75m "MEM 37 GB: MTP Head SEED probe, L1-20 rerun seed 1 + code/math walk (~40 min; the orchestrator 01:2x)" > data/seed_lock.log 2>&1
rc=$?
log "ticket rc=$rc: $(tail -1 data/seed_lock.log)"
if [ $rc -ne 0 ] || ! grep -q "quiet lock held" data/seed_lock.log; then log "NO GRANT: stop"; exit 4; fi
if [ "$(date +%s)" -ge "$LASTSTART" ]; then log "granted after 09:25: too late to end by 10:15"; peer unlock 2>&1 | tail -1; exit 6; fi
if ! $PY -I -m py_compile sticky/seed_train.py; then log "seed_train.py does not compile"; peer unlock 2>&1 | tail -1; exit 7; fi
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
    if [ "$(date +%s)" -ge "$STOPAT" ]; then log "10:12: SIGTERM (the chunk must end by 10:15)"; pkill -TERM -f "$pat"; break; fi
  done
  wait $pid; local rc=$?
  log "rc=$rc (swap $(swap_mb) MB)"
  return $rc
}
log "3. training"
LOGF=data/seed_train.log guarded "seed_trai[n].py" /usr/bin/time -l $PY sticky/seed_train.py --seed 1 --epochs 2 --minutes 50 \
  --lr 2e-5 --S 4 --A 64 --val-every 100 --micro 1 --cache-gb 2.0 --bs 32 --pmask 0.5 --data ~/Leap/data/l120 --out $OUT
rc=$?
if [ $rc -eq 0 ] && grep -q "DONE: saved" data/seed_train.log; then
  LOGF=data/seed_walk.log guarded "d02b_wal[k].py" env TL_DRAFT=$OUT TL_WALK_TAG=_seed1 /usr/bin/time -l $PY sticky/d02b_walk.py
  log "walk: $(grep -E 'kept per check' data/seed_walk.log | cut -c1-140 | tr '\n' ' ')"
else
  log "training did not finish (rc=$rc): no walk"
fi
peer unlock 2>&1 | tail -1
log "DONE"
