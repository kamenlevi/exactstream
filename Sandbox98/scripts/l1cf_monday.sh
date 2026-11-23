#!/bin/bash
# L1CF Monday runner (results/LEVERS_A.md, section L1CF; orchestrator 21:4x). It runs one epoch of sb/lever1cf.py by day
# on Mon 16 Nov.
# - Start gate:
#   - not before 07:00 (L1CF_START), and no first start after 20:30 (L1CF_LATEST);
#   - not while HOLD, PAUSE or a shut lid (with no external display) holds;
#   - no quiet ticket, no other lock above MEM 4 GB, and no shared run at >= 50 % GPU (its MEM tag may be cut);
#   - NO-SWAP >= 14 GB (the 7 GB cap + Kamen's 5 GB floor + 2).
# - Hard end: at 22:30 (L1CF_END) the job gets a TERM and saves its state, leaving room for Monday night's runs.
# - While the job runs (guard_lib's guarded):
#   - a quiet ticket, PAUSE, a shut lid or a big slot makes it yield: it saves its state and resumes when the Mac is free;
#   - swap +1 GB or NO-SWAP < 5 GB twice (Kamen's floor) kills it and stops the runner;
#   - a HOLD stops it (TERM; the job saves its state) and ends the runner.
# - The share lock says "MEM 7 GB", the orchestrator's cap, with the smoke's measured peak in its note. It is re-sized
#   from the run's own pace.
# - Every end writes one status line to logs/l1cf_END. A background watcher in the Sandbox98 session waits on that file.
#   A plain kill (TERM) of this runner stops the job cleanly too; -9 skips the marker.
# Tests: L1CF_DRY=1 swaps in a stub job (scripts/l1cf_stub.py) and stub lock calls, and writes logs/l1cf_END_dry.
#   L1CF_QDIR is a test quiet dir, and L1CF_LIDTEST is a file whose presence means "lid shut".
cd "$(dirname "$0")/.."
export GUARD_QDIR=${L1CF_QDIR:-$HOME/.local/share/quiet}
source scripts/guard_lib.sh
DRY=${L1CF_DRY:-0}
at() { date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-16 $1:00" +%s; }
START=${L1CF_START:-$(at 07:00)}; LATEST=${L1CF_LATEST:-$(at 20:30)}; HARD=${L1CF_END:-$(at 22:30)}
if [ "$DRY" = 1 ]; then
  END_MARK=logs/l1cf_END_dry; PIDF=logs/.l1cf_job_dry.pid; RUN=l1cf_dry; LOG=logs/l1cf_stub.log
  JOB=(/usr/bin/time -l $PY scripts/l1cf_stub.py)
  peer() { case "$1" in lock) echo "shared run reserved (dry stub)" ;; unlock) echo "unlocked (dry stub)" ;; *) command peer "$@" ;; esac; }
else
  END_MARK=logs/l1cf_END; PIDF=logs/.l1cf_job.pid; RUN=l1cf; LOG=logs/lever1cf_a.log
  JOB=(/usr/bin/time -l $PY sb/lever1cf.py --tag a)
fi
STATUS="armed"; GEXIT=""
say() { echo "$(date '+%a %H:%M:%S') $*"; }
lid_shut() {   # EvalSuite's runner-64 check: lid closed and no external display
  if [ -n "${L1CF_LIDTEST:-}" ]; then [ -e "$L1CF_LIDTEST" ]; return; fi
  ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes || return 1
  (( $(system_profiler SPDisplaysDataType -detailLevel mini 2>/dev/null | grep -c "Resolution:") > 1 )) && return 1
  return 0
}
stop_fn() {    # GUARD_STOP_FN: Kamen's HOLD or the hard end stop the job for good
> (a quote from the project owner, kept private)
  [ $(date +%s) -ge $HARD ] && { GSTOP="the $(date -r $HARD +%H:%M) end, room for Monday night's runs"; return 0; }
  return 1
}
why_wait() {
  local w=""
  [ -e $QD/PAUSE ] && w="$w PAUSE;"; lid_shut && w="$w lid shut;"
  [ "$(quiet_others)" -gt 0 ] && w="$w a quiet ticket;"; [ "$(mem_others)" -gt 0 ] && w="$w a lock above MEM 4 GB;"
  [ "$(gpu_others)" -gt 0 ] && w="$w a shared run at >= 50 % GPU;"; [ "$(noswap)" -lt $GUARD_NOSWAP ] && w="$w NO-SWAP $(noswap) GB;"
  echo "${w:- room now}"
}
finish() {
  local rc=$?
  if [ -f $PIDF ]; then
    local sp=$(cat $PIDF)
    pgrep -g $sp >/dev/null 2>&1 && { say "stopping the job (process group $sp); it saves its state"; stop_group $sp 120; }
    rm -f $PIDF; peer unlock >/dev/null 2>&1
  fi
  [ -n "$GEXIT" ] && STATUS="$GEXIT"
  echo "$(date '+%a %H:%M:%S') rc $rc: $STATUS" > $END_MARK
  say "END rc $rc: $STATUS"
}
trap finish EXIT
trap 'STATUS="stopped by a signal during: $STATUS"; exit 143' TERM INT
export MEMGB=7 GUARD_MEMNOTE="cap (smoke peak ${L1CF_SMOKE_PEAK:-?} GB)" GUARD_NOSWAP=14 NO_MEM_OTHERS=1 GUARD_MEM_OTHERS_GT=4 \
  GUARD_GPU_OTHERS_MAX=50 GUARD_BIG_GB=10 GUARD_KILL_WAIT=120 GUARD_SWAP_KILL_MB=1000 GUARD_NOSWAP_KILL=${GUARD_NOSWAP_KILL:-5} GUARD_GPU=60 \
  GUARD_TAIL_STEPS=500 GUARD_SETTLE=${GUARD_SETTLE:-30} GUARD_HELD_FN=lid_shut GUARD_STOP_FN=stop_fn GUARD_PIDFILE=$PIDF \
  GUARD_RUN=$RUN
rm -f $END_MARK
say "armed (pid $$, dry $DRY): start from $(date -r $START '+%a %H:%M'), no first start after $(date -r $LATEST '+%a %H:%M'), stop at $(date -r $HARD '+%a %H:%M')"
STATUS="waiting for $(date -r $START '+%a %H:%M')"
while [ $(date +%s) -lt $START ]; do
  [ -e $QD/HOLD ] && { STATUS="stopped by HOLD before the start (no run)"; exit 8; }
  sleep ${L1CF_POLL:-20}
done
STATUS="waiting for room"; n=0
while true; do
  SWAP0=$(swap_mb)                                    # swap that grows while nothing of mine runs is not mine
  stop_fn && { STATUS="no start: $GSTOP"; exit 8; }
  ready && break
  [ $(date +%s) -ge $LATEST ] && { STATUS="no start by $(date -r $LATEST +%H:%M): no room ($(why_wait))"; exit 6; }
  [ $((n % 15)) = 0 ] && say "waiting:$(why_wait)"
  n=$((n + 1)); sleep ${L1CF_POLL:-20}
done
say "room:$(why_wait); swap $(swap_mb) MB, NO-SWAP $(noswap) GB; starting the guarded run"
STATUS="running (guarded)"
guarded l1cf_a $LOG 45 "${JOB[@]}"; rc=$?
case $rc in
  0) STATUS="done: $(grep -E 'DONE' $LOG | tail -1)" ;;
  8) STATUS="stopped: $GSTOP; the job saved its state and resumes from it when re-armed" ;;
  *) STATUS="failed (rc $rc): $(grep -vE '^ +[0-9]+ ' $LOG | tail -2 | tr '\n' ' ' | cut -c1-300)" ;;
esac
exit $rc
