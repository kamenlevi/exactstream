#!/bin/bash
# Guarded runner for the drift screens (THINK_1116c; orchestrator 18:2x: <= 5 GB, <= 30 min each; NewIdeas picks).
#   scripts/drift_run.sh PLAN            start detached: scripts/detach.sh logs/drift_NAME_runner.log scripts/drift_run.sh plans/NAME.sh
# PLAN is a bash file that sets NAME and defines plan(), whose body calls
#   step STEP LOG MINUTES CMD...   a GPU step under guard_lib's guarded() with a share lock "Sandbox98 MEM ${DRIFT_MEM} GB
#                                  cap: step STEP" (done steps leave logs/.done/drift_NAME/STEP and are skipped on re-arm)
# plus plain CPU commands, and may set VERDICT (it goes into the END line). Each step's start gate: no HOLD, PAUSE or
# shut lid (with no external display); no quiet ticket; no lock >= ${GUARD_BIG_GB:-30} GB (e.g. BigMoE's 45 GB key
# test); NO-SWAP >= ${GUARD_NOSWAP:-13} GB (the 5 GB cap + 8). While a step runs: a quiet ticket, PAUSE, a shut lid or a
# big lock make it yield and resume; swap +1 GB or NO-SWAP < 3 GB twice kill it and stop the runner; a HOLD stops it.
# AT-MAC YIELD (Kamen 11-16 21:0x; the orchestrator's BREAK98 asks): while Kamen is at the Mac (HIDIdleTime under
# DRIFT_AWAY_MIN = 900 s, ~/.local/share/quiet/away_secs.sh) no step starts and a running step yields, like a shut lid
# (GUARD_HELD_FN = held). DRIFT_ATMAC_YIELD=0 turns it off; DRIFT_ATMACTEST = a test file whose presence means "at the Mac".
# Optional: DRIFT_START_FILE (no start before that file exists) + DRIFT_START_DELAY s.
# Optional start window (epoch seconds): DRIFT_START_AT = no start before then, unless DRIFT_GO_FILE exists (an early
# go, e.g. from the queue's owner); DRIFT_LAST_START = no step STARTS after then (a running step finishes; the runner
# ends with "window closed", finished steps are kept and a re-arm resumes).
# Every end writes one line to logs/drift_NAME_END: "rc 0: done -- VERDICT" or why it stopped. TERM stops cleanly
# (the trap runs once the current 20 s poll sleep ends); never -9 (no marker, the lock stays).
# Tests: DRIFT_DRY=1 runs scripts/l1cf_stub.py for every step with stub lock calls and writes logs/drift_NAME_END_dry;
# DRIFT_QDIR a test quiet dir; DRIFT_LIDTEST a file whose presence means "lid shut".
cd "$(dirname "$0")/.."
PLAN=$1
[ -f "$PLAN" ] || { echo "usage: $0 PLAN (bash: NAME=...; plan() { step ...; })"; exit 2; }
export GUARD_QDIR=${DRIFT_QDIR:-$HOME/.local/share/quiet}
source scripts/guard_lib.sh
NAME=""; VERDICT=""
source "$PLAN"
[ -n "$NAME" ] && declare -F plan >/dev/null || { echo "$PLAN must set NAME and define plan()"; exit 2; }
DRY=${DRIFT_DRY:-0}
if [ "$DRY" = 1 ]; then
  END_MARK=logs/drift_${NAME}_END_dry; PIDF=logs/.drift_job_dry.pid; RUN=drift_dry_$NAME
  peer() { case "$1" in lock) echo "shared run reserved (dry stub)" ;; unlock) echo "unlocked (dry stub)" ;; ls) cat "${DRIFT_FAKE_LS:-/dev/null}" ;; *) command peer "$@" ;; esac; }
else
  END_MARK=logs/drift_${NAME}_END; PIDF=logs/.drift_job.pid; RUN=drift_$NAME
fi
STATUS="armed"; GEXIT=""
say() { echo "$(date '+%a %H:%M:%S') $*"; }
lid_shut() {
  if [ -n "${DRIFT_LIDTEST:-}" ]; then [ -e "$DRIFT_LIDTEST" ]; return; fi
  ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes || return 1
  (( $(system_profiler SPDisplaysDataType -detailLevel mini 2>/dev/null | grep -c "Resolution:") > 1 )) && return 1
  return 0
}
at_mac() {
  if [ -n "${DRIFT_ATMACTEST:-}" ]; then [ -e "$DRIFT_ATMACTEST" ]; return; fi
  [ "${DRIFT_ATMAC_YIELD:-1}" = 1 ] || return 1
  local a; a=$(bash $HOME/.local/share/quiet/away_secs.sh 2>/dev/null)
  [ -n "$a" ] && [ "$a" -lt ${DRIFT_AWAY_MIN:-900} ]
}
held() { lid_shut || at_mac; }
> (a quote from the project owner, kept private)
why_wait() {
  local w=""
> (a quote from the project owner, kept private)
  [ "$(quiet_others)" -gt 0 ] && w="$w a quiet ticket;"; [ "$(big_others)" -gt 0 ] && w="$w a lock >= ${GUARD_BIG_GB} GB;"
  [ "$(noswap)" -lt $GUARD_NOSWAP ] && w="$w NO-SWAP $(noswap) GB;"
  echo "${w:- room now}"
}
finish() {
  local rc=$?
  if [ -f $PIDF ]; then
    local sp=$(cat $PIDF)
    pgrep -g $sp >/dev/null 2>&1 && { say "stopping the step (process group $sp)"; stop_group $sp 120; }
    rm -f $PIDF; peer unlock >/dev/null 2>&1
  fi
  [ -n "$GEXIT" ] && STATUS="$GEXIT"
  echo "$(date '+%a %H:%M:%S') rc $rc: $STATUS" > $END_MARK
  say "END rc $rc: $STATUS"
}
trap finish EXIT
trap 'STATUS="stopped by a signal during: $STATUS"; exit 143' TERM INT
MEMGB=${DRIFT_MEM:-5}
export MEMGB GUARD_MEMNOTE="cap" GUARD_NOSWAP=${GUARD_NOSWAP:-$((MEMGB + 8))} GUARD_BIG_GB=${GUARD_BIG_GB:-30} GUARD_KILL_WAIT=120 \
  GUARD_SWAP_KILL_MB=1000 GUARD_NOSWAP_KILL=${GUARD_NOSWAP_KILL:-3} GUARD_GPU=40 GUARD_SETTLE=${GUARD_SETTLE:-30} \
  GUARD_HELD_FN=held GUARD_STOP_FN=stop_fn GUARD_PIDFILE=$PIDF GUARD_RUN=$RUN SB_MEM_GB=$MEMGB SB_CACHE_GB=${SB_CACHE_GB:-0.25}
rm -f $END_MARK
T=/usr/bin/time
step() {   # NAME LOG MINUTES CMD...: wait for room (swap baseline reset while waiting), then the guarded step
  local name=$1 log=$2 mins=$3; shift 3
  [ -f logs/.done/$RUN/$name ] && { say "$name: done before, skip"; return 0; }
  STATUS="$name: waiting for room"; local n=0
  while true; do
    SWAP0=$(swap_mb)
    stop_fn && { STATUS="$name: not started ($GSTOP)"; exit 8; }
    if [ -n "${DRIFT_LAST_START:-}" ] && [ $(date +%s) -gt $DRIFT_LAST_START ]; then
      STATUS="$name: not started, window closed at $(date -r $DRIFT_LAST_START '+%a %H:%M'); finished steps are kept, re-arm resumes"; exit 9
    fi
    ready && break
    [ $((n % 15)) = 0 ] && say "$name waiting:$(why_wait)"
    n=$((n + 1)); sleep ${DRIFT_POLL:-20}
  done
  say "$name: room ($(why_wait | sed 's/^ //')); swap $(swap_mb) MB, NO-SWAP $(noswap) GB"
  STATUS="$name: running (guarded)"
  [ "$DRY" = 1 ] && log=logs/drift_stub_$name.log
  if [ "$DRY" = 1 ]; then guarded $name $log $mins $PY scripts/l1cf_stub.py; else guarded $name $log $mins "$@"; fi
  local rc=$?
  case $rc in
    0) say "$name done: $(grep -E 'DONE|VERDICT|\[ev\]' $log 2>/dev/null | tail -1 | cut -c1-200)" ;;
    8) STATUS="stopped in $name: $GSTOP; finished steps are kept, re-arm resumes"; exit 8 ;;
    *) STATUS="$name failed (rc $rc): $(grep -vE '^ +[0-9]+ ' $log 2>/dev/null | tail -2 | tr '\n' ' ' | cut -c1-300)"; exit $rc ;;
  esac
}
say "armed (pid $$, dry $DRY, plan $PLAN, name $NAME, MEM $MEMGB GB, NO-SWAP gate $GUARD_NOSWAP GB)"
if [ -n "${DRIFT_START_FILE:-}" ]; then
  STATUS="waiting for $DRIFT_START_FILE"
  say "no start before $DRIFT_START_FILE exists, then ${DRIFT_START_DELAY:-0} s"
  while [ ! -e "$DRIFT_START_FILE" ]; do
    stop_fn && { STATUS="not started ($GSTOP)"; exit 8; }; sleep ${DRIFT_POLL:-20}
  done
  T1=$(( $(date +%s) + ${DRIFT_START_DELAY:-0} ))
  say "$DRIFT_START_FILE is there; start gate opens at $(date -r $T1 '+%a %H:%M:%S')"
  while [ $(date +%s) -lt $T1 ]; do
    stop_fn && { STATUS="not started ($GSTOP)"; exit 8; }; sleep ${DRIFT_POLL:-20}
  done
fi
if [ -n "${DRIFT_START_AT:-}" ]; then
  STATUS="waiting for $(date -r $DRIFT_START_AT '+%a %H:%M')"
  say "no start before $(date -r $DRIFT_START_AT '+%a %d %b %H:%M:%S')${DRIFT_GO_FILE:+ unless $DRIFT_GO_FILE appears}${DRIFT_LAST_START:+; no step starts after $(date -r $DRIFT_LAST_START '+%a %d %b %H:%M:%S')}"
  while [ $(date +%s) -lt $DRIFT_START_AT ]; do
    stop_fn && { STATUS="not started ($GSTOP)"; exit 8; }
    if [ -n "${DRIFT_GO_FILE:-}" ] && [ -e "$DRIFT_GO_FILE" ]; then say "$DRIFT_GO_FILE is there: the start gate opens now"; break; fi
    sleep ${DRIFT_POLL:-20}
  done
fi
plan
STATUS="done, rc 0 -- ${VERDICT:-(no verdict line)}"
exit 0
