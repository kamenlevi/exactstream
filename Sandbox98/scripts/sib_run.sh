#!/bin/bash
# SIB runner (results/LEVERS_A.md, section SIB; NewIdeas 12:2x, PLAN900 bet 9b). Steps, each under guard_lib's guarded()
# with a share lock "Sandbox98 MEM 5 GB cap": the smoke (3 + 3 windows) and its harness check, the sibling rows of
# HE 60 + MBPP 100, the frozen s30's score, the pool's sibling rows, the fine-tune (ctl vs sibmix, one pass over the
# pool), the tuned heads' score, and the verdict (CPU). A finished step leaves logs/.done/sib/NAME and is skipped when
# the runner is re-armed.
# - Start gate (each step): no HOLD, PAUSE or shut lid (with no external display); no quiet ticket; no lock >= 30 GB
#   (it runs in EvalSuite's pause, orchestrator 13:1x: EvalSuite pauses when logs/sib_READY exists and resumes on
#   logs/sib_END); NO-SWAP >= 13 GB (the 5 GB cap + 8).
# - While a step runs: a quiet ticket, PAUSE, a shut lid or a >= 30 GB lock make it yield and resume; swap +1 GB or
#   NO-SWAP < 3 GB twice kill it and stop the runner; a HOLD stops it.
# - Phases (orchestrator 12:4x): SIB_PHASE=1 runs the smoke and the main read-out and ends with logs/sib_END (EvalSuite
#   resumes on it); SIB_PHASE=2 runs the fine-tune variant (optionally not before SIB_START_AFTER) and ends with
#   logs/sib_ft_END. Every end writes one status line there. A plain TERM stops cleanly; -9 skips the marker.
# Tests: SIB_DRY=1 swaps every step for scripts/l1cf_stub.py (stub lock calls), writes logs/sib_END_dry; SIB_QDIR is a
# test quiet dir; SIB_LIDTEST a file whose presence means "lid shut".
cd "$(dirname "$0")/.."
export GUARD_QDIR=${SIB_QDIR:-$HOME/.local/share/quiet}
source scripts/guard_lib.sh
DRY=${SIB_DRY:-0}
PHASE=${SIB_PHASE:-1}   # 1: smoke + the main read-out (END logs/sib_END: EvalSuite resumes on it); 2: the fine-tune
EP=$([ "$PHASE" = 2 ] && echo _ft)
if [ "$DRY" = 1 ]; then
  END_MARK=logs/sib${EP}_END_dry; PIDF=logs/.sib_job_dry.pid; RUN=sib_dry
  peer() { case "$1" in lock) echo "shared run reserved (dry stub)" ;; unlock) echo "unlocked (dry stub)" ;; ls) cat "${SIB_FAKE_LS:-/dev/null}" ;; *) command peer "$@" ;; esac; }   # tests see no other sessions unless SIB_FAKE_LS
else
  END_MARK=logs/sib${EP}_END; PIDF=logs/.sib_job.pid; RUN=sib
fi
STATUS="armed"; GEXIT=""
say() { echo "$(date '+%a %H:%M:%S') $*"; }
lid_shut() {
  if [ -n "${SIB_LIDTEST:-}" ]; then [ -e "$SIB_LIDTEST" ]; return; fi
  ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes || return 1
  (( $(system_profiler SPDisplaysDataType -detailLevel mini 2>/dev/null | grep -c "Resolution:") > 1 )) && return 1
  return 0
}
> (a quote from the project owner, kept private)
why_wait() {
  local w=""
  [ -e $QD/PAUSE ] && w="$w PAUSE;"; lid_shut && w="$w lid shut;"
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
export MEMGB=5 GUARD_MEMNOTE="cap" GUARD_NOSWAP=${GUARD_NOSWAP:-13} GUARD_BIG_GB=${GUARD_BIG_GB:-30} GUARD_KILL_WAIT=120 GUARD_SWAP_KILL_MB=1000 \
  GUARD_NOSWAP_KILL=${GUARD_NOSWAP_KILL:-3} GUARD_GPU=40 GUARD_SETTLE=${GUARD_SETTLE:-30} GUARD_HELD_FN=lid_shut \
  GUARD_STOP_FN=stop_fn GUARD_PIDFILE=$PIDF GUARD_RUN=$RUN SB_MEM_GB=5 SB_CACHE_GB=${SB_CACHE_GB:-0.25}
rm -f $END_MARK
step() {   # NAME LOG MINUTES CMD...: wait for room (swap baseline reset while waiting), then the guarded step
  local name=$1 log=$2 mins=$3; shift 3
  [ -f logs/.done/$RUN/$name ] && { say "$name: done before, skip"; return 0; }
  STATUS="$name: waiting for room"; local n=0
  while true; do
    SWAP0=$(swap_mb)
    stop_fn && { STATUS="$name: not started ($GSTOP)"; exit 8; }
    ready && break
    [ $((n % 15)) = 0 ] && say "$name waiting:$(why_wait)"
    n=$((n + 1)); sleep ${SIB_POLL:-20}
  done
  say "$name: room ($(why_wait | sed 's/^ //')); swap $(swap_mb) MB, NO-SWAP $(noswap) GB"
  STATUS="$name: running (guarded)"
  [ "$DRY" = 1 ] && log=logs/sib_stub_$name.log
  if [ "$DRY" = 1 ]; then guarded $name $log $mins $PY scripts/l1cf_stub.py; else guarded $name $log $mins "$@"; fi
  local rc=$?
  case $rc in
    0) say "$name done: $(grep -E 'DONE|SMOKECHECK|VERDICT' $log 2>/dev/null | tail -1 | cut -c1-200)" ;;
    8) STATUS="stopped in $name: $GSTOP; finished steps are kept, re-arm resumes"; exit 8 ;;
    *) STATUS="$name failed (rc $rc): $(grep -vE '^ +[0-9]+ ' $log 2>/dev/null | tail -2 | tr '\n' ' ' | cut -c1-300)"; exit $rc ;;
  esac
}
say "armed (pid $$, dry $DRY, phase $PHASE)"
T=/usr/bin/time
if [ -n "${SIB_START_AFTER:-}" ]; then     # either phase: not before this time (e.g. let EvalSuite take its lock first)
  STATUS="waiting for $(date -r $SIB_START_AFTER '+%a %H:%M')"
  say "no start before $(date -r $SIB_START_AFTER '+%a %H:%M')"
  while [ $(date +%s) -lt $SIB_START_AFTER ]; do
    stop_fn && { STATUS="not started ($GSTOP)"; exit 8; }; sleep ${SIB_POLL:-20}
  done
fi
if [ -n "${SIB_START_FILE:-}" ]; then      # either phase: not before this file exists, then SIB_START_DELAY s more (15:0x:
  # phase 2 slipped in while EvalSuite waited for memory; it now waits for EvalSuite's chain end, results/T1C_RESF2_END,
  # and lets the jobs gated on the same file take their locks first)
  STATUS="waiting for $SIB_START_FILE"
  say "no start before $SIB_START_FILE exists, then ${SIB_START_DELAY:-0} s"
  while [ ! -e "$SIB_START_FILE" ]; do
    stop_fn && { STATUS="not started ($GSTOP)"; exit 8; }; sleep ${SIB_POLL:-20}
  done
  T1=$(( $(date +%s) + ${SIB_START_DELAY:-0} ))
  say "$SIB_START_FILE is there; start gate opens at $(date -r $T1 '+%a %H:%M:%S')"
  while [ $(date +%s) -lt $T1 ]; do
    stop_fn && { STATUS="not started ($GSTOP)"; exit 8; }; sleep ${SIB_POLL:-20}
  done
fi
if [ "$PHASE" = 1 ]; then
  step sib_smoke_build logs/sib_smoke_build.log 15 $T -l $PY sb/sibstate.py build --sets he,mbpp --n 3 --tag _smoke
  step sib_smoke_score logs/sib_smoke_score.log 15 $T -l $PY sb/sibstate.py score --heads s30 --sets he,mbpp --n 3 --tag _smoke --out results/sib_smoke.json
  if [ "$DRY" != 1 ]; then
    $PY -I sb/sibstate.py smokecheck --sets he,mbpp --tag _smoke --out results/sib_smoke.json > logs/sib_smokecheck.log 2>&1 \
      || { STATUS="smoke check failed: $(tail -1 logs/sib_smokecheck.log | cut -c1-250)"; exit 5; }
    say "$(tail -1 logs/sib_smokecheck.log)"
  fi
  step sib_build_eval logs/sib_build_eval.log 25 $T -l $PY sb/sibstate.py build --sets he,mbpp
  step sib_score_s30 logs/sib_score_s30.log 20 $T -l $PY sb/sibstate.py score --heads s30 --sets he,mbpp --out results/sib_s30.json
  if [ "$DRY" != 1 ]; then
    $PY -I sb/sibstate.py verdict --frozen results/sib_s30.json --out results/sib_verdict_main.json > logs/sib_verdict_main.log 2>&1
    STATUS="main read-out done, rc 0 -- $(tail -1 logs/sib_verdict_main.log | cut -c1-250)"
  else
    STATUS="main read-out done, rc 0 -- (dry)"
  fi
  exit 0
fi
step sib_build_pool logs/sib_build_pool.log 25 $T -l $PY sb/sibstate.py build --sets pool
step sib_ft logs/sib_ft.log 25 $T -l $PY sb/train_multi.py --tag sib1 --arms ctl:-,sib:sibmix --files pool --steps 579 --depth 12 --lr 1e-5 --mem 5
step sib_score_ft logs/sib_score_ft.log 30 $T -l $PY sb/sibstate.py score --heads sib1_ctl,sib1_sib --sets he,mbpp --out results/sib_ft.json
if [ "$DRY" != 1 ]; then
  $PY -I sb/sibstate.py verdict --frozen results/sib_s30.json --ft results/sib_ft.json --out results/sib_verdict.json > logs/sib_verdict.log 2>&1
  STATUS="done: $(tail -1 logs/sib_verdict.log | cut -c1-250)"
else
  STATUS="done (dry)"
fi
exit 0
