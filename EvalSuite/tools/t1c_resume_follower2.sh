#!/bin/bash
# t1c_resume_follower2.sh: the 1c-tiers resume after the orchestrator's pause, second form (PREREG_lossygate amendment 4h;
# the orchestrator 13:0x). It replaces t1c_resume_follower.sh (pid 23473). Two pauses, ONE end marker.
# RESUME 1 (V1 -> ...): the paused chain (pid CHAIN_PREVPID) gone, then the first of:
#   - Sandbox98's logs/sib_END written after the pause began (CHAIN_PAUSE1_AT) whose line reads
#     "<day HH:MM:SS> rc 0: main read-out done ..." (sib_run.sh's format; a failed or stopped attempt's "rc 5" / "rc 143"
#     line never counts);
#   - CHAIN_RESUME_AT (13:30), unless a SIB job runs then (logs/.sib_job.pid's process group alive, or a Sandbox98 shared
#     run in peer ls): then it waits on for the rc 0 line, with CHAIN_RESUME_CAP (15:30) as the cap;
#   - results/T1C_RESUME_NOW written after this arming (EvalSuite's manual override, on the orchestrator's word).
# PAUSE 2 (before B2): when the resumed chain logs B2's start (after V1's PASS and its speed read-out) and
#   logs/sib_READY exists newer than CHAIN_READY_AFTER (13:06), the chain is stopped (TERM; B2's runner is in its CPU
#   setup) and resumed on an rc 0 line written after that stop, CHAIN_PAUSE2_MIN (75) min after it, or T1C_RESUME_NOW
#   written after it, whichever first.
# Each run is tools/t1c_chain2.sh (phase wed on CHAIN_DAY, cutoff CHAIN_CUTOFF, no at-Mac rule), tag mon3, then mon4 after
# pause 2: the done steps are skipped. results/T1C_RESF2_END is written on EVERY end of this follower, and only then: it,
# with this pid gone, is the real end that BigMoE's G1 key test and Breakthrough's map screen wait for. Stop it by pid
# only, with a plain kill (TERM); it passes the TERM to the chain it runs.
#   CHAIN_MYPID=<this session's Claude pid> tools/t1c_resume_follower2.sh
# Tests only: CHAIN_SCRIPT, CHAIN_SBDIR (the Sandbox98 root).
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
L=results/t1c_chain.log; END=results/T1C_RESF2_END; NOW=results/T1C_RESUME_NOW
MYPID=${CHAIN_MYPID:?set CHAIN_MYPID to this session s Claude pid}
PREVPID=${CHAIN_PREVPID:-60135}; SB=${CHAIN_SBDIR:-$HOME/Sandbox98}; CH=${CHAIN_SCRIPT:-tools/t1c_chain2.sh}
MARK=$SB/logs/sib_END; READY=$SB/logs/sib_READY; JOBPID=$SB/logs/.sib_job.pid
FDAY=${CHAIN_DAY:-2026-11-16}; CUTS=${CHAIN_CUTOFF:-2026-11-16 22:30}
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1:00" +%s; }
P1AT=$(ep "${CHAIN_PAUSE1_AT:-2026-11-16 13:06}"); AT=$(ep "${CHAIN_RESUME_AT:-2026-11-16 13:30}")
CAP=$(ep "${CHAIN_RESUME_CAP:-2026-11-16 15:30}"); RAFTER=$(ep "${CHAIN_READY_AFTER:-2026-11-16 13:06}")
P2MIN=${CHAIN_PAUSE2_MIN:-75}; LAST=$(( $(ep "$CUTS") - 900 )); ARMED=$(date +%s)
say() { echo "$(date '+%m-%d %H:%M:%S') [resf2] $*" >> "$L"; }
STATUS="armed"; CPID=""; RC=0
finish() {
  local rc=$?
  if [ -n "$CPID" ] && kill -0 "$CPID" 2>/dev/null; then    # stopped mid-run: the chain stops its runner, unlocks, writes
    kill -TERM "$CPID" 2>/dev/null; wait "$CPID" 2>/dev/null      # its own marker
  fi
  echo "T1C RESF2 END $(date '+%m-%d %H:%M:%S'): $STATUS" > "$END"
  say "end (rc $rc): $STATUS; wrote $END"
}
trap finish EXIT
trap 'STATUS="stopped by a signal during: $STATUS"; exit 143' TERM INT
alive() { ps -o command= -p "$1" 2>/dev/null | grep -q t1c_chain; }   # a chain, not a reused pid
fresh() { [ -e "$1" ] && [ "$(stat -f %m "$1")" -ge "$2" ]; }        # file, epoch: written at or after it
done0() {      # epoch: sib_END written at or after it, reading "<day HH:MM:SS> rc 0: main read-out done"
  fresh "$MARK" "$1" && head -1 "$MARK" | grep -qE '^([A-Z][a-z]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} )?rc 0: main read-out done'
}
sibrun() {     # 0 = a SIB job runs now: its process group alive, or a Sandbox98 shared run in peer ls
  { [ -s "$JOBPID" ] && pgrep -g "$(cat "$JOBPID")" > /dev/null 2>&1; } && return 0
  peer ls 2>/dev/null | grep -i 'small-model sandbox' | grep -q 'shared run'
}
runchain() {   # tag: t1c_chain2.sh as a child; for mon3, pause 2 at B2's start when sib_READY is fresh (P2=1)
  local tag=$1 n0; n0=$(grep -c '^' "$L"); P2=0
  STATUS="the resumed chain (tag $tag)"
  CHAIN_PHASE=wed CHAIN_DAY=$FDAY CHAIN_CUTOFF="$CUTS" CHAIN_NOYIELD=1 CHAIN_TAG=$tag CHAIN_MYPID=$MYPID "$CH" &
  CPID=$!; say "the resumed chain (tag $tag) runs as pid $CPID"
  while kill -0 "$CPID" 2>/dev/null; do
    if [ "$tag" = mon3 ] && [ "$P2" = 0 ] && tail -n +$((n0 + 1)) "$L" | grep -q '\[mon3\] B2 base reverse: CARD_'; then
      if fresh "$READY" "$RAFTER"; then
        say "pause 2: B2 starts and $(basename "$READY") is fresh ($(date -r "$(stat -f %m "$READY")" '+%H:%M:%S')): stopping the chain (pid $CPID) before B2's GPU work"
        kill -TERM "$CPID"; P2=1
      else
        P2=2; say "B2 starts with no fresh $(basename "$READY"): no pause 2"
      fi
    fi
    sleep 2
  done
  wait "$CPID"; RC=$?; CPID=""
}
waitfor() {    # label, from-epoch, then-epoch: until an rc 0 line or T1C_RESUME_NOW after from-epoch, or then-epoch
  local lab=$1 from=$2 until_=$3 n=0
  while :; do
    [ "$(date +%s)" -lt "$LAST" ] || { STATUS="no start by $(date -r "$LAST" '+%m-%d %H:%M') ($lab): no run"; say "$STATUS"; exit 6; }
    done0 "$from" && { WHY="$(basename "$MARK"): $(head -1 "$MARK" | cut -c1-120)"; return 0; }
    fresh "$NOW" "$(( from > ARMED ? from : ARMED ))" && { WHY="$NOW (EvalSuite's manual resume)"; return 0; }
    [ "$(date +%s)" -ge "$until_" ] && { WHY="the time $(date -r "$until_" '+%m-%d %H:%M')"; return 0; }
    n=$((n + 1)); [ $((n % 30)) = 0 ] && say "$lab: waiting for an rc 0 line or $(date -r "$until_" '+%H:%M')"
    sleep 30
  done
}
say "follower2 (pid $$; session $MYPID): resume 1 once pid $PREVPID is gone, on an rc 0 line after $(date -r "$P1AT" '+%H:%M'), at $(date -r "$AT" '+%H:%M') unless a SIB job runs (then the rc 0 line, cap $(date -r "$CAP" '+%H:%M')); pause 2 before B2 if $(basename "$READY") is newer than $(date -r "$RAFTER" '+%H:%M') ($P2MIN min or an rc 0 line); one end marker $END"
n=0; SAIDW=0
while :; do                                                   # resume 1
  [ "$(date +%s)" -lt "$LAST" ] || { STATUS="no start by $(date -r "$LAST" '+%m-%d %H:%M'): no run"; say "$STATUS"; exit 6; }
  if ! alive "$PREVPID"; then
    done0 "$P1AT" && { WHY="$(basename "$MARK"): $(head -1 "$MARK" | cut -c1-120)"; break; }
    fresh "$NOW" "$ARMED" && { WHY="$NOW (EvalSuite's manual resume)"; break; }
    t=$(date +%s)
    [ "$t" -ge "$CAP" ] && { WHY="the cap $(date -r "$CAP" '+%m-%d %H:%M')"; break; }
    if [ "$t" -ge "$AT" ]; then
      if sibrun; then
        [ "$SAIDW" = 1 ] || { say "$(date -r "$AT" '+%H:%M'): a SIB job runs: waiting for its rc 0 line (cap $(date -r "$CAP" '+%H:%M'))"; SAIDW=1; }
      else
        WHY="$(date -r "$AT" '+%H:%M') with no SIB job running"; break
      fi
    fi
  fi
  n=$((n + 1)); [ $((n % 30)) = 0 ] && say "waiting (resume 1): pid $PREVPID $(alive "$PREVPID" && echo 'still running' || echo gone)"
  sleep 30
done
say "resume 1: $WHY"
runchain mon3
if [ "$P2" = 1 ]; then                                        # pause 2, then the rest (B2 -> V2)
  P2AT=$(date +%s); STATUS="pause 2 (before B2)"; say "pause 2 began: the chain ended rc $RC"
  waitfor "pause 2" "$P2AT" "$(( P2AT + P2MIN * 60 ))"
  say "resume 2: $WHY"
  runchain mon4
  STATUS="done: the resumed chain mon4 ended rc $RC ($(head -c 160 results/T1C_MON4_END 2>/dev/null | tr '\n' ' '))"
else
  STATUS="done: the resumed chain mon3 ended rc $RC ($(head -c 160 results/T1C_MON3_END 2>/dev/null | tr '\n' ' '))"
fi
exit "$RC"
