#!/bin/bash
# t1c_resume_follower.sh: resume the 1c-tiers gate after the orchestrator's pause (PREREG_lossygate amendment 4g; the
# orchestrator 12:4x: chain2 stopped at the clean boundary after B1, V1 not started, so Sandbox98's SIB test gets the GPU).
# Start: the paused chain (pid CHAIN_PREVPID, tools/t1c_chain2.sh) is gone, then EITHER Sandbox98's end marker
# (CHAIN_RESUME_MARK) written after the pause began (first seen with the chain gone) whose first line starts with
# CHAIN_RESUME_PREFIX ("rc 0: main read-out done": the orchestrator 12:5x; a stopped attempt's "rc 143: ..." never
# counts), OR the resume time (CHAIN_RESUME_AT), whichever comes first.
# Then it runs tools/t1c_chain2.sh (phase wed on CHAIN_DAY, tag CHAIN_TAG, cutoff CHAIN_CUTOFF, no at-Mac rule): the done
# steps are skipped (a recorded VOID / INCOMPLETE stops it), then V1 -> B2 -> V2 under that chain's own guards and
# end marker (results/T1C_<TAG>_END). No start after the cutoff - 15 min. EVERY end of this follower writes
# results/T1C_RESF_END. Stop it by pid only, with a plain kill (TERM); it passes the TERM to the chain it started.
#   CHAIN_MYPID=<this session's Claude pid> tools/t1c_resume_follower.sh
# Settings (defaults = today's plan): CHAIN_PREVPID 60135, CHAIN_RESUME_MARK ~/Sandbox98/logs/sib_END, CHAIN_RESUME_AT
# "2026-11-16 15:30", CHAIN_DAY 2026-11-16, CHAIN_CUTOFF "2026-11-16 22:30", CHAIN_TAG mon3. Tests only: CHAIN_SCRIPT.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
L=results/t1c_chain.log; END=results/T1C_RESF_END
MYPID=${CHAIN_MYPID:?set CHAIN_MYPID to this session s Claude pid}
PREVPID=${CHAIN_PREVPID:-60135}; MARK=${CHAIN_RESUME_MARK:-$HOME/Sandbox98/logs/sib_END}; CH=${CHAIN_SCRIPT:-tools/t1c_chain2.sh}
PFX=${CHAIN_RESUME_PREFIX:-rc 0: main read-out done}; PAUSED=""
FDAY=${CHAIN_DAY:-2026-11-16}; CUTS=${CHAIN_CUTOFF:-2026-11-16 22:30}; TAG=${CHAIN_TAG:-mon3}
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1:00" +%s; }
AT=$(ep "${CHAIN_RESUME_AT:-2026-11-16 15:30}"); LAST=$(( $(ep "$CUTS") - 900 )); ARMED=$(date +%s)
say() { echo "$(date '+%m-%d %H:%M:%S') [resf] $*" >> "$L"; }
STATUS="armed"; CPID=""
finish() {
  local rc=$?
  if [ -n "$CPID" ] && kill -0 "$CPID" 2>/dev/null; then    # stopped mid-run: the chain stops its runner, unlocks, writes
    kill -TERM "$CPID" 2>/dev/null; wait "$CPID" 2>/dev/null      # its own marker
  fi
  echo "T1C RESF END $(date '+%m-%d %H:%M:%S'): $STATUS" > "$END"
  say "end (rc $rc): $STATUS; wrote $END"
}
trap finish EXIT
trap 'STATUS="stopped by a signal during: $STATUS"; exit 143' TERM INT
alive() { ps -o command= -p "$PREVPID" 2>/dev/null | grep -q t1c_chain; }   # the paused chain, not a reused pid
marked() {      # 0 = the marker was written after the pause began and its first line starts with the done prefix
  [ -n "$PAUSED" ] && [ -e "$MARK" ] && [ "$(stat -f %m "$MARK")" -ge "$PAUSED" ] || return 1
  case "$(head -1 "$MARK")" in "$PFX"*) return 0 ;; *) return 1 ;; esac
}
say "follower (pid $$; session $MYPID): resumes the 1c-tiers gate (tag $TAG, cutoff $CUTS) once pid $PREVPID is gone AND ($MARK written after the pause began, starting '$PFX', OR $(date -r "$AT" '+%m-%d %H:%M'))"
n=0
while :; do
  [ "$(date +%s)" -lt "$LAST" ] || { STATUS="no start by $(date -r "$LAST" '+%m-%d %H:%M'): no run"; say "$STATUS"; exit 6; }
  if ! alive; then
    [ -n "$PAUSED" ] || { PAUSED=$(date +%s); say "the paused chain (pid $PREVPID) is gone: the pause began (marker reads count from now)"; }
    if marked; then WHY="$MARK written $(date -r "$(stat -f %m "$MARK")" '+%H:%M:%S'): $(head -1 "$MARK" | cut -c1-120)"; break; fi
    [ "$(date +%s)" -ge "$AT" ] && { WHY="the resume time $(date -r "$AT" '+%m-%d %H:%M')"; break; }
  fi
  n=$((n + 1)); [ $((n % 30)) = 0 ] && say "waiting: pid $PREVPID $(alive && echo 'still running' || echo gone); $(basename "$MARK") $(marked && echo written || echo 'not yet')"
  sleep 30
done
say "start: $WHY"
STATUS="the resumed chain (tag $TAG): V1 -> B2 -> V2"
CHAIN_PHASE=wed CHAIN_DAY=$FDAY CHAIN_CUTOFF="$CUTS" CHAIN_NOYIELD=1 CHAIN_TAG=$TAG CHAIN_MYPID=$MYPID "$CH" &
CPID=$!; say "the resumed chain runs as pid $CPID"; wait "$CPID"; rc=$?; CPID=""
UT=$(echo "$TAG" | tr a-z A-Z)
STATUS="done: the resumed chain ended rc $rc ($(head -c 200 "results/T1C_${UT}_END" 2>/dev/null | tr '\n' ' '))"
exit "$rc"                                                   # the chain's own rc
