#!/bin/bash
# t1c_tue_follower.sh: Tuesday's B2 -> V2 for the 1c-tiers gate (NewIdeas 09:3x: if V1 passes, the reverse pair runs on
# TUESDAY by day, not Wednesday; PREREG_lossygate amendment 4d). A separate follower, because the armed chain's file
# (tools/t1c_chain.sh, sha256 d7d9001d6549811d) serves today's phase (pid 86890) and must not change under it.
# Start: results/T1C_TUE_END written after this follower was armed and saying "V1 PASS" (the tue phase found or made
# V1's PASS), with the tue phase (pid 86926) gone; read twice, 30 s apart. Any other tue end (a KILL, a FAIL, a stop, a
# cut, a HOLD, no run) or the tue phase gone without a new marker: no run.
# Then it runs tools/t1c_chain.sh's wed phase with Tuesday's date (CHAIN_PHASE=wed CHAIN_DAY=2026-11-17): the done steps
# are skipped, then B2 -> V2, under that chain's own guards (HOLD / PAUSE / lid / Kamen at the Mac; no start after
# 14:15; cutoff 14:30) and its end marker results/T1C_WED_END (its log lines read "[wed]" with Tuesday's date).
# No start after Tue 14:15 here either. EVERY end of this follower writes results/T1C_TUEF_END. Stop it by pid only, with
# a plain kill (TERM); it passes the TERM to the chain it started.
# CHAIN_DRY=1: the chain's wed phase as a dry run, straight away.
#   CHAIN_MYPID=<this session's Claude pid> [CHAIN_DRY=1] tools/t1c_tue_follower.sh
# Tests only: CHAIN_TUEEND, CHAIN_TUEPID, CHAIN_LAST (YYYY-mm-dd HH:MM), CHAIN_SCRIPT, CHAIN_FDAY (the chain's day).
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
DRY=${CHAIN_DRY:-0}; L=results/t1c_chain.log; END=results/T1C_TUEF_END
MYPID=${CHAIN_MYPID:?set CHAIN_MYPID to this session s Claude pid}
PREV=${CHAIN_TUEEND:-results/T1C_TUE_END}; PREVPID=${CHAIN_TUEPID:-86926}; CH=${CHAIN_SCRIPT:-tools/t1c_chain.sh}
FDAY=${CHAIN_FDAY:-2026-11-17}
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1:00" +%s; }
LAST=$(ep "${CHAIN_LAST:-2026-11-17 14:15}"); ARMED=$(date +%s)
say() { echo "$(date '+%m-%d %H:%M:%S') [tuef] $*" >> "$L"; }
STATUS="armed"; CPID=""
finish() {
  local rc=$?
  if [ -n "$CPID" ] && kill -0 "$CPID" 2>/dev/null; then    # stopped mid-run: the chain stops its runner, unlocks, writes
    kill -TERM "$CPID" 2>/dev/null; wait "$CPID" 2>/dev/null      # its own marker
  fi
  [ "$DRY" = 1 ] && { say "DRY end (rc $rc): no marker"; exit "$rc"; }
  echo "T1C TUEF END $(date '+%m-%d %H:%M:%S'): $STATUS" > "$END"
  say "end (rc $rc): $STATUS; wrote $END"
}
trap finish EXIT
trap 'STATUS="stopped by a signal during: $STATUS"; exit 143' TERM INT
alive() { ps -o command= -p "$PREVPID" 2>/dev/null | grep -q t1c_chain; }   # the tue phase, not a reused pid
prevend() {     # 0 = the tue phase ended with V1's PASS (a marker newer than this arming, its process gone); 1 = not yet;
  alive && return 1                                                          # 5 = no run
  { [ -e "$PREV" ] && [ "$(stat -f %m "$PREV")" -ge "$ARMED" ]; } \
    || { PREVWHY="the tue phase (pid $PREVPID) is gone without a new end marker"; return 5; }
  PREVWHY=$(head -c 300 "$PREV" | tr '\n' ' ')
  echo "$PREVWHY" | grep -q 'done for the day (tue): V1 PASS' || return 5
}
say "follower (pid $$$([ "$DRY" = 1 ] && echo ', DRY'); session $MYPID): Tuesday's B2 -> V2 once the tue phase (pid $PREVPID) ends with V1 PASS"
if [ "$DRY" != 1 ]; then
  say "waits for $PREV newer than this arming with 'V1 PASS', the tue phase gone; no start after $(date -r "$LAST" '+%m-%d %H:%M')"
  n=0
  while :; do
    [ "$(date +%s)" -lt "$LAST" ] || { STATUS="no start by $(date -r "$LAST" '+%m-%d %H:%M') (the tue phase $(alive && echo 'still running' || echo ended)): no run"; say "$STATUS"; exit 6; }
    prevend; p=$?
    [ "$p" = 5 ] && { STATUS="no run: the tue phase did not end with V1 PASS ($PREVWHY)"; say "$STATUS"; exit 5; }
    if [ "$p" = 0 ]; then
      sleep 30
      if prevend; then say "start: the tue phase ended ($PREVWHY); read again 30 s later"; break; fi
    fi
    n=$((n + 1)); [ $((n % 30)) = 0 ] && say "waiting: the tue phase $(alive && echo 'still running' || echo ended)"
    sleep 60
  done
fi
STATUS="the chain's wed phase on $FDAY (B2 -> V2)"
CHAIN_PHASE=wed CHAIN_DAY=$FDAY CHAIN_MYPID=$MYPID "$CH" &
CPID=$!; wait "$CPID"; rc=$?; CPID=""
if [ "$DRY" = 1 ]; then STATUS="DRY: the chain's wed phase on $FDAY ended rc $rc"
else STATUS="done: the chain's wed phase on $FDAY ended rc $rc ($(head -c 200 results/T1C_WED_END 2>/dev/null | tr '\n' ' '))"; fi
exit "$rc"                                                   # the chain's own rc
