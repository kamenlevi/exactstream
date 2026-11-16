#!/bin/bash
# t1c_resume_follower3.sh: ADOPTS the running 1c-tiers chain (tools/t1c_chain2.sh, tag mon3, pid CHAIN_ADOPT_PID) from
# t1c_resume_follower2.sh and applies pause 2 as the orchestrator set it at 13:1x (PREREG_lossygate amendment 4h).
# follower2 is stopped by SIGKILL on its own pid only, so its trap neither stops the chain (V1 runs on) nor writes the end
# marker; this follower then owns the chain's end. Pause 2 comes at B2's start (after V1's PASS and its speed read-out):
#   (a) logs/sib_READY newer than CHAIN_READY_AFTER (13:06): Leap's c25 engine A/Bs (quiet, <= 30 min) then SIB's main
#       read-out (~15 min); resume on a sib_END "<day HH:MM:SS> rc 0: main read-out done" line written after the pause
#       began, or CHAIN_PAUSE2A_MIN (60) min after it;
#   (b) no fresh sib_READY: Leap's c25 alone; resume when a quiet lock seen after the pause began is released, or
#       CHAIN_PAUSE2B_MIN (35) min after it.
#   Either way results/T1C_RESUME_NOW written after the pause began resumes at once (EvalSuite's manual override).
# The rest (B2 -> V2) is tools/t1c_chain2.sh with tag mon4 (done steps skipped). If the adopted chain ends without
# reaching B2 (e.g. V1 did not pass), there is no pause and no mon4. results/T1C_RESF2_END is written on EVERY end of
# this follower, and only then: it is the real end BigMoE's G1 key test and Breakthrough's map screen wait for (with no
# t1c_chain2.sh process). Stop it by pid only, with a plain kill (TERM): it stops the chain it owns.
#   CHAIN_MYPID=<this session's Claude pid> CHAIN_ADOPT_PID=<the mon3 chain's pid> tools/t1c_resume_follower3.sh
# Tests only: CHAIN_SCRIPT, CHAIN_SBDIR (the Sandbox98 root), CHAIN_PEER (the peer command).
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
L=results/t1c_chain.log; END=results/T1C_RESF2_END; NOW=results/T1C_RESUME_NOW
MYPID=${CHAIN_MYPID:?set CHAIN_MYPID to this session s Claude pid}; AP=${CHAIN_ADOPT_PID:?set CHAIN_ADOPT_PID}
TAG=${CHAIN_ADOPT_TAG:-mon3}; SB=${CHAIN_SBDIR:-$HOME/Sandbox98}; CH=${CHAIN_SCRIPT:-tools/t1c_chain2.sh}; PEER=${CHAIN_PEER:-peer}
MARK=$SB/logs/sib_END; READY=$SB/logs/sib_READY
FDAY=${CHAIN_DAY:-2026-11-16}; CUTS=${CHAIN_CUTOFF:-2026-11-16 22:30}
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1:00" +%s; }
RAFTER=$(ep "${CHAIN_READY_AFTER:-2026-11-16 13:06}"); PA=${CHAIN_PAUSE2A_MIN:-60}; PB=${CHAIN_PAUSE2B_MIN:-35}
LAST=$(( $(ep "$CUTS") - 900 ))
say() { echo "$(date '+%m-%d %H:%M:%S') [resf3] $*" >> "$L"; }
STATUS="adopting"; CPID=""
finish() {
  local rc=$?
  for p in "$CPID" "$AP"; do                                 # stopped mid-run: stop the chain this follower owns
    if [ -n "$p" ] && ps -o command= -p "$p" 2>/dev/null | grep -q t1c_chain; then
      kill -TERM "$p" 2>/dev/null; for _ in $(seq 1 90); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    fi
  done
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
quiet() {      # another session's quiet lock, held or queued (peer ls segments; my own row is "You:")
  "$PEER" ls 2>/dev/null | grep '^- ' | grep -oE '· (benchmarking|waiting for a quiet window)[^·]*' | head -1
}
b2seen() {     # the adopted chain logged B2's start (after its own start line)
  awk -v s="[$TAG] chain2 (pid $AP;" 'index($0, s) { on = 1 } on && index($0, "[" T "] B2 base reverse: CARD_")' T="$TAG" "$L" | grep -q .
}
alive "$AP" || { STATUS="no chain to adopt: pid $AP is not a running chain"; say "$STATUS"; exit 2; }
say "follower3 (pid $$; session $MYPID): adopts the chain pid $AP (tag $TAG); pause 2 at B2's start: (a) $(basename "$READY") newer than $(date -r "$RAFTER" '+%H:%M') -> an rc 0 line or $PA min; (b) otherwise -> a quiet lock's release or $PB min; one end marker $END"
STATUS="the adopted chain (tag $TAG)"; CASE=""
while alive "$AP"; do
  if [ -z "$CASE" ] && b2seen; then
    if fresh "$READY" "$RAFTER"; then CASE=a; else CASE=b; fi
    say "pause 2 ($CASE): B2 starts$([ "$CASE" = a ] && echo " and $(basename "$READY") is fresh ($(date -r "$(stat -f %m "$READY")" '+%H:%M:%S'))"): stopping the chain (pid $AP) before B2's GPU work"
    kill -TERM "$AP"
  fi
  sleep 2
done
if [ -z "$CASE" ]; then
  STATUS="done: the adopted chain $TAG ended without reaching B2 ($(head -c 160 "results/T1C_$(echo "$TAG" | tr a-z A-Z)_END" 2>/dev/null | tr '\n' ' '))"
  say "$STATUS"; exit 0
fi
P2AT=$(date +%s); STATUS="pause 2 ($CASE, before B2)"; say "pause 2 ($CASE) began: the chain ended"
seen=0; n=0
while :; do
  [ "$(date +%s)" -lt "$LAST" ] || { STATUS="no start of B2 by $(date -r "$LAST" '+%m-%d %H:%M') (pause 2): no run"; say "$STATUS"; exit 6; }
  fresh "$NOW" "$P2AT" && { WHY="$NOW (EvalSuite's manual resume)"; break; }
  if [ "$CASE" = a ]; then
    done0 "$P2AT" && { WHY="$(basename "$MARK"): $(head -1 "$MARK" | cut -c1-120)"; break; }
    [ "$(date +%s)" -ge $(( P2AT + PA * 60 )) ] && { WHY="$PA min after the pause began"; break; }
  else
    q=$(quiet)
    if [ -n "$q" ]; then [ "$seen" = 1 ] || say "pause 2 (b): a quiet lock is up: $(echo "$q" | cut -c1-120)"; seen=1
    elif [ "$seen" = 1 ]; then WHY="the quiet lock was released"; break; fi
    [ "$(date +%s)" -ge $(( P2AT + PB * 60 )) ] && { WHY="$PB min after the pause began"; break; }
  fi
  n=$((n + 1)); [ $((n % 30)) = 0 ] && say "pause 2 ($CASE): waiting"
  sleep 10
done
say "resume 2: $WHY"
STATUS="the resumed chain (tag mon4): B2 -> V2"
CHAIN_PHASE=wed CHAIN_DAY=$FDAY CHAIN_CUTOFF="$CUTS" CHAIN_NOYIELD=1 CHAIN_TAG=mon4 CHAIN_MYPID=$MYPID "$CH" &
CPID=$!; say "the resumed chain (tag mon4) runs as pid $CPID"; wait "$CPID"; rc=$?; CPID=""
STATUS="done: the resumed chain mon4 ended rc $rc ($(head -c 160 results/T1C_MON4_END 2>/dev/null | tr '\n' ' '))"
exit "$rc"
