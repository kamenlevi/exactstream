#!/bin/bash
# t1c_tue_follower2.sh: Tuesday's B2 -> V2 of 1c-tiers (PREREG_lossygate amendment 4j, the lift of 4d's cutoff; NewIdeas
# 19:2x: B2 -> V2 cannot start tonight, so 4i's follower lapses at 20:00 and the Tuesday route takes it). It replaces 4d's
# follower (tools/t1c_tue_follower.sh, pid 95134) and the tue phase it waited on (pid 86926); both are stopped by TERM
# before this one is armed. Its stages:
#   1. T0A_1117_END (any end; mtime >= Mon 08:40, as filed) with the tier-0a follower (pid 45029) gone, read twice 30 s
#      apart.
#   2. The night window, only if B2 -> V2 is priced to END by 07:00, re-priced at every read: 190 min (B1 85 + V1 67 min
#      on their own path, + 25 %). Not before 03:00 (E14's armed end), so an early T0A_1117_END from a failed or held
#      night1117 cannot put B2 -> V2 into tonight's earlier slots. Then the start conditions, then tools/t1c_chain2.sh's
#      wed phase on 11-17, tag tuen, cutoff 07:00; the runner prices each process before the cutoff. A cut by 07:00 (B2
#      or V2 rc 75) resumes in stage 3; any other end of that chain is final.
#   3. Otherwise: once Leap's slot #1 (run_c25_tue1.sh, window 09:30-12:00) is over, on two reads 30 s apart: no
#      run_c25_tue1 process (any pid, so a re-arm counts as the slot) AND either its end marker
#      ~/Leap/data/week98/c25/C25_tue1_DONE (written on any end; a re-arm removes it) dated 09:30 or later and not
#      "stopped by a signal" (a stop for a re-arm), or the window's end, 12:00 (NewIdeas 19:5x: key on the marker, not
#      the pid). Then the start conditions, then chain2's wed phase on 11-17, tag tued, cutoff 17:00. No start after
#      14:30.
#   The start conditions, at every read: no HOLD, PAUSE or shut lid (no Kamen-at-the-Mac rule: Kamen 11:5x, as in 4f);
#   no BigMoE key-test process (it stops itself when a day runner appears); no other session's shared MEM run above 5 GB
#   (runner 65's yield rule); NO-SWAP >= 43 GB on two reads 30 s apart (runner 65), or, from 10 min after the stage's
#   start condition, >= 42 (runner 66 at CARD_GATE=42: amendment 4e, carried over as in 4i). The runner's own gates, its
#   +2 GB swap guard and its stops are unchanged.
# results/T1C_TUEF2_END is written on EVERY end of this follower, and only then, and copied to results/T1C_RESF2_END,
# which keeps meaning the 1c-tiers end (amendment 4i). Stop it by pid only, with a plain kill (TERM): it stops the chain
# it owns, which stops its runner and the runner's children, unlocks and writes its marker.
#   CHAIN_MYPID=<this session's Claude pid> tools/t1c_tue_follower2.sh
# Tests only: CHAIN_SCRIPT, CHAIN_RES (the results dir), CHAIN_T0A, CHAIN_T0APID, CHAIN_LEAPMK, CHAIN_LEAPPAT, CHAIN_KTPAT,
# CHAIN_QDIR, CHAIN_FREEGB / CHAIN_MEMBUSY / CHAIN_LIDSHUT (commands), CHAIN_POLL, CHAIN_PAIR, CHAIN_FB_S, CHAIN_PRICE_S,
# and the times CHAIN_NIGHTOPEN, CHAIN_NIGHTCUT, CHAIN_SLOTSTART, CHAIN_SLOTEND, CHAIN_LASTSTART, CHAIN_DAYCUT.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
MYPID=${CHAIN_MYPID:?set CHAIN_MYPID to this session s Claude pid}; CH=${CHAIN_SCRIPT:-tools/t1c_chain2.sh}
RES=${CHAIN_RES:-results}; L=$RES/t1c_chain.log; END=$RES/T1C_TUEF2_END; V2D=$RES/CARD_t1cx45t1c1_rev_DONE
T0A=${CHAIN_T0A:-results/T0A_1117_END}; T0APID=${CHAIN_T0APID:-45029}
LMK=${CHAIN_LEAPMK:-$HOME/Leap/data/week98/c25/C25_tue1_DONE}; LPAT=${CHAIN_LEAPPAT:-run_c25_tue1[^ /]*\.sh}
KTPAT=${CHAIN_KTPAT:-scripts/g1_follo[w]\.sh|scripts/g1_keytes[t]\.py}; Q=${CHAIN_QDIR:-$HOME/.local/share/quiet}
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1:00" +%s; }
hm() { date -r "$1" '+%m-%d %H:%M'; }
FDAY=2026-11-17; T0AAFTER=$(ep "2026-11-16 08:40")
NOPEN=$(ep "${CHAIN_NIGHTOPEN:-2026-11-17 03:00}"); NCUTS=${CHAIN_NIGHTCUT:-2026-11-17 07:00}; NCUT=$(ep "$NCUTS")
SSTART=$(ep "${CHAIN_SLOTSTART:-2026-11-17 09:30}"); SEND=$(ep "${CHAIN_SLOTEND:-2026-11-17 12:00}")
LAST=$(ep "${CHAIN_LASTSTART:-2026-11-17 14:30}"); DCUTS=${CHAIN_DAYCUT:-2026-11-17 17:00}
PRICE=${CHAIN_PRICE_S:-11400}; FB=${CHAIN_FB_S:-600}; POLL=${CHAIN_POLL:-60}; PAIR=${CHAIN_PAIR:-30}
ARMED=$(date +%s)
say() { echo "$(date '+%m-%d %H:%M:%S') [tuef2] $*" >> "$L"; }
freegb() {
  if [ -n "${CHAIN_FREEGB:-}" ]; then $CHAIN_FREEGB
  else .venv/bin/python -c "import sys; sys.path.insert(0, '.'); from evalsuite.cli import free_gb; print(int(free_gb()))"; fi
}
membusy() {     # runner 65's yield rule: another session's shared run above 5 GB (reasons "evalsuite: MEM" are ours)
  if [ -n "${CHAIN_MEMBUSY:-}" ]; then $CHAIN_MEMBUSY; return; fi
  peer ls 2>/dev/null | grep "shared run" | grep -v "evalsuite: MEM" | grep -oE "MEM [0-9]+ GB" | awk '$2 > 5 {f = 1} END {exit !f}'
}
lidshut() {     # chain2's rule: the lid shut with no external display
  if [ -n "${CHAIN_LIDSHUT:-}" ]; then $CHAIN_LIDSHUT; return; fi
  ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes || return 1
  (( $(system_profiler SPDisplaysDataType -detailLevel mini 2>/dev/null | grep -c "Resolution:") > 1 )) && return 1
  return 0
}
STATUS="waiting for T0A_1117_END"; CPID=""
finish() {
  local rc=$?
  if [ -n "$CPID" ] && ps -o command= -p "$CPID" 2>/dev/null | grep -q t1c_chain; then   # stopped mid-run: the chain stops
    kill -TERM "$CPID" 2>/dev/null                                           # its runner (TERM, KILL after 60 s) and unlocks
    for _ in $(seq 1 120); do kill -0 "$CPID" 2>/dev/null || break; sleep 1; done
  fi
  echo "T1C TUEF2 END $(date '+%m-%d %H:%M:%S'): $STATUS" > "$END"
  cp "$END" "$RES/T1C_RESF2_END"                             # 4i: T1C_RESF2_END keeps meaning the 1c-tiers end
  say "end (rc $rc): $STATUS; wrote $END and T1C_RESF2_END"
}
trap finish EXIT
trap 'STATUS="stopped by a signal during: $STATUS"; exit 143' TERM INT
blocked() {     # 0 = a start condition other than memory does not hold now; WHY says which
  [ -e "$Q/HOLD" ] && { WHY="HOLD"; return 0; }
  [ -e "$Q/PAUSE" ] && { WHY="PAUSE"; return 0; }
  lidshut && { WHY="the lid shut, no external display"; return 0; }
  pgrep -f -- "$KTPAT" > /dev/null && { WHY="BigMoE's key test is armed or running (pid $(pgrep -f -- "$KTPAT" | head -3 | tr '\n' ' '))"; return 0; }
  membusy && { WHY="another session's shared MEM run above 5 GB"; return 0; }
  return 1
}
nightok() { [ $(( $(date +%s) + PRICE )) -le "$NCUT" ]; }
dayok() { [ "$(date +%s)" -lt "$LAST" ]; }
startgate() {   # $1 = when the stage's start condition held; $2 = a test that holds while a start is still allowed.
  local a b g n=0 last=""                          # 0 = start now (RUN 65 or 66); 1 = no start allowed any more
  while :; do
    "$2" || return 1
    if blocked; then
      { [ "$WHY" != "$last" ] || [ $((n % 30)) = 0 ]; } && say "waiting: $WHY"; last=$WHY
    else
      last=""
      a=$(freegb); g=43
      [ "$(date +%s)" -ge $(( $1 + FB )) ] && [ "${a:-0}" -lt 43 ] && g=42
      if [ "${a:-0}" -ge "$g" ]; then
        sleep "$PAIR"; b=$(freegb)
        if [ "${b:-0}" -ge "$g" ] && ! blocked && "$2"; then
          if [ "$g" = 42 ]; then RUN=66; else RUN=65; fi
          say "start conditions hold: NO-SWAP $a / $b GB >= $g: runner $RUN$([ "$g" = 42 ] && echo " (amendment 4e's fallback: 43 not open $((FB / 60)) min after the start condition)")"
          return 0
        fi
      fi
      [ $((n % 30)) = 0 ] && say "waiting for memory: NO-SWAP ${a:-?} GB (gate $g)"
    fi
    n=$((n + 1)); sleep "$POLL"
  done
}
runchain() {    # tag, cutoff: chain2's wed phase on Tuesday (the done steps skipped, a cut tier-1 step resumed); CEND = its end
  local tag=$1 cut=$2 utag cs rc
  utag=$(echo "$tag" | tr a-z A-Z); cs=$(date +%s)
  STATUS="the chain (tag $tag, runner $RUN, cutoff $cut): B2 -> V2"
  if [ "$RUN" = 66 ]; then
    CHAIN_RUNNER=tools/day_runner66.sh CARD_GATE=42 CHAIN_PHASE=wed CHAIN_DAY=$FDAY CHAIN_CUTOFF="$cut" CHAIN_NOYIELD=1 CHAIN_TAG=$tag CHAIN_MYPID=$MYPID "$CH" &
  else
    CHAIN_PHASE=wed CHAIN_DAY=$FDAY CHAIN_CUTOFF="$cut" CHAIN_NOYIELD=1 CHAIN_TAG=$tag CHAIN_MYPID=$MYPID "$CH" &
  fi
  CPID=$!; say "the chain (tag $tag, runner $RUN, cutoff $cut) runs as pid $CPID"; wait "$CPID"; rc=$?; CPID=""
  if [ -e "$RES/T1C_${utag}_END" ] && [ "$(stat -f %m "$RES/T1C_${utag}_END")" -ge "$cs" ]; then
    CEND=$(head -c 300 "$RES/T1C_${utag}_END" | tr '\n' ' ')
  else CEND="(no end marker from this run)"; fi
  say "the chain (tag $tag) ended rc $rc: $CEND"; return "$rc"
}
v2done() { [ -e "$V2D" ] && [ "$(stat -f %m "$V2D")" -ge "$ARMED" ] && head -1 "$V2D" | grep -q '^LOSSY tier1' && ! head -1 "$V2D" | grep -qE 'VOID|INCOMPLETE'; }
t0aend() { [ -e "$T0A" ] && [ "$(stat -f %m "$T0A")" -ge "$T0AAFTER" ] && ! ps -o command= -p "$T0APID" 2>/dev/null | grep -q night1117_t0a; }
leapup() {      # a slot #1 process, any pid (a re-armed slot included): a shell RUNNING a run_c25_tue1*.sh script, so a
  ps -axo command= | grep -qE "^(/bin/)?(ba|z)?sh [^ ]*$LPAT( |\$)"     # monitor or a command naming it does not count
}
leapdone() {    # 0 = slot #1's end marker: an end at or after its window's start, not a stop by a signal (a re-arm)
  [ -e "$LMK" ] && [ "$(stat -f %m "$LMK")" -ge "$SSTART" ] && ! grep -q 'stopped by a signal' "$LMK"
}
slotover() { ! leapup && { leapdone || [ "$(date +%s)" -ge "$SEND" ]; }; }
say "follower2 (pid $$; session $MYPID): Tuesday's B2 -> V2 (amendment 4j): after T0A_1117_END (pid $T0APID gone); in the night window from $(hm "$NOPEN") only if priced ($((PRICE / 60)) min) to end by $(hm "$NCUT"); else once Leap's slot #1 is over ($(basename "$LMK"), window to $(hm "$SEND")), cutoff $DCUTS, no start after $(hm "$LAST"); one end marker $END"
n=0
while :; do
  dayok || { STATUS="no run: no T0A_1117_END by $(hm "$LAST")"; say "$STATUS"; exit 6; }
  if t0aend; then sleep "$PAIR"; t0aend && break; fi
  n=$((n + 1)); [ $((n % 30)) = 0 ] && say "waiting: no T0A_1117_END yet"
  sleep "$POLL"
done
say "T0A_1117_END (read twice, $PAIR s apart): $(head -c 160 "$T0A" | tr '\n' ' ')"
S=$(date +%s); [ "$S" -lt "$NOPEN" ] && S=$NOPEN
if [ $(( S + PRICE )) -le "$NCUT" ]; then
  if [ "$(date +%s)" -lt "$NOPEN" ]; then
    STATUS="waiting for the night window ($(hm "$NOPEN"))"; say "the night window opens at $(hm "$NOPEN"): waiting"
    while [ "$(date +%s)" -lt "$NOPEN" ]; do sleep "$POLL"; done
  fi
  STATUS="the night window's start conditions"
  say "the night window: B2 -> V2 priced $((PRICE / 60)) min, to end by $(hm "$NCUT"): the start conditions"
  if startgate "$(date +%s)" nightok; then
    runchain tuen "$NCUTS"; rc=$?
    v2done && { STATUS="done: B2 -> V2 ran in the night window (tag tuen): $CEND"; exit "$rc"; }
    if ! echo "$CEND" | grep -qE '(B2|V2) rc 75|no start by'; then
      STATUS="ended without V2's verdict: the night chain (tag tuen) ended rc $rc: $CEND"; say "$STATUS"; exit "$rc"
    fi
    say "the night chain was cut by $(hm "$NCUT") (it resumes): after Leap's slot #1"
  else
    say "no night start: B2 -> V2 would no longer end by $(hm "$NCUT"): after Leap's slot #1"
  fi
else
  say "the night window: B2 -> V2 ($((PRICE / 60)) min) cannot end by $(hm "$NCUT"): after Leap's slot #1"
fi
STATUS="waiting for Leap's slot #1 to be over"; n=0; lastw=""
while :; do
  dayok || { STATUS="no run: Leap's slot #1 not over by $(hm "$LAST")"; say "$STATUS"; exit 6; }
  if slotover; then sleep "$PAIR"; slotover && break; fi
  w=$(if leapup; then echo "a slot #1 process runs or waits for its window"; else echo "no slot #1 process; its end marker: $(head -c 100 "$LMK" 2>/dev/null | tr '\n' ' ')$([ -e "$LMK" ] || echo none) (the window is kept to $(hm "$SEND"))"; fi)
  { [ "$w" != "$lastw" ] || [ $((n % 30)) = 0 ]; } && say "waiting: $w"; lastw=$w
  n=$((n + 1)); sleep "$POLL"
done
if leapdone; then say "Leap's slot #1 is over (read twice, $PAIR s apart): $(head -c 160 "$LMK" | tr '\n' ' ')"
else say "Leap's slot #1 window is over ($(hm "$SEND")), with no slot #1 process and no in-window end marker ($(head -c 120 "$LMK" 2>/dev/null | tr '\n' ' ')$([ -e "$LMK" ] || echo none))"; fi
TC=$(date +%s)
STATUS="the day start conditions"
if ! startgate "$TC" dayok; then STATUS="no run: the start conditions did not hold by $(hm "$LAST")"; say "$STATUS"; exit 6; fi
runchain tued "$DCUTS"; rc=$?
if v2done; then STATUS="done: B2 -> V2 ran after Leap's slot #1 (tag tued): $CEND"
else STATUS="ended without V2's verdict: the day chain (tag tued) ended rc $rc: $CEND"; fi
exit "$rc"
