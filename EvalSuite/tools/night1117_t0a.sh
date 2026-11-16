#!/bin/bash
# night1117_t0a.sh: the tier-0a REPLAY re-run on snapshot rpl2 (docs/PREREG_lossygate amendments 2b / 2c), LAST in
# night1117: the orchestrator's condition (06:4x / 06:5x: last in night1117 only if NewIdeas agrees), NewIdeas' yes
# (Mon 08:2x). A separate follower, so the armed chain (tools/night1117_chain.sh, pid 52881, sha256 d434e7f7866d2bc2)
# is not touched.
# Start: night1117's end marker (results/CHAIN_1117_END) written after this follower was armed and saying "done:" (its
# close-call arms ran to their verdicts or their cutoff), with the chain's process gone; read twice, 30 s apart. Any
# other end (a HOLD, a signal, no run) or the chain gone without a new marker: no run. Nothing starts while HOLD, PAUSE,
# a shut lid (no external display) or Kamen at the Mac, nor after 13:15 (the three arms need ~45-90 min).
# The runs (amendment 2b, through runner 64, unchanged): step 3 only, p2x45rpl2 (NO-SWAP gate 43), m1bx45rpl2 (47),
# then lossycalx45rpl2 (47): each its REPLAY pass on the merged margins of Sunday night's step 2, then its mini-panel.
# The cutoff is 14:30, or 2 h after the first arm starts if that is earlier, so a memory wait cannot hold up Tuesday's
# day slot. An arm that ends without a verdict (rc != 0) stops the follower: the lines need all three.
# Kamen at the Mac (away_secs.sh < 900): no start; during a run, two reads 30 s apart make it yield at once (the runner
# and its children stop, peer unlock). A REPLAY pass does not resume (its merge is per runner start), so once nothing
# holds the arm is re-run whole, its partial run dirs moved aside (<dir>.yield<N>) first. A HOLD before or during a run
# stops the follower, the runner and its children too.
# EVERY end writes results/T0A_1117_END, after the runner and its children exit and peer unlock. Stop it by pid only,
# with a plain kill (TERM): it then stops its runner and the runner's children. kill -9 skips the marker.
# CHAIN_DRY=1: every runner call as a dry run (CARD_DRY=1); no wait, no GPU, no marker.
#   CHAIN_MYPID=<this session's Claude pid> [CHAIN_DRY=1] tools/night1117_t0a.sh
# Tests only: CHAIN_QDIR (HOLD / PAUSE), CHAIN_PREV (night1117's marker), CHAIN_PREVPID (its pid), CHAIN_LAST /
# CHAIN_CUT (YYYY-mm-dd HH:MM), CHAIN_AWAY (a command that prints away seconds), CHAIN_RUNNER (in place of runner 64).
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
DRY=${CHAIN_DRY:-0}; L=results/night1117_t0a.log; END=results/T0A_1117_END
MYPID=${CHAIN_MYPID:?set CHAIN_MYPID to this session s Claude pid}
PREV=${CHAIN_PREV:-results/CHAIN_1117_END}; PREVPID=${CHAIN_PREVPID:-52881}
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1:00" +%s; }
LAST=$(ep "${CHAIN_LAST:-2026-11-17 13:15}"); CUT=${CHAIN_CUT:-2026-11-17 14:30}; ARMED=$(date +%s)
Q=${CHAIN_QDIR:-$HOME/.local/share/quiet}                  # HOLD and PAUSE live here
AWAY=${CHAIN_AWAY:-$HOME/.local/share/quiet/away_secs.sh}; RUNNER=${CHAIN_RUNNER:-tools/day_runner64.sh}
BMS=$HOME/BigMoE/data/stack; EXACT=ref2tnax,ref2tnaxn.r3,ref2tnaxn.r2; MARG=results/card/replay_margins_x45rpl1.json
RPL2=data/r57/snap/rpl2=7038e9a8ca1e98f16a0e2b88df75fd5bd3efc097,data/r57/snap/frozen_rpl1=db72c3c1367e2c68f9a1365ed8d6137fce6c8af6,bigmoe=d92cdb4b9e81d386e4170323d50f11b34147aa5f
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$L"; }
STATUS="armed"; RPID=""; RSTART=0
stoprunner() {  # the runner and its children: TERM, KILL after 60 s
  { [ -n "$RPID" ] && kill -0 "$RPID" 2>/dev/null; } || { RPID=""; return 0; }
  say "stopping runner $RPID and its children"
  pkill -TERM -P "$RPID"; kill -TERM "$RPID" 2>/dev/null
  for _ in 1 2 3 4 5 6; do kill -0 "$RPID" 2>/dev/null || break; sleep 10; done
  pkill -KILL -P "$RPID" 2>/dev/null; kill -KILL "$RPID" 2>/dev/null
  wait "$RPID" 2>/dev/null; RPID=""
}
finish() {
  local rc=$?
  stoprunner                                                 # stopped mid-run: the runner and its children first
  [ "$DRY" = 1 ] && { say "DRY end (rc $rc): no marker"; exit "$rc"; }
  peer unlock > /dev/null 2>&1
  echo "T0A 1117 END $(date '+%m-%d %H:%M:%S'): $STATUS" > "$END"
  say "end (rc $rc): $STATUS; wrote $END"
}
trap finish EXIT
trap 'STATUS="stopped by a signal during: $STATUS"; exit 143' TERM INT
away() { "$AWAY" 2>/dev/null; }
held() {        # 0 = held: Kamen's HOLD, the PAUSE file, the lid shut with no external display (runner 64's lid_shut), or
  [ -e "$Q/HOLD" ] && { HELD="HOLD"; return 0; }          # Kamen at the Mac (one read; no reading counts as away)
  [ -e "$Q/PAUSE" ] && { HELD="PAUSE"; return 0; }
> (a quote from the project owner, kept private)
  ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes || return 1
  (( $(system_profiler SPDisplaysDataType -detailLevel mini 2>/dev/null | grep -c "Resolution:") > 1 )) && return 1
  HELD="the lid shut, no external display"; return 0
}
back() {        # 0 = Kamen is at the Mac on two reads 30 s apart: a running step yields
  local a; a=$(away); [ "${a:-900}" -lt 900 ] 2>/dev/null || return 1
  sleep 30; a=$(away); [ "${a:-900}" -lt 900 ] 2>/dev/null
}
alive() { ps -o command= -p "$PREVPID" 2>/dev/null | grep -q night1117_chain; }   # night1117's chain, not a reused pid
prevend() {     # 0 = night1117 ended "done:" (a marker newer than this arming, its process gone); 1 = not yet; 5 = no run
  alive && return 1
  { [ -e "$PREV" ] && [ "$(stat -f %m "$PREV")" -ge "$ARMED" ]; } \
    || { PREVWHY="night1117's chain (pid $PREVPID) is gone without a new end marker"; return 5; }
  PREVWHY=$(head -c 300 "$PREV" | tr '\n' ' ')
  echo "$PREVWHY" | grep -qE '^CHAIN 1117 END [0-9-]+ [0-9:]+: done:' || return 5
}
run() {         # label, the run's TG (for its DONE file), then runner 64's K=V settings; the runner is a stoppable child.
  local lab=$1 tg=$2 rc; shift 2                   # 75: it yielded to Kamen (not started, or stopped; nothing left running)
  [ -e "$Q/HOLD" ] && { STATUS="stopped by HOLD before: $STATUS"; say "HOLD: $lab not started"; exit 8; }
> (a quote from the project owner, kept private)
  say "$lab: $*"
  RSTART=$(date +%s)
  env CARD_DRY="$DRY" "$@" "$RUNNER" >> results/night1117_t0a_runner.out 2>&1 &
  RPID=$!
  while kill -0 "$RPID" 2>/dev/null; do     # a HOLD mid-run: the finish trap stops the runner and its children
    [ -e "$Q/HOLD" ] && { STATUS="stopped by HOLD during: $STATUS"; say "HOLD: stopping $lab"; exit 8; }
    if [ "$DRY" != 1 ] && back && kill -0 "$RPID" 2>/dev/null; then
> (a quote from the project owner, kept private)
      stoprunner; peer unlock > /dev/null 2>&1; return 75
    fi
    sleep 15
  done
  wait "$RPID"; rc=$?; RPID=""
  local df=results/CARD_${tg}_DONE
  say "$lab: rc $rc; $(if [ -e "$df" ] && [ "$(stat -f %m "$df")" -ge "$RSTART" ]; then cut -c1-220 "$df"; else echo 'no DONE from this run'; fi)"
  return "$rc"
}
waitfree() {    # after a yield: until nothing holds (a HOLD ends the follower); 1 = no start left before the cutoff ($1)
  local cut n=0 h; cut=$(ep "$1")
  while :; do
    [ -e "$Q/HOLD" ] && { STATUS="stopped by HOLD while yielding: $STATUS"; say "HOLD while yielding"; exit 8; }
    h=$(held && echo "$HELD")
    [ -z "$h" ] && { say "nothing holds (away_secs $(away) s): continuing"; return 0; }
    [ $(( $(date +%s) + 900 )) -le "$cut" ] || { say "still held at $(date +%H:%M) ($h): no start left before $1"; return 1; }
    [ $((n % 30)) = 0 ] && say "yielding: $h"
    n=$((n + 1)); sleep 60
  done
}
say "follower (pid $$$([ "$DRY" = 1 ] && echo ', DRY'); session $MYPID): the tier-0a re-run on rpl2 (amendments 2b / 2c), last in night1117"
if [ "$DRY" != 1 ]; then
  { [ -s "$MARG" ] && [ -d results/card_x45rpl1_t0code ]; } \
    || { STATUS="no run: step 1's rows or step 2's margins are missing"; say "$STATUS"; exit 9; }
> (a quote from the project owner, kept private)
  n=0; lasth=""
  while :; do
    h=$(held && echo "$HELD")
    if [ "$h" != "$lasth" ]; then
      if [ -n "$h" ]; then say "held: $h (no start while held)"; else say "no longer held"; fi
      lasth=$h
    fi
    [ "$(date +%s)" -lt "$LAST" ] || { STATUS="no start by $(date -r "$LAST" +%H:%M) (${h:+held: $h; }night1117 $(alive && echo 'still running' || echo ended)): no run"; say "$STATUS"; exit 6; }
    prevend; p=$?
    [ "$p" = 5 ] && { STATUS="no run: night1117 did not end done ($PREVWHY)"; say "$STATUS"; exit 5; }
    if [ "$p" = 0 ] && [ -z "$h" ]; then
      sleep 30
      if prevend && ! held; then say "start: night1117 ended ($PREVWHY); read again 30 s later"; break; fi
      say "the start condition did not hold 30 s later: waiting"
    fi
    n=$((n + 1))
    [ $((n % 30)) = 0 ] && say "waiting: ${h:+held: $h; }night1117 $(alive && echo 'still running' || echo ended)"
    sleep 60
  done
  C2=$(( $(date +%s) + 7200 )); [ "$C2" -lt "$(ep "$CUT")" ] && CUT=$(date -r "$C2" '+%Y-%m-%d %H:%M')
  say "cutoff $CUT; free NO-SWAP $(.venv/bin/python -c "import sys; sys.path.insert(0, '.'); from evalsuite.cli import free_gb; print(int(free_gb()))") GB; swap $(sysctl -n vm.swapusage | awk '{print $3}')"
fi
T0=(CARD_TIER=0 CARD_SCOPE=code "CARD_REF=$EXACT" "CARD_TREES=$RPL2" CARD_SCREEN_SHA=d74d8b119e34 CARD_SO_SHA=ff69fa24a745
    "CARD_REPLAY=$MARG" CARD_REPLAY_ROWS=card_x45rpl1_t0code "CARD_CUTOFF=$CUT")
R=""
for a in p2x45rpl2:ff30bd2afd75c146b7676d2dfd15ba0d0eab9315:43 m1bx45rpl2:a59e13e024504b7083cea083b30400d1d0e4a8e3:47 \
    lossycalx45rpl2:741c4d2f0af44aca58c11304b61fd22963a0b4b1:47; do
  IFS=: read -r t b g <<< "$a"
  A=(CARD_TAG="$t" "CARD_FENV=$BMS/${t}_env.txt" "CARD_FBLOB=$b" "CARD_MEM=$((g - 4))")
  STATUS="S3 $t"; k=0
  run "S3 $t" "$t" "${T0[@]}" "${A[@]}"; rc=$?
  while [ "$rc" = 75 ]; do       # a yield: a REPLAY pass does not resume, so the arm is re-run whole
    waitfree "$CUT" || break
    k=$((k + 1))
    for d in "results/card_${t}_t0code_rpl" "results/card_${t}_t0code"; do
      [ -e "$d" ] && mv "$d" "$d.yield$k" && say "moved aside: $d -> $d.yield$k"
    done
    run "S3 $t, again (whole)" "$t" "${T0[@]}" "${A[@]}"; rc=$?
  done
  R="$R $t rc $rc;"
> (a quote from the project owner, kept private)
done
STATUS="done:${R%;} (A / F_hi in results/card_<tag>_replay.txt, verdicts in CARD_<tag>_DONE; 75: yielded, no start left)"
