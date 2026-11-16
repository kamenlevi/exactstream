#!/bin/bash
# night1117_chain.sh: Mon -> Tue 16-17 Nov at 45 GB, last in the orchestrator's Monday order (17:2x, 18:1x): BigMoE's
# TREE16 M4 23:05 -> 00:25, Leap's exact engine A/Bs E1-E4 00:30 -> hard end 03:00, then THIS chain: the close-call arms
# of docs/PREREG_lossygate amendments 3 / 3a / 3b, through runner 64, against the cached lossy-45 base lx45cc1.
#   1. delta 1 (ccx45cc1): forward; if it PASSES, the reverse (CARD_FWD_RUN / CARD_FWD_REF: both orders pooled).
#      A FAIL or an interim STOP-FAIL ends the arm.
#   2. delta 2 (ccx45cc1d2): the same, after delta 1 whatever its verdict. Each arm has its own verdict.
#   After each run with a verdict: tools/lossy_speed.py <arm TG> <base TG> -> results/lossygate/ (information).
#   A runner call that ends without a verdict (rc != 0) stops the chain.
# Mon 06:5x, night1116's three fixes (the orchestrator 06:4x: "yes, replace night1117's 10:00 last start with the away
# rule, as tonight"):
# Start: behind Leap's E1-E4: its end marker (data/week98/screens/E1E4_1117_DONE) written after E1-E4's run started (the
# "[run_e14] run (power mode" line in Leap's data/week98/e14/run_e14.log; run_e14.sh writes the marker on ANY end, a stop
# before a re-arm included, as at Mon 00:33), or from 03:15 if E1-E4 has not started and no other session's lock is
# above 5 GB. Never before Tue 00:30. Either is read twice, 30 s apart, and nothing starts while HOLD, PAUSE, a shut lid
# (no external display) or Kamen at the Mac. The base must be complete: both orders' DONE files from Sunday night's
# chain, neither VOID nor INCOMPLETE.
# Memory: runner 64 gates each process on NO-SWAP >= CARD_MEM + 4 GB; gate 43 GB (the base lx45cc1 peaked at 42.2 GB
# on Mon 16 Nov), so CARD_MEM 39 and the shared lock's reason reads "MEM 39" (the gate shift; the runs stay MEM 45). The
# verdicts and the speed read-out (kept tokens per check) are outputs, so the tighter start biases neither. Runner 64
# keeps its +2 GB swap guard at each process start, its per-process yield to other sessions' runs above 5 GB and its
# share lock behind any quiet lock or queue, and waits out PAUSE and a shut lid itself.
# Kamen at the Mac (away_secs.sh < 900): no start; during a run, two reads 30 s apart make it yield at once (keep >= 5 GB
# free): the runner and its children stop, peer unlock. Once nothing holds, the run continues (CARD_RESUME=1). No fixed
# last start: every run's cutoff is 14:30, so nothing starts after 14:15.
# A HOLD before or during a run stops the chain, the runner and its children too.
# EVERY end writes results/CHAIN_1117_END, after the runner and its children exit and peer unlock. Stop the chain by pid
# only, with a plain kill (TERM): it then stops its runner and the runner's children. kill -9 skips the marker.
# CHAIN_DRY=1: every runner call as a dry run (CARD_DRY=1), both orders of each arm; no wait, no GPU, no marker.
#   CHAIN_MYPID=<this session's Claude pid> [CHAIN_DRY=1] tools/night1117_chain.sh
# Tests only: CHAIN_QDIR (HOLD / PAUSE), CHAIN_E14 (the marker), CHAIN_E14LOG, CHAIN_NOTBEFORE / CHAIN_FALLBACK /
# CHAIN_CUT (YYYY-mm-dd HH:MM), CHAIN_AWAY (a command that prints away seconds), CHAIN_RUNNER (in place of runner 64).
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
DRY=${CHAIN_DRY:-0}; L=results/night1117_chain.log; END=results/CHAIN_1117_END
MYPID=${CHAIN_MYPID:?set CHAIN_MYPID to this session s Claude pid}
MARK=${CHAIN_E14:-$HOME/Leap/data/week98/screens/E1E4_1117_DONE}   # run_e14.sh: removed at each arming, written on any end
E14LOG=${CHAIN_E14LOG:-$HOME/Leap/data/week98/e14/run_e14.log}
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1:00" +%s; }
NOTBEFORE=$(ep "${CHAIN_NOTBEFORE:-2026-11-17 00:30}"); FALLBACK=$(ep "${CHAIN_FALLBACK:-2026-11-17 03:15}")
CUT=${CHAIN_CUT:-2026-11-17 14:30}; GIVEUP=$(( $(ep "$CUT") - 900 )); BASEAFTER=$(ep "2026-11-15 21:22")
GATE=43                                                    # NO-SWAP GB at each process start = runner 64's CARD_MEM + 4
Q=${CHAIN_QDIR:-$HOME/.local/share/quiet}                  # HOLD and PAUSE live here
AWAY=${CHAIN_AWAY:-$HOME/.local/share/quiet/away_secs.sh}; RUNNER=${CHAIN_RUNNER:-tools/day_runner64.sh}
BMS=$HOME/BigMoE/data/stack
CC=data/r57/snap/cc1=85a88a46a945b3a4d21cc2c6fb216372fa549eab,data/r57/snap/frozen_cc1=ec97bdbb7ac3ba7fdd1cdf845762e4957d6b4c06,bigmoe=d92cdb4b9e81d386e4170323d50f11b34147aa5f
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
  echo "CHAIN 1117 END $(date '+%m-%d %H:%M:%S'): $STATUS" > "$END"
  say "end (rc $rc): $STATUS; wrote $END"
}
trap finish EXIT
trap 'STATUS="stopped by a signal during: $STATUS"; exit 143' TERM INT
biglock() {     # another live session's lock above 5 GB: quiet, "MEM n" with n >= 6, or no MEM and GPU >= 30 (a lapsed one counts)
  python3 - "$MYPID" <<'BL'
import glob, json, os, re, sys, time
for f in glob.glob(os.path.expanduser("~/.claude-peers/locks/*.json")):
    pid = os.path.basename(f)[:-5]
    if pid == sys.argv[1]:
        continue
    try:
        d = json.load(open(f)); os.kill(int(pid), 0)
    except Exception:
        continue
    if not d.get("mode") or d.get("until", 0) < time.time():
        continue
    m = re.search(r"MEM (\d+)", d.get("why", ""))
    if d["mode"] == "quiet" or (m and int(m.group(1)) >= 6) or (not m and d.get("gpu", 0) >= 30):
        print(f"{pid} {d['mode']} gpu {d.get('gpu')}: {d.get('why', '')[:90]}"); sys.exit(0)
sys.exit(1)
BL
}
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
e14_run() {     # E1-E4's run start (epoch): the last "[run_e14] run (power mode" line dated 11-17; empty: none
  local t; t=$(grep -E '^11-17 [0-9:]{8} \[run_e14\] run \(power mode' "$E14LOG" 2>/dev/null | tail -1 | awk '{print $2}')
  [ -n "$t" ] && date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-17 $t" +%s
}
startok() {     # 0 = go: nothing holds, past 00:30, and Leap's E1-E4 marker written after its run started, or (E1-E4 not
  held && return 1                                 # started) from 03:15 with no other lock above 5 GB. A marker with no
  [ "$(date +%s)" -ge "$NOTBEFORE" ] || return 1   # run before it does not count (run_e14.sh writes one when stopped)
  local t0; t0=$(e14_run)
  if [ -n "$t0" ]; then
    { [ -e "$MARK" ] && [ "$(stat -f %m "$MARK")" -ge "$t0" ]; } || return 1
    WHY="Leap's E1-E4 marker after its run ($(head -c 100 "$MARK" | tr '\n' ' '))"; return 0
  fi
  { [ "$(date +%s)" -ge "$FALLBACK" ] && ! biglock > /dev/null; } || return 1
  WHY="03:15, E1-E4 not started, no other lock above 5 GB"; return 0
}
baseok() {      # the cached base: both orders' DONE files from Sunday night's chain, neither VOID nor INCOMPLETE
  local f
  for f in results/CARD_lx45cc1_DONE results/CARD_lx45cc1_rev_DONE; do
    if [ ! -s "$f" ] || [ "$(stat -f %m "$f")" -lt "$BASEAFTER" ] || grep -qE "VOID|INCOMPLETE" "$f"; then
      BASEWHY="$f: $(head -c 120 "$f" 2>/dev/null || echo missing)"; return 1
    fi
  done
}
run() {         # label, the run's TG (for its DONE file), then runner 64's K=V settings; the runner is a stoppable child.
  local lab=$1 tg=$2 rc; shift 2                   # 75: it yielded to Kamen (not started, or stopped; nothing left running)
  [ -e "$Q/HOLD" ] && { STATUS="stopped by HOLD before: $STATUS"; say "HOLD: $lab not started"; exit 8; }
> (a quote from the project owner, kept private)
  say "$lab: $*"
  RSTART=$(date +%s)
  env CARD_DRY="$DRY" "$@" "$RUNNER" >> results/night1117_chain_runner.out 2>&1 &
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
waitfree() {    # after a yield: until nothing holds (a HOLD ends the chain); 1 = no start left before the cutoff ($1)
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
step() {        # label, TG, run dir, then runner 64's settings: run(); after a yield, wait until nothing holds and continue
  local lab=$1 tg=$2 rd=$3 rc; shift 3             # (CARD_RESUME=1 once the run dir exists); 75: no start left by $CUT
  run "$lab" "$tg" "$@"; rc=$?
  while [ "$rc" = 75 ]; do
    waitfree "$CUT" || return 75
    if [ -e "results/$rd" ]; then run "$lab, resumed" "$tg" "$@" CARD_RESUME=1; else run "$lab, again" "$tg" "$@"; fi; rc=$?
  done
  return "$rc"
}
passed() {      # the run's own DONE (written since it started) is the tier-1 PASS line
  local f=results/CARD_$1_DONE
  [ -e "$f" ] && [ "$(stat -f %m "$f")" -ge "$RSTART" ] && [ "$(head -1 "$f" | cut -d';' -f1)" = "LOSSY tier1 (CODE-ONLY mode): PASS" ]
}
speed() {       # information beside the verdict (amendment 3a): kept per check, arm vs base, all answers and clean ones
  [ "$DRY" = 1 ] && return
  local o=results/lossygate/speed_$1_2026-11-17.txt
  .venv/bin/python tools/lossy_speed.py "$1" "$2" > "$o" 2>&1
  say "speed $1 vs $2 (rc $?): $(tail -1 "$o" | cut -c1-160)"
}
say "chain (pid $$$([ "$DRY" = 1 ] && echo ', DRY'); session $MYPID): close-call delta 1, then delta 2 (cutoff $CUT); NO-SWAP gate $GATE GB per process"
if [ "$DRY" != 1 ]; then
  baseok && say "the base is complete (both orders)" || say "NOTE: the base is not complete yet ($BASEWHY); read again at the start"
> (a quote from the project owner, kept private)
  n=0; lasth=""
  while :; do
    h=$(held && echo "$HELD")
    if [ "$h" != "$lasth" ]; then
      if [ -n "$h" ]; then say "held: $h (no start while held)"; else say "no longer held"; fi
      lasth=$h
    fi
    [ "$(date +%s)" -lt "$GIVEUP" ] || { STATUS="no start by $(date -r "$GIVEUP" +%H:%M) (${h:+held: $h; }no E1-E4 marker, E1-E4 running, or another lock above 5 GB): no run"; say "$STATUS"; exit 6; }
    if startok; then
      w1=$WHY; sleep 30
      if startok; then say "start: $w1; read again 30 s later: $WHY"; break; fi
      say "the start condition did not hold 30 s later: waiting"
    fi
    n=$((n + 1))
    [ $((n % 30)) = 0 ] && say "waiting: ${h:+held: $h; }$(t0=$(e14_run); [ -n "$t0" ] && echo "E1-E4 started $(date -r "$t0" +%H:%M:%S)" || echo 'E1-E4 not started'); $([ -e "$MARK" ] && echo 'an E1-E4 marker' || echo 'no E1-E4 marker'); $(biglock | head -1)"
    sleep 60
  done
  baseok || { STATUS="no run: the base is not complete ($BASEWHY)"; say "$STATUS"; exit 9; }
  say "free NO-SWAP $(.venv/bin/python -c "import sys; sys.path.insert(0, '.'); from evalsuite.cli import free_gb; print(int(free_gb()))") GB; swap $(sysctl -n vm.swapusage | awk '{print $3}')"
fi
C=(CARD_TIER=1 CARD_SCOPE=code "CARD_TREES=$CC" CARD_SCREEN_SHA=efc1f9b6f531 CARD_SO_SHA=ff69fa24a745
   CARD_ENGAGE=close_call:accepted CARD_ENGAGE_MODE=any CARD_IFEVAL_IDENT=1 "CARD_CUTOFF=$CUT" "CARD_MEM=$((GATE - 4))")
R=""
for a in ccx45cc1:46bdf06bc43d05bc768219c7298793013e846dab ccx45cc1d2:a5cb28ef13ec3221d24655c199b8e65e6ffb2ad1; do
  t=${a%%:*}; A=(CARD_TAG="$t" "CARD_FENV=$BMS/${t}_env.txt" "CARD_FBLOB=${a#*:}"); r=-
  STATUS="$t forward"
  step "$t forward" "$t" "card_${t}_t1code" "${C[@]}" "${A[@]}" CARD_REF=card_lx45cc1_t1code; f=$?
  if [ "$f" = 0 ]; then
    speed "$t" lx45cc1
    if [ "$DRY" = 1 ] || passed "$t"; then
      STATUS="$t reverse"
      step "$t reverse" "${t}_rev" "card_${t}_t1code_rev" "${C[@]}" "${A[@]}" CARD_ORDER=rev CARD_REF=card_lx45cc1_t1code_rev \
        "CARD_FWD_RUN=card_${t}_t1code" CARD_FWD_REF=card_lx45cc1_t1code; r=$?
      [ "$r" = 0 ] && speed "${t}_rev" lx45cc1_rev
    else
      say "$t reverse skipped: the forward did not pass"
    fi
  fi
  R="$R $t forward rc $f, reverse rc $r;"
> (a quote from the project owner, kept private)
done
> (a quote from the project owner, kept private)
