#!/bin/bash
# night1117_chain.sh: Mon -> Tue 16-17 Nov at 45 GB, last in the orchestrator's Monday order (17:2x, 18:1x): BigMoE's
# TREE16 M4 23:05 -> 00:25, Leap's exact engine A/Bs E1-E4 00:30 -> hard end 03:00, then THIS chain: the close-call arms
# of docs/PREREG_lossygate amendments 3 / 3a / 3b, through runner 64, against the cached lossy-45 base lx45cc1.
#   1. delta 1 (ccx45cc1): forward; if it PASSES, the reverse (CARD_FWD_RUN / CARD_FWD_REF: both orders pooled).
#      A FAIL or an interim STOP-FAIL ends the arm.
#   2. delta 2 (ccx45cc1d2): the same, after delta 1 whatever its verdict. Each arm has its own verdict.
#   After each run with a verdict: tools/lossy_speed.py <arm TG> <base TG> -> results/lossygate/ (information).
#   A runner call that ends without a verdict (rc != 0) stops the chain.
# Start: Leap's E1E4_1117_DONE (newer than the arming; Leap writes it on ANY end, a HOLD included), or from 03:15 with
# no other session's quiet lock or lock above 5 GB. Either is read twice, 30 s apart, never before Tue 00:30, and neither
# starts while HOLD, PAUSE or a shut lid (no external display) holds. The base must be complete: both orders' DONE files
# from Sunday night's chain, neither VOID nor INCOMPLETE. No start after 10:00. Every run's cutoff is 14:30 (apps closed
# until 15:00). Runner 64 gates each process on NO-SWAP memory, takes its share lock, and waits out PAUSE and a shut lid
# itself. A HOLD before or during a run stops the chain, the runner and its children too.
# EVERY end writes results/CHAIN_1117_END, after the runner and its children exit and peer unlock. Stop the chain by pid
# only, with a plain kill (TERM): it then stops its runner and the runner's children. kill -9 skips the marker.
# CHAIN_DRY=1: every runner call as a dry run (CARD_DRY=1), both orders of each arm; no wait, no GPU, no marker.
#   CHAIN_MYPID=<this session's Claude pid> [CHAIN_DRY=1] tools/night1117_chain.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
DRY=${CHAIN_DRY:-0}; L=results/night1117_chain.log; END=results/CHAIN_1117_END
MYPID=${CHAIN_MYPID:?set CHAIN_MYPID to this session s Claude pid}
MARK=$HOME/Leap/data/week98/screens/E1E4_1117_DONE
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1:00" +%s; }
NOTBEFORE=$(ep "2026-11-17 00:30"); FALLBACK=$(ep "2026-11-17 03:15"); LATEST=$(ep "2026-11-17 10:00")
CUT="2026-11-17 14:30"; BASEAFTER=$(ep "2026-11-15 21:22"); ARMED=$(date +%s)
Q=${CHAIN_QDIR:-$HOME/.local/share/quiet}                  # HOLD and PAUSE live here; CHAIN_QDIR: tests only
BMS=$HOME/BigMoE/data/stack
CC=data/r57/snap/cc1=85a88a46a945b3a4d21cc2c6fb216372fa549eab,data/r57/snap/frozen_cc1=ec97bdbb7ac3ba7fdd1cdf845762e4957d6b4c06,bigmoe=d92cdb4b9e81d386e4170323d50f11b34147aa5f
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$L"; }
STATUS="armed"; RPID=""; RSTART=0
finish() {
  local rc=$?
  if [ -n "$RPID" ] && kill -0 "$RPID" 2>/dev/null; then       # stopped mid-run: the runner and its children first
    say "stopping runner $RPID and its children"
    pkill -TERM -P "$RPID"; kill -TERM "$RPID" 2>/dev/null
    for _ in 1 2 3 4 5 6; do kill -0 "$RPID" 2>/dev/null || break; sleep 10; done
    pkill -KILL -P "$RPID" 2>/dev/null; kill -KILL "$RPID" 2>/dev/null
  fi
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
held() {        # 0 = held: Kamen's HOLD, the PAUSE file, or the lid shut with no external display (runner 64's lid_shut)
  [ -e "$Q/HOLD" ] && { HELD="HOLD"; return 0; }
  [ -e "$Q/PAUSE" ] && { HELD="PAUSE"; return 0; }
  ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes || return 1
  (( $(system_profiler SPDisplaysDataType -detailLevel mini 2>/dev/null | grep -c "Resolution:") > 1 )) && return 1
  HELD="the lid shut, no external display"; return 0
}
startok() {     # 0 = go: not held, past 00:30, and Leap's marker newer than the arming or past 03:15 with no lock above 5 GB
  held && return 1
  [ "$(date +%s)" -ge "$NOTBEFORE" ] || return 1
  if [ -e "$MARK" ] && [ "$(stat -f %m "$MARK")" -ge "$ARMED" ]; then
    WHY="Leap's end marker ($(head -c 100 "$MARK" | tr '\n' ' '))"; return 0
  fi
  if [ "$(date +%s)" -ge "$FALLBACK" ] && ! biglock > /dev/null; then WHY="the 03:15 fallback, no other quiet lock or lock above 5 GB"; return 0; fi
  return 1
}
baseok() {      # the cached base: both orders' DONE files from Sunday night's chain, neither VOID nor INCOMPLETE
  local f
  for f in results/CARD_lx45cc1_DONE results/CARD_lx45cc1_rev_DONE; do
    if [ ! -s "$f" ] || [ "$(stat -f %m "$f")" -lt "$BASEAFTER" ] || grep -qE "VOID|INCOMPLETE" "$f"; then
      BASEWHY="$f: $(head -c 120 "$f" 2>/dev/null || echo missing)"; return 1
    fi
  done
}
run() {         # label, the run's TG (for its DONE file), then runner 64's K=V settings; the runner is a stoppable child
  local lab=$1 tg=$2 rc; shift 2
  [ -e "$Q/HOLD" ] && { STATUS="stopped by HOLD before: $STATUS"; say "HOLD: $lab not started"; exit 8; }
  say "$lab: $*"
  RSTART=$(date +%s)
  env CARD_DRY="$DRY" "$@" tools/day_runner64.sh >> results/night1117_chain_runner.out 2>&1 &
  RPID=$!
  while kill -0 "$RPID" 2>/dev/null; do     # a HOLD mid-run: the finish trap stops the runner and its children
    [ -e "$Q/HOLD" ] && { STATUS="stopped by HOLD during: $STATUS"; say "HOLD: stopping $lab"; exit 8; }
    sleep 15
  done
  wait "$RPID"; rc=$?; RPID=""
  local df=results/CARD_${tg}_DONE
  say "$lab: rc $rc; $(if [ -e "$df" ] && [ "$(stat -f %m "$df")" -ge "$RSTART" ]; then cut -c1-220 "$df"; else echo 'no DONE from this run'; fi)"
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
say "chain (pid $$$([ "$DRY" = 1 ] && echo ', DRY'); session $MYPID): close-call delta 1, then delta 2 (cutoff $CUT)"
if [ "$DRY" != 1 ]; then
  baseok && say "the base is complete (both orders)" || say "NOTE: the base is not complete yet ($BASEWHY); read again at the start"
  say "waits for $MARK (newer than the arming) or the 03:15 fallback; not before 00:30; no start while HOLD, PAUSE or a shut lid, nor after 10:00"
  n=0; lasth=""
  while :; do
    h=$(held && echo "$HELD")
    if [ "$h" != "$lasth" ]; then
      if [ -n "$h" ]; then say "held: $h (no start while held)"; else say "no longer held"; fi
      lasth=$h
    fi
    if startok; then
      w1=$WHY; sleep 30
      if startok; then say "start: $w1; read again 30 s later: $WHY"; break; fi
      say "the start condition did not hold 30 s later: waiting"
    fi
    [ "$(date +%s)" -lt "$LATEST" ] || { STATUS="no start by 10:00 (${h:+held: $h; }no marker, or another lock above 5 GB): no run"; say "$STATUS"; exit 6; }
    n=$((n + 1))
    [ $((n % 30)) = 0 ] && say "waiting: ${h:+held: $h; }$([ -e "$MARK" ] && echo 'the marker predates the arming' || echo 'no marker yet'); $(biglock | head -1)"
    sleep 60
  done
  baseok || { STATUS="no run: the base is not complete ($BASEWHY)"; say "$STATUS"; exit 9; }
  say "free NO-SWAP $(.venv/bin/python -c "import sys; sys.path.insert(0, '.'); from evalsuite.cli import free_gb; print(int(free_gb()))") GB; swap $(sysctl -n vm.swapusage | awk '{print $3}')"
fi
C=(CARD_TIER=1 CARD_SCOPE=code "CARD_TREES=$CC" CARD_SCREEN_SHA=efc1f9b6f531 CARD_SO_SHA=ff69fa24a745
   CARD_ENGAGE=close_call:accepted CARD_ENGAGE_MODE=any CARD_IFEVAL_IDENT=1 "CARD_CUTOFF=$CUT")
R=""
for a in ccx45cc1:46bdf06bc43d05bc768219c7298793013e846dab ccx45cc1d2:a5cb28ef13ec3221d24655c199b8e65e6ffb2ad1; do
  t=${a%%:*}; A=(CARD_TAG="$t" "CARD_FENV=$BMS/${t}_env.txt" "CARD_FBLOB=${a#*:}"); r=-
  STATUS="$t forward"
  run "$t forward" "$t" "${C[@]}" "${A[@]}" CARD_REF=card_lx45cc1_t1code; f=$?
  if [ "$f" = 0 ]; then
    speed "$t" lx45cc1
    if [ "$DRY" = 1 ] || passed "$t"; then
      STATUS="$t reverse"
      run "$t reverse" "${t}_rev" "${C[@]}" "${A[@]}" CARD_ORDER=rev CARD_REF=card_lx45cc1_t1code_rev \
        "CARD_FWD_RUN=card_${t}_t1code" CARD_FWD_REF=card_lx45cc1_t1code; r=$?
      [ "$r" = 0 ] && speed "${t}_rev" lx45cc1_rev
    else
      say "$t reverse skipped: the forward did not pass"
    fi
  fi
  R="$R $t forward rc $f, reverse rc $r;"
  if [ "$f" != 0 ] || { [ "$r" != - ] && [ "$r" != 0 ]; }; then say "stops: $t ended without a verdict (rc $f / $r)"; break; fi
done
STATUS="done:${R%;} (verdicts in the CARD_*_DONE files)"
