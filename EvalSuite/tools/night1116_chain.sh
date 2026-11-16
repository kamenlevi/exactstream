#!/bin/bash
# night1116_chain.sh: Sun -> Mon 15-16 Nov at 45 GB. Armed Sun 21:22 behind BigMoE's night (it started 23:41); re-armed
# Mon 03:2x for the orchestrator's 03:0x ask: runner 64 had waited since 23:41 for NO-SWAP 49 GB (MEM 45 + 4) while the
# Mac sat at 41-42 GB. All runs go through runner 64.
#   1. The tier-0a REPLAY calibration (docs/PREREG_lossygate amendment 2):
#      S1 x45rpl1's own tier-0 rows; S2 its REPLAY=exact self-check, which writes the margins, then
#      tools/replay_unset_check.py; S3 p2x45rpl1, lossycalx45rpl1, then m1bx45rpl1 (MEM 47) on those margins. m1b goes
#      last, with its own 90-min cutoff, so a memory shortfall cannot hold up the base.
#   2. The close-call BASE lx45cc1 (amendment 3): forward, then reverse if the forward completed.
# Start: behind Leap's a45: its end marker (data/week98/cc3/A45_1116_DONE) written after today's a45 run started (its
# "run a45" line in Leap's cc3.log), or from 03:45 if a45 has not started and no other session's lock is above 5 GB.
# Either is read twice, 30 s apart, and nothing starts while HOLD, PAUSE, a shut lid (no external display) or Kamen at
# the Mac.
# Memory: runner 64 gates each process on NO-SWAP >= CARD_MEM + 4 GB; the orchestrator: no higher than MEM. Gate 43 GB
# for the MEM 45 runs (Leap measured exact-45 C 256's lifetime peak at 42.5 GB), 47 (its MEM) for m1b: CARD_MEM 39 / 43,
# so the shared lock's reason reads "MEM 39" / "MEM 43" (the gate shift; the runs stay MEM 45 / 47). The calibration
# rows are outputs and margins, and the base's verdict and speed read-out (kept tokens per check) are outputs too, so the
# tighter start biases none of them. Runner 64 keeps its +2 GB swap guard at each process start and its share lock
# behind any quiet lock or queue, and waits out PAUSE and a shut lid itself.
# Kamen at the Mac (away_secs.sh < 900): no start; during a run, two reads 30 s apart make it yield at once (keep >= 5 GB
# free): the runner and its children stop, peer unlock. Once nothing holds, S1, B1 and B2 continue (CARD_RESUME=1); a
# yield in S2 or S3 ends the calibration (their REPLAY merges are per runner start) and the base follows.
# A HOLD before or during a run stops the chain, the runner and its children too. Cutoffs: 09:30 for the calibration,
# 14:00 for the base; no start after 13:45.
# EVERY end writes results/CHAIN_1116_END, after the runner and its children exit and peer unlock; a HOLD stop says so.
# Stop the chain by pid only, with a plain kill (TERM): it then stops its runner and the runner's children. kill -9 skips
# the marker.
# CHAIN_DRY=1: every runner call as a dry run (CARD_DRY=1); no wait, no GPU, no marker.
#   CHAIN_MYPID=<this session's Claude pid> [CHAIN_DRY=1] tools/night1116_chain.sh
# Tests only: CHAIN_QDIR (HOLD / PAUSE), CHAIN_A45 (the marker), CHAIN_A45LOG, CHAIN_FALLBACK (YYYY-mm-dd HH:MM),
# CHAIN_AWAY (a command that prints away seconds), CHAIN_RUNNER (in place of tools/day_runner64.sh).
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
DRY=${CHAIN_DRY:-0}; L=results/night1116_chain.log; END=results/CHAIN_1116_END
MYPID=${CHAIN_MYPID:?set CHAIN_MYPID to this session s Claude pid}
MARK=${CHAIN_A45:-$HOME/Leap/data/week98/cc3/A45_1116_DONE}        # a45_last.sh: removed at each arming, written on any end
A45LOG=${CHAIN_A45LOG:-$HOME/Leap/data/week98/cc3/cc3.log}
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1:00" +%s; }
FALLBACK=$(ep "${CHAIN_FALLBACK:-2026-11-16 03:45}")
CALCUT="2026-11-16 09:30"; BASECUT="2026-11-16 14:00"; GIVEUP=$(( $(ep "$BASECUT") - 900 ))
GATE=43; GATE_M1B=47                                         # NO-SWAP GB at each process start = runner 64's CARD_MEM + 4
Q=${CHAIN_QDIR:-$HOME/.local/share/quiet}                  # HOLD and PAUSE live here
AWAY=${CHAIN_AWAY:-$HOME/.local/share/quiet/away_secs.sh}; RUNNER=${CHAIN_RUNNER:-tools/day_runner64.sh}
BMS=$HOME/BigMoE/data/stack; EXACT=ref2tnax,ref2tnaxn.r3,ref2tnaxn.r2
RPL=data/r57/snap/rpl1=0fb256faba39d467950e5c451b751c6d3b347786,data/r57/snap/frozen_rpl1=db72c3c1367e2c68f9a1365ed8d6137fce6c8af6,bigmoe=d92cdb4b9e81d386e4170323d50f11b34147aa5f
CC=data/r57/snap/cc1=85a88a46a945b3a4d21cc2c6fb216372fa549eab,data/r57/snap/frozen_cc1=ec97bdbb7ac3ba7fdd1cdf845762e4957d6b4c06,bigmoe=d92cdb4b9e81d386e4170323d50f11b34147aa5f
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$L"; }
STATUS="armed"; RPID=""
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
  echo "CHAIN 1116 END $(date '+%m-%d %H:%M:%S'): $STATUS" > "$END"
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
a45_run() {     # the epoch of today's last "run a45[bc]: ..." line in Leap's cc3.log (cc3_runner's run start); empty: none
  local t; t=$(grep -E '^11-16 [0-9:]{8} run a45' "$A45LOG" 2>/dev/null | tail -1 | awk '{print $2}')
  [ -n "$t" ] && date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-16 $t" +%s
}
startok() {     # 0 = go: nothing holds, and Leap's a45 marker written after its run started, or (a45 not started) from 03:45
  held && return 1                                 # with no other lock above 5 GB. A marker with no run before it does not
  local t0; t0=$(a45_run)                          # count (a45_last.sh writes one when stopped, as at 03:13 before a re-arm)
  if [ -n "$t0" ]; then
    { [ -e "$MARK" ] && [ "$(stat -f %m "$MARK")" -ge "$t0" ]; } || return 1
    WHY="Leap's a45 marker after its run ($(head -c 100 "$MARK" | tr '\n' ' '))"; return 0
  fi
  { [ "$(date +%s)" -ge "$FALLBACK" ] && ! biglock > /dev/null; } || return 1
  WHY="03:45, a45 not started, no other lock above 5 GB"; return 0
}
run() {         # label, the run's TG (for its DONE file), then runner 64's K=V settings; the runner is a stoppable child.
  local lab=$1 tg=$2 rc; shift 2                   # 75: it yielded to Kamen (not started, or stopped; nothing left running)
  [ -e "$Q/HOLD" ] && { STATUS="stopped by HOLD before: $STATUS"; say "HOLD: $lab not started"; exit 8; }
> (a quote from the project owner, kept private)
  say "$lab: $*"
  env CARD_DRY="$DRY" "$@" "$RUNNER" >> results/night1116_chain_runner.out 2>&1 &
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
  say "$lab: rc $rc; $(cut -c1-220 "results/CARD_${tg}_DONE" 2>/dev/null)"
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
step() {        # label, TG, run dir, cutoff, then runner 64's settings: run(); after a yield, wait until nothing holds and
  local lab=$1 tg=$2 rd=$3 cut=$4 rc; shift 4      # continue (CARD_RESUME=1 once the run dir exists); 75: no start left
  run "$lab" "$tg" "$@"; rc=$?
  while [ "$rc" = 75 ]; do
    waitfree "$cut" || return 75
    if [ -e "results/$rd" ]; then run "$lab, resumed" "$tg" "$@" CARD_RESUME=1; else run "$lab, again" "$tg" "$@"; fi; rc=$?
  done
  return "$rc"
}
say "chain (pid $$$([ "$DRY" = 1 ] && echo ', DRY'); session $MYPID): calibration (cutoff $CALCUT), then the base (cutoff $BASECUT); NO-SWAP gate $GATE GB per process ($GATE_M1B for m1b)"
if [ "$DRY" != 1 ]; then
> (a quote from the project owner, kept private)
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
    [ "$(date +%s)" -lt "$GIVEUP" ] || { STATUS="no start by $(date -r "$GIVEUP" +%H:%M) (${h:+held: $h; }no a45 marker, a45 running, or another lock above 5 GB): no run"; say "$STATUS"; exit 6; }
    n=$((n + 1))
    [ $((n % 30)) = 0 ] && say "waiting: ${h:+held: $h; }$(t0=$(a45_run); [ -n "$t0" ] && echo "a45 started $(date -r "$t0" +%H:%M:%S)" || echo 'a45 not started'); $([ -e "$MARK" ] && echo 'an a45 marker' || echo 'no a45 marker'); $(biglock | head -1)"
    sleep 60
  done
  say "free NO-SWAP $(.venv/bin/python -c "import sys; sys.path.insert(0, '.'); from evalsuite.cli import free_gb; print(int(free_gb()))") GB; swap $(sysctl -n vm.swapusage | awk '{print $3}')"
fi
T0=(CARD_TIER=0 CARD_SCOPE=code "CARD_REF=$EXACT" "CARD_TREES=$RPL" CARD_SCREEN_SHA=0d3d45ac9b72 "CARD_CUTOFF=$CALCUT" "CARD_MEM=$((GATE - 4))")
X45=(CARD_TAG=x45rpl1 "CARD_FENV=$BMS/x45rpl1_env.txt" CARD_FBLOB=dcfe4196dec6663fe90988c06cd5a3fc2cd69b33 CARD_SO_SHA=5333899cb514)
STATUS="calibration S1 (x45rpl1's own rows)"
step "S1 x45rpl1 own rows" x45rpl1 card_x45rpl1_t0code "$CALCUT" "${T0[@]}" "${X45[@]}"; r1=$?; r2=-
if [ "$r1" = 0 ]; then
  [ "$DRY" = 1 ] || cp results/CARD_x45rpl1_DONE results/CARD_x45rpl1_DONE.step1
  STATUS="calibration S2 (x45rpl1 REPLAY=exact)"
  run "S2 x45rpl1 REPLAY=exact" x45rpl1 "${T0[@]}" "${X45[@]}" CARD_RESUME=1 CARD_REPLAY=exact CARD_REPLAY_ROWS=card_x45rpl1_t0code; r2=$?
  if [ "$DRY" != 1 ]; then
    .venv/bin/python tools/replay_unset_check.py card_x45rpl1_t0code "$EXACT" results/card/replay_margins_x45rpl1.json \
      results/card/lossy_keys_x45rpl1_t0code.txt > results/card_x45rpl1_unset.txt 2>&1
    say "S1's unset-path check (rc $?): $(tail -1 results/card_x45rpl1_unset.txt)"
  fi
  if [ "$r2" = 0 ] && { [ "$DRY" = 1 ] || [ -s results/card/replay_margins_x45rpl1.json ]; }; then
    for a in p2x45rpl1:fb592bfa1f41cf76deadd587f9847ffb49941068 lossycalx45rpl1:8cb1a7ca3cd3334b70a8215840c59372f33dc4ca \
        m1bx45rpl1:532f0c44eedd23c7f5e188267e11dd3b438c3f1b; do
      t=${a%%:*}; ex=(); cut=$CALCUT
      if [ "$t" = m1bx45rpl1 ]; then
        ex=("CARD_MEM=$((GATE_M1B - 4))"); c90=$(( $(date +%s) + 5400 ))
        [ "$c90" -lt "$(ep "$CALCUT")" ] && cut=$(date -r "$c90" '+%Y-%m-%d %H:%M')
      fi
      STATUS="calibration S3 ($t)"
      run "S3 $t" "$t" "${T0[@]}" "CARD_CUTOFF=$cut" CARD_TAG="$t" "CARD_FENV=$BMS/${t}_env.txt" "CARD_FBLOB=${a#*:}" \
        CARD_SO_SHA=ff69fa24a745 CARD_REPLAY=results/card/replay_margins_x45rpl1.json CARD_REPLAY_ROWS=card_x45rpl1_t0code "${ex[@]}"
      [ $? = 75 ] && { say "S3 $t yielded: the rest of the calibration is skipped (a REPLAY pass does not resume)"; break; }
    done
  else
    say "S3 skipped: S2 rc $r2, or no merged margins"
  fi
else
  say "S2 and S3 skipped: S1 rc $r1"
fi
B=(CARD_TIER=1 CARD_SCOPE=code CARD_TAG=lx45cc1 "CARD_FENV=$BMS/lx45cc1_env.txt" CARD_FBLOB=216c8e20d35494ad32c855eb83c2712e86c94d18
   "CARD_REF=$EXACT" "CARD_TREES=$CC" CARD_SCREEN_SHA=efc1f9b6f531 CARD_SO_SHA=ff69fa24a745 "CARD_CUTOFF=$BASECUT" "CARD_MEM=$((GATE - 4))")
STATUS="the base lx45cc1, forward"
step "B1 lx45cc1 forward" lx45cc1 card_lx45cc1_t1code "$BASECUT" "${B[@]}"; b1=$?; b2=-
if [ "$b1" = 0 ]; then
  STATUS="the base lx45cc1, reverse"
  step "B2 lx45cc1 reverse" lx45cc1_rev card_lx45cc1_t1code_rev "$BASECUT" "${B[@]}" CARD_ORDER=rev; b2=$?
else
  say "B2 skipped: the forward ended rc $b1"
fi
> (a quote from the project owner, kept private)
