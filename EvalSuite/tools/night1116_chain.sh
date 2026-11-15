#!/bin/bash
# night1116_chain.sh: Sun -> Mon 15-16 Nov at 45 GB, step 4 of the orchestrator's 15:0x ORDER (Leap a45, Leap C7c, BigMoE's
# M1 check + ng1 bit test, THIS chain, then Breakthrough's profile on results/CHAIN_1116_END). All runs go through runner 64.
#   1. The tier-0a REPLAY calibration (docs/PREREG_lossygate amendment 2):
#      S1 x45rpl1's own tier-0 rows; S2 its REPLAY=exact self-check, which writes the margins, then
#      tools/replay_unset_check.py; S3 p2x45rpl1, lossycalx45rpl1, then m1bx45rpl1 (MEM 47) on those margins. m1b goes
#      last, with its own 90-min cutoff, so a memory shortfall cannot hold up the base.
#   2. The close-call BASE lx45cc1 (amendment 3): forward, then reverse if the forward completed.
# Start: BigMoE's end marker (newer than the arming), or from 01:30 with no other session's lock above 5 GB. Either is
# read twice, 30 s apart. No start after 08:00. Runner 64 then gates each process on NO-SWAP memory and takes its share
# lock, behind any quiet lock or queue. Cutoffs: 09:30 for the calibration, 14:00 for the base (apps closed until 15:00).
# EVERY end writes results/CHAIN_1116_END, after the runner and its children exit and peer unlock. Breakthrough starts
# on it; never touch its GO file. Stop the chain by pid only: it then stops its runner and the runner's children.
# CHAIN_DRY=1: every runner call as a dry run (CARD_DRY=1); no wait, no GPU, no marker.
#   CHAIN_MYPID=<this session's Claude pid> [CHAIN_DRY=1] tools/night1116_chain.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
DRY=${CHAIN_DRY:-0}; L=results/night1116_chain.log; END=results/CHAIN_1116_END
MYPID=${CHAIN_MYPID:?set CHAIN_MYPID to this session s Claude pid}
MARK=$HOME/BigMoE/data/deepdraft/bank/NIGHT1115_BIGMOE_DONE
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1:00" +%s; }
FALLBACK=$(ep "2026-11-16 01:30"); LATEST=$(ep "2026-11-16 08:00")
CALCUT="2026-11-16 09:30"; BASECUT="2026-11-16 14:00"; ARMED=$(date +%s)
BMS=$HOME/BigMoE/data/stack; EXACT=ref2tnax,ref2tnaxn.r3,ref2tnaxn.r2
RPL=data/r57/snap/rpl1=0fb256faba39d467950e5c451b751c6d3b347786,data/r57/snap/frozen_rpl1=db72c3c1367e2c68f9a1365ed8d6137fce6c8af6,bigmoe=d92cdb4b9e81d386e4170323d50f11b34147aa5f
CC=data/r57/snap/cc1=85a88a46a945b3a4d21cc2c6fb216372fa549eab,data/r57/snap/frozen_cc1=ec97bdbb7ac3ba7fdd1cdf845762e4957d6b4c06,bigmoe=d92cdb4b9e81d386e4170323d50f11b34147aa5f
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$L"; }
STATUS="armed"; RPID=""
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
  echo "CHAIN 1116 END $(date '+%m-%d %H:%M:%S'): $STATUS" > "$END"
  say "end (rc $rc): $STATUS; wrote $END"
}
trap finish EXIT
trap 'STATUS="stopped by a signal during: $STATUS"; exit 143' TERM INT
biglock() {     # another live session's lock above 5 GB: quiet, "MEM n" with n >= 6, or GPU >= 30 (a lapsed one counts)
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
    if d["mode"] == "quiet" or (m and int(m.group(1)) >= 6) or d.get("gpu", 0) >= 30:
        print(f"{pid} {d['mode']} gpu {d.get('gpu')}: {d.get('why', '')[:90]}"); sys.exit(0)
sys.exit(1)
BL
}
startok() {     # 0 = go: BigMoE's marker newer than the arming, or past 01:30 with no other lock above 5 GB
  if [ -e "$MARK" ] && [ "$(stat -f %m "$MARK")" -ge "$ARMED" ]; then
    WHY="BigMoE's end marker ($(head -c 100 "$MARK" | tr '\n' ' '))"; return 0
  fi
  if [ "$(date +%s)" -ge "$FALLBACK" ] && ! biglock > /dev/null; then WHY="the 01:30 fallback, no other lock above 5 GB"; return 0; fi
  return 1
}
run() {         # label, the run's TG (for its DONE file), then runner 64's K=V settings; the runner is a stoppable child
  local lab=$1 tg=$2 rc; shift 2
  say "$lab: $*"
  env CARD_DRY="$DRY" "$@" tools/day_runner64.sh >> results/night1116_chain_runner.out 2>&1 &
  RPID=$!; wait "$RPID"; rc=$?; RPID=""
  say "$lab: rc $rc; $(cut -c1-220 "results/CARD_${tg}_DONE" 2>/dev/null)"
  return "$rc"
}
say "chain (pid $$$([ "$DRY" = 1 ] && echo ', DRY'); session $MYPID): calibration (cutoff $CALCUT), then the base (cutoff $BASECUT)"
if [ "$DRY" != 1 ]; then
  say "waits for $MARK (newer than the arming) or the 01:30 fallback; no start after 08:00"
  n=0
  while :; do
    if startok; then
      w1=$WHY; sleep 30
      if startok; then say "start: $w1; read again 30 s later: $WHY"; break; fi
      say "the start condition did not hold 30 s later: waiting"
    fi
    [ "$(date +%s)" -lt "$LATEST" ] || { STATUS="no start by 08:00 (no marker, or another lock above 5 GB): no run"; say "$STATUS"; exit 6; }
    n=$((n + 1))
    [ $((n % 30)) = 0 ] && say "waiting: $([ -e "$MARK" ] && echo 'the marker predates the arming' || echo 'no marker yet'); $(biglock | head -1)"
    sleep 60
  done
  say "free NO-SWAP $(.venv/bin/python -c "import sys; sys.path.insert(0, '.'); from evalsuite.cli import free_gb; print(int(free_gb()))") GB; swap $(sysctl -n vm.swapusage | awk '{print $3}')"
fi
T0=(CARD_TIER=0 CARD_SCOPE=code "CARD_REF=$EXACT" "CARD_TREES=$RPL" CARD_SCREEN_SHA=0d3d45ac9b72 "CARD_CUTOFF=$CALCUT")
X45=(CARD_TAG=x45rpl1 "CARD_FENV=$BMS/x45rpl1_env.txt" CARD_FBLOB=dcfe4196dec6663fe90988c06cd5a3fc2cd69b33 CARD_SO_SHA=5333899cb514)
STATUS="calibration S1 (x45rpl1's own rows)"
run "S1 x45rpl1 own rows" x45rpl1 "${T0[@]}" "${X45[@]}"; r1=$?; r2=-
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
        ex=(CARD_MEM=47); c90=$(( $(date +%s) + 5400 ))
        [ "$c90" -lt "$(ep "$CALCUT")" ] && cut=$(date -r "$c90" '+%Y-%m-%d %H:%M')
      fi
      STATUS="calibration S3 ($t)"
      run "S3 $t" "$t" "${T0[@]}" "CARD_CUTOFF=$cut" CARD_TAG="$t" "CARD_FENV=$BMS/${t}_env.txt" "CARD_FBLOB=${a#*:}" \
        CARD_SO_SHA=ff69fa24a745 CARD_REPLAY=results/card/replay_margins_x45rpl1.json CARD_REPLAY_ROWS=card_x45rpl1_t0code "${ex[@]}"
    done
  else
    say "S3 skipped: S2 rc $r2, or no merged margins"
  fi
else
  say "S2 and S3 skipped: S1 rc $r1"
fi
B=(CARD_TIER=1 CARD_SCOPE=code CARD_TAG=lx45cc1 "CARD_FENV=$BMS/lx45cc1_env.txt" CARD_FBLOB=216c8e20d35494ad32c855eb83c2712e86c94d18
   "CARD_REF=$EXACT" "CARD_TREES=$CC" CARD_SCREEN_SHA=efc1f9b6f531 CARD_SO_SHA=ff69fa24a745 "CARD_CUTOFF=$BASECUT")
STATUS="the base lx45cc1, forward"
run "B1 lx45cc1 forward" lx45cc1 "${B[@]}"; b1=$?; b2=-
if [ "$b1" = 0 ]; then
  STATUS="the base lx45cc1, reverse"
  run "B2 lx45cc1 reverse" lx45cc1_rev "${B[@]}" CARD_ORDER=rev; b2=$?
else
  say "B2 skipped: the forward ended rc $b1"
fi
STATUS="done: calibration S1 rc $r1, S2 rc $r2 (S3 in the log); base forward rc $b1, reverse rc $b2"
