#!/bin/bash
# TREE16 M4 engine A/B, TUESDAY night 17 -> 18 Nov (m4_night.sh re-dated for NewIdeas' Tue gates: after the 23:30 apps quit,
# latest start Wed 01:30, hard end 02:05; one quiet ticket; NO-SWAP >= 36 GB; away >= 900 per run; C T T C on Sunday's M1 PASS). The END MARKER
# is data/r57/NIGHT1117_BIGMOE_END on ANY end, NOT RUN included (EvalSuite's g1_chain + Leap's slot 2 key on it). Pre-reg: docs/TREE16_M4_PREREG.md.
# - Waits for Tue 23:35 (after the 23:30 apps quit). M1 CONDITION (the orchestrator Sun 20:5x, after the 18:52 HOLD took
#   Sunday night's runs): with "#3 M1 rc 0: PASS" in data/r57/night1115.log the A/B starts at once; without it, M1 (t16a,
#   the same 3 keys, tag t16am1w) runs FIRST under the same ticket; on M1 FAIL (or rc != 0) the runner stops (marker).
# - PAUSE (the orchestrator 21:0x): waited out up to the latest start that still fits (01:15 with M1 needed, 01:30 with a
#   Sunday-night PASS; between runs, the next run's latest start). HOLD / the lid closed / HEALTH ACT stop at once.
# - Runs C1 T1 T2 C2 on snapshot t16b (arm T = R57_TREE16=1), one process each. A run starts only if its expected length
#   (the last run's, first guess 12 min, + 10 %) ends by 02:05; a watchdog stops the runner at 02:05 sharp anyway.
# - On ANY end: the running job stopped and waited for, the ticket released, tree16_m4ab.py on whatever ran, then the
#   marker data/r57/NIGHT1117_BIGMOE_END (unless data/r57/m4_1117.NOMARK exists: a deliberate stop to re-arm).
# M4_HEADTEST=1: no waits, gates or ticket: each run's exact command through scripts/headtest_r57.py (the real loop up to
# its model load); prints PASS / FAIL per run.
> (a quote from the project owner, kept private)
# ~6 GB waits for away_secs (HIDIdleTime) >= 900, checked before and after the memory reads): AWAYMIN (default 900) is
# required before the ticket, after the grant, after the NO-SWAP reads (Kamen back there: wait, then read memory again)
# and before each run; waited out up to the same latest start as a PAUSE. If M4 cannot start by its latest start (01:30)
# it ends NOT RUN WITH the marker (11-17: followers key on it) and writes
# data/r57/m4_1117_AWAY_SKIP (a record). Between runs a timeout ends the runner at
# that process boundary WITH the marker (the A/B analysed on what ran).
# M4_SIM=1 (simulation only): SIMD = the log / marker dir; START now + 5 s; HARD now + M4_SIM_MIN (6) min; latest start
# HARD - M4_SIM_LATE (150) s; first run guess M4_SIM_EST (5) s; a fake grant; FAKE_NOSWAP / FAKE_AWAYF = files holding
# NO-SWAP GB / away seconds; reads M4_SIM_GS (3) s apart; each run = sleep M4_SIM_RUN (3) s; no analysis.
cd ~/BigMoE || exit 9
L=data/r57/m4_1117.log
MK=data/r57/NIGHT1117_BIGMOE_END
NOMARK=data/r57/m4_1117.NOMARK
Q=$HOME/.local/share/quiet
START=$(date -j -f '%Y-%m-%d %H:%M:%S' '2026-11-17 23:35:00' +%s)
HARD=$(date -j -f '%Y-%m-%d %H:%M:%S' '2026-11-18 02:05:00' +%s)
ENVA=$(sed -n 2p data/stack/x45rpl1_env.txt | tr ',' ' ')
KEYS=humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,mbpp/2,mbpp/13,mbpp/16,mbpp/17,mbpp/19,mbpp/28,mbpp/29,mbpp/30
COMMON="R57_PRESET=45 R57_REPS=1 R57_WARM=1 R57_CHATSW=1 R57_MTP_ALL=1 R57_MTP_CKPT=$HOME/datasets/deepdraft/ckpt/dd_t30b.safetensors R57_KEYS=$KEYS"
SCREEN=data/r57/snap/t16b/r57_screen_t16b.py
M1KEYS=humaneval/HumanEval/1,mbpp/16,ifeval/1219              # Sunday night's M1 (night1115d #3), verbatim
M1ENV="R57_PRESET=45 R57_REPS=1 R57_WARM=1 R57_TAG=t16am1w R57_KEYS=$M1KEYS R57_CHATSW=1 R57_MTP_ALL=1 R57_MTP_CKPT=$HOME/datasets/deepdraft/ckpt/dd_t30b.safetensors R57_TREE16_M1TEST=1"
M1SCREEN=data/r57/snap/t16a/r57_screen_t16a.py
ORDER="C1 T1 T2 C2"
AWAYMIN=${AWAYMIN:-900}; SKIP=data/r57/m4_1117_AWAY_SKIP; LATE_A=2100; LATE_M1=3000; EST0=720; GS=30; WS=30; SETTLE=20
SIM=${M4_SIM:-0}
case "$AWAYMIN" in ""|*[!0-9]*) echo "AWAYMIN must be a whole number of seconds" >&2; exit 2;; esac
if [ "$SIM" = 1 ]; then                                  # simulation only (the header)
  W=${SIMD:?SIMD is required with M4_SIM=1}; L=$W/m4.log; MK=$W/M4_DONE; NOMARK=$W/m4.NOMARK; SKIP=$W/AWAY_SKIP
  START=$(( $(date +%s) + 5 )); HARD=$(( $(date +%s) + 60 * ${M4_SIM_MIN:-6} ))
  LATE_A=${M4_SIM_LATE:-150}; LATE_M1=$LATE_A; EST0=${M4_SIM_EST:-5}; GS=${M4_SIM_GS:-3}; WS=1; SETTLE=1
fi
armenv() { case $1 in C*) echo "R57_TAG=${2:-t16b}$1";; T*) echo "R57_TAG=${2:-t16b}$1 R57_TREE16=1";; esac; }
if [ "${M4_HEADTEST:-0}" = 1 ]; then
  out=$(env $ENVA ${M1ENV/R57_TAG=t16am1w/R57_TAG=t16am1whead} .venv/bin/python scripts/headtest_r57.py $M1SCREEN 160 128 2>&1 | tail -1)
  case "$out" in *"reached the model load"*) echo "M1 PASS (t16a, the in-window fallback)";; *) echo "M1 FAIL: $out";; esac
  for r in $ORDER; do
    out=$(env $ENVA $COMMON $(armenv $r t16bhead) .venv/bin/python scripts/headtest_r57.py $SCREEN 160 256 2>&1 | tail -1)
    case "$out" in *"reached the model load"*) echo "$r PASS ($(armenv $r))";; *) echo "$r FAIL: $out";; esac
  done
  exit 0
fi
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
noswap() { if [ "$SIM" = 1 ] && [ -n "${FAKE_NOSWAP:-}" ]; then cat "$FAKE_NOSWAP"; return; fi
           vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
away_ok() {    # Kamen away >= AWAYMIN s (no reading = at the Mac); AW = the last reading
  local a; if [ "$SIM" = 1 ] && [ -n "${FAKE_AWAYF:-}" ]; then a=$(cat "$FAKE_AWAYF" 2>/dev/null); else a=$(bash $Q/away_secs.sh 2>/dev/null); fi
  AW=${a:-none}; case "$a" in ""|*[!0-9]*) return 1;; esac; [ "$a" -ge "$AWAYMIN" ]; }
STATUS="not started"; LOCKED=0; CP=""; GUARD=""; WD=""; RAN=0
killtree() { local c; for c in $(pgrep -P $1 2>/dev/null); do killtree $c; done; kill -TERM $1 2>/dev/null; }
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null; [ -n "$WD" ] && kill $WD 2>/dev/null
            if [ -n "$CP" ] && kill -0 $CP 2>/dev/null; then         # a run still going: stop it and WAIT until it is gone
              killtree $CP
              for i in $(seq 1 120); do kill -0 $CP 2>/dev/null || break; sleep 1; done
              kill -0 $CP 2>/dev/null && { kill -KILL $CP 2>/dev/null; sleep 2; }
            fi
            [ $LOCKED = 1 ] && { [ "$SIM" = 1 ] || peer unlock >/dev/null 2>&1; say "ticket released"; }
            [ $RAN = 1 ] && [ "$SIM" != 1 ] && say "$(.venv/bin/python scripts/tree16_m4ab.py 2>&1 | tail -1)"
            if [ -e $NOMARK ]; then
              say "END without the marker (NOMARK): $STATUS"
            else
              echo "$(date '+%m-%d %H:%M:%S') BigMoE M4 17->18: $STATUS" > $MK
              say "END: $STATUS (marker $MK written after the runs exited and the ticket was released; NO-SWAP $(noswap) GB, swap $(swapmb) MB)"
            fi; }
trap cleanup EXIT; trap 'STATUS="$STATUS; stopped by a signal"; exit 1' INT TERM
stop_checks() { # $1 = the latest time (epoch) to wait out a PAUSE (the orchestrator 21:0x, as Leap 5a32a17: a PAUSE is usually
                # fanwatch's transient heat pause); HOLD, the lid closed and HEALTH ACT end the runner at once
  local dl=${1:-0}
  [ -e $Q/HOLD ] && { STATUS="$STATUS; STOP: HOLD"; exit 0; }
  lid_closed && { STATUS="$STATUS; STOP: lid closed"; exit 0; }
  grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null && { STATUS="$STATUS; STOP: HEALTH ACT"; exit 0; }
  if [ -e $Q/PAUSE ]; then
    say "PAUSE present: waiting it out until $(date -r $dl '+%H:%M:%S') at the latest"
    while [ -e $Q/PAUSE ] && [ "$(date +%s)" -lt $dl ]; do
      sleep 10
      [ -e $Q/HOLD ] && { STATUS="$STATUS; STOP: HOLD (during a PAUSE)"; exit 0; }
      lid_closed && { STATUS="$STATUS; STOP: lid closed (during a PAUSE)"; exit 0; }
    done
    [ -e $Q/PAUSE ] && { STATUS="$STATUS; STOP: PAUSE still present at $(date -r $dl '+%H:%M:%S') (the latest start)"; exit 0; }
    say "PAUSE cleared"
  fi
  if [ "$AWAYMIN" -gt 0 ] && ! away_ok; then     # the away gate (the header): waited out to the same latest time
    say "Kamen on the Mac (away_secs $AW < $AWAYMIN): waiting until $(date -r $dl '+%H:%M:%S') at the latest"
    while ! away_ok && [ "$(date +%s)" -lt $dl ]; do
      sleep 10
      [ -e $Q/HOLD ] && { STATUS="$STATUS; STOP: HOLD (during the away wait)"; exit 0; }
      lid_closed && { STATUS="$STATUS; STOP: lid closed (during the away wait)"; exit 0; }
    done
    if ! away_ok; then
      if [ $RAN = 0 ]; then       # never started: NOT RUN, NOMARK (no marker) + the skip file for the key test
        :   # 11-17: no NOMARK here; the marker is written on any end, NOT RUN included
        echo "$(date '+%m-%d %H:%M:%S') BigMoE M4 17->18 NOT RUN: Kamen on the Mac (away_secs $AW < $AWAYMIN) until the latest start $(date -r $dl '+%H:%M:%S')" > $SKIP
> (a quote from the project owner, kept private)
      fi
> (a quote from the project owner, kept private)
    fi
    say "away again (away_secs $AW)"
  fi; }
rm -f $MK $SKIP
say "armed: waits for Tue 23:35; M1 condition (a Sunday-night PASS -> the A/B, else M1 first); a PAUSE is waited out to 01:15 (M1 needed) / 01:30; C1 T1 T2 C2 on t16b; hard end 02:05; away >= $AWAYMIN s before the ticket, after the grant, after the memory reads and before each run (NOT RUN = the marker + $SKIP)$( [ "$SIM" = 1 ] && echo "; SIMULATION")"
while [ "$(date +%s)" -lt $START ]; do sleep $WS; done
if grep -q "#3 M1 rc 0: PASS" data/r57/night1115.log; then M1OK=1; STATUS="Sunday night's M1 PASS"
else M1OK=0; STATUS="no Sunday-night M1 PASS: M1 runs first in the window"; fi
say "$STATUS"
if [ $M1OK = 1 ]; then LATEST=$(( HARD - LATE_A )); else LATEST=$(( HARD - LATE_M1 )); fi   # the latest start that still fits: 01:30 / 01:15
stop_checks $LATEST
[ "$(date +%s)" -lt $HARD ] || { STATUS="$STATUS; past the hard end before the ticket"; exit 7; }
say "queue the quiet ticket (NO-SWAP $(noswap) GB)"
if [ "$SIM" = 1 ]; then say "SIMULATION: a fake quiet grant"; else
PEER_QUEUE_TIMEOUT=$(( HARD - $(date +%s) )) peer lock quiet --queue 80m "BigMoE M4: TREE16 engine A/B (t16b C T T C), exact-45 at C 160, MEM 32 GB, hard end 02:05" >> $L 2>&1 \
  || { STATUS="$STATUS; the quiet ticket left the queue without a grant: not run"; exit 4; }
fi
LOCKED=1; say "granted; settle $SETTLE s"; sleep $SETTLE
stop_checks $LATEST
( while [ "$(date +%s)" -lt $HARD ]; do sleep 5; done; echo "$(date '+%m-%d %H:%M:%S') HARD END 02:05: stopping" >> $L; kill -TERM $$ ) & WD=$!
while :; do
  ok=0; for i in $(seq 1 20); do a=$(noswap); sleep $GS; b=$(noswap); say "  gate 36: reads $a / $b GB"; [ $a -ge 36 ] && [ $b -ge 36 ] && { ok=1; break; }; done
  [ $ok = 1 ] || { STATUS="$STATUS; NO-SWAP below 36 GB for 10 min under the ticket: not run"; exit 3; }
  { [ "$AWAYMIN" = 0 ] || away_ok; } && break
  say "Kamen back on the Mac after the memory reads (away_secs $AW < $AWAYMIN): wait, then read memory again"
  stop_checks $LATEST
done
s0=$(swapmb); ( while :; do [ $(( $(swapmb) - s0 )) -gt 2048 ] && { echo "$(date +%T) SWAP GUARD: $(swapmb) MB" >> $L; kill -TERM $$; exit; }; sleep 5; done ) & GUARD=$!
if [ $M1OK = 0 ]; then                                   # the M1 condition: t16a, Sunday night's #3 verbatim (tag t16am1w)
  [ $(( $(date +%s) + 660 )) -le $HARD ] || { STATUS="$STATUS; no room for M1 before the hard end"; exit 7; }
  say "M1 start (t16a in the window; NO-SWAP $(noswap) GB)"
  rm -f data/r57/t16am1w_runs.json; RAN=1
  ( exec env $ENVA $M1ENV .venv/bin/python $M1SCREEN 160 128 ) >> data/r57/t16a_m1.log 2>&1 &
  CP=$!; wait $CP; RC=$?; CP=""
  M1V=$(python3 - <<'PY' 2>&1
import json
try:
    R = json.load(open("data/r57/t16am1w_runs.json"))
except Exception as e:
    print(f"no runs.json ({e})"); raise SystemExit
ok = bool(R) and all((r.get("m1t") or {}).get("checks", 0) > 0 and r["m1t"]["match"] == r["m1t"]["checks"] for r in R)
print(("PASS " if ok else "FAIL ") + "; ".join(f"{r['key']}: {r.get('m1t')}" for r in R))
PY
)
  say "M1 (in the window) rc $RC: $M1V"; STATUS="$STATUS; M1 rc $RC ${M1V:0:160}"
  case "$M1V" in PASS*) [ $RC = 0 ] || { STATUS="$STATUS; M1 rc $RC: the A/B is not run"; exit 6; };;
                 *) STATUS="$STATUS; M1 not PASS: the A/B is not run"; exit 6;; esac
  stop_checks $(( HARD - 720 * 11 / 10 ))                # a PAUSE: waited out while C1 can still start
fi
EST=$EST0                                                  # a first guess of one run (s): load + warm-up + 16 keys x 256 tokens
for r in $ORDER; do
  [ $(( $(date +%s) + EST * 11 / 10 )) -le $HARD ] || { STATUS="$STATUS; stopped before $r at a process boundary (~$EST s per run, hard end 02:05)"; break; }
  say "run $r start ($(armenv $r); NO-SWAP $(noswap) GB)"
  t0=$(date +%s); RAN=1
  if [ "$SIM" = 1 ]; then ( exec sleep ${M4_SIM_RUN:-3} ) >> $L 2>&1 &
  else ( exec env $ENVA $COMMON $(armenv $r) .venv/bin/python $SCREEN 160 256 ) >> data/r57/m4_1117_runs.log 2>&1 &   # exec env: $! = the job
  fi
  CP=$!; wait $CP; RC=$?; CP=""
  EST=$(( $(date +%s) - t0 ))
  say "run $r rc $RC ($EST s)"; STATUS="$STATUS; $r rc $RC"
  [ $RC = 0 ] || { STATUS="$STATUS; stopped after $r rc $RC"; exit 5; }
  [ "$r" = "${ORDER##* }" ] || stop_checks $(( HARD - EST * 11 / 10 ))   # a PAUSE / Kamen on the Mac: waited out while the next
                                                         # run can still start (none after the last run)
done
