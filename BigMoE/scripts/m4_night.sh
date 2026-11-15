#!/bin/bash
# TREE16 M4 engine A/B, Monday night 16 -> 17 Nov (the orchestrator, Sun 17:1x: FIRST in the order, 23:05 -> HARD END 00:25,
# stopping at a process boundary; one quiet ticket; NO-SWAP >= 36 GB; C T T C; gated on Sunday night's M1 PASS; an end
# marker on any end, like night1115b). Pre-registration: docs/TREE16_M4_PREREG.md. Analysis: scripts/tree16_m4ab.py.
# - Waits for Mon 23:05 (after the 23:00 apps quit). Gate: "#3 M1 rc 0: PASS" in data/r57/night1115.log, else the slot is
#   dropped (status in the log + marker; the orchestrator is told by hand).
# - Runs C1 T1 T2 C2 on snapshot t16b (arm T = R57_TREE16=1), one process each. A run starts only if its expected length
#   (the last run's, first guess 12 min, + 10 %) ends by 00:25; a watchdog stops the runner at 00:25 sharp anyway.
# - On ANY end: the running job stopped and waited for, the ticket released, tree16_m4ab.py on whatever ran, then the
#   marker data/deepdraft/bank/M4_1116_BIGMOE_DONE (unless data/r57/m4_1116.NOMARK exists: a deliberate stop to re-arm).
# M4_HEADTEST=1: no waits, gates or ticket: each run's exact command through scripts/headtest_r57.py (the real loop up to
# its model load); prints PASS / FAIL per run.
cd ~/BigMoE || exit 9
L=data/r57/m4_1116.log
MK=data/deepdraft/bank/M4_1116_BIGMOE_DONE
NOMARK=data/r57/m4_1116.NOMARK
Q=$HOME/.local/share/quiet
START=$(date -j -f '%Y-%m-%d %H:%M:%S' '2026-11-16 23:05:00' +%s)
HARD=$(date -j -f '%Y-%m-%d %H:%M:%S' '2026-11-17 00:25:00' +%s)
ENVA=$(sed -n 2p data/stack/x45rpl1_env.txt | tr ',' ' ')
KEYS=humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,mbpp/2,mbpp/13,mbpp/16,mbpp/17,mbpp/19,mbpp/28,mbpp/29,mbpp/30
COMMON="R57_PRESET=45 R57_REPS=1 R57_WARM=1 R57_CHATSW=1 R57_MTP_ALL=1 R57_MTP_CKPT=$HOME/datasets/deepdraft/ckpt/dd_t30b.safetensors R57_KEYS=$KEYS"
SCREEN=data/r57/snap/t16b/r57_screen_t16b.py
ORDER="C1 T1 T2 C2"
armenv() { case $1 in C*) echo "R57_TAG=${2:-t16b}$1";; T*) echo "R57_TAG=${2:-t16b}$1 R57_TREE16=1";; esac; }
if [ "${M4_HEADTEST:-0}" = 1 ]; then
  for r in $ORDER; do
    out=$(env $ENVA $COMMON $(armenv $r t16bhead) .venv/bin/python scripts/headtest_r57.py $SCREEN 160 256 2>&1 | tail -1)
    case "$out" in *"reached the model load"*) echo "$r PASS ($(armenv $r))";; *) echo "$r FAIL: $out";; esac
  done
  exit 0
fi
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
STATUS="not started"; LOCKED=0; CP=""; GUARD=""; WD=""; RAN=0
killtree() { local c; for c in $(pgrep -P $1 2>/dev/null); do killtree $c; done; kill -TERM $1 2>/dev/null; }
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null; [ -n "$WD" ] && kill $WD 2>/dev/null
            if [ -n "$CP" ] && kill -0 $CP 2>/dev/null; then         # a run still going: stop it and WAIT until it is gone
              killtree $CP
              for i in $(seq 1 120); do kill -0 $CP 2>/dev/null || break; sleep 1; done
              kill -0 $CP 2>/dev/null && { kill -KILL $CP 2>/dev/null; sleep 2; }
            fi
            [ $LOCKED = 1 ] && { peer unlock >/dev/null 2>&1; say "ticket released"; }
            [ $RAN = 1 ] && say "$(.venv/bin/python scripts/tree16_m4ab.py 2>&1 | tail -1)"
            if [ -e $NOMARK ]; then
              say "END without the marker (NOMARK): $STATUS"
            else
              echo "$(date '+%m-%d %H:%M:%S') BigMoE M4 16->17: $STATUS" > $MK
              say "END: $STATUS (marker $MK written after the runs exited and the ticket was released; NO-SWAP $(noswap) GB, swap $(swapmb) MB)"
            fi; }
trap cleanup EXIT; trap 'STATUS="$STATUS; stopped by a signal"; exit 1' INT TERM
stop_checks() { [ -e $Q/HOLD ] && { STATUS="$STATUS; STOP: HOLD"; exit 0; }
                [ -e $Q/PAUSE ] && { STATUS="$STATUS; STOP: PAUSE"; exit 0; }
                lid_closed && { STATUS="$STATUS; STOP: lid closed"; exit 0; }
                grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null && { STATUS="$STATUS; STOP: HEALTH ACT"; exit 0; }; }
rm -f $MK
say "armed: waits for Mon 23:05; gate: Sunday night's M1 PASS; C1 T1 T2 C2 on t16b; hard end 00:25"
while [ "$(date +%s)" -lt $START ]; do sleep 30; done
grep -q "#3 M1 rc 0: PASS" data/r57/night1115.log || { STATUS="Sunday night's M1 is not PASS (night1115.log): the slot is dropped"; exit 6; }
STATUS="M1 PASS"
stop_checks
[ "$(date +%s)" -lt $HARD ] || { STATUS="$STATUS; past the hard end before the ticket"; exit 7; }
say "queue the quiet ticket (NO-SWAP $(noswap) GB)"
PEER_QUEUE_TIMEOUT=$(( HARD - $(date +%s) )) peer lock quiet --queue 80m "BigMoE M4: TREE16 engine A/B (t16b C T T C), exact-45 at C 160, MEM 32 GB, hard end 00:25" >> $L 2>&1 \
  || { STATUS="$STATUS; the quiet ticket left the queue without a grant: not run"; exit 4; }
LOCKED=1; say "granted; settle 20 s"; sleep 20
stop_checks
( while [ "$(date +%s)" -lt $HARD ]; do sleep 5; done; echo "$(date '+%m-%d %H:%M:%S') HARD END 00:25: stopping" >> $L; kill -TERM $$ ) & WD=$!
ok=0; for i in $(seq 1 20); do a=$(noswap); sleep 30; b=$(noswap); say "  gate 36: reads $a / $b GB"; [ $a -ge 36 ] && [ $b -ge 36 ] && { ok=1; break; }; done
[ $ok = 1 ] || { STATUS="$STATUS; NO-SWAP below 36 GB for 10 min under the ticket: not run"; exit 3; }
s0=$(swapmb); ( while :; do [ $(( $(swapmb) - s0 )) -gt 2048 ] && { echo "$(date +%T) SWAP GUARD: $(swapmb) MB" >> $L; kill -TERM $$; exit; }; sleep 5; done ) & GUARD=$!
EST=720                                                  # a first guess of one run (s): load + warm-up + 16 keys x 256 tokens
for r in $ORDER; do
  [ $(( $(date +%s) + EST * 11 / 10 )) -le $HARD ] || { STATUS="$STATUS; stopped before $r at a process boundary (~$EST s per run, hard end 00:25)"; break; }
  say "run $r start ($(armenv $r); NO-SWAP $(noswap) GB)"
  t0=$(date +%s); RAN=1
  ( exec env $ENVA $COMMON $(armenv $r) .venv/bin/python $SCREEN 160 256 ) >> data/r57/m4_1116_runs.log 2>&1 &   # exec env: $! = the job
  CP=$!; wait $CP; RC=$?; CP=""
  EST=$(( $(date +%s) - t0 ))
  say "run $r rc $RC ($EST s)"; STATUS="$STATUS; $r rc $RC"
  [ $RC = 0 ] || { STATUS="$STATUS; stopped after $r rc $RC"; exit 5; }
  stop_checks
done
