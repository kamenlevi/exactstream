#!/bin/bash
# run_c25.sh (Leap, Mon 16 Nov 2026 12:4x; CHECK25): one run of c25_loop.py (BigMoE's lossy-45 + 1c-tiers loop in the G32T
# test mode, c25_engine's arms on 32-row checks) with c25_env.py, in a slot NewIdeas gives (Tue by day ~09:30-10:00; a smoke
# today when memory frees). Waits, holding nothing, until START; never starts if it cannot end by HARD (latest start =
# HARD - EST); then: HOLD / PAUSE / lid / HEALTH ACT checks, the head test (the real loop up to its model load), a quiet ticket
# (queued until the latest start), settle 20 s, the NO-SWAP gate on the lower of two reads 30 s apart (>= GATE, default 46 =
# t1cx45t1c1's key-test peak 42.4 + 4), a swap guard (+2 GB stops it), the run killed at HARD. On ANY end: the ticket
# released, then the marker ~/Leap/data/week98/c25/C25_<TAG>_DONE = "c25 end <time>: rc N (why)".
# Env knobs (passed through to c25_env.py): C25_TAG, C25_ARMS, C25_ROUNDS, C25_WARM, C25_KEYS; C25_GATE, C25_EST_MIN;
# C25_AFTER=<marker path>: also wait for that file, newer than arming (another job's end marker).
# DRY=1: fake lock, 1 s waits, gate 1, START now, C25_CMD (default: sleep 3) in place of the loop.
# Usage: run_c25.sh "<START YYYY-MM-DD HH:MM:SS>" "<HARD END ...>"
E=/Users/kamen/Leap/data/week98/c25; H=/Users/kamen/Leap/scripts/week98/c25; D=/Users/kamen/Leap/data/week98/screens
Q=$HOME/.local/share/quiet; PY=/Users/kamen/BigMoE/.venv/bin/python
DRY=${DRY:-0}; GATE=${C25_GATE:-46}; EST=${C25_EST_MIN:-40}; ZZ=30; ST=20; TAG=${C25_TAG:-c25}
[ $DRY = 1 ] && { GATE=1; ZZ=1; ST=1; TAG=${TAG}_dry; }
export C25_TAG=$TAG
mkdir -p $E
MARK=$E/C25_${TAG}_DONE; LOG=$E/run_${TAG}.log
START=$(date -j -f "%Y-%m-%d %H:%M:%S" "${1:?START}" +%s) || exit 2
HARD=$(date -j -f "%Y-%m-%d %H:%M:%S" "${2:?HARD END}" +%s) || exit 2
[ $DRY = 1 ] && START=$(date +%s)
LATEST=$(( HARD - EST * 60 ))
T0=$(date +%s)
say() { echo "$(date '+%m-%d %H:%M:%S') [run_c25 $TAG] $*" | tee -a $LOG >> $D/screens.log; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
pmode() { pmset -g | awk '/^ *powermode/ {print $2; exit}'; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
killtree() { local c; for c in $(pgrep -P $1 2>/dev/null); do killtree $c; done; kill -TERM $1 2>/dev/null; }
RC=9; WHY="started"; LOCKED=0; CP=""; GUARD=""
fin() {
  [ -n "$GUARD" ] && kill $GUARD 2>/dev/null
  [ -n "$CP" ] && kill -0 $CP 2>/dev/null && killtree $CP
  if [ $LOCKED = 1 ]; then [ $DRY = 1 ] || peer unlock >/dev/null 2>&1; say "ticket released"; fi
  echo "c25 end $(date '+%m-%d %H:%M:%S'): rc $RC ($WHY)" > $MARK
  say "wrote $MARK (rc $RC, $WHY); NO-SWAP $(noswap) GB, swap $(swap) MB"
}
trap fin EXIT
trap 'WHY="stopped by a signal"; RC=1; exit 1' INT TERM
stop_checks() { [ -e $Q/HOLD ] && { WHY="HOLD"; RC=3; exit 3; }
                [ -e $Q/PAUSE ] && { WHY="PAUSE"; RC=3; exit 3; }
                lid_closed && { WHY="lid closed"; RC=3; exit 3; }
                grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null && { WHY="HEALTH ACT"; RC=3; exit 3; }; }
gate_ok() { local a b; a=$(noswap); sleep $ZZ; b=$(noswap); say "  gate $GATE: read 1 $a GB; read 2 (+$ZZ s) $b GB"
            [ $a -ge $GATE ] && [ $b -ge $GATE ]; }
rm -f $MARK
say "armed (pid $$, DRY $DRY): start $(date -r $START '+%m-%d %H:%M'), latest start $(date -r $LATEST '+%m-%d %H:%M'), hard end $(date -r $HARD '+%m-%d %H:%M'); gate $GATE GB; arms ${C25_ARMS:-all}; rounds ${C25_ROUNDS:-12}"
while [ $(date +%s) -lt $START ]; do sleep $(( DRY == 1 ? 1 : 20 )); done
if [ -n "${C25_AFTER:-}" ]; then   # wait for another job's end marker, newer than arming (or the latest start)
  say "waiting for $C25_AFTER (newer than arming)"
  while ! { [ -f "$C25_AFTER" ] && [ "$(stat -f %m "$C25_AFTER")" -ge $T0 ]; }; do
    [ $(date +%s) -ge $LATEST ] && { WHY="no marker $C25_AFTER before the latest start"; RC=3; exit 3; }
    sleep $(( DRY == 1 ? 1 : 20 ))
  done
  say "marker in: $(head -c 160 "$C25_AFTER")"
fi
[ $(date +%s) -ge $LATEST ] && { WHY="no start before the latest start"; RC=3; exit 3; }
cd /Users/kamen/Leap || { WHY="no ~/Leap"; RC=2; exit 2; }
stop_checks
if [ $DRY = 1 ] && [ -n "${C25_SKIP_HEAD:-}" ]; then say "head test skipped (dry)"; else
  ( eval "$(python3 $H/c25_env.py)" && $PY -I -B $H/c25_headtest.py ) >> $LOG 2>&1 || { WHY="the head test failed (log)"; RC=5; exit 5; }
  say "head test PASS"
fi
to=$(( LATEST - $(date +%s) )); [ $to -lt 60 ] && { WHY="no time left to queue"; RC=3; exit 3; }
say "queue the quiet ticket (timeout ${to} s; NO-SWAP $(noswap) GB)"
if [ $DRY = 1 ]; then echo "(dry) fake ticket" >> $LOG; else
  PEER_QUEUE_TIMEOUT=$to peer lock quiet --queue 60m "Leap CHECK25: lossy 32-row check engine A/Bs (c25), MEM 43 GB" >> $LOG 2>&1 \
    || { WHY="the ticket left the queue without a grant"; RC=4; exit 4; }
fi
LOCKED=1; say "granted (power mode $(pmode)); settle $ST s"; sleep $ST
stop_checks
gate_ok || { WHY="NO-SWAP below $GATE GB after the grant"; RC=3; exit 3; }
[ $(date +%s) -ge $LATEST ] && { WHY="granted after the latest start"; RC=3; exit 3; }
s0=$(swap); ( while :; do [ $(( $(swap) - s0 )) -gt 2048 ] && { echo "$(date +%T) [run_c25] SWAP GUARD: $(swap) MB" >> $LOG; kill -TERM $$; exit; }; sleep 5; done ) & GUARD=$!
say "run (power mode $(pmode), swap $(swap) MB, NO-SWAP $(noswap) GB)"
if [ $DRY = 1 ]; then ( bash -c "${C25_CMD:-sleep 3}" ) >> $LOG 2>&1 & else
  ( eval "$(python3 $H/c25_env.py)" && exec $PY -I -B $H/c25_loop.py 256 2048 ) > $E/${TAG}_run.txt 2>&1 &
fi
CP=$!
while kill -0 $CP 2>/dev/null; do
  if [ $(date +%s) -ge $HARD ]; then say "HARD END: stopping the run"; killtree $CP; sleep 5; WHY="cut at the hard end"; RC=6; exit 6; fi
  stop_checks_soft() { [ -e $Q/HOLD ] || [ -e $Q/PAUSE ] || lid_closed; }
  if stop_checks_soft; then say "HOLD / PAUSE / lid during the run: stopping"; killtree $CP; sleep 5; WHY="stopped (HOLD / PAUSE / lid)"; RC=3; exit 3; fi
  sleep $(( DRY == 1 ? 1 : 10 ))
done
wait $CP; RC=$?
case $RC in 0) WHY="ran";; *) WHY="the run failed (${TAG}_run.txt)";; esac
exit $RC
