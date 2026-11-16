#!/bin/bash
# run_e14.sh (Leap, Sun 15 Nov 2026 20:5x): Monday night's E1-E4 in Leap's slot (the orchestrator 17:2x / 18:1x: after
# BigMoE's M4, from 00:30 to a HARD END of 03:00; an end marker on ANY end; EvalSuite's close-call arms gate on it).
# One run of e14_loop.py (the exact-45 loop + e14_engine's 8 arms on G32T's 3 windows) with e14_env.sh.
#   - Arming deletes an old marker. The run waits, holding nothing, until START (default Tue 00:30) AND:
#       BigMoE's M4 end marker newer than arming; or, from START on, no M4 marker and no BigMoE quiet lock held or queued
#       (the orchestrator 17:3x: if tonight's M1 fails BigMoE drops M4 and writes its marker at once, but a hard kill
#       writes none).
#   - It never starts if it cannot end by the hard end: the latest start is HARD - EST (EST default 45 min, the heat
#     stretch included). The run is killed (tree) at the hard end.
#   - Then (21:0x): a PAUSE (fanwatch's heat pause is transient) is waited out up to the latest start; HOLD exits.
#     Then: stop checks (HOLD / PAUSE / lid / HEALTH ACT), the head test (e14_headtest.py, the real loop up to its model
#     load), a quiet ticket (its queue timeout ends at the latest start), settle 20 s, the NO-SWAP gate on the lower of two
#     reads 30 s apart (>= GATE, default 39 = Breakthrough's measured engine peak ~31 + 8), a swap guard (+2 GB stops it),
#     the power mode logged.
#   - On ANY end (ran, skipped, failed, killed by a signal): the ticket released, then the marker
#     ~/Leap/data/week98/screens/E1E4_1117_DONE = "E1-E4 end <time>: rc N (why)".
# DRY=1: the head test of this script: fake lock, 1 s waits, gate 1 GB, START now, E14_CMD (default: sleep 3) in place
# of the loop. Usage: run_e14.sh ["<START YYYY-MM-DD HH:MM:SS>" ["<HARD END ...>"]]
D=/Users/kamen/Leap/data/week98/screens; E=/Users/kamen/Leap/data/week98/e14; H=/Users/kamen/Leap/scripts/week98/e14
MARK=$D/E1E4_1117_DONE; M4=/Users/kamen/BigMoE/data/deepdraft/bank/M4_1116_BIGMOE_DONE; LOG=$E/run_e14.log
Q=$HOME/.local/share/quiet; PY=/Users/kamen/BigMoE/.venv/bin/python
DRY=${DRY:-0}; GATE=${E14_GATE:-39}; EST=${E14_EST_MIN:-45}; ZZ=30; ST=20
[ $DRY = 1 ] && { GATE=1; ZZ=1; ST=1; }
mkdir -p $E
[ $DRY = 1 ] && { MARK=$D/E1E4_DRY_DONE; LOG=$E/run_e14_dry.log; }   # a dry run never writes the real marker
START=$(date -j -f "%Y-%m-%d %H:%M:%S" "${1:-2026-11-17 00:30:00}" +%s) || exit 2
HARD=$(date -j -f "%Y-%m-%d %H:%M:%S" "${2:-2026-11-17 03:00:00}" +%s) || exit 2
[ $DRY = 1 ] && START=$(date +%s)
LATEST=$(( HARD - EST * 60 ))
T0=$(date +%s)
say() { echo "$(date '+%m-%d %H:%M:%S') [run_e14$([ $DRY = 1 ] && echo _dry)] $*" | tee -a $LOG >> $D/screens.log; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
pmode() { pmset -g | awk '/^ *powermode/ {print $2; exit}'; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
killtree() { local c; for c in $(pgrep -P $1 2>/dev/null); do killtree $c; done; kill -TERM $1 2>/dev/null; }
bm_lock() {   # BigMoE holds or queues a quiet lock (peer ls segments; the memory peer-ls-lock-segments)
  peer ls 2>/dev/null | grep "^- " | grep "larger-than-RAM MoE" | grep -oE "· (benchmarking|waiting for a quiet window)[^·]*" | grep -q .; }
RC=9; WHY="started"; LOCKED=0; CP=""; GUARD=""
fin() {
  [ -n "$GUARD" ] && kill $GUARD 2>/dev/null
  [ -n "$CP" ] && kill -0 $CP 2>/dev/null && killtree $CP
  if [ $LOCKED = 1 ]; then [ $DRY = 1 ] || peer unlock >/dev/null 2>&1; say "ticket released"; fi
  echo "E1-E4 end $(date '+%m-%d %H:%M:%S'): rc $RC ($WHY)" > $MARK
  say "wrote $MARK (rc $RC, $WHY); NO-SWAP $(noswap) GB, swap $(swap) MB"
}
trap fin EXIT
trap 'WHY="stopped by a signal"; RC=1; exit 1' INT TERM
stop_checks() { [ -e $Q/HOLD ] && { WHY="HOLD"; RC=3; exit 3; }
                [ -e $Q/PAUSE ] && { WHY="PAUSE"; RC=3; exit 3; }
                lid_closed && { WHY="lid closed"; RC=3; exit 3; }
                grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null && { WHY="HEALTH ACT"; RC=3; exit 3; }; }
pause_wait() {  # before the ticket: fanwatch's heat pause is transient, so wait it out (up to the latest start); HOLD exits
  local said=0
  while [ -e $Q/PAUSE ] && [ ! -e $Q/HOLD ]; do
    [ $said = 0 ] && { say "PAUSE set: waiting (heat pauses are transient), at most to the latest start"; said=1; }
    [ $(date +%s) -ge $LATEST ] && { WHY="PAUSE until the latest start"; RC=3; exit 3; }
    sleep $(( DRY == 1 ? 1 : 30 ))
  done
  [ $said = 1 ] && say "PAUSE cleared"
  return 0; }
away_ok() {   # the orchestrator 21:0x (Kamen: lower-RAM tests while he is at the Mac): away >= 900 s around the reads
  local s; s=$(bash ${LEAP_AWAY_SH:-/Users/kamen/.local/share/quiet/away_secs.sh} 2>/dev/null)
> (a quote from the project owner, kept private)
gate_ok() { local a b; away_ok || return 1; a=$(noswap); sleep $ZZ; b=$(noswap); say "  gate $GATE: read 1 $a GB; read 2 (+$ZZ s) $b GB"
            [ $a -ge $GATE ] && [ $b -ge $GATE ] && away_ok; }
rm -f $MARK
say "armed (pid $$, DRY $DRY): start $(date -r $START '+%m-%d %H:%M'), latest start $(date -r $LATEST '+%m-%d %H:%M'), hard end $(date -r $HARD '+%m-%d %H:%M'); gate $GATE GB; arms ${E14_ARMS:-all 8}"
# 1. wait for the slot: START, then M4's marker (newer than arming) or the fallback (no marker, no BigMoE quiet lock)
while :; do
  now=$(date +%s)
  if [ $now -ge $LATEST ]; then WHY="no start before the latest start (the slot would not end by the hard end)"; RC=3; exit 3; fi
  if [ $now -ge $START ]; then
    if [ -f $M4 ] && [ "$(stat -f %m $M4)" -ge $T0 ]; then say "M4's end marker is in: $(head -c 160 $M4)"; break; fi
    if [ $DRY = 1 ] || ! bm_lock; then say "no M4 marker and no BigMoE quiet lock held or queued: start (the fallback)"; break; fi
  fi
  sleep $(( DRY == 1 ? 1 : 30 ))
done
cd /Users/kamen/Leap || { WHY="no ~/Leap"; RC=2; exit 2; }
pause_wait
stop_checks
# 2. the head test with the run's env (the real loop up to its model load)
if [ $DRY = 1 ] && [ -n "${E14_SKIP_HEAD:-}" ]; then say "head test skipped (dry)"; else
  ( source $H/e14_env.sh && $PY -I -B $H/e14_headtest.py ) >> $LOG 2>&1 || { WHY="the head test failed (log above)"; RC=5; exit 5; }
  say "head test PASS"
fi
# 3. the quiet ticket, queued only until the latest start
to=$(( LATEST - $(date +%s) )); [ $to -lt 60 ] && { WHY="no time left to queue"; RC=3; exit 3; }
say "queue the quiet ticket (timeout ${to} s; NO-SWAP $(noswap) GB)"
if [ $DRY = 1 ]; then echo "(dry) fake ticket" >> $LOG; else
  PEER_QUEUE_TIMEOUT=$to peer lock quiet --queue 60m "Leap E1-E4: exact check engine A/Bs (8 arms, G32T windows), MEM 33 GB" >> $LOG 2>&1 \
    || { WHY="the ticket left the queue without a grant"; RC=4; exit 4; }
fi
LOCKED=1; say "granted (power mode $(pmode)); settle $ST s"; sleep $ST
stop_checks
> (a quote from the project owner, kept private)
[ $(date +%s) -ge $LATEST ] && { WHY="granted after the latest start"; RC=3; exit 3; }
# 4. the run, killed at the hard end; swap guard +2 GB
s0=$(swap); ( while :; do [ $(( $(swap) - s0 )) -gt 2048 ] && { echo "$(date +%T) [run_e14] SWAP GUARD: $(swap) MB" >> $LOG; kill -TERM $$; exit; }; sleep 5; done ) & GUARD=$!
say "run (power mode $(pmode), swap $(swap) MB, NO-SWAP $(noswap) GB)"
if [ $DRY = 1 ]; then ( bash -c "${E14_CMD:-sleep 3}" ) >> $LOG 2>&1 & else
  ( source $H/e14_env.sh && exec $PY -I -B $H/e14_loop.py 160 256 ) > $E/e14_run.txt 2>&1 &
fi
CP=$!
while kill -0 $CP 2>/dev/null; do
  if [ $(date +%s) -ge $HARD ]; then say "HARD END: stopping the run"; killtree $CP; sleep 5; WHY="cut at the hard end (partial windows dumped)"; RC=6; exit 6; fi
  sleep $(( DRY == 1 ? 1 : 10 ))
done
wait $CP; RC=$?
case $RC in 0) WHY="ran";; *) WHY="the run failed (e14_run.txt)";; esac
exit $RC
