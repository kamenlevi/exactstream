#!/bin/bash
# c7c_night.sh (Leap, Sun 15 Nov 2026 15:1x): C7c tonight AFTER a45, in the orchestrator's order (15:0x):
#   a45 -> C7c -> BigMoE's M1 check + ng1 -> EvalSuite (tier-0a, lx45cc1) -> Breakthrough's profile.
# Waits for the a45 runner's pid to exit (it writes DONE and exits by its 01:00 cutoff), then runs C7c through
# qscreen.sh (its own quiet ticket, the NO-SWAP gate 26 GB on two reads, WAIT 1800 s, time cap est + 120).
# Writes data/week98/screens/C7C_DONE on ANY end (ran, skipped, failed, stopped by a signal): BigMoE gates on it.
# Safety: if a45 still runs at 02:30, C7c is skipped (marker written) so the chain is never held.
# usage: c7c_night.sh <a45 runner pid>       head test: C7C_CMD="<cmd>" c7c_night.sh <pid of something short>
D=/Users/kamen/Leap/data/week98/screens; MARK=$D/C7C_DONE; LOG=$D/screens.log
A45PID=${1:?the a45 runner pid}
STOP=$(date -j -f "%Y-%m-%d %H:%M:%S" "${C7C_STOP:-2026-11-16 02:30:00}" +%s)
say() { echo "$(date '+%m-%d %H:%M:%S') [c7c_night] $*" >> $LOG; }
RC=9; WHY="started"
fin() { echo "C7c end $(date '+%m-%d %H:%M:%S'): rc $RC ($WHY)" > $MARK; say "wrote C7C_DONE (rc $RC, $WHY)"; }
trap fin EXIT
trap 'WHY="stopped by a signal"; RC=1; exit 1' INT TERM
rm -f $MARK
say "armed (pid $$): waits for the a45 runner (pid $A45PID) to exit; skip at $(date -r $STOP '+%m-%d %H:%M')"
while kill -0 $A45PID 2>/dev/null; do
  if [ $(date +%s) -ge $STOP ]; then WHY="a45 still running at the safety stop: C7c skipped"; RC=3; exit 3; fi
  sleep 30
done
say "the a45 runner ended; C7c starts"
cd /Users/kamen/Leap || { WHY="no ~/Leap"; RC=2; exit 2; }
if [ -n "${C7C_CMD:-}" ]; then
  bash -c "$C7C_CMD"; RC=$?
else
  WAIT=1800 scripts/week98/qscreen.sh c7c_one32 26 360 "Leap C7c one-pass 32-row check timing (slice, per-shape forms), MEM 21 GB" \
    /Users/kamen/BigMoE/.venv/bin/python scripts/week98/c7c_one32.py
  RC=$?
fi
case $RC in 0) WHY="ran";; 3) WHY="skipped by its gate";; 4) WHY="the ticket left the queue";; *) WHY="failed";; esac
exit $RC
