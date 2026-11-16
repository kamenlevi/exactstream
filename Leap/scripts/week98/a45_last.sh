#!/bin/bash
# a45_last.sh (Leap, Sun 15 Nov 2026 23:4x): exact-45's a45 (cc3 screen 3: the shipped exact-45 32-row check split into
# dense / experts / misses) LAST in tonight's order. The orchestrator 23:4x: "YES, re-arm a45 LAST in tonight's order":
#   - gate: Breakthrough's lever-2 quiet-bench follower ends (pid 88412, after the profile 76388); if 88412 is gone, 76388's
#     end plus 88412's absence. Here: wait, holding nothing, until BOTH have ended (each checked by pid AND command);
#   - NO-SWAP >= 47 GB on two reads; its own quiet ticket; HOLD / PAUSE / lid checks as usual (cc3_runner.sh does these
#     per run, plus HEALTH ACT, the no-other-engine check and the +2 GB swap guard);
#   - CUTOFF: every run ENDS by 06:45 and none starts after ~06:00. cc3_runner gets the cutoff LATEST + 420 s (its run
#     check is now + est <= cutoff, est 420 s), so no a45 run starts after LATEST (06:00); a watchdog TERMs the runner at
#     HARD (06:45); its trap kills the run tree and releases the ticket;
#   - an end marker on ANY end: data/week98/cc3/A45_1116_DONE.
# Mon 16 Nov 03:1x, the orchestrator 03:0x: re-armed with a45's start gate at its measured peak, CC3_G45=42 on the
# lower of two reads (the 11-13 ad5f655 precedent; cc3_runner v7 reads CC3_G45); the +2 GB swap guard, HOLD / PAUSE /
# lid / HEALTH ACT and the 06:00 latest start stay.
# A HOLD while waiting ends it (marker written); a PAUSE is left to cc3_runner (it waits, holding nothing).
# usage: a45_last.sh ["<LATEST START YYYY-MM-DD HH:MM:SS>" ["<HARD END ...>"]]   (defaults Mon 06:00:00 / 06:45:00)
# Dry simulation only: DRY=1 (passed to cc3_runner: fake ticket, 1 GB gates, flagcheck.py in place of run.py), CC3_DIR (its
# log dir), A45_GATE ("pid:command-pattern ..." in place of the two gate pids), A45_MARK (the marker path).
D=${CC3_DIR:-/Users/kamen/Leap/data/week98/cc3}; LOG=$D/cc3.log; W=/Users/kamen/Leap/scripts/week98
MARK=${A45_MARK:-/Users/kamen/Leap/data/week98/cc3/A45_1116_DONE}; Q=$HOME/.local/share/quiet
GATE=${A45_GATE:-"76388:run_c55_engine 88412:run_l2_quiet"}
LATEST=$(date -j -f "%Y-%m-%d %H:%M:%S" "${1:-2026-11-16 06:00:00}" +%s) || exit 2
HARD=$(date -j -f "%Y-%m-%d %H:%M:%S" "${2:-2026-11-16 06:45:00}" +%s) || exit 2
CUTS=$(date -r $(( LATEST + 420 )) '+%Y-%m-%d %H:%M:%S')
GIVEUP=$(( LATEST - 300 ))          # cc3_runner queues no ticket after cutoff - 660 s = LATEST - 240 s
mkdir -p $D
say() { echo "$(date '+%m-%d %H:%M:%S') [a45_last] $*" >> $LOG; }
alive() { ps -p ${1%%:*} -o command= 2>/dev/null | grep -q -- "${1#*:}"; }
gate_alive() { local g; for g in $GATE; do alive $g && return 0; done; return 1; }
RC=9; WHY="started"; CP=""; L0=0
outcome() { tail -n +$(( L0 + 1 )) $LOG 2>/dev/null | grep -E "a45[bc]? rc|SKIP|STOP|did not run|left the queue|SWAP GUARD|stopped by" \
            | sed -E 's/^[0-9-]+ //' | tail -3 | tr '\n' ';'; }
fin() {
  if [ -n "$CP" ] && kill -0 $CP 2>/dev/null; then kill -TERM $CP; wait $CP 2>/dev/null; fi
  local o; o=$(outcome)
  echo "a45 end $(date '+%m-%d %H:%M:%S'): rc $RC ($WHY)${o:+; runner: $o}" > $MARK
  say "wrote $(basename $MARK) (rc $RC, $WHY)"
}
trap fin EXIT
trap 'WHY="stopped by a signal"; RC=1; exit 1' INT TERM
rm -f $MARK
say "armed (pid $$, a45 start gate ${CC3_G45:-47} GB$([ "${DRY:-0}" = 1 ] && echo ", DRY")): waits for the gate pids ($GATE) to end; then cc3_runner a45 with cutoff $CUTS (no run starts after $(date -r $LATEST '+%m-%d %H:%M')); hard end $(date -r $HARD '+%m-%d %H:%M')"
while gate_alive; do
  [ -e $Q/HOLD ] && { WHY="HOLD while waiting"; RC=3; exit 3; }
  [ $(date +%s) -ge $GIVEUP ] && { WHY="the chain was still running at $(date -r $GIVEUP '+%H:%M'): skipped (no start after $(date -r $LATEST '+%H:%M'))"; RC=3; exit 3; }
  sleep $([ "${DRY:-0}" = 1 ] && echo 1 || echo 30)
done
[ -e $Q/HOLD ] && { WHY="HOLD"; RC=3; exit 3; }
[ $(date +%s) -ge $GIVEUP ] && { WHY="the gate opened too late ($(date '+%H:%M')): skipped"; RC=3; exit 3; }
cd /Users/kamen/Leap || { WHY="no ~/Leap"; RC=2; exit 2; }
L0=$(wc -l < $LOG 2>/dev/null || echo 0); L0=${L0// /}
say "the gate pids have ended; cc3_runner (CC3_ONLY=a45) starts now, cutoff $CUTS"
CC3_ONLY=a45 $W/cc3_runner.sh "$(date '+%Y-%m-%d %H:%M:%S')" "$CUTS" > $D/a45_last_runner.out 2>&1 &
CP=$!
while kill -0 $CP 2>/dev/null; do
  if [ $(date +%s) -ge $HARD ]; then
    say "HARD END $(date -r $HARD '+%H:%M'): stopping the runner (pid $CP)"
    kill -TERM $CP; wait $CP 2>/dev/null; CP=""; WHY="cut at the hard end"; RC=6; exit 6
  fi
  sleep $([ "${DRY:-0}" = 1 ] && echo 1 || echo 10)
done
wait $CP; RC=$?; CP=""
o=$(outcome)
case "$o" in *"a45 rc 0"*|*"a45b rc 0"*|*"a45c rc 0"*) WHY="ran";; *"did not run"*|*SKIP*) WHY="skipped";; *) WHY="runner ended";; esac
exit $RC
