#!/bin/bash
# Start BigMoE's slot (the capture, then the TTT30C arms 300 s later) only after EvalSuite's 45 GB P2 gate has really
# ended (Fri 13 Nov 13:4x; the orchestrator: "keep both slot jobs from starting until EvalSuite's resumed chain (pid 20328)
# ends, then re-arm them on AFTER_PID 20328"). The first arming waited on chain pid 5975 alone, which ended at 13:31 on a
# VOID that the chain then resumed under a new pid, so the capture started beside the gate for 3 min.
# Conditions: (a) GATE_PID has ended, and (b) EvalSuite's peer line has shown no "P2 gate" lock for 10 consecutive minutes,
# or ~/EvalSuite/results/CARD_p2_45_DONE exists. ~/FableGod/deepdraft/STOP_SLOT ends the wait (checked after each minute).
# usage (a harness background task): GATE_PID=20328 bash scripts/dd_slot_after_gate.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
source scripts/dd_slot_lock.sh                           # STOP
LOG=$HOME/BigMoE/data/deepdraft/ttt30c/slot.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
DONEF=$HOME/EvalSuite/results/CARD_p2_45_DONE
gate_lock() { peer ls 2>/dev/null | grep "^- Eval suite" | grep -oE "· (shared run|benchmarking|waiting for a quiet window)[^·]*" | grep -q "P2 gate"; }   # the lock segment only: its status text names the gate too
say "after-gate waiter (pid $$): GATE_PID ${GATE_PID:?set GATE_PID}, then 10 min with no EvalSuite P2-gate lock (or $DONEF)"
ok=0
while :; do
  sleep 60
  [ -e "$STOP" ] && { say "after-gate waiter: STOP_SLOT, nothing started"; exit 0; }
  if kill -0 "$GATE_PID" 2>/dev/null; then ok=0; continue; fi
  if [ -e "$DONEF" ]; then say "after-gate waiter: pid $GATE_PID ended and $(basename $DONEF) exists"; break; fi
  if gate_lock; then ok=0; else ok=$((ok + 1)); fi
  [ $ok -ge 10 ] && { say "after-gate waiter: pid $GATE_PID ended, no P2-gate lock for 10 min"; break; }
done
DD_C=100 bash scripts/dd_capture_slot.sh &
CP=$!
sleep 300                                                # the capture's memory shows before the arms size themselves
[ -e "$STOP" ] || bash scripts/dd_ttt_slot.sh
wait $CP
say "after-gate waiter: both launchers ended"
