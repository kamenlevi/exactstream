#!/bin/bash
# Queued 20:18 (orchestrator's OK 20:1x): the tapmlp screen once BigMoE's bank is done, finished before Leap's 20:40 quiet.
cd "$(dirname "$0")/.."
BANK=$HOME/BigMoE/data/deepdraft/bank/bank.log
LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-14 20:33:00" +%s)
until grep -q "BANK-DONE" $BANK 2>/dev/null; do
  [ "$(date +%s)" -gt "$LAST" ] && { echo "$(date +%H:%M:%S) no BANK-DONE by 20:33: not started (yield to Leap's 20:40 quiet ticket)"; exit 0; }
  sleep 15
done
[ "$(date +%s)" -gt "$LAST" ] && { echo "$(date +%H:%M:%S) BANK-DONE came after 20:33: not started (Leap's 20:40 quiet ticket)"; exit 0; }
[ -f $HOME/.local/share/quiet/PAUSE ] && { echo "$(date +%H:%M:%S) PAUSE: not started"; exit 0; }
echo "$(date +%H:%M:%S) BANK-DONE; swap: $(sysctl -n vm.swapusage)"
peer lock share 12m --cpu 200 --gpu 25 "Sandbox98: tapmlp screen (300 steps + 80 windows), MEM 3.6 GB measured peak; ends before Leap's 20:40" 2>&1 | head -1
LINE=$(sed -n '/^Line/,$p' results/screens/tapmlp_LINE.md | tr '\n' ' ')
./screen.sh tapmlp tapmlp "$LINE" 300 50 30
peer unlock 2>&1 | tail -1
echo "$(date +%H:%M:%S) swap after: $(sysctl -n vm.swapusage)"
echo "after_bank done"
