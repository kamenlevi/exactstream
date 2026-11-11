#!/bin/sh
# MoonShots, one quiet lock (launch holding it): M-04 VD + LP1 confirmation, lpom_harness.py R2 4 F 0.28 F1 0.26 (a random 28 % of the
# rank >= 4 uses at 2 code planes + a random 26 % of rank-1 uses at MSB-3, the rest at v3), GPU, capped at 9.5 min. The run has its own process group: time cap, a
# lost lock, PAUSE or swap +2 GB stop the whole tree.
D=$HOME/MoonShots; PY=$HOME/BigMoE/.venv/bin/python; LOG=$D/data/kl125/vdlp_runner.log
LOCKF=$HOME/.claude-peers/locks/$CLAUDE_PID.json
mkdir -p $D/data/kl125
sh $D/scripts/keepalive_while.sh $$ & GL=$!
trap 'peer unlock > /dev/null 2>&1; if [ -n "$GL" ]; then kill $GL 2>/dev/null; fi' EXIT
lockok() { /usr/bin/python3 -I -S -c '
import json, sys, time
try: d = json.load(open(sys.argv[1]))
except Exception: sys.exit(1)
sys.exit(0 if d.get("mode") == "quiet" and d.get("until", 0) > time.time() else 1)' "$LOCKF"; }
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
S0=$(swapmb)
echo "$(date +%H:%M:%S) runner started (swap ${S0} MB)" >> $LOG
set -m
$PY $D/scripts/lpom_harness.py 4 0.28 0.26 > $D/data/kl125/screen_vdlp.txt 2>&1 &
P=$!
i=0; why=""
while kill -0 $P 2>/dev/null; do
  sleep 5; i=$((i + 5))
  if [ $i -ge 570 ]; then why="time cap"; break; fi
  if [ -f $HOME/.local/share/quiet/PAUSE ]; then why="PAUSE"; break; fi
  if [ $((i % 30)) -eq 0 ] && ! lockok; then why="lost the lock"; break; fi
  if [ $(( $(swapmb) - S0 )) -gt 2048 ]; then why="swap +2 GB"; break; fi
done
if [ -n "$why" ] && kill -0 $P 2>/dev/null; then
  kill -- -$P 2>/dev/null; sleep 3; kill -9 -- -$P 2>/dev/null
  echo "$(date +%H:%M:%S) run stopped: $why" >> $LOG
fi
wait $P 2>/dev/null
echo "$(date +%H:%M:%S) result: $(grep -E '^(loaded|lpom_|target|ar4|plan4433333333|Traceback|[A-Za-z]*Error)' $D/data/kl125/screen_vdlp.txt | cut -c1-200 | tr '\n' ' ')" >> $LOG
echo "$(date +%H:%M:%S) runner finished" >> $LOG
