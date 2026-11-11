#!/bin/sh
# MoonShots, one quiet lock (launch holding it): M-09d TOPK-KLD, topk_harness.py 8 norenorm (v3 tiers, per-row top-8 without renormalization), GPU, capped at 9.5 min. The run has its own process group: time cap, a
# lost lock, PAUSE or swap +2 GB stop the whole tree.
D=$HOME/MoonShots; PY=$HOME/BigMoE/.venv/bin/python; LOG=$D/data/kl125/topk_runner.log
LOCKF=$HOME/.claude-peers/locks/$CLAUDE_PID.json
mkdir -p $D/data/kl125
# wait for room: >= 24 GB available (vm_stat), PAUSE clear, start no later than 20:30 (Kamen's window ends 20:45)
avail() { vm_stat | awk '/page size of/ {ps=$8} /Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} END {gsub("\\.","",f); gsub("\\.","",i); gsub("\\.","",s); print int((f+i+s)*ps/1073741824)}'; }
while :; do
  [ -f $HOME/.local/share/quiet/HOLD ] && { echo "$(date +%H:%M:%S) gave up: HOLD" >> $LOG; exit 0; }
  if [ ! -f $HOME/.local/share/quiet/PAUSE ] && [ $(avail) -ge 24 ]; then
    peer lock share 15m --cpu 200 --gpu 50 "MoonShots M-09d top-8 no-renorm KLD (tf screen, C 64, ~16 GB)" >> $LOG 2>&1 && break
  fi
  sleep 30
done
export TOPK_C=64
sh $D/scripts/keepalive_while.sh $$ & GL=$!
trap 'peer unlock > /dev/null 2>&1; if [ -n "$GL" ]; then kill $GL 2>/dev/null; fi' EXIT
lockok() { /usr/bin/python3 -I -S -c '
import json, sys, time
try: d = json.load(open(sys.argv[1]))
except Exception: sys.exit(1)
sys.exit(0 if d.get("mode") in ("quiet", "share") and d.get("until", 0) > time.time() else 1)' "$LOCKF"; }
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
S0=$(swapmb)
echo "$(date +%H:%M:%S) runner started (swap ${S0} MB)" >> $LOG
set -m
$PY $D/scripts/topk_harness.py ${K:-8} ${MODE:-norenorm} > $D/data/kl125/screen_topk${K:-8}_${MODE:-norenorm}.txt 2>&1 &
P=$!
i=0; why=""
while kill -0 $P 2>/dev/null; do
  sleep 5; i=$((i + 5))
  if [ $i -ge 900 ]; then why="time cap"; break; fi
  if [ -f $HOME/.local/share/quiet/HOLD ]; then why="HOLD"; break; fi
  if [ $(( $(swapmb) - S0 )) -gt 2048 ]; then why="swap +2 GB"; break; fi
done
if [ -n "$why" ] && kill -0 $P 2>/dev/null; then
  kill -- -$P 2>/dev/null; sleep 3; kill -9 -- -$P 2>/dev/null
  echo "$(date +%H:%M:%S) run stopped: $why" >> $LOG
fi
wait $P 2>/dev/null
echo "$(date +%H:%M:%S) result: $(grep -E '^(loaded|topk|target|ar4|plan4433333333|Traceback|[A-Za-z]*Error)' $D/data/kl125/screen_topk${K:-8}_${MODE:-norenorm}.txt | cut -c1-200 | tr '\n' ' ')" >> $LOG
echo "$(date +%H:%M:%S) runner finished" >> $LOG
