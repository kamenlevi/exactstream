#!/bin/sh
# MoonShots, one quiet lock (launch holding it): W24-sp step 2 screen re-run (trace-length fix) + R-16 bytes/token replay (CPU).
D=$HOME/MoonShots; PY=$HOME/BigMoE/.venv/bin/python; LOG=$D/data/w24sp_runner.log
sh $D/scripts/keepalive_while.sh $$ & GL=$!
trap 'peer unlock > /dev/null 2>&1; if [ -n "$GL" ]; then kill $GL 2>/dev/null; fi' EXIT
echo "$(date +%H:%M:%S) runner2 started" >> $LOG
$PY $D/w24sp/screen.py full > $D/data/w24sp_screen.log 2>&1
echo "$(date +%H:%M:%S) W24-sp: $(grep -E '^(VERDICT|L[0-9]+: bitwise|Traceback|[A-Za-z]*Error)' $D/data/w24sp_screen.log | cut -c1-220 | tr '\n' ' ')" >> $LOG
$PY $D/scripts/w24ec_replay.py > $D/data/w24ec_replay.txt 2>&1
echo "$(date +%H:%M:%S) W24-ec replay done" >> $LOG
for ch in 16384 32768 65536; do
  W24EC_CH=$ch W24EC_LAYERS=24 $PY $D/scripts/w24ec.py 32 > $D/data/w24ec_ch$ch.log 2>&1
  echo "$(date +%H:%M:%S) W24-ec chunk $ch: $(grep -E '^(VERDICT|  latency|Traceback|[A-Za-z]*Error)' $D/data/w24ec_ch$ch.log | cut -c1-160 | tr '\n' ' ')" >> $LOG
done
echo "$(date +%H:%M:%S) runner2 finished" >> $LOG
