#!/bin/sh
# MoonShots, one 10-min quiet lock (launch holding it): W24-sp step 2 screen (11-bit scale / bias 3c: convert 3 layers,
# bitwise S = 1 / 4 / 16, 16-row timing), R-17 T3-45 pricing (CPU, instant), then R-16 W24-ec (GPU rANS decode screen)
# only if <= 5.5 min have passed, so the lock stays within its 10 minutes.
D=$HOME/MoonShots; PY=$HOME/BigMoE/.venv/bin/python; LOG=$D/data/w24sp_runner.log
sh $D/scripts/keepalive_while.sh $$ & GL=$!
trap 'peer unlock > /dev/null 2>&1; if [ -n "$GL" ]; then kill $GL 2>/dev/null; fi' EXIT
T0=$(date +%s)
echo "$(date +%H:%M:%S) runner started" >> $LOG
$PY $D/w24sp/screen.py full > $D/data/w24sp_screen.log 2>&1
echo "$(date +%H:%M:%S) W24-sp: $(grep -E '^(VERDICT|L[0-9]+: bitwise|Traceback|[A-Za-z]*Error)' $D/data/w24sp_screen.log | cut -c1-220 | tr '\n' ' ')" >> $LOG
/usr/bin/python3 $D/scripts/t3_45.py > $D/data/t3_45.txt 2>&1
echo "$(date +%H:%M:%S) T3-45 priced" >> $LOG
if [ $(( $(date +%s) - T0 )) -lt 330 ]; then
  $PY $D/scripts/w24ec.py 32 > $D/data/w24ec.log 2>&1
  echo "$(date +%H:%M:%S) W24-ec: $(grep -E '^(VERDICT|Traceback|[A-Za-z]*Error)' $D/data/w24ec.log | cut -c1-220 | tr '\n' ' ')" >> $LOG
else
  echo "$(date +%H:%M:%S) W24-ec skipped (over 5.5 min)" >> $LOG
fi
echo "$(date +%H:%M:%S) runner finished" >> $LOG
