#!/bin/sh
# Breakthrough D2 follow-up: white-noise arms at the trellis arms' expected NMSE (L=8: k=3 ~0.026, k=4 ~0.0075), all
# routed experts of the 8-bit 35B, so trellis-vs-noise KLD can be compared at equal error. Waits for day_d2b.sh.
OUT=$HOME/Breakthrough/data/noise35; LOG=$HOME/Breakthrough/data/d2/summary.txt
PY=$HOME/BigMoE/.venv/bin/python; S=$HOME/Breakthrough/scripts
export PYTHONDONTWRITEBYTECODE=1 PEER_QUEUE_TIMEOUT=14400
R=$(cat $HOME/Breakthrough/data/d2/runner_d2b.pid 2>/dev/null)
while [ -n "$R" ] && kill -0 "$R" 2>/dev/null; do sleep 20; done
peer lock quiet --queue 60m "Breakthrough: D2 noise arms at matched NMSE (~3 min)" > /dev/null 2>&1 || { echo "$(date +%H:%M) no lock (d2c)" >> "$LOG"; exit 1; }
echo "$(date +%H:%M) start experts noise arms (d2c)" >> "$LOG"
$PY $S/noise35.py experts n0.026 n0.0075 >> "$OUT/experts_noise_matched.txt" 2>&1
peer unlock > /dev/null 2>&1
echo "$(date +%H:%M) done d2c: $(grep -E '^experts' "$OUT/experts_noise_matched.txt" | tail -2 | tr '\n' ' ' | cut -c1-400)" >> "$LOG"
