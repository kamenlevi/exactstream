#!/bin/sh
# Breakthrough D2 follow-up (re-run of day_d2c.sh, which got no lock at 13:58): white-noise arms at the measured NMSE of
# the trellis arms (L=8: k=3 0.0262, k=4 0.0081), all routed experts of the 8-bit 35B, so trellis-vs-noise KLD is compared
# at equal error directly instead of on the interpolated curve.
OUT=$HOME/Breakthrough/data/noise35; LOG=$HOME/Breakthrough/data/d2/summary.txt
PY=$HOME/BigMoE/.venv/bin/python; S=$HOME/Breakthrough/scripts
export PYTHONDONTWRITEBYTECODE=1 PEER_QUEUE_TIMEOUT=3600
peer lock quiet --queue 30m "Breakthrough: D2 noise arms at matched NMSE (~4 min)" > /dev/null 2>&1 || { echo "$(date +%H:%M) no lock (d2d)" >> "$LOG"; exit 1; }
echo "$(date +%H:%M) start experts noise arms n0.026 n0.0081 (d2d)" >> "$LOG"
$PY $S/noise35.py experts n0.026 n0.0081 >> "$OUT/experts_noise_matched.txt" 2>&1
peer unlock > /dev/null 2>&1
echo "$(date +%H:%M) done d2d: $(grep -E '^experts' "$OUT/experts_noise_matched.txt" | tail -2 | tr '\n' ' ' | cut -c1-400)" >> "$LOG"
