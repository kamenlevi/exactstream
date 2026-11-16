#!/bin/sh
# WEEK98 BET 7 (a') inside a quiet ticket (NewIdeas 14:4x): fl_touch.py (stock / fused2 / pre / cap 4 MB / late, 30
# rounds) under the memory guard (5 GB, swap +1 GB); unlocks the moment it ends.
cd "$HOME/MoonShots/week98/bet7" || exit 1
PY="$HOME/BigMoE/.venv/bin/python"
gpu() { ioreg -r -d 1 -w 0 -c IOAccelerator 2>/dev/null | sed -nE 's/.*"Device Utilization %"=([0-9]+).*/\1/p' | head -1; }
echo "[quiet] granted $(date '+%H:%M:%S'); GPU busy $(gpu) $(sleep 1; gpu) %" > fl_touch_quiet.log
MAXGB=5 SWAPGB=1 LOG=fl_touch_quiet.log sh ../jcc_guard.sh "$PY" -I -B fl_touch.py --rounds 30 --cap 4 --out "$HOME/MoonShots/week98/bet7/fl_touch_quiet.json"
echo "[quiet] fl_touch rc $? at $(date '+%H:%M:%S')" >> fl_touch_quiet.log
peer unlock >> fl_touch_quiet.log 2>&1
echo "[quiet] unlocked $(date '+%H:%M:%S')" >> fl_touch_quiet.log
