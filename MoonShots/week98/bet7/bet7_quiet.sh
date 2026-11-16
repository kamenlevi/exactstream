#!/bin/sh
# WEEK98 BET 7 deliverable (3) inside NewIdeas' quiet slot (Mon 16 Nov 13:3x): the fused level's timing (fl_time.py,
# stock vs fused, 2 screens x 2 cache lengths, 30 rounds) then the per-step profile (fl_prof.py), each under the memory
# guard (4 GB, swap +1 GB); unlocks the moment both end.
cd "$HOME/MoonShots/week98/bet7" || exit 1
PY="$HOME/BigMoE/.venv/bin/python"
gpu() { ioreg -r -d 1 -w 0 -c IOAccelerator 2>/dev/null | sed -nE 's/.*"Device Utilization %"=([0-9]+).*/\1/p' | head -1; }
echo "[quiet] granted $(date '+%H:%M:%S'); GPU busy $(gpu) $(sleep 1; gpu) %" > fl_time_quiet.log
MAXGB=4 SWAPGB=1 LOG=fl_time_quiet.log sh ../jcc_guard.sh "$PY" -I -B fl_time.py --rounds 30 --out "$HOME/MoonShots/week98/bet7/fl_time_quiet.json"
echo "[quiet] fl_time rc $? at $(date '+%H:%M:%S')" >> fl_time_quiet.log
MAXGB=4 SWAPGB=1 LOG=fl_prof_quiet.log sh ../jcc_guard.sh "$PY" -I -B fl_prof.py --n 30 --out "$HOME/MoonShots/week98/bet7/fl_prof_quiet.json"
echo "[quiet] fl_prof rc $? at $(date '+%H:%M:%S')" >> fl_time_quiet.log
peer unlock >> fl_time_quiet.log 2>&1
echo "[quiet] unlocked $(date '+%H:%M:%S')" >> fl_time_quiet.log
