#!/bin/sh
# WEEK98 PILLAR 3 M0-M2 timing inside NewIdeas' quiet slot (Mon 16 Nov 2026, after S10 PASS2): starts when the queued
# quiet lock is granted; runs only if Kamen is away (away_secs >= 900), else defers. Two processes back to back under
# the memory guard (4 GB), then unlocks at once.
cd "$HOME/MoonShots/week98" || exit 1
LOG=p3_timing.log
gpu() { ioreg -r -d 1 -w 0 -c IOAccelerator 2>/dev/null | sed -nE 's/.*"Device Utilization %"=([0-9]+).*/\1/p' | head -1; }
echo "[timing] lock granted $(date '+%H:%M:%S')" >> $LOG
AW=$("$HOME/.local/share/quiet/away_secs.sh" 2>/dev/null); AW=${AW:-0}
echo "[timing] away_secs $AW; GPU busy $(gpu) $(sleep 1; gpu) $(sleep 1; gpu) %" >> $LOG
if [ "$AW" -lt 900 ]; then
> (a quote from the project owner, kept private)
  peer unlock >> $LOG 2>&1
  exit 3
fi
for st in stock fused; do
  MAXGB=4 SWAPGB=1 LOG=p3_m012_$st.log sh jcc_guard.sh "$HOME/BigMoE/.venv/bin/python" -I -B p3_m012.py --set $st
  echo "[timing] $st rc $? at $(date '+%H:%M:%S')" >> $LOG
done
peer unlock >> $LOG 2>&1
echo "[timing] unlocked $(date '+%H:%M:%S')" >> $LOG
