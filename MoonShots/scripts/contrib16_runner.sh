#!/bin/sh
# M-09 CONTRIB16 runner (MoonShots, Fri 2026-10-30): waits for a moment with no quiet lock, takes a SHARED lock
# (Kamen 18:45: proof-of-concept screens share the GPU), gates on PAUSE and free memory (expected peak ~4 GB + 8 GB spare),
# runs scripts/contrib16.py with the swap guard (+2 GB -> kill the process group), then unlocks.
cd "$HOME/MoonShots" || exit 1
LOG=data/contrib16_runner.log
PY="$HOME/BigMoE/.venv/bin/python"
echo "start $(date)" >> "$LOG"
tries=0
while :; do
  tries=$((tries + 1)); [ $tries -gt 12 ] && { echo "gave up waiting $(date)" >> "$LOG"; exit 1; }
  peer wait --timeout 60m >> "$LOG" 2>&1
  while [ -f "$HOME/.local/share/quiet/PAUSE" ]; do sleep 30; done
  free_pct=$(memory_pressure | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
  free_gb=$((free_pct * 64 / 100))
  if [ "$free_gb" -lt 12 ]; then echo "memory gate: ${free_gb} GB free < 12, retry in 5 min $(date)" >> "$LOG"; sleep 300; continue; fi
  if peer lock share 12m --cpu 150 --gpu 40 "MoonShots M-09 CONTRIB16 proof-of-concept screen (rough, ~3 min)" >> "$LOG" 2>&1; then
    break
  fi
  echo "share lock refused (rc $?), retry $(date)" >> "$LOG"; sleep 60
done
swap0=$(sysctl -n vm.swapusage | sed 's/.*used = \([0-9.]*\)M.*/\1/' | cut -d. -f1)
echo "running: free ${free_gb} GB, swap ${swap0} MB $(date)" >> "$LOG"
taskpolicy -b "$PY" scripts/contrib16.py >> "$LOG" 2>&1 &
P=$!
while kill -0 $P 2>/dev/null; do
  sw=$(sysctl -n vm.swapusage | sed 's/.*used = \([0-9.]*\)M.*/\1/' | cut -d. -f1)
  if [ $((sw - swap0)) -gt 2048 ]; then echo "SWAP GUARD: ${swap0} -> ${sw} MB, killing $(date)" >> "$LOG"; kill $P; sleep 2; kill -9 $P 2>/dev/null; break; fi
  sleep 5
done
wait $P; rc=$?
echo "done rc $rc $(date)" >> "$LOG"
peer unlock >> "$LOG" 2>&1
