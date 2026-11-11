#!/bin/sh
# Generic shared-GPU runner (MoonShots, Fri 2026-10-30; from the M-09 runner): usage share_runner.sh scripts/<x>.py
# Waits for a moment with no quiet lock, takes a SHARED lock (Kamen 18:45: proof-of-concept screens share the GPU),
# gates on PAUSE and free memory (>= 12 GB free: a few-GB peak + 8 GB spare), runs the script with the swap guard
# (+2 GB -> kill), then unlocks. Log: data/<x>_runner.log
cd "$HOME/MoonShots" || exit 1
S="$1"
LOG="data/$(basename "$S" .py)_runner.log"
PY="$HOME/BigMoE/.venv/bin/python"
echo "start $(date)" >> "$LOG"
tries=0
while :; do
  tries=$((tries + 1)); [ $tries -gt 12 ] && { echo "gave up waiting $(date)" >> "$LOG"; exit 1; }
  peer wait --timeout 60m >> "$LOG" 2>&1
  while [ -f "$HOME/.local/share/quiet/PAUSE" ]; do sleep 30; done
  # available = free + inactive + speculative pages (vm_stat); memory_pressure's free % misses wired Metal memory held
  # by concurrent MLX jobs (Leap / NewIdeas 21:1x)
  free_gb=$(vm_stat | awk '/page size of/ {ps=$8} /Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} END {gsub("\\.","",f); gsub("\\.","",i); gsub("\\.","",s); print int((f+i+s)*ps/1073741824)}')
  if [ "$free_gb" -lt 12 ]; then echo "memory gate: ${free_gb} GB free < 12, retry in 5 min $(date)" >> "$LOG"; sleep 300; continue; fi
  if peer lock share 12m --cpu 150 --gpu 40 "MoonShots $(basename "$S" .py) proof-of-concept screen (rough)" >> "$LOG" 2>&1; then
    break
  fi
  echo "share lock refused, retry $(date)" >> "$LOG"; sleep 60
done
swap0=$(sysctl -n vm.swapusage | sed 's/.*used = \([0-9.]*\)M.*/\1/' | cut -d. -f1)
echo "running $S: free ${free_gb} GB, swap ${swap0} MB $(date)" >> "$LOG"
taskpolicy -b "$PY" "$S" >> "$LOG" 2>&1 &
P=$!
while kill -0 $P 2>/dev/null; do
  sw=$(sysctl -n vm.swapusage | sed 's/.*used = \([0-9.]*\)M.*/\1/' | cut -d. -f1)
  if [ $((sw - swap0)) -gt 2048 ]; then echo "SWAP GUARD: ${swap0} -> ${sw} MB, killing $(date)" >> "$LOG"; kill $P; sleep 2; kill -9 $P 2>/dev/null; break; fi
  sleep 5
done
wait $P; rc=$?
echo "done rc $rc $(date)" >> "$LOG"
peer unlock >> "$LOG" 2>&1
