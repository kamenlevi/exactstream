#!/bin/bash
# run_c55.sh (Breakthrough, CHECK55 step 0): the profile on the 12-layer slice under a QUIET ticket.
# Gate: available (free + purgeable + file-backed) >= 31 GB (peak ~23 + 8) after the grant and a 20 s settle; swap guard +2 GB
# (kill); time cap 15 min; the ticket released at the end. Usage: run_c55.sh <tag>
set -u
cd ~/Breakthrough
TAG=${1:-c55_$(date +%H%M)}
LOG=data/m1/check55/$TAG.log
NEED=31
avail() { vm_stat | awk '/page size of/ {ps=$8} /Pages free/ {f=$3} /Pages purgeable/ {p=$3} /File-backed pages/ {fb=$3} END {gsub("[.]","",f); gsub("[.]","",p); gsub("[.]","",fb); printf "%.1f", (f+p+fb)*ps/1e9}'; }
swapmb() { sysctl -n vm.swapusage | awk '{print $6}' | tr -d 'M'; }
echo "$(date '+%a %H:%M:%S') gate pre-lock: avail $(avail) GB, swap $(swapmb) MB" | tee -a "$LOG"
peer lock quiet --queue 40m "Breakthrough CHECK55 exact-check profile (12-layer slice ~23 GB, ~5 min)" 2>&1 | tail -2 | tee -a "$LOG"
sleep 20
a=$(avail)
echo "$(date '+%a %H:%M:%S') gate after grant + settle: avail $a GB, swap $(swapmb) MB; $(pmset -g | grep -i powermode | head -1)" | tee -a "$LOG"
if [ "$(echo "$a < $NEED" | bc)" = 1 ]; then echo "SKIP: avail $a < $NEED GB" | tee -a "$LOG"; peer unlock; exit 3; fi
s0=$(swapmb)
C55_TAG=$TAG ~/BigMoE/.venv/bin/python -I -B scripts/m1/check55/c55_profile.py >> "$LOG" 2>&1 &
pid=$!
end=$(( $(date +%s) + 900 ))
while kill -0 $pid 2>/dev/null; do
  s=$(swapmb); d=$(echo "$s - $s0" | bc)
  if [ "$(echo "$d > 2048" | bc)" = 1 ]; then echo "SWAP GUARD: +$d MB, killing $pid" | tee -a "$LOG"; kill $pid; sleep 2; kill -9 $pid 2>/dev/null; break; fi
  if [ "$(date +%s)" -gt "$end" ]; then echo "TIME CAP: killing $pid" | tee -a "$LOG"; kill $pid; sleep 2; kill -9 $pid 2>/dev/null; break; fi
  sleep 5
done
wait $pid; rc=$?
echo "$(date '+%a %H:%M:%S') exit $rc; swap $(swapmb) MB (start $s0)" | tee -a "$LOG"
peer unlock 2>&1 | tail -1 | tee -a "$LOG"
