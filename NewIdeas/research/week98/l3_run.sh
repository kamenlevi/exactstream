#!/bin/bash
# L3 microbench runner (NewIdeas, Sun 15 Nov 17:3x): wait for MoonShots' TREE-E2E shared run to end (poll every 2 min),
# then a quiet ticket (queue <= 30 min), check memory and swap, run l3_slc_bench.py (full), unlock. Give up if not started
# by 21:45 (Leap's a45 queues for 23:00; we must be done by 22:30).
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd ~/NewIdeas/research/week98
LOG=l3_run.log; say() { echo "[l3 $(date '+%H:%M:%S')] $*" | tee -a $LOG; }
END=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-15 21:45:00" +%s)
say "armed: waiting for MoonShots TREE-E2E to end"
while peer ls 2>/dev/null | grep -q "MoonShots TREE-E2E"; do
  [ $(date +%s) -ge $END ] && { say "MoonShots still running at 21:45: SKIP (not run)"; exit 3; }
  sleep 120
done
say "MoonShots done; queueing a quiet ticket"
peer lock quiet --queue 30m "NewIdeas: L3 HC SLC staging microbench (~3 min, MEM 2 GB; pre-registered cceee56)" >> $LOG 2>&1 || { say "no quiet ticket: SKIP"; exit 4; }
[ $(date +%s) -ge $END ] && { peer unlock >> $LOG 2>&1; say "granted after 21:45: SKIP"; exit 5; }
sleep 30
ns=$(vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}')
say "NO-SWAP ${ns} GB, swap $(sysctl -n vm.swapusage | awk '{print $6}')"
if [ "$ns" -lt 6 ]; then peer unlock >> $LOG 2>&1; say "NO-SWAP < 6 GB: SKIP"; exit 6; fi
~/BigMoE/.venv/bin/python -I -B l3_slc_bench.py > l3_slc_bench.txt 2>&1; rc=$?
peer unlock >> $LOG 2>&1
say "bench rc $rc; lock released"
cat l3_slc_bench.txt | grep -E "^STEP|MLX peak" | tee -a $LOG
