#!/bin/sh
# Run D-24 (turn63) under a shared lock when NO-SWAP >= 21 GB; never start after 22:00 (day4_chain queues its quiet lock at
# 23:00 and the session must hold one lock request at a time); up to 4 slots.
cd $HOME/TrainLab
LOG=logs/turns.log
say() { echo "$(date '+%m-%d %H:%M:%S') d24_wait: $*" >> $LOG; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
END=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-01 22:00" +%s)
n=0
while [ ! -s data/d24/code_b5.npz ] && [ $(date +%s) -lt $END ] && [ $n -lt 4 ]; do
  if [ "$(noswap)" -ge 21 ]; then
    n=$((n + 1)); say "D-24, shared slot $n ($(noswap) GB NO-SWAP)"
    N=$(wc -l < logs/turn_job.out)
    ./run_shared.sh 20 300 40 turns/turn63.sh "D-24: MTP head top-3 at chain misses (teacher-forced, held-out sets)"
    tail -n +$((N + 1)) logs/turn_job.out | grep -q "Traceback" && { say "D-24 crashed (see logs/turn_job.out)"; exit 1; }
    tail -n +$((N + 1)) logs/turn_job.out | grep -q "not starting" && sleep 600
  else
    sleep 120
  fi
done
say "done (D-24 saved: $([ -s data/d24/code_b5.npz ] && echo yes || echo no))"
