#!/bin/sh
# After HPD-1's shared lock (run_shared.sh, pid $1) ends (MTP Head, Mon 11-02 08:4x): D-23e (the chat switch at C 224, REPORT
# "D-23e ... pre-registration") in one quiet 40-min FIFO ticket, then HPD-1 again under a shared lock if it has not finished
# (its phases resume). D-23e needs NO-SWAP >= 48 GB at its gate: the ticket is queued only once NO-SWAP has read >= 50 GB
# (checked every 2 min), so a short day does not burn a ticket. Queue deadline 18:00 (Kamen: no day hold Monday); after that
# D-23e waits for tonight (queued by hand). One lock request at a time: each is taken only after the previous release.
cd $HOME/TrainLab
LOG=logs/turns.log
say() { echo "$(date '+%m-%d %H:%M:%S') after_hpd: $*" >> $LOG; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
STOPAT=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-02 18:00" +%s)
while kill -0 $1 2>/dev/null; do sleep 20; done
say "HPD-1's shared run ended; waiting for NO-SWAP >= 50 GB to queue D-23e"
while [ "$(noswap)" -lt 50 ]; do
  [ $(date +%s) -ge $STOPAT ] && { say "18:00 without 50 GB NO-SWAP: D-23e left for tonight"; exit 0; }
  sleep 120
done
LEFT=$(( STOPAT - $(date +%s) ))
say "queueing D-23e ($(noswap) GB NO-SWAP; 40 min; queue timeout ${LEFT} s = 18:00)"
if PEER_QUEUE_TIMEOUT=$LEFT peer lock quiet --queue 40m "MTP Head D-23e: the chat switch at C 224 (timing, ~44 GB), ~25 min" >> $LOG 2>&1; then
  ./run_locked.sh 40 turns/turn74.sh
  say "D-23e ticket done: $(grep 'PRIMARY chat' data/d23d_read.txt 2>/dev/null | tail -1 | cut -c1-200)"
else
  say "D-23e not granted by 18:00"
fi
if [ ! -e data/HPD_DONE ]; then
  say "HPD-1 not finished: resuming it under a shared lock (120 min)"
  HPD_MARGIN=8 ./run_shared.sh 120 300 60 turns/hpd1.sh "HPD-1: drafter training screen (~15-20 GB, no timing; resumes from checkpoints)"
  say "HPD-1 shared run 2 ended: $(grep 'hpd1:' $LOG | tail -1 | cut -c1-200)"
fi
