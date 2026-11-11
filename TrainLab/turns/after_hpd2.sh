#!/bin/sh
# Replaces after_hpd.sh (MTP Head, Mon 11-02 10:0x): HPD-1 has finished (data/HPD_DONE). D-23e (REPORT "D-23e" + amendment 1:
# gate 47 GB) in one quiet 40-min FIFO ticket once NO-SWAP reads >= 47 GB (checked every 2 min), then D-23x (NewIdeas'
# first-check diagnostic: read trace by pool and layer; not a timing result) in one quiet 25-min ticket, each queued only
# after the previous release. Queue deadline 18:00 (Kamen: no day hold Monday).
cd $HOME/TrainLab
LOG=logs/turns.log
say() { echo "$(date '+%m-%d %H:%M:%S') after_hpd2: $*" >> $LOG; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
STOPAT=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-02 18:00" +%s)
waitmem() {
  while [ "$(noswap)" -lt $1 ]; do
    [ $(date +%s) -ge $STOPAT ] && { say "18:00 without $1 GB NO-SWAP: $2 left for tonight"; return 1; }
    sleep 120
  done
}
waitmem 47 D-23e || exit 0
LEFT=$(( STOPAT - $(date +%s) ))
say "queueing D-23e ($(noswap) GB NO-SWAP; 40 min; queue timeout ${LEFT} s = 18:00)"
if PEER_QUEUE_TIMEOUT=$LEFT peer lock quiet --queue 40m "MTP Head D-23e: the chat switch at C 224 (timing, ~43 GB), ~25 min" >> $LOG 2>&1; then
  ./run_locked.sh 40 turns/turn74.sh
  say "D-23e ticket done: $(grep 'PRIMARY chat' data/d23d_read.txt 2>/dev/null | tail -1 | cut -c1-200)"
else
  say "D-23e not granted by 18:00"; exit 0
fi
waitmem 44 D-23x || exit 0
LEFT=$(( STOPAT - $(date +%s) ))
say "queueing D-23x ($(noswap) GB NO-SWAP; 25 min; queue timeout ${LEFT} s = 18:00)"
if PEER_QUEUE_TIMEOUT=$LEFT peer lock quiet --queue 25m "MTP Head D-23x: first-check read trace (C 192, ~38 GB), ~15 min" >> $LOG 2>&1; then
  ./run_locked.sh 25 turns/turn75.sh
  say "D-23x ticket done: $(grep 'agent base reads per layer, check 1' data/d23x_read.txt 2>/dev/null | tail -1 | cut -c1-200)"
else
  say "D-23x not granted by 18:00"
fi
