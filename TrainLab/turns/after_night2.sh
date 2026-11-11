#!/bin/sh
# After after_night.sh (MTP Head, Mon 11-02 evening): DEPTH32 (Fable mission 23) in Tuesday's big window, one shared slot
# ending by 15:00 (latest start 13:30); its verdict to NewIdeas + the orchestrator (address looked up at send time).
cd $HOME/TrainLab
LOG=logs/turns.log
say() { echo "$(date '+%m-%d %H:%M:%S') after_night2: $*" >> $LOG; }
orch() { $HOME/.local/share/pacer/orch_addr.sh 2>/dev/null | sed 's|.*/||; s|\.sock$||'; }
fwd() { peer say --to ojox-bd "$1" > /dev/null 2>&1; O=$(orch); [ -n "$O" ] && echo "$1" | peer send "$O" - > /dev/null 2>&1; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
hm() { date +%H%M | sed 's/^0*//;s/^$/0/'; }
STOPAT=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-03 15:00" +%s)
say "start: waiting for after_night.sh, then DEPTH32 in Tuesday's big window"
while pgrep -f "turns/after_night.sh" > /dev/null; do sleep 60; done
while pgrep -f "turns/(et_job|dd_job|dv_job|kw_job|turn75)" > /dev/null; do sleep 20; done
t=$(hm)
if ! { [ "$(date +%d)" = "03" ] && [ "$t" -ge 30 ] && [ "$t" -lt 1330 ]; }; then say "outside the window / past 13:30: DEPTH32 not run"; exit 0; fi
while [ "$(noswap)" -lt 23 ]; do [ "$(hm)" -ge 1330 ] && { say "DEPTH32: no 23 GB by 13:30"; exit 0; }; sleep 120; done
m=$(( (STOPAT - $(date +%s)) / 60 )); [ $m -gt 90 ] && m=90
say "DEPTH32 ($(noswap) GB NO-SWAP; slot $m min)"
./run_shared.sh $m 300 60 turns/d23h_job.sh "DEPTH32: MTP chain hits to depth 31 + DFlash2 every-position walk (mission 23; ~15 GB, no timing)"
L=$(grep 'LINE' data/d23h_read.txt 2>/dev/null | tail -1 | sed 's/^ *//')
say "DEPTH32: ${L:-not finished}"; fwd "MTP Head (auto) DEPTH32 (Fable mission 23): ${L:-not finished}"
say "end"
