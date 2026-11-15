#!/bin/bash
# S4 DEPTHC rerun, held for BigMoE's G32 (orchestrator 01:56: no new job > 5 GB from 02:20 until G32 has loaded).
# Starts at >= 03:30 AND NO-SWAP >= 20 GB (free + purgeable + file-backed); then the paused runner as pre-registered.
O=$HOME/TrainLab/data/sb98
ns() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
until [ "$(date +%H%M)" -ge 0330 ] && [ "$(ns)" -ge 20 ]; do sleep 60; done
echo "[$(date '+%a %H:%M:%S')] S4 start (NO-SWAP $(ns) GB)"
rm -f $O/mhd1_RESULT.txt
$HOME/TrainLab/sb98/depth_copy_p.sh > $O/mhd1_runner3.log 2>&1
[ -f $O/mhd1_RESULT.txt ] && peer say --to "AI/LLM project recommendations" "MTP Head (auto): $(head -c 240 $O/mhd1_RESULT.txt)" >/dev/null 2>&1
echo "[$(date '+%a %H:%M:%S')] S4 done: $(cat $O/mhd1_RESULT.txt 2>/dev/null)"
