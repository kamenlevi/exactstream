#!/bin/bash
# Night chain (MTP Head, Sun 11-15 01:3x; sub2 is blocked ~01:45 -> ~06:00, so this runs with nobody watching):
#  1. waits for CARRY400 2a's RESULT file and posts its line to the orchestrator (peer say);
#  2. then runs CARRY400 2b (allrows alone, vs 2a's n1; scratch tag mhx, overwritten by later screens) and posts it;
#  3. waits for S4 DEPTHC's RESULT file (its stop line at step 380 is enforced inside depth_copy.sh) and posts it.
# Every verdict is also in data/sb98/*_RESULT.txt with its compare files beside it.
O=$HOME/TrainLab/data/sb98; L=$HOME/TrainLab/sb98/levers
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
say() { peer say --to "AI/LLM project recommendations" "MTP Head (auto): $(head -c 240 "$1")" 2>&1 | tail -1; }
log "night chain armed"
until [ -f $O/mhc2a_RESULT.txt ]; do sleep 60; done
log "2a: $(cat $O/mhc2a_RESULT.txt)"; say $O/mhc2a_RESULT.txt
if [ -f $HOME/Sandbox98/ckpt/mhc2a_n1.safetensors ]; then
  log "2b: allrows vs mhc2a_n1"
  TAG=mhx ARMS="a1:$L/allrows" CTL=mhc2a_n1 MEMGB=7 $HOME/TrainLab/sb98/carry400.sh > $O/mhx_runner.log 2>&1
  [ -f $O/mhx_RESULT.txt ] && { log "2b: $(cat $O/mhx_RESULT.txt)"; say $O/mhx_RESULT.txt; } || log "2b: no RESULT (see mhx_runner.log)"
else
  log "2b skipped: 2a's control checkpoint is missing"
fi
until [ -f $O/mhd1_RESULT.txt ]; do sleep 60; done
log "S4: $(cat $O/mhd1_RESULT.txt)"; say $O/mhd1_RESULT.txt
log "night chain done"
