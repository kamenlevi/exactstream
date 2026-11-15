#!/bin/bash
# Night chain v2 (MTP Head, Sun 11-15 01:4x; sub2 is blocked ~01:45 -> ~06:00). Sequential, pause-don't-kill:
#  1. wait for the running CARRY400 2a (carry400.sh, guarded, try 3) to exit; if it left no RESULT, rerun 2a paused;
#  2. CARRY400 2b (allrows vs mhc2a_n1, scratch tag mhx), paused;
#  3. S4 DEPTHC again (its first run gave up after 3 yields at step 9; that run's read-out was stopped and is VOID),
#     paused, stop line at step 380 as pre-registered; the training must log its DONE line before the final read-out.
# Each RESULT line is posted to the orchestrator by peer say.
O=$HOME/TrainLab/data/sb98; L=$HOME/TrainLab/sb98/levers; S=$HOME/TrainLab/sb98
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
say() { peer say --to "AI/LLM project recommendations" "MTP Head (auto): $(head -c 240 "$1")" 2>&1 | tail -1; }
log "night chain v2 armed"
while pgrep -f "sb98/carry400.sh" >/dev/null; do sleep 30; done
if [ ! -f $O/mhc2a_RESULT.txt ]; then
  log "2a left no RESULT (yielded out): rerun paused"
  $S/carry400p.sh > $O/mhc2a_runner2.log 2>&1
fi
[ -f $O/mhc2a_RESULT.txt ] && { log "2a: $(cat $O/mhc2a_RESULT.txt)"; say $O/mhc2a_RESULT.txt; }
if [ -f $O/mhc2a_RESULT.txt ] && grep -q "DONE mhc2a" $O/mhc2a_train.log; then
  log "2b: allrows vs mhc2a_n1"
  TAG=mhx ARMS="a1:$L/allrows" CTL=mhc2a_n1 MEMGB=7 $S/carry400p.sh > $O/mhx_runner.log 2>&1
  [ -f $O/mhx_RESULT.txt ] && { log "2b: $(cat $O/mhx_RESULT.txt)"; say $O/mhx_RESULT.txt; } || log "2b: no RESULT (see mhx_runner.log)"
else
  log "2b skipped: 2a did not finish"
fi
log "S4 DEPTHC rerun (paused)"
rm -f $O/mhd1_RESULT.txt
$S/depth_copy_p.sh > $O/mhd1_runner2.log 2>&1
[ -f $O/mhd1_RESULT.txt ] && { log "S4: $(cat $O/mhd1_RESULT.txt)"; say $O/mhd1_RESULT.txt; } || log "S4: no RESULT (see mhd1_runner2.log)"
log "night chain v2 done"
