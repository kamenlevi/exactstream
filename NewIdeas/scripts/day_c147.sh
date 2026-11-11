#!/bin/bash
# NewIdeas, Wed 2026-11-04 07:4x. Screen 4's masks (the 24 GB pool, QUEUE 07:4x) + the top-1 byte prices, CPU only
# (<= 4 cores, MEM < 3 GB), under one share lock taken at a gap between other sessions' quiet windows:
#   1. masks qom_0_k1_live_bq_c147(_t3) and qom_0_live_bq_c147(_t3) (two processes), bit-checked t3 vs 4-bit twin
#      (scripts/qom_t3_check.py), READY marker on a pass;
#   2. scripts/ledger_top1.py (2 procs).
# Waits while another session holds or waits for a quiet window (gives up at 12:30); pauses (SIGSTOP) during one;
# stops everything if swap grows > 2 GB. Run ATTACHED (Bash run_in_background) from ~/NewIdeas.
cd ~/NewIdeas || exit 1
PY=~/BigMoE/.venv/bin/python
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
quiet_others() { peer ls 2>/dev/null | grep -v "^You:" | grep -qE "quiet lock until|benchmarking: quiet lock|waiting for a quiet window"; }
tree() { local p=$1 c; echo $p; for c in $(pgrep -P $p); do tree $c; done; }
sig() { local s=$1; shift; for p in "$@"; do kill -0 $p 2>/dev/null && kill -$s $(tree $p) 2>/dev/null; done; }
until ! quiet_others; do
  [ "$(date +%H%M)" -ge 1230 ] && { echo "$(date +%H:%M:%S) no gap before 12:30: not starting"; exit 2; }
  sleep 60
done
peer lock share 45m --cpu 400 --gpu 0 "NewIdeas: screen 4 masks (24 GB pool) + top-1 byte replay (CPU only, 4 cores, MEM < 3 GB)" || exit 1
S0=$(swap_mb); echo "$(date +%H:%M:%S) start; swap ${S0} MB"
taskpolicy -b $PY -I scripts/dm_oea_masks.py ~/NewIdeas/data/dm_masks qom_0_k1_live_bq_c147 qom_0_k1_live_bq_c147_t3 \
  > data/dm_masks/build_c147_B.log 2>&1 & MB=$!
taskpolicy -b $PY -I scripts/dm_oea_masks.py ~/NewIdeas/data/dm_masks qom_0_live_bq_c147 qom_0_live_bq_c147_t3 \
  > data/dm_masks/build_c147_C.log 2>&1 & MC=$!
TXL_PROCS=2 taskpolicy -b $PY -I scripts/ledger_top1.py > data/ledger/ledger_top1.log 2>&1 & L=$!
paused=0
while kill -0 $MB 2>/dev/null || kill -0 $MC 2>/dev/null || kill -0 $L 2>/dev/null; do
  sleep 15
  if [ $(( $(swap_mb) - S0 )) -gt 2048 ]; then
    echo "$(date +%H:%M:%S) swap grew > 2 GB ($(swap_mb) MB): stopping"; sig CONT $MB $MC $L; sig TERM $MB $MC $L; break
  fi
  if quiet_others; then
    [ $paused -eq 0 ] && { echo "$(date +%H:%M:%S) quiet window (others): pausing"; sig STOP $MB $MC $L; paused=1; }
  elif [ $paused -eq 1 ]; then
    echo "$(date +%H:%M:%S) quiet window gone: resuming"; sig CONT $MB $MC $L; paused=0
  fi
done
wait $MB; eb=$?; wait $MC; ec=$?; wait $L; el=$?
echo "$(date +%H:%M:%S) masks B exit $eb, masks C exit $ec, ledger exit $el"
if [ $eb -eq 0 ] && [ $ec -eq 0 ]; then
  if $PY -I scripts/qom_t3_check.py data/dm_masks qom_0_k1_live_bq_c147_t3 qom_0_live_bq_c147_t3 > data/dm_masks/check_c147.log 2>&1; then
    touch data/dm_masks/qom_0_k1_live_bq_c147_t3/READY data/dm_masks/qom_0_live_bq_c147_t3/READY; echo "bit-check ALL OK: READY"
  else echo "bit-check FAILED: see data/dm_masks/check_c147.log"; fi
fi
peer unlock
echo "$(date +%H:%M:%S) done; swap $(swap_mb) MB"
