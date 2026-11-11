#!/bin/bash
# NewIdeas, Wed 2026-11-04 03:3x. One share lock, CPU only (<= 4 cores, MEM < 10 GB):
#   1. QOM mask sets for Breakthrough's after_bigmoe.sh runner: qom_0.5_live_t3, qom_0_live_t3, qom_0.5_live_c164
#      (bit-checked and marked READY afterwards, by hand: scripts/qom_t3_check.py).
#   2. BYTES10 ledger replay (scripts/ten_x_ledger.py A B C, 3 procs): the cross-check of Fable C's BYTES10.md.
# Stops both if swap grows > 2 GB; pauses them (SIGSTOP) while another session holds or waits for a quiet window.
# Run ATTACHED (Bash run_in_background) from ~/NewIdeas.
cd ~/NewIdeas || exit 1
PY=~/BigMoE/.venv/bin/python
mkdir -p data/ledger
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
quiet_others() { peer ls 2>/dev/null | grep -v "^You:" | grep -qE "quiet lock until|benchmarking: quiet lock|waiting for a quiet window"; }
tree() { local p=$1 c; echo $p; for c in $(pgrep -P $p); do tree $c; done; }
sig() { local s=$1; shift; for p in "$@"; do kill -0 $p 2>/dev/null && kill -$s $(tree $p) 2>/dev/null; done; }
quiet_others && { echo "$(date +%H:%M:%S) another session's quiet window: not starting"; exit 2; }
peer lock share 60m --cpu 400 --gpu 5 "NewIdeas: QOM t3 masks + BYTES10 ledger replay (CPU only, 4 cores, MEM < 10 GB)" || exit 1
S0=$(swap_mb); echo "$(date +%H:%M:%S) start; swap ${S0} MB"
taskpolicy -b $PY -I scripts/dm_oea_masks.py ~/NewIdeas/data/dm_masks qom_0.5_live_t3 qom_0_live_t3 qom_0.5_live_c164 \
  > data/dm_masks/build_t3_c164.log 2>&1 & M=$!
TXL_PROCS=3 taskpolicy -b $PY -I scripts/ten_x_ledger.py A B C > data/ledger/run.log 2>&1 & L=$!
paused=0
while kill -0 $M 2>/dev/null || kill -0 $L 2>/dev/null; do
  sleep 15
  if [ $(( $(swap_mb) - S0 )) -gt 2048 ]; then
    echo "$(date +%H:%M:%S) swap grew > 2 GB ($(swap_mb) MB): stopping both"; sig TERM $M $L; break
  fi
  if quiet_others; then
    [ $paused -eq 0 ] && { echo "$(date +%H:%M:%S) quiet window (others): pausing"; sig STOP $M $L; paused=1; }
  elif [ $paused -eq 1 ]; then
    echo "$(date +%H:%M:%S) quiet window gone: resuming"; sig CONT $M $L; paused=0
  fi
done
wait $M; echo "$(date +%H:%M:%S) masks exit $?"; wait $L; echo "$(date +%H:%M:%S) ledger exit $?"
peer unlock
echo "$(date +%H:%M:%S) done; swap $(swap_mb) MB"
