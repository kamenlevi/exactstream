#!/bin/bash
# Decision map, the rest of night 1 after the 04:10 stop (Breakthrough, Mon 2026-11-02). Run it ATTACHED to the session
# that holds the quiet lock (Bash run_in_background, no nohup / disown): a detached runner's lock lapses after 8 min
# because the lock sees no process of its session (what happened at 04:07). Takes no lock itself; unlocks at the end.
# Resumes with --skip-done (baseline, topk10, topk2 are done; the gate re-checks baseline.json). Swap guard 2 GB.
cd ~/FableGod/decision_map || exit 1
mkdir -p out
LOG=out/runner.log
PY=~/BigMoE/.venv/bin/python
NAMES="baseline gate topk10 topk2 topk2r hc_static topk5 shared_zero att_reuse_prev gdn_reuse_prev moe_reuse_prev_g0 moe_reuse_prev_g1 moe_reuse_prev_g2 moe_reuse_prev_g3 hc_q4 gdn_q4 att_q4 hc_mean_raw"
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
log() { echo "[dm-rest] $* $(date '+%a %H:%M:%S'); swap $(swap_mb) MB; free disk $(df -g / | tail -1 | awk '{print $4}') GB" >> "$LOG"; }
log "start (pid $$, attached; lock held by the session)"
S0=$(swap_mb)
$PY run_map.py $NAMES --skip-done >> "$LOG" 2>&1 &
P=$!
while kill -0 $P 2>/dev/null; do
  sleep 20
  if [ $(( $(swap_mb) - S0 )) -gt 2048 ]; then log "swap grew > 2 GB: stopping pid $P"; kill $P; sleep 5; kill -9 $P 2>/dev/null; fi
done
wait $P; RC=$?
peer unlock >> "$LOG" 2>&1
log "run_map done (rc $RC); unlocked"
grep -v "moe-sorted-gather" "$LOG" | tail -24
