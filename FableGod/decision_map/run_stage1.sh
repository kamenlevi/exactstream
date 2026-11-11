#!/bin/bash
# Decision map night 2, STAGE 1 (NewIdeas Mon 2026-11-02 06:4x; Breakthrough). Run it ATTACHED to the session that holds
# the quiet lock (Bash run_in_background, no nohup / disown; see run_rest.sh). Takes no lock itself; unlocks at the end.
# One ~25-min ticket: (1) MoE reuse on layers 12-35 at once (lag 1); the orchestrator's chained variant (sure rows take
# the last computed row's output, the 125B's own class as the oracle); (2) lag 2 and lag 4 on 12-35; (3) the band's
# edges, 4-layer groups 8-11 and 36-39. ~3.8 min per run (night 1's MoE reuse runs: 215-233 s). The CPU unit test of the
# new reuse indexing runs first (seconds); a failure stops before the model loads. Swap guard 2 GB.
cd ~/FableGod/decision_map || exit 1
mkdir -p out
LOG=out/runner.log
PY=~/BigMoE/.venv/bin/python
NAMES="gate moe_reuse_prev_L12_35 moe_reuse_chain_L12_35 moe_reuse_lag2_L12_35 moe_reuse_lag4_L12_35 moe_reuse_prev_L08_11 moe_reuse_prev_L36_39"
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
log() { echo "[dm-s1] $* $(date '+%a %H:%M:%S'); swap $(swap_mb) MB; free disk $(df -g / | tail -1 | awk '{print $4}') GB" >> "$LOG"; }
log "start (pid $$, attached; lock held by the session): $NAMES"
$PY test_reuse.py > out/test_reuse.txt 2>&1
grep -v "moe-sorted-gather" out/test_reuse.txt >> "$LOG"
if ! tail -1 out/test_reuse.txt | grep -q "ALL OK"; then
  peer unlock >> "$LOG" 2>&1; log "unit test FAILED (out/test_reuse.txt): nothing ran; unlocked"; exit 4
fi
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
grep -v "moe-sorted-gather" "$LOG" | grep -E "dm-s1|done in|FAILED|WARNING|GATE|^\| moe_reuse_(prev_L|chain|lag)" | tail -30
