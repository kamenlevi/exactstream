#!/bin/bash
# Decision map night 2, STAGE 2a (proposed by Breakthrough Mon 07:2x after stage 1; NewIdeas orders the queue).
# Run ATTACHED to the session that holds the quiet lock (run_in_background; see run_rest.sh). Takes no lock; unlocks at
# the end. One ~30-min ticket, ~3.8 min per run: the bounded chain (dm_common.bound_runs: every (S+1)-th row of a sure run
# refreshes, so no reuse is more than S rows old) with S=1 on 12-35 and 8-35, S=2 on 8-35, S=1 on 8-35 stacked with top-2
# (Q1), and the band's low edge (4-7, 0-3 at lag 1). The CPU unit test runs first.
cd ~/FableGod/decision_map || exit 1
mkdir -p out
LOG=out/runner.log
PY=~/BigMoE/.venv/bin/python
NAMES="gate moe_reuse_chain1_L12_35 moe_reuse_chain1_L08_35 moe_reuse_chain2_L08_35 moe_reuse_chain1_L08_35_topk2 moe_reuse_prev_L04_07 moe_reuse_prev_L00_03"
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
log() { echo "[dm-s2a] $* $(date '+%a %H:%M:%S'); swap $(swap_mb) MB; free disk $(df -g / | tail -1 | awk '{print $4}') GB" >> "$LOG"; }
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
for n in moe_reuse_chain1_L12_35 moe_reuse_chain1_L08_35 moe_reuse_chain2_L08_35 moe_reuse_chain1_L08_35_topk2; do
  [ -e out/$n.json ] && $PY stage1_extra.py $n >> "$LOG" 2>&1
done
peer unlock >> "$LOG" 2>&1
log "run_map done (rc $RC); unlocked"
grep -v "moe-sorted-gather" "$LOG" | grep -E "dm-s2a|done in|FAILED|WARNING|GATE|^\| moe_reuse_(prev_L|chain|lag)|^\| (code|ifeval|multi) " | tail -45
