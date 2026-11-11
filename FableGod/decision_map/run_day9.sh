#!/bin/bash
# DAY9 (Kamen Mon 2026-11-02 ~10:3x via the orchestrator: one decisive test per ROUTE_350 condition today) - Breakthrough's
# conditions 1-3 on the decision map, under a SHARE lock (flip rates, not timings). Run it ATTACHED to the session that
# holds the lock (Bash run_in_background, no nohup / disown). Takes no lock itself; unlocks at the end.
#   capture           the baseline again + rank-sorted top-10 routing per row and layer (out/routing/, MoonShots' bytes
#                     replay); its flips vs baseline must be 0 (a self-check that the capture changes nothing)
#   gate_top2_cf099   Q1: top-2 at the model's own weights on rows whose DFlash2 cf >= 0.99 (the realizable gate), the
#                     rest exact; gate_top2_cf0999 the same at cf >= 0.999
#   gate_bands        Q2 (NEW-1): cf >= 0.999 top-2, 0.99-0.999 top-4, 0.9-0.99 top-8, else exact
#   hc8w / hc8w_mix   Q3: the HC mixers + PLE key / value (hc8w) or the mixers only (hc8w_mix) at 8-bit affine, group 64
# ~4 min per run. The CPU unit test runs first; a failure stops before the model loads. Swap guard 2 GB.
cd ~/FableGod/decision_map || exit 1
mkdir -p out
LOG=out/runner.log
PY=~/BigMoE/.venv/bin/python
NAMES=${DAY9_NAMES:-"gate capture gate_top2_cf099 gate_top2_cf0999 gate_bands hc8w hc8w_mix"}
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
log() { echo "[dm-d9] $* $(date '+%a %H:%M:%S'); swap $(swap_mb) MB; free disk $(df -g / | tail -1 | awk '{print $4}') GB" >> "$LOG"; }
log "start (pid $$, attached; share lock held by the session): $NAMES"
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
grep -v "moe-sorted-gather" "$LOG" | grep -E "dm-d9|done in|FAILED|WARNING|GATE|hc8w:" | tail -30
