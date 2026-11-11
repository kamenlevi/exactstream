#!/bin/bash
# Decision map, night 1 (WIRED by Breakthrough, Mon 2026-11-02; Fable's original in run_night1.sh.orig).
# ONE process, one model load (D-02a's configuration: C 51, ~20 GB peak; see dm_common's WIRED note), one quiet FIFO
> (a quote from the project owner, kept private)
#   baseline -> gate (vs D-02a's stored top-1, >= 99.9 %, else stop) -> topk10 (wrapper identity: 0 flips expected) ->
#   topk2 / topk2r (Q1: top-2 experts, mass / renorm weights) -> Fable's night-1 list -> hc_mean_raw (Fable's literal HC).
# ctx2048 / ctx512 dropped: every answer here starts within 442 tokens, so the cut removes nothing (out/TRIED_NOTES.md).
# ~36 windows (~16K tokens) per run, ~1-3 min each; ~30-60 min in all. Guards: waits while HOLD exists; BigMoE passes
# wait while PAUSE exists (pausewait); stops the run if swap grows > 2 GB. --skip-done: a restart resumes.
# Log: out/runner.log. Usage: nohup bash run_night1.sh >/dev/null 2>&1 &
cd ~/FableGod/decision_map || exit 1
mkdir -p out
LOG=out/runner.log
Q=~/.local/share/quiet
PY=~/BigMoE/.venv/bin/python
NAMES="baseline gate topk10 topk2 topk2r hc_static topk5 shared_zero att_reuse_prev gdn_reuse_prev moe_reuse_prev_g0 moe_reuse_prev_g1 moe_reuse_prev_g2 moe_reuse_prev_g3 hc_q4 gdn_q4 att_q4 hc_mean_raw"
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
log() { echo "[dm-runner] $* $(date '+%a %H:%M:%S'); swap $(swap_mb) MB; free disk $(df -g / | tail -1 | awk '{print $4}') GB" >> "$LOG"; }
log "start (pid $$): $NAMES"
while true; do
  while [ -e $Q/HOLD ]; do sleep 60; done
  if PEER_QUEUE_TIMEOUT=43200 peer lock quiet --queue 75m "Breakthrough: Fable's decision map night 1 (Q1 top-2 first; D-02a config C 51, ~20 GB; ~30-60 min)" >> "$LOG" 2>&1; then
    log "lock taken"
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
    [ $RC -eq 3 ] && log "GATE FAILED: nothing after the baseline ran (out/gate.txt)"
    break
  fi
  log "no lock (queue timeout); re-queueing"
done
log "done"
