#!/bin/sh
# Leap DAY9 (Mon 2026-11-02 11:0x): C0 again at C 160 (at C 112 the base pool holds 103 experts, so a 16-row probe call fell
# to the slot path and asserted) + the bandwidth probe for C2's floor. Waits for runner_c0q to finish. Was: one quiet ticket for C0's per-layer timeline (c0_timeline.py at C 112, ~24 GB, ~5 min;
# code HumanEval/68 + chat multi/bg0) and the C2 / B3 kernel benches (k62p_floor.py, ~3 GB, ~4 min). Attached (run in the
# session's background) so the lock sees its processes. C0 only if NO-SWAP >= 28 GB at the grant; swap guard +2 GB.
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
log() { echo "[c0b] $* $(date '+%H:%M:%S'); swap $(sysctl -n vm.swapusage | cut -d'(' -f1 | sed 's/ *$//')"; }
noswap10() { vm_stat | awk '/page size of/ {ps = $8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v = $NF; sub("[.]", "", v); s += v} END {printf "%d", s * ps / 1e8}'; }
swap_mb() { sysctl -n vm.swapusage | awk '{for (i = 1; i <= NF; i++) if ($i == "used") {v = $(i + 2); sub("M", "", v); print int(v)}}'; }
tree() { echo "$1"; for c in $(pgrep -P "$1"); do tree "$c"; done; }
guard() {  # guard <pid> <tag>: swap +2 GB -> stop
  SW0=$(swap_mb)
  while kill -0 $1 2>/dev/null; do
    sleep 5
    if [ $(( $(swap_mb) - SW0 )) -gt 2048 ]; then log "SWAP GUARD: stopping $2"; T=$(tree $1); kill $T 2>/dev/null; sleep 20; kill -9 $T 2>/dev/null; break; fi
  done
  wait $1; log "$2: rc $?"
}
while ! grep -q "\[c0q\] done" logs/runner_c0q.log; do sleep 10; done
log "queueing the quiet ticket"
PEER_QUEUE_TIMEOUT=14400 peer lock quiet --queue 20m "Leap DAY9 #5: C0 per-layer timeline at C 160 (~29 GB, ~7 min) + bandwidth probe for C2 (~1 min)" >/dev/null 2>&1 || { log "no grant"; exit 1; }
log "quiet lock (NO-SWAP $(noswap10) / 10 GB)"
if [ "$(noswap10)" -ge 330 ] && [ ! -e $HOME/.local/share/quiet/PAUSE ] && [ ! -e $HOME/.local/share/quiet/HOLD ]; then
  env R57_PRESET=45 BIGMOE_EXT_BUILD=build_next R57_REPS=1 R57_WARM=1 R57_TAG=c0_c160 R57_KEYS=humaneval/HumanEval/68,multi/bg0 \
      C0_AT=1,2,5,10,15,30 R57_REFTOK=$HOME/BigMoE/data/r57/s45t_B_tokens.json:gate+t \
      perl -e 'alarm shift; exec @ARGV' 700 $PY scripts/c0_timeline.py 160 256 > logs/c0_c160.out 2>&1 &
  guard $! c0_c160
else
  log "C0 skipped (NO-SWAP $(noswap10) / 10 GB < 33, or PAUSE / HOLD)"
fi
perl -e 'alarm shift; exec @ARGV' 300 $PY scripts/bw_probe.py > logs/bw_probe.out 2>&1 &
guard $! bw_probe
peer unlock >/dev/null 2>&1
log "done"
