#!/bin/sh
# Leap DAY9 (Mon 2026-11-02; v2 11:0x adds r86 A2/A4 + r85 D7): one quiet ticket for C0's per-layer timeline (c0_timeline.py at C 112, ~24 GB, ~5 min;
# code HumanEval/68 + chat multi/bg0) and the C2 / B3 kernel benches (k62p_floor.py, ~3 GB, ~4 min). Attached (run in the
# session's background) so the lock sees its processes. C0 only if NO-SWAP >= 28 GB at the grant; swap guard +2 GB.
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
log() { echo "[c0q] $* $(date '+%H:%M:%S'); swap $(sysctl -n vm.swapusage | cut -d'(' -f1 | sed 's/ *$//')"; }
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
log "queueing the quiet ticket"
PEER_QUEUE_TIMEOUT=14400 peer lock quiet --queue 20m "Leap DAY9 #5/#4/#9/#6: C0 per-layer timeline (C 112, ~24 GB, ~6 min) + kernel benches C2/B3, A2/A4, D7 (~4 GB, ~9 min)" >/dev/null 2>&1 || { log "no grant"; exit 1; }
log "quiet lock (NO-SWAP $(noswap10) / 10 GB)"
if [ "$(noswap10)" -ge 280 ] && [ ! -e $HOME/.local/share/quiet/PAUSE ] && [ ! -e $HOME/.local/share/quiet/HOLD ]; then
  env R57_PRESET=45 BIGMOE_EXT_BUILD=build_next R57_REPS=1 R57_WARM=1 R57_TAG=c0_c112 R57_KEYS=humaneval/HumanEval/68,multi/bg0 \
      C0_AT=1,2,5,10,15,30 R57_REFTOK=$HOME/BigMoE/data/r57/s45t_B_tokens.json:gate+t \
      perl -e 'alarm shift; exec @ARGV' 700 $PY scripts/c0_timeline.py 112 256 > logs/c0_c112.out 2>&1 &
  guard $! c0_c112
else
  log "C0 skipped (NO-SWAP $(noswap10) / 10 GB < 28, or PAUSE / HOLD)"
fi
perl -e 'alarm shift; exec @ARGV' 600 $PY scripts/k62p_floor.py > logs/k62p_floor.out 2>&1 &
guard $! k62p_floor
perl -e 'alarm shift; exec @ARGV' 300 $PY scripts/r86_a2a4.py > logs/r86_a2a4.out 2>&1 &
guard $! r86_a2a4
perl -e 'alarm shift; exec @ARGV' 600 $PY scripts/r85_d7.py > logs/r85_d7.out 2>&1 &
guard $! r85_d7
peer unlock >/dev/null 2>&1
log "done"
