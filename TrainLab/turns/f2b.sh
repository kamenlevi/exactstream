#!/bin/bash
# F2b (MTP Head): the data-rich re-run of F2a stage 1 (REPORT 'F2A night: pre-registration', amendment 5). Same recipe and
# lines as F2a: b5 continued on the 125B's own chat features (UltraChat + Aya shard, ext/chatcorpus), D-22 trainer, 2 epochs,
# lr b5 x 0.5; verdict f2b vs b5 (PASS held-out kept >= x1.08 AND on-policy IFEval >= x1.05; KILL held-out < x1.03).
# Differences from F2a's night (runner v3):
#   - the window comes from the queue: F2B_NB and F2B_END ("YYYY-MM-DD HH:MM"); capture ends 4 h 30 min before F2B_END
#     (2 epochs on a full ring take <= ~3.9 h at the measured 2.23 s per window-step), training 25 min before, eval 5 min;
#   - the ring continues from Sunday's (241 windows, cursor.json) and the 24 on-policy eval windows are reused (--eval 0);
#   - the ring budget is 25 GiB (~1.3M tokens); the expert pool is gated at the measured peak + 8 GB (C 128 swapped at +4);
#   - a ring under 1,200 windows (~0.5M tokens) is not trained on: NOT A TEST of the data-rich plan;
#   - no stage 2. DONE file data/f2a/F2B_DONE; last log line "F2b DONE".
# Guards per phase (as v3): swap +2 GB -> SIGTERM (the phase saves), free disk < 25 GB -> SIGTERM, 10 min past its
# deadline -> SIGTERM; PAUSE -> the scripts wait. Capture also stops at free disk < 28 GB.
# Memory: C 128 measured MLX peak 37.4 GB (Sun 23:29); model ~21 GB + 0.118 GB per slot -> NO-SWAP gate (peak + 8):
#   C 256 >= 61, C 192 >= 53, C 128 >= 46, C 64 >= 37, else C 16.
# DRY=1 ROOT=<dir> PY=<stub>: control-flow test (times from now, no lock, no git).
# Usage (attached, run_in_background): cd ~/TrainLab && F2B_NB="2026-11-12 01:00" F2B_END="2026-11-12 13:30" \
#   ./turns/f2b.sh > data/f2a/f2b.log 2>&1
cd "${ROOT:-$HOME/TrainLab}" || exit 1
mkdir -p data/f2a
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
at() { date -j -f "%Y-%m-%d %H:%M" "$1" +%s; }
PY=${PY:-$HOME/BigMoE/.venv/bin/python3}
MINWIN=${F2B_MINWIN:-1200}
if [ -n "$DRY" ]; then
  n=$(date +%s); NB=$n; END=$((n + 900)); CAP=$((n + 300)); TR=$((n + 600)); EV=$((n + 840))
  peer() { echo "lock held (dry: peer $1)"; }; git() { echo "(dry) git $1 $2"; }
else
  [ -n "$F2B_NB" ] && [ -n "$F2B_END" ] || { log "set F2B_NB and F2B_END"; exit 2; }
  NB=$(at "$F2B_NB") && END=$(at "$F2B_END") || { log "bad time"; exit 2; }
  CAP=$((END - 270 * 60)); TR=$((END - 25 * 60)); EV=$((END - 5 * 60))
  [ "$CAP" -gt "$NB" ] || { log "window too short: capture would end before it starts"; exit 2; }
fi
DONE=data/f2a/F2B_DONE
rm -f "$DONE"
STARTED=0; FIN=0; CHILD=""
swap_mb() { sysctl -n vm.swapusage | awk '{print $6}' | tr -d M; }
noswap_gb() { vm_stat | awk '/page size of/ {ps = $8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v = $NF; sub("[.]", "", v); s += v} END {printf "%d", s * ps / 2^30}'; }
disk_gb() { df -g / | awk 'NR == 2 {print $4}'; }
nwin() { find data/f2a/ring -name 'w_*.npz' 2>/dev/null | wc -l | tr -d ' '; }
frac() { local A; A=$(noswap_gb); FR=0.03   # gate: measured peak + 8
  [ "$A" -ge 37 ] && FR=0.125; [ "$A" -ge 46 ] && FR=0.25; [ "$A" -ge 53 ] && FR=0.375; [ "$A" -ge 61 ] && FR=0.5
  log "NO-SWAP $A GB -> capture frac $FR"; }
commit() {
  git add data/f2a/*.txt data/f2a/*.log ckpt/d22_f2b*.json 2>/dev/null
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "committed: $1"
}
finish() {   # $1 = what to record. After the start only: stop a live phase, release the lock, DONE file, then the last line.
  [ "$STARTED" = 1 ] && [ "$FIN" = 0 ] || return 0
  FIN=1
  if [ -n "$CHILD" ] && kill -0 "$CHILD" 2>/dev/null; then log "stopping the live phase (pid $CHILD)"; kill -TERM "$CHILD"; wait "$CHILD"; fi
  peer unlock 2>&1 | tail -1
  printf '%s\n' "F2b $(date '+%a %H:%M:%S'): $1" > "$DONE.tmp" && mv "$DONE.tmp" "$DONE"
  log "DONE file: $DONE ($1)"
  log "F2b DONE"
}
trap 'finish "runner exit (rc $?)"' EXIT
trap 'exit 143' TERM INT
log "F2b armed: not before $(date -r "$NB" '+%a %H:%M'), window end $(date -r "$END" '+%a %H:%M') (capture to $(date -r "$CAP" '+%H:%M'), train to $(date -r "$TR" '+%H:%M'), eval to $(date -r "$EV" '+%H:%M')); ring now $(nwin) windows; disk $(disk_gb) GB free"
while [ "$(date +%s)" -lt "$NB" ]; do sleep 30; done
STARTED=1
[ "$(date +%s)" -ge "$CAP" ] && { log "past the capture end: nothing runs"; exit 5; }
mins=$(( (END - $(date +%s)) / 60 ))
out=$(peer lock quiet --queue ${mins}m "MEM 44 GB: MTP Head F2b chat-head capture + training (to $(date -r "$END" '+%H:%M'))" 2>&1)
log "lock: $(printf '%s' "$out" | tail -1)"
printf '%s' "$out" | grep -qi "lock held\|reserved" || { log "NO LOCK"; exit 4; }
phase() {   # $1 name, $2 deadline (epoch), rest: the command
  local nm=$1 dl=$2; shift 2
  local s0; s0=$(swap_mb)
  log "phase $nm start (swap $s0 MB, NO-SWAP $(noswap_gb) GB, disk $(disk_gb) GB, ring $(nwin) windows): $*"
  env F2A_UNTIL=$dl "$@" > data/f2a/f2b_$nm.log 2>&1 &
  CHILD=$!
  while kill -0 $CHILD 2>/dev/null; do
    sleep 15
    if awk -v x="$(swap_mb)" -v y="$s0" 'BEGIN{exit !(x-y>2048)}'; then log "$nm: swap +2 GB -> SIGTERM"; kill -TERM $CHILD; fi
    if [ "$(disk_gb)" -lt 25 ]; then log "$nm: disk under 25 GB -> SIGTERM"; kill -TERM $CHILD; fi
    if [ "$(date +%s)" -ge $((dl + 600)) ]; then log "$nm: 10 min past its deadline -> SIGTERM"; kill -TERM $CHILD; fi
  done
  wait $CHILD; local rc=$?
  CHILD=""
  log "phase $nm rc=$rc: $(tail -1 data/f2a/f2b_$nm.log | cut -c1-200)"
  return $rc
}
frac
phase capture "$CAP" env F2A_RING_GB=25 F2A_DISK_MIN=28 $PY -I sticky/f2a_capture.py --frac $FR --eval 0
n1=$(nwin)
if [ "$n1" -ge "$MINWIN" ]; then
  phase train "$TR" env F2A_EPOCHS=2 F2A_INIT=b5 $PY -I sticky/f2a_train.py --run f2b
  phase eval "$EV" $PY -I sticky/f2a_eval.py f2b
  tail -2 data/f2a/f2b_eval.txt 2>/dev/null
  commit "F2b: ring = $n1 windows, f2b vs b5 and f2a (verdict in data/f2a/f2b_eval.txt)"
  V="$(tail -2 data/f2a/f2b_eval.txt 2>/dev/null | head -1 | cut -c1-170)"
else
  log "NOT A TEST: the ring has $n1 windows (< $MINWIN, ~0.5M tokens): no training"
  commit "F2b: NOT A TEST (ring $n1 windows < $MINWIN)"
  V="NOT A TEST (ring $n1 windows)"
fi
finish "$V"
