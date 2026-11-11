#!/bin/bash
# F2a night v3 (MTP Head; quiet whole-Mac window Sun 11-08 ~23:15 -> ~03:30 per ~/Papers/MONDAY.md, behind the 23:00 PF-D ticket; corpus under Kamen's
# standing download OK, ext/chatcorpus). Pre-registered: REPORT 'F2A night: pre-registration' + amendments 1-3.
# Stage 1 (the pre-registered verdict; amendment 3: the only stage tonight):
#   1. capture: 24 on-policy IFEval eval windows, then ring 1 (data/f2a/ring) until the ring budget, free disk < 28 GB
#      (amendment 2) or 02:00
#   2. train: b5 -> ckpt/d22_f2a, 2 epochs, until 03:00     3. eval: f2a vs b5 -> data/f2a/f2a_eval.txt by 03:15; commit
# Then: peer unlock, write data/f2a/STAGE1_DONE, and log "F2a night DONE" as the LAST line of data/f2a/night.log
# (EvalSuite's chain starts on either). Every exit after 23:00 does the same (trap), so the chain never waits on a dead runner.
# Stage 2 (amendment 2, moved after the release by amendment 3) runs ONLY if data/f2a/FULL_NIGHT exists when stage 1 ends
# (by 03:45): delete ring 1; ring 2 to 05:10; f2a -> f2a2 to 06:05; eval to 06:30; then unlock + DONE as above.
# A ring under 200 windows is not trained on (NO DATA).
# Guards per phase: swap +2 GB -> SIGTERM (the phase saves), free disk < 25 GB -> SIGTERM, 10 min past its deadline -> SIGTERM;
# PAUSE -> the scripts wait. Memory (smoke Sun 01:17): peak ~21 GB + 0.118 GB per pool slot -> C 256 at NO-SWAP >= 54, C 192
# >= 46, C 128 >= 38, C 64 >= 31, else C 16.
# DRY=1 ROOT=<dir> PY=<stub>: control-flow test (times from now, no lock, no git).
# Usage (attached, run_in_background): cd ~/TrainLab && ./turns/f2a_night.sh > data/f2a/night.log 2>&1
cd "${ROOT:-$HOME/TrainLab}" || exit 1
mkdir -p data/f2a
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
at() { date -j -f "%Y-%m-%d %H:%M" "$1" +%s; }
NB=$(at "2026-11-08 23:01"); CAP1=$(at "2026-11-09 02:15"); TR1=$(at "2026-11-09 03:15"); EV1=$(at "2026-11-09 03:30")   # amendment 4: queued behind the 23:00 PF-D ticket
S2=$(at "2026-11-09 03:45"); CAP2=$(at "2026-11-09 05:10"); TR2=$(at "2026-11-09 06:05"); EV2=$(at "2026-11-09 06:30")
END=$(at "2026-11-09 06:50")
PY=${PY:-$HOME/BigMoE/.venv/bin/python3}
if [ -n "$DRY" ]; then
  n=$(date +%s); NB=$n; CAP1=$((n + 600)); TR1=$CAP1; EV1=$CAP1; S2=$CAP1; CAP2=$CAP1; TR2=$CAP1; EV2=$CAP1; END=$((n + 900))
  peer() { echo "lock held (dry: peer $1)"; }; git() { echo "(dry) git $1 $2"; }
fi
DONE=data/f2a/STAGE1_DONE
rm -f "$DONE" data/f2a/FULL_NIGHT.used
STARTED=0; FIN=0; CHILD=""
swap_mb() { sysctl -n vm.swapusage | awk '{print $6}' | tr -d M; }
noswap_gb() { vm_stat | awk '/page size of/ {ps = $8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v = $NF; sub("[.]", "", v); s += v} END {printf "%d", s * ps / 2^30}'; }
disk_gb() { df -g / | awk 'NR == 2 {print $4}'; }
nwin() { find data/f2a/ring -name 'w_*.npz' 2>/dev/null | wc -l | tr -d ' '; }
frac() { local A; A=$(noswap_gb); FR=0.03   # gate: measured peak + 4
  [ "$A" -ge 31 ] && FR=0.125; [ "$A" -ge 38 ] && FR=0.25; [ "$A" -ge 46 ] && FR=0.375; [ "$A" -ge 54 ] && FR=0.5
  log "NO-SWAP $A GB -> capture frac $FR"; }
commit() {
  git add data/f2a/*.txt data/f2a/*.log ckpt/d22_f2a*.json 2>/dev/null
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "committed: $1"
}
finish() {   # $1 = what to record. After 23:00 only: stop a live phase, release the lock, DONE file, then the last log line.
  [ "$STARTED" = 1 ] && [ "$FIN" = 0 ] || return 0
  FIN=1
  if [ -n "$CHILD" ] && kill -0 "$CHILD" 2>/dev/null; then log "stopping the live phase (pid $CHILD)"; kill -TERM "$CHILD"; wait "$CHILD"; fi
  peer unlock 2>&1 | tail -1
  printf '%s\n' "F2a $(date '+%a %H:%M:%S'): $1" > "$DONE.tmp" && mv "$DONE.tmp" "$DONE"
  log "DONE file: $DONE ($1)"
  log "F2a night DONE"
}
trap 'finish "runner exit (rc $?)"' EXIT
trap 'exit 143' TERM INT
log "F2a night v3 armed: not before $(date -r "$NB" '+%a %H:%M'); stage 1 only, queued behind the 23:00 ticket (capture to 02:15, train to 03:15, eval to 03:30, then unlock, the DONE file $DONE and the closing log line); stage 2 only if data/f2a/FULL_NIGHT exists; disk $(disk_gb) GB free"
while [ "$(date +%s)" -lt "$NB" ]; do sleep 30; done
STARTED=1
[ "$(date +%s)" -ge "$END" ] && { log "past the end: nothing runs"; exit 5; }
mins=$(( (END - $(date +%s)) / 60 ))
out=$(peer lock quiet --queue ${mins}m "MEM 44 GB: MTP Head F2a chat-head capture + training (stage 1, to ~03:30)" 2>&1)
log "lock: $(printf '%s' "$out" | tail -1)"
printf '%s' "$out" | grep -qi "lock held\|reserved" || { log "NO LOCK"; exit 4; }
phase() {   # $1 name, $2 deadline (epoch), rest: the command
  local nm=$1 dl=$2; shift 2
  local s0; s0=$(swap_mb)
  log "phase $nm start (swap $s0 MB, NO-SWAP $(noswap_gb) GB, disk $(disk_gb) GB, ring $(nwin) windows): $*"
  env F2A_UNTIL=$dl "$@" > data/f2a/$nm.log 2>&1 &
  CHILD=$!
  while kill -0 $CHILD 2>/dev/null; do
    sleep 15
    if awk -v x="$(swap_mb)" -v y="$s0" 'BEGIN{exit !(x-y>2048)}'; then log "$nm: swap +2 GB -> SIGTERM"; kill -TERM $CHILD; fi
    if [ "$(disk_gb)" -lt 25 ]; then log "$nm: disk under 25 GB -> SIGTERM"; kill -TERM $CHILD; fi
    if [ "$(date +%s)" -ge $((dl + 600)) ]; then log "$nm: 10 min past its deadline -> SIGTERM"; kill -TERM $CHILD; fi
  done
  wait $CHILD; local rc=$?
  CHILD=""
  log "phase $nm rc=$rc: $(tail -1 data/f2a/$nm.log | cut -c1-200)"
  return $rc
}
# ---- stage 1 (pre-registered)
frac
phase capture "$CAP1" env F2A_RING_GB=50 F2A_DISK_MIN=28 $PY -I sticky/f2a_capture.py --frac $FR --eval 24
n1=$(nwin)
if [ "$n1" -ge 200 ]; then
  phase train "$TR1" env F2A_EPOCHS=2 $PY -I sticky/f2a_train.py
  phase eval "$EV1" $PY -I sticky/f2a_eval.py f2a
  tail -1 data/f2a/f2a_eval.txt 2>/dev/null
  commit "F2a night stage 1: ring 1 = $n1 windows, f2a vs b5 (verdict in data/f2a/f2a_eval.txt)"
else
  log "NO DATA: ring 1 has $n1 windows (< 200): no training"
fi
V1="stage 1: $(tail -1 data/f2a/f2a_eval.txt 2>/dev/null | cut -c1-170)"
# ---- stage 2 (amendment 2; amendment 3: only on the FULL_NIGHT flag)
if [ -f data/f2a/FULL_NIGHT ] && [ "$(date +%s)" -lt "$S2" ] && [ -f ckpt/d22_f2a.safetensors ] && [ "$n1" -ge 200 ]; then
  mv data/f2a/FULL_NIGHT data/f2a/FULL_NIGHT.used
  log "stage 2 (FULL_NIGHT flag): deleting ring 1 ($n1 windows); ring 2 continues from cursor.json"
  find data/f2a/ring -name 'w_*.npz' -delete
  frac
  phase capture2 "$CAP2" env F2A_RING_GB=50 F2A_DISK_MIN=28 $PY -I sticky/f2a_capture.py --frac $FR --eval 0
  n2=$(nwin)
  if [ "$n2" -ge 200 ]; then
    phase train2 "$TR2" env F2A_EPOCHS=2 F2A_INIT=f2a $PY -I sticky/f2a_train.py --run f2a2
    phase eval2 "$EV2" $PY -I sticky/f2a_eval.py f2a2
    tail -2 data/f2a/f2a2_eval.txt 2>/dev/null
    commit "F2a night stage 2: ring 2 = $n2 windows, f2a2 vs b5 and f2a (data/f2a/f2a2_eval.txt)"
  else
    log "NO DATA: ring 2 has $n2 windows (< 200): no stage-2 training"
  fi
else
  log "stage 2 not run (amendment 3: stage 1 only unless data/f2a/FULL_NIGHT exists; flag $([ -f data/f2a/FULL_NIGHT ] && echo present || echo absent), time $(date '+%H:%M'))"
fi
commit "F2a night: logs (data/f2a)"
finish "$V1"
