#!/bin/bash
# MTP Head, Wed 11-04 18:0x: P-CAP (REPORT 'P-CAP'; the orchestrator's 19:05 GPU slot, after Breakthrough's map batch).
# Waits until 19:05 and >= 34 GB available (free + inactive + speculative + purgeable; 18:2x baseline ~37 GB with the map batch idle)
# with no other session declaring MEM >= 5 GB for 3 polls in a row (Breakthrough 18:4x: not a gap between its arms; gives up 21:50, the orchestrator 18:4x re-slot), takes a share lock (MEM 32 GB), trains the 10-layer
# drafter (sticky/pcap_train.py; swap guard: SIGTERM at +2 GB -> the trainer checkpoints and exits 3), walks it
# (sticky/d02b_walk.py, TL_EXPAND=1), unlocks. Logs: data/pcap_run.log, data/pcap_train.log, data/pcap_walk.log.
cd ~/TrainLab || exit 1
log() { echo "[$(date +%H:%M:%S)] $*"; }
swap_mb() { sysctl -n vm.swapusage | awk '{print $6}' | tr -d M; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} /Pages purgeable/ {p=$3} END {printf "%d", (f+i+s+p)*ps/2^30}'; }
PY=~/BigMoE/.venv/bin/python3
OUT=~/TrainLab/ext/pcap/pcap10.safetensors
mkdir -p ~/TrainLab/ext/pcap
clear=0
while :; do
  now=$(date +%H%M)
  if [ "$now" -ge 1905 ]; then
    big=$(peer ls 2>/dev/null | grep -v "MTP Head" | grep -E "shared run|quiet lock" | sed "s/.*·//" | grep -oE "MEM [0-9]+ GB" | awk '$2 >= 5' | wc -l | tr -d " ")
    av=$(avail_gb)
    if [ "$big" -eq 0 ] && [ "$av" -ge 34 ]; then clear=$((clear + 1)); else clear=0; fi
    [ "$clear" -ge 3 ] && { log "slot free for 3 polls: ${av} GB available"; break; }   # not a gap between another job's arms
    [ "$now" -ge 2150 ] && { log "GIVE UP at 21:50: big=$big, ${av} GB available"; exit 3; }
  fi
  sleep 60
done
ok=0
for t in 1 2 3; do
  out=$(peer lock share 150m --cpu 300 --gpu 90 "MEM 32 GB: MTP Head P-CAP (10-layer drafter: train ~60 min + walk ~10 min; the orchestrator 18:4x: after Breakthrough's map arms)" 2>&1)
  log "lock try $t: $(printf '%s' "$out" | head -1)"
  printf '%s' "$out" | grep -q "reserved" && { ok=1; break; }
  sleep 60
done
[ $ok -eq 1 ] || { log "NO LOCK"; exit 4; }

guarded() {   # $1 = pattern for pkill, rest = command; output to the file in $LOGF
  local pat=$1; shift
  local s0; s0=$(swap_mb); log "start (swap $s0 MB): $*"
  "$@" > "$LOGF" 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 20
    local s; s=$(swap_mb)
    if awk -v a="$s" -v b="$s0" 'BEGIN{exit !(a-b>2048)}'; then
      log "swap $s MB = +2 GB over the start: SIGTERM"; pkill -TERM -f "$pat"; break
    fi
  done
  wait $pid; local rc=$?
  log "rc=$rc (swap $(swap_mb) MB)"
  return $rc
}
LOGF=data/pcap_train.log guarded "pcap_trai[n].py" /usr/bin/time -l $PY sticky/pcap_train.py --expand --lr-new 2e-4 \
  --A 32 --S 8 --micro 1 --epochs 2 --minutes 75 --lr 2e-5 --bs 32 --pmask 0.5 --data ~/Leap/data/l120 --out $OUT
rc=$?
if [ $rc -eq 0 ] && grep -q "DONE: saved" data/pcap_train.log; then
  LOGF=data/pcap_walk.log guarded "d02b_wal[k].py" env TL_EXPAND=1 TL_DRAFT=$OUT TL_WALK_TAG=_pcap /usr/bin/time -l $PY sticky/d02b_walk.py
else
  log "training did not finish (rc=$rc): no walk"
fi
peer unlock 2>&1 | tail -1
log "DONE"
