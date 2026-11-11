#!/bin/bash
# MTP Head, Wed 11-04 18:1x: P-OFF training + walk (REPORT 'P-OFF'). The folder (sticky/poff_data.py: L1-20's windows +
# the first 60K off-policy assistant tokens, amendment 18:1x), then sticky/poff_train.py from L1-20 for 1 epoch (L1-20's recipe), then
# the D-02b walk. Share lock MEM 34 GB (L1-20's own peak 33.7 GB); swap guard +2 GB (SIGTERM: the trainer checkpoints,
# --resume continues). Started on the orchestrator's slot. Logs: data/poff_t.log, data/poff_train.log, data/poff_walk.log.
cd ~/TrainLab || exit 1
log() { echo "[$(date +%H:%M:%S)] $*"; }
swap_mb() { sysctl -n vm.swapusage | awk '{print $6}' | tr -d M; }
PY=~/BigMoE/.venv/bin/python3
OUT=$HOME/TrainLab/ext/poff/poff60.safetensors
$PY sticky/poff_data.py --tokens 60000 --out $HOME/TrainLab/ext/poff/train60 || { log "data folder failed"; exit 2; }
ok=0
for t in 1 2 3; do
  out=$(peer lock share 90m --cpu 300 --gpu 90 "MEM 34 GB: MTP Head P-OFF training (L1-20 + 60K off-policy code tokens, ~55 min) + code walk; then Breakthrough s7_trl" 2>&1)
  log "lock try $t: $(printf '%s' "$out" | head -1)"
  printf '%s' "$out" | grep -q "reserved" && { ok=1; break; }
  sleep 60
done
[ $ok -eq 1 ] || { log "NO LOCK"; exit 4; }
guarded() {
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
LOGF=data/poff_train.log guarded "poff_trai[n].py" /usr/bin/time -l $PY sticky/poff_train.py --init ~/Leap/data/l120_ft.safetensors \
  --epochs 1 --minutes 55 --lr 2e-5 --S 4 --A 64 --micro 1 --bs 32 --pmask 0.5 --data ~/TrainLab/ext/poff/train60 --out $OUT
rc=$?
if [ $rc -eq 0 ] && grep -q "DONE: saved" data/poff_train.log; then
  LOGF=data/poff_walk.log guarded "d02b_wal[k].py" env TL_WALK_SETS="code=$HOME/Leap/data/dfl2d_code/*.npz" TL_DRAFT=$OUT TL_WALK_TAG=_poff /usr/bin/time -l $PY sticky/d02b_walk.py
else
  log "training did not finish (rc=$rc): no walk"
fi
peer unlock 2>&1 | tail -1
log "DONE"
