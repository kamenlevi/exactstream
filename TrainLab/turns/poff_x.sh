#!/bin/bash
# MTP Head, Wed 11-04 18:1x: P-OFF extraction (REPORT 'P-OFF'). Leap's ~/Leap/scripts/dfl2c_extract.py, used read-only
# and unchanged, on ext/poff/poff_windows.jsonl -> ext/poff/x/<key>.npz (resumable: done windows are skipped).
# 19:4x (REPORT 'P-OFF amendment 2'): only the first $POFF_LIMIT windows (default 260 = the 60K-token training prefix,
# 137K tokens). Leap's own log says ~7 s per short window (56 tok/s, SSD-bound expert streaming), not the 370 tok/s I
# assumed at 18:1x, so all 1,698 windows would take ~3.3 h. Smoke = the 3 longest windows of the prefix
# (ext/poff/smoke3.jsonl: 1271 / 1213 / 974 tokens), so its footprint bounds the run's (the extractor clears its cache per
# window); stops if it is over 27 GB. Share lock MEM 28 GB (Leap's log: MLX peak 22.4 GB on <= 609-token windows); swap
# guard +2 GB. Logs: data/poff_x_smoke.log, data/poff_x.log.
cd ~/TrainLab || exit 1
log() { echo "[$(date +%H:%M:%S)] $*"; }
swap_mb() { sysctl -n vm.swapusage | awk '{print $6}' | tr -d M; }
PY=~/BigMoE/.venv/bin/python3
LIMIT=${POFF_LIMIT:-260}
export DFL2C_IN=$HOME/TrainLab/ext/poff/poff_windows.jsonl DFL2C_OUT=$HOME/TrainLab/ext/poff/x
ok=0
for t in 1 2 3; do
  out=$(peer lock share 60m --cpu 300 --gpu 90 "MEM 28 GB: MTP Head P-OFF extraction (the 125B's prefill of the first $LIMIT code windows, ~137K tokens, ~35 min; the orchestrator 19:4x)" 2>&1)
  log "lock try $t: $(printf '%s' "$out" | head -1)"
  printf '%s' "$out" | grep -q "reserved" && { ok=1; break; }
  sleep 60
done
[ $ok -eq 1 ] || { log "NO LOCK"; exit 4; }
log "smoke (the 3 longest windows of the prefix)"
DFL2C_IN=$HOME/TrainLab/ext/poff/smoke3.jsonl /usr/bin/time -l $PY ~/Leap/scripts/dfl2c_extract.py > data/poff_x_smoke.log 2>&1
src=$?
fp=$(grep "peak memory footprint" data/poff_x_smoke.log | awk '{printf "%.1f", $1/2^30}')
log "smoke rc=$src footprint ${fp} GB; $(grep -c '\[dfl2c-x\] [0-9]' data/poff_x_smoke.log) windows; $(grep -oE 'peak [0-9.]+ GB' data/poff_x_smoke.log)"
if [ $src -ne 0 ] || [ -z "$fp" ] || awk -v f="$fp" 'BEGIN{exit !(f > 27)}'; then
  log "smoke failed or footprint over 27 GB: stop, re-declare MEM"; peer unlock 2>&1 | tail -1; exit 5
fi
s0=$(swap_mb); log "full run, first $LIMIT windows (swap $s0 MB)"
DFL2C_LIMIT=$LIMIT /usr/bin/time -l $PY ~/Leap/scripts/dfl2c_extract.py > data/poff_x.log 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  sleep 20
  s=$(swap_mb)
  if awk -v a="$s" -v b="$s0" 'BEGIN{exit !(a-b>2048)}'; then
    log "swap $s MB = +2 GB over the start: stopping (resumable)"; pkill -TERM -f "dfl2c_extrac[t].py"; break
  fi
done
wait $pid; log "rc=$? (swap $(swap_mb) MB); $(grep -c '\[dfl2c-x\] [0-9]' data/poff_x.log) windows this run; $(ls ext/poff/x | wc -l | tr -d ' ') files"
peer unlock 2>&1 | tail -1
log "DONE"
