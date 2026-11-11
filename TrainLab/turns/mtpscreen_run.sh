#!/bin/bash
# MTP Head, Wed 11-04 15:3x (the orchestrator 15:2x: run MTPSCREEN by day in the one > 5 GB slot, after BigMoE's by-day
# share lock ends ~15:44-15:52). Waits until no other session holds a share lock declaring MEM >= 5 GB (gives up 17:30),
# takes `peer lock share 30m --cpu 300 --gpu 50` with MEM 11 GB, runs sticky/mtpscreen.py (swap guard: stop at +2 GB),
# unlocks. Log: data/mtpscreen_run.log (+ data/mtpscreen.log).
cd ~/TrainLab || exit 1
log() { echo "[$(date +%H:%M:%S)] $*"; }
swap_mb() { sysctl -n vm.swapusage | awk '{print $6}' | tr -d M; }
while :; do
  big=$(peer ls 2>/dev/null | grep -v "MTP Head" | grep -E "shared run|quiet lock" | sed "s/.*·//" | grep -oE "MEM [0-9]+ GB" | awk "\$2 >= 5" | wc -l | tr -d " ")
  [ "$big" -eq 0 ] && break
  [ "$(date +%H%M)" -ge 1730 ] && { log "GIVE UP: a > 5 GB job still holds the slot at 17:30"; exit 3; }
  sleep 60
done
log "slot free"
ok=0
for t in 1 2 3; do
  out=$(peer lock share 30m --cpu 300 --gpu 50 "MEM 11 GB: MTP Head MTPSCREEN (screened MTP head for the MTP-assisted pick; Leap's ask), ~15-20 min" 2>&1)
  log "lock try $t: $(printf '%s' "$out" | head -1)"
  printf '%s' "$out" | grep -q "reserved" && { ok=1; break; }
  sleep 60
done
[ $ok -eq 1 ] || { log "NO LOCK"; exit 4; }
s0=$(swap_mb); log "start (swap $s0 MB)"
/usr/bin/time -l ~/BigMoE/.venv/bin/python sticky/mtpscreen.py > data/mtpscreen.log 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  sleep 20
  s=$(swap_mb)
  if awk -v a="$s" -v b="$s0" 'BEGIN{exit !(a-b>2048)}'; then
    log "swap $s MB = +2 GB over the start: stopping"; pkill -f "mtpscree[n].py"; break
  fi
done
wait $pid; log "rc=$? (swap $(swap_mb) MB)"
peer unlock 2>&1 | tail -1
log "DONE"
