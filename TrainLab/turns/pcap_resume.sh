#!/bin/bash
# MTP Head, Wed 11-04 20:4x: P-CAP RESUME (the orchestrator 20:4x) after its swap-guard stop at step 148/188 (REPORT
# 'P-CAP: run STOPPED'): the last 40 steps (sticky/pcap_train.py --resume) + the walk (code + math), ~25 min, MEM 36 GB.
#  1. It goes FIRST after BigMoE's early 45 GB run releases its lock. If that run hasn't started by 21:15, it runs at
#     23:00 instead: then the resume starts as soon as the GPU is free, and P-OFF's extraction (turns/poff_x.sh, ~35 min)
#     follows only if it can start by 21:45 (ends by ~22:20). Otherwise P-OFF's extraction is Thursday's.
#  2. Start rule: 6 clear polls (3 min): no other quiet lock, no other share lock with MEM >= 5 GB, >= 36 GB available
#     (P-CAP's pre-registered guard). Gives up at 22:25 (the trellis / chat screens own the GPU after it; s7_trl's last
#     start is 22:53).
# Deviation (recorded in REPORT): the MLX trainer's resume restarts AdamW's moments with a 10-step learning-rate ramp.
# No port tests run while this holds the GPU. Logs: data/pcap_resume.log, data/pcap_train2.log, data/pcap_walk.log.
cd ~/TrainLab || exit 1
log() { echo "[$(date +%H:%M:%S)] $*"; }
hm() { date +%H%M; }
swap_mb() { sysctl -n vm.swapusage | awk '{print $6}' | tr -d M; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} /Pages purgeable/ {p=$3} END {printf "%d", (f+i+s+p)*ps/2^30}'; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }   # peer ls segments: share lock / held quiet lock or bench / queued quiet ticket
nquiet() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
nbig() { others | locks | grep -oE "MEM [0-9]+ GB" | awk '$2 >= 5' | wc -l | tr -d " "; }
nbigmoe() { others | grep "~/BigMoE" | locks | grep -cE "^· benchmarking|MEM ([5-9]|[1-9][0-9]+) GB"; }
wait_clear() {   # $1 = minimum GB available, $2 = give-up HHMM
  local clear=0 q b a
  while :; do
    q=$(nquiet); b=$(nbig); a=$(avail_gb)
    if [ "$q" -eq 0 ] && [ "$b" -eq 0 ] && [ "$a" -ge "$1" ]; then clear=$((clear + 1)); else clear=0; fi
    [ "$clear" -ge 6 ] && { log "clear for 6 polls (3 min): ${a} GB available"; return 0; }
    [ "$(hm)" -ge "$2" ] && { log "not clear by $2: quiet=$q big=$b avail=${a} GB"; return 1; }
    sleep 30
  done
}
guarded() {   # $1 = pkill pattern, rest = command; output to $LOGF; SIGTERM at swap +2 GB (the trainer checkpoints)
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
PY=~/BigMoE/.venv/bin/python3
OUT=~/TrainLab/ext/pcap/pcap10.safetensors
log "1. waiting for BigMoE's early 45 GB run to come and go (if it hasn't started by 21:15, it runs at 23:00)"
seen=0; late=0
while :; do
  if [ "$(nbigmoe)" -gt 0 ]; then
    [ $seen -eq 0 ] && log "BigMoE's lock is on: $(others | grep '~/BigMoE' | locks | cut -c1-150)"
    seen=1
  elif [ $seen -eq 1 ]; then log "BigMoE's lock released"; break
  elif [ "$(hm)" -ge 2115 ]; then log "no BigMoE lock by 21:15: it runs at 23:00; the resume goes as soon as the GPU is free"; late=1; break; fi
  [ "$(hm)" -ge 2225 ] && { log "GIVE UP at 22:25 (BigMoE's run still on): the resume moves to Thursday"; exit 3; }
  sleep 30
done
log "2. waiting for a clear slot (>= 36 GB available; gives up 22:25)"
wait_clear 36 2225 || { log "GIVE UP at 22:25: the resume moves to Thursday"; exit 3; }
ok=0
for t in 1 2 3; do
  out=$(peer lock share 60m --cpu 300 --gpu 90 "MEM 36 GB: MTP Head P-CAP resume (last 40 steps + walk, ~25 min; the orchestrator 20:4x)" 2>&1)
  log "lock try $t: $(printf '%s' "$out" | head -1)"
  printf '%s' "$out" | grep -q "reserved" && { ok=1; break; }
  sleep 60
done
[ $ok -eq 1 ] || { log "NO LOCK"; exit 4; }
LOGF=data/pcap_train2.log guarded "pcap_trai[n].py" /usr/bin/time -l $PY sticky/pcap_train.py --expand --lr-new 2e-4 \
  --A 32 --S 8 --micro 1 --epochs 2 --minutes 75 --lr 2e-5 --bs 32 --pmask 0.5 --data ~/Leap/data/l120 --out $OUT --resume
rc=$?
if [ $rc -eq 0 ] && grep -q "DONE: saved" data/pcap_train2.log; then
  LOGF=data/pcap_walk.log guarded "d02b_wal[k].py" env TL_EXPAND=1 TL_DRAFT=$OUT TL_WALK_TAG=_pcap /usr/bin/time -l $PY sticky/d02b_walk.py
  log "walk rc=$?: $(grep -E 'code|math' data/pcap_walk.log | tail -3 | tr '\n' ' ' | cut -c1-300)"
else
  log "training did not finish (rc=$rc): no walk"
fi
peer unlock 2>&1 | tail -1
if [ $late -eq 1 ]; then
  if [ "$(hm)" -le 2145 ]; then
    log "3. BigMoE runs at 23:00 and it is <= 21:45: P-OFF's extraction (ends by ~22:20)"
    ./turns/poff_x.sh > data/poff_x_run.log 2>&1
    log "extraction runner rc=$?: $(tail -2 data/poff_x_run.log | tr '\n' ' ')"
    rate=$(grep -oE "on-path rate \(assistant\) [0-9.]+" data/poff_x.log | awk '{print $NF}')
    log "P-OFF on-path rate (assistant): ${rate:-none} (alive line 0.75; training is Thursday's either way)"
  else
    log "3. too late for P-OFF's extraction to end by ~22:20: Thursday"
  fi
fi
log "DONE"
