#!/bin/bash
# MTP Head, Wed 11-04 20:1x: P-OFF EXTRACTION ONLY tonight (the orchestrator 20:0x, yes to my option): after P-CAP and
# BigMoE's early 45 GB run; it gives tonight's on-path rate = P-OFF's alive / KILL line. Training moves to Thursday.
#  1. Wait for P-CAP's runner to be gone.
#  2. Wait for BigMoE's lock to come and go ("quiet lock until", or a share lock with MEM >= 5 GB, on its ~/BigMoE line);
#     if none has started by 21:00, go on.
#  3. 6 clear polls (3 min): no other session's quiet lock, no other share lock declaring MEM >= 5 GB, and >= 30 GB
#     available (free + inactive + speculative + purgeable; a guard in case a lock line is missed: the extraction needs
#     <= 27 GB). The extraction gives up if it can't start by 22:00 (then Thursday). ~35 min, done by ~22:35.
#  4. turns/poff_x.sh: the first 260 windows. Then the rate: >= 0.75 = alive (training on Thursday), < 0.75 = KILL by the
#     pre-registered diagnostic (REPORT 'P-OFF amendment'). Breakthrough's s7_trl goes after it (or before, if its store
#     is whole first: the same one-job rule).
# Log: data/poff_night.log.
cd ~/TrainLab || exit 1
log() { echo "[$(date +%H:%M:%S)] $*"; }
hm() { date +%H%M; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} /Pages purgeable/ {p=$3} END {printf "%d", (f+i+s+p)*ps/2^30}'; }
others() { peer ls 2>/dev/null | grep "^- "; }   # other sessions only (mine is the "You:" line; a status that mentions MTP Head still counts)
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }   # peer ls segments: share lock / held quiet lock or bench / queued quiet ticket   # the lock segment only, never the status text
nquiet() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
nbig() { others | locks | grep -oE "MEM [0-9]+ GB" | awk '$2 >= 5' | wc -l | tr -d " "; }
nbigmoe() { others | grep "~/BigMoE" | locks | grep -cE "^· benchmarking|MEM ([5-9]|[1-9][0-9]+) GB"; }   # its quiet lock, or a share lock declaring MEM >= 5 GB
wait_clear() {   # $1 = minimum GB available, $2 = give-up HHMM; 0 when clear for 6 polls (3 min)
  local clear=0 q b a
  while :; do
    q=$(nquiet); b=$(nbig); a=$(avail_gb)
    if [ "$q" -eq 0 ] && [ "$b" -eq 0 ] && [ "$a" -ge "$1" ]; then clear=$((clear + 1)); else clear=0; fi
    [ "$clear" -ge 6 ] && { log "clear for 6 polls (3 min): ${a} GB available"; return 0; }
    [ "$(hm)" -ge "$2" ] && { log "not clear by $2: quiet=$q big=$b avail=${a} GB"; return 1; }
    sleep 30
  done
}
log "1. waiting for P-CAP's runner"
while pgrep -f "pcap_ru[n].sh" > /dev/null; do sleep 30; done
log "P-CAP's runner is gone. 2. waiting for BigMoE's 45 GB run to come and go (or 21:00 if it hasn't started)"
seen=0
while :; do
  if [ "$(nbigmoe)" -gt 0 ]; then
    [ $seen -eq 0 ] && log "BigMoE's quiet lock is on: $(others | grep '~/BigMoE' | sed 's/.*·//' | cut -c1-150)"
    seen=1
  elif [ $seen -eq 1 ]; then log "BigMoE's quiet lock released"; break
  elif [ "$(hm)" -ge 2100 ]; then log "no BigMoE quiet lock by 21:00: going on"; break; fi
  [ "$(hm)" -ge 2200 ] && { log "GIVE UP at 22:00 (BigMoE's run still on): P-OFF's extraction goes to Thursday"; exit 3; }
  sleep 30
done
log "3. waiting for a clear slot (>= 30 GB available; gives up 22:00)"
wait_clear 30 2200 || { log "GIVE UP at 22:00: P-OFF's extraction goes to Thursday"; exit 3; }
./turns/poff_x.sh > data/poff_x_run.log 2>&1
log "4. extraction runner rc=$?: $(tail -2 data/poff_x_run.log | tr '\n' ' ')"
rate=$(grep -oE "on-path rate \(assistant\) [0-9.]+" data/poff_x.log | awk '{print $NF}')
log "on-path rate (assistant): ${rate:-none}"
if [ -z "$rate" ]; then log "no rate line: extraction incomplete, no training"; exit 4; fi
if awk -v r="$rate" 'BEGIN{exit !(r < 0.75)}'; then log "on-path rate < 0.75: P-OFF = KILL by the pre-registered diagnostic (no training)"; exit 0; fi
log "on-path rate >= 0.75: P-OFF is alive; its training moves to Thursday (the orchestrator 20:0x)"
log "DONE"
