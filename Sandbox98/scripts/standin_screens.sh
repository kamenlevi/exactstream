#!/bin/bash
# The orchestrator's three learned-stand-in screens (20:2x), queued 20:33. Polite: never during another session's quiet
# lock or queued quiet ticket (checked before each screen and every 20 s during it; on one, this screen is killed and
# rerun from scratch after the quiet lock ends); not before 20:41 (Leap's 20:40 ticket); swap and memory checked first.
cd "$(dirname "$0")/.."
PY=$HOME/BigMoE/.venv/bin/python
NOT_BEFORE=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-14 20:41:00" +%s)
quiet_others() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (benchmarking|waiting for a quiet window)[^·]*" | grep -c . ; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
SWAP0=$(swap_mb)
ready() {
  [ "$(date +%s)" -ge "$NOT_BEFORE" ] || return 1
  [ -f $HOME/.local/share/quiet/PAUSE ] && return 1
  [ "$(quiet_others)" -eq 0 ] || return 1
  [ "$(noswap)" -ge 12 ] || return 1                                   # NO-SWAP >= peak 3.6 + 8 GB
  [ $(( $(swap_mb) - SWAP0 )) -lt 1500 ] || { echo "$(date +%H:%M:%S) swap grew > 1.5 GB since the start: stopping"; exit 0; }
  return 0
}
run_one() {   # NAME LEVER
  local tries=0
  while true; do
    until ready; do sleep 20; done
    sleep 60; ready || continue                                       # still clear a minute later
    echo "$(date +%H:%M:%S) start $1 (swap $(swap_mb) MB, NO-SWAP $(noswap) GB)"
    peer lock share 12m --cpu 200 --gpu 25 "Sandbox98: stand-in screen $1 (300 steps + 80 windows), MEM 3.6 GB measured peak" 2>&1 | head -1
    LINE=$(cat results/screens/standin_LINES.md | tr '\n' ' ' | cut -c1-900)
    $PY -I -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' ./screen.sh $1 $2 "$LINE" 300 50 30 > logs/screen_$1.log 2>&1 &
    local SP=$!
    local yielded=0
    while kill -0 $SP 2>/dev/null; do
      sleep 20
      if [ "$(quiet_others)" -gt 0 ]; then
        echo "$(date +%H:%M:%S) quiet ticket appeared: yield (kill screen $1)"; kill -TERM -- -$SP 2>/dev/null; sleep 5; kill -KILL -- -$SP 2>/dev/null
        yielded=1; break
      fi
    done
    wait $SP 2>/dev/null
    peer unlock 2>&1 | tail -1
    if [ $yielded = 0 ]; then
      grep -A 12 "== screen $1" results/SCREENS.log | tail -13
      echo "$(date +%H:%M:%S) done $1 (swap $(swap_mb) MB)"
      return 0
    fi
    tries=$((tries + 1)); [ $tries -ge 3 ] && { echo "$(date +%H:%M:%S) $1: yielded 3 times, giving up"; return 1; }
  done
}
run_one distill distill
run_one headfeat headfeat
run_one noisehead noisehead
echo "standin screens done"
