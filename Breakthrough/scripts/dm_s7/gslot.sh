#!/bin/bash
# A guarded run of one command on the shared GPU (Breakthrough, Thu 2026-11-05): the s67 map runner's guards for a command that
# is not a map arm (the scores capture, its smoke). Clear = no map process, no other session's quiet window held or waited
# for, no other session's shared run declaring MEM > 5 GB, lid open, no PAUSE, NO-SWAP >= 25 GB: 3 clear checks 10 s apart.
# Share lock "MEM <mem> GB". Every 15 s: the hard stop, swap +2 GB, a quiet window / lid closed (3 strikes), a non-fanwatch
# PAUSE over 5 min stop the run (SIGKILL after 5 s if frozen; never SIGCONT). cwd ~/Breakthrough (fanwatch freezes it with
# the other research jobs). Done = the command exits 0; else it is retried (3 runs at most) while it can still start.
# Usage: gslot.sh <name> <est min> <mem GB> <command ...>; env GS_LASTGO / GS_DEADLINE "YYYY-mm-dd HH:MM:SS" (the last
# start, the hard stop; default now + 1 h / + 2 h). Log: ~/Breakthrough/data/gslot/<name>.log. Exit 0 = done, 1 = not done.
NAME=$1 EST=$2 MEM=$3; shift 3; [ $# -ge 1 ] || { echo "usage: $0 <name> <est min> <mem GB> <command ...>"; exit 1; }
LOG=~/Breakthrough/data/gslot/$NAME.log; mkdir -p ~/Breakthrough/data/gslot; Q=$HOME/.local/share/quiet
ts() { date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s; }
LASTGO=$(ts "${GS_LASTGO:-$(date -v+1H '+%Y-%m-%d %H:%M:%S')}"); DEADLINE=$(ts "${GS_DEADLINE:-$(date -v+2H '+%Y-%m-%d %H:%M:%S')}")
say() { echo "[gslot $NAME] $* $(date '+%a %H:%M:%S')" >> $LOG; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
quiet_others() { peer ls 2>/dev/null | grep -v "^You:" | grep -qE "quiet lock until|benchmarking: quiet lock|waiting for a quiet window"; }
big_others() { peer ls 2>/dev/null | grep -v "^You:" | grep "shared run" | grep -oE "MEM [0-9]+ GB" | awk '$2 > 5 {f = 1} END {exit !f}'; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
kill_tree() { local p=$1 c; for c in $(pgrep -P $p); do kill_tree $c; done; kill -TERM $p 2>/dev/null; }
stop_run() { kill_tree $1; sleep 5; ps -o stat= -p $1 2>/dev/null | grep -q T && kill -KILL $1 2>/dev/null; }
why_not() {
  if pgrep -f "run_ma[p].py" > /dev/null; then echo "a map process is running"
  elif quiet_others; then echo "a quiet window (others)"; elif big_others; then echo "another session's shared run > 5 GB"
  elif lid_closed; then echo "lid closed"
  elif [ -e "$Q/PAUSE" ]; then echo "PAUSE"; elif [ "$(noswap)" -lt 25 ]; then echo "NO-SWAP < 25 GB"; fi
}
say "armed: $*; est $EST min, MEM $MEM GB; last start $(date -r $LASTGO '+%a %H:%M'), hard stop $(date -r $DEADLINE '+%a %H:%M')"
need=3; last=; runs=0
while :; do
  [ $(date +%s) -ge $LASTGO ] && { say "last start passed; not done"; exit 1; }
  [ $(( $(date +%s) + EST * 60 )) -le $DEADLINE ] || { say "no longer fits by the hard stop; not done"; exit 1; }
  w=$(why_not)
  if [ -n "$w" ]; then need=3; [ "$w" != "$last" ] && say "waiting: $w"; last=$w; sleep 10; continue; fi
  need=$((need - 1)); [ $need -gt 0 ] && { sleep 10; continue; }
  last=; runs=$((runs + 1)); [ $runs -gt 3 ] && { say "3 runs used; not done"; exit 1; }
  peer lock share $(( 3 * EST + 10 ))m --cpu 200 --gpu 100 "Breakthrough $NAME: MEM $MEM GB" >> $LOG 2>&1 || { say "share lock not granted"; need=3; sleep 30; continue; }
  S0=$(swap_mb); say "go (NO-SWAP $(noswap) GB, swap $S0 MB)"
  (cd ~/Breakthrough && exec "$@") >> $LOG 2>&1 & P=$!
  bad=0; pz=0; stopped=
  while kill -0 $P 2>/dev/null; do
    sleep 15; w=
    [ $(date +%s) -ge $DEADLINE ] && { stopped="the hard stop"; stop_run $P; break; }
    [ $(( $(swap_mb) - S0 )) -gt 2048 ] && { stopped="swap grew > 2 GB"; stop_run $P; break; }
    if [ -e "$Q/PAUSE" ] && [ ! -e "$Q/PAUSE.fanwatch" ]; then pz=$((pz + 1)); else pz=0; fi
    if quiet_others; then w="a quiet window (others)"; elif lid_closed; then w="lid closed"; elif [ $pz -ge 20 ]; then w="PAUSE over 5 min"; fi
    if [ -n "$w" ]; then bad=$((bad + 1)); else bad=0; fi
    [ $bad -ge 3 ] && { stopped="$w"; stop_run $P; break; }
  done
  wait $P 2>/dev/null; rc=$?; peer unlock >> $LOG 2>&1
  if [ -z "$stopped" ] && [ $rc = 0 ]; then say "done (rc 0); unlocked"; exit 0; fi
  need=3; say "not done (rc $rc${stopped:+, stopped for: $stopped}); unlocked"
  [ $(date +%s) -ge $DEADLINE ] && exit 1
done
