#!/bin/bash
# wait_map.sh + a start window (Breakthrough, Mon 11-02 23:3x): usage: wait_map2.sh "<names>" <read-out script> T_LAST
# [T_FIRST], both HH:MM; no start before T_FIRST or after T_LAST, and each rolls to tomorrow when already past, so a
# watcher armed in the evening can wait for the next day's slot (NewIdeas' GPU order). Everything else is wait_map.sh's:
# starts run_day9.sh with "gate <names>" after 3 clear checks in a row (60 s apart): NO-SWAP >= 25 GB; no other
# session's quiet window (held or waited for), the lid open, no A6 waiter, E2 not waiting for memory; no PAUSE, no
# soft_end ramp, no other session's MEM-tagged share lock over 5 GB (the day rule). The guard (every 30 s) stops the map
# for the hard reasons after 3 failing checks, at once for another MEM job over 5 GB, and after 5 min of PAUSE (a map
# paused in pausewait still spins a core in bigmoe_ext). --skip-done keeps finished names across restarts. When every
# name's out/<name>.json exists, it runs the read-out and exits. Run ATTACHED (Bash run_in_background).
cd ~/FableGod/decision_map || exit 1
LOG=out/runner.log
NAMES=$1; READ=$2
OUTS=$(for n in $NAMES; do echo out/$n.json; done)
Q="$HOME/.local/share/quiet"
NOW=$(date +%s); LAST=$(date -j -f "%H:%M" "${3:-21:30}" +%s); FIRST=${4:+$(date -j -f "%H:%M" "$4" +%s)}
[ -n "$FIRST" ] && [ "$FIRST" -lt "$NOW" ] && FIRST=$((FIRST + 86400))
# [T_STOP HH:MM] (5th arg): a running map is stopped then, whatever is left waits (NewIdeas Tue 01:4x: finish or pause
# by 05:45, MTP Head's chain starts at 06:00)
STOP=${5:+$(date -j -f "%H:%M" "$5" +%s)}; [ -n "$STOP" ] && [ "$STOP" -lt "$NOW" ] && STOP=$((STOP + 86400))
if [ "$LAST" -lt "$NOW" ] || { [ -n "$FIRST" ] && [ "$LAST" -lt "$FIRST" ]; }; then LAST=$((LAST + 86400)); fi
say() { echo "[dm-wm] $* $(date '+%a %H:%M:%S')" >> $LOG; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
quiet_others() { peer ls 2>/dev/null | grep -v "^You:" | grep -qE "quiet lock until|benchmarking: quiet lock|waiting for a quiet window"; }
desc() { local p; for p in $(pgrep -P $1); do echo $p; desc $p; done; }
e2_wants_mem() {
  tail -1 ~/TrainLab/logs/turns.log 2>/dev/null | grep -q "waiting for NO-SWAP" && return 0
  local p q up
  for p in $(pgrep -f "run_shared.sh .*turns/e2_"); do
    up=0; for q in $(desc $p); do ps -o command= -p $q 2>/dev/null | grep -q "Python" && { up=1; break; }; done
    [ $up -eq 0 ] && return 0
  done
  return 1
}
hard() {         # reasons to stop a running map (and not to start one)
  if quiet_others; then echo "a quiet window (others)"
  elif ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; then echo "lid closed"
  elif pgrep -f "turn45_a6.sh" >/dev/null; then echo "A6 first"
  elif e2_wants_mem; then echo "E2 waits for memory (first)"
  elif [ -n "$WAIT_EARLYTAP" ] && pgrep -f "turns/et_job.sh" >/dev/null; then echo "EARLYTAP runs (first)"   # orchestrator Tue 06:5x
  fi
}
big_others() {   # the day rule (pacer RULES.md, Ultra section, Mon 20:50): ONE job over ~5 GB at a time
  [ -n "$NO_DAYRULE" ] && return 1   # a slot granted beside others' share jobs (the orchestrator Tue 06:4x: BIG window)
  peer ls 2>/dev/null | grep -v "^You:" | grep -v "Breakthrough decision map" | grep -oE "shared run.*MEM [0-9]+ GB" | grep -oE "MEM [0-9]+" | awk '$2 >= 5 {f = 1} END {exit !f}'
}
soft() {         # reasons not to start
  if [ -e "$Q/PAUSE" ]; then echo "PAUSE (heat or a window)"
  elif pgrep -f "quiet/soft_end.sh" >/dev/null; then echo "soft_end ramp"
  elif big_others; then echo "another job over 5 GB (day rule)"
  elif [ -n "$ONE_GPU" ] && peer ls 2>/dev/null | grep -v "^You:" | grep -v "Breakthrough decision map" | grep -q "shared run"; then
    echo "another GPU job (one at a time, orchestrator Tue 09:11)"
  fi
}
why_not() { local w; w=$(hard); [ -z "$w" ] && w=$(soft); echo "$w"; }
say "watching (NO-SWAP $(noswap) GB; no start after $(date -r $LAST '+%a %H:%M')${FIRST:+, none before $(date -r $FIRST '+%a %H:%M')}): $NAMES"
if [ -n "$FIRST" ]; then while [ $(date +%s) -lt $FIRST ]; do sleep 60; done; say "start window open"; fi
ok=0; last=; fails=0
while ! ls $OUTS > /dev/null 2>&1; do
  [ $(date +%s) -ge $LAST ] && { say "gave up: no start after $(date -r $LAST '+%a %H:%M')"; exit 3; }
  [ -n "$STOP" ] && [ $(date +%s) -ge $STOP ] && { say "the slot ended at $(date -r $STOP '+%H:%M'); the rest waits"; exit 4; }
  w=$(why_not); f=$(noswap)
  [ -z "$w" ] && [ "$f" -lt 25 ] && w="NO-SWAP < 25 GB"
  if [ -n "$w" ]; then
    ok=0; [ "$w" != "$last" ] && say "waiting: $w (NO-SWAP $f GB)"; last=$w; sleep 60; continue
  fi
  ok=$((ok + 1)); last=
  [ $ok -lt 3 ] && { sleep 60; continue; }
  if ! peer lock share 120m --cpu 200 --gpu 50 "Breakthrough decision map: MEM 17 GB: $NAMES (flips, no timing)" >> $LOG 2>&1; then
    say "share lock not granted"; ok=0; sleep 60; continue
  fi
  say "go: NO-SWAP $f GB"
  DAY9_NAMES="gate $NAMES" bash run_day9.sh > /dev/null & R=$!
  bad=0; stopped=0; pz=0
  while kill -0 $R 2>/dev/null; do
    sleep 30
    w=$(hard)
    if [ -e "$Q/PAUSE" ]; then pz=$((pz + 1)); else pz=0; fi
    if [ -z "$w" ] && [ $pz -ge 10 ]; then w="PAUSE over 5 min (a paused map spins a core)"; bad=2; fi
    if [ -z "$w" ] && big_others; then w="another job over 5 GB (day rule)"; bad=2; fi
    if [ -n "$STOP" ] && [ $(date +%s) -ge $STOP ]; then w="the slot ends ($(date -r $STOP '+%H:%M'))"; bad=2; fi
    if [ -n "$w" ]; then bad=$((bad + 1)); else bad=0; fi
    if [ $bad -ge 3 ]; then
      P=$(pgrep -P $R -f run_map.py | head -1)
      [ -n "$P" ] && { say "stopping the map for: $w (pid $P)"; kill -TERM $P; stopped=1; }
      bad=0
    fi
  done
  wait $R; ok=0
  if [ $stopped -eq 0 ] && ! ls $OUTS > /dev/null 2>&1; then
    fails=$((fails + 1)); say "run ended without all names and not stopped by the guard ($fails)"
    [ $fails -ge 2 ] && { say "giving up after 2 failed runs (see the log above)"; exit 5; }
  fi
done
say "all names written; read-out $READ"
~/BigMoE/.venv/bin/python $READ > out/$(basename $READ .py).log 2>&1
say "read-out rc $?"
