#!/bin/bash
# Map runs beside the day's first-in-line jobs (Breakthrough, Mon 11-02 18:2x; replaces wait_t3b.sh, whose E2 check
# looked for E2's programs under its CURRENT chain only: MTP Head swapped chain7 for chain8 at 17:40 while x4 trained on,
# so x4's program had no chain ancestor and the watcher read "E2 waiting" for hours). The orchestrator's order: E2 and
# A6 first, my map runs only beside them if memory allows; the two-pass check (mission 20) is first on my own list.
# usage: wait_map.sh "<names>" <read-out script> [T_LAST HH:MM, default 21:30]. Run ATTACHED (Bash run_in_background).
# Starts run_day9.sh with "gate <names>" after 3 clear checks in a row (60 s apart):
#  (a) NO-SWAP >= 25 GB (free + speculative + purgeable + file-backed, the measure E2 and A6 gate on; map peak ~17 + 8);
#  (b) no other session holds or waits for a quiet window; no PAUSE, the lid open, no soft_end ramp;
#  (c) BigMoE's A6 waiter (turn45_a6.sh) gone;
#  (d) E2 not waiting for memory: its log's last line is no memory wait, and every E2 step under run_shared.sh has its
#      Python up (a step that has taken its lock but not loaded yet is the race of 13:45).
# A guard re-checks every 30 s under the run and stops the map after 3 failing checks in a row (TERM to run_map.py;
# run_day9.sh unlocks) for the hard reasons only: another session's quiet window, the lid closed, A6, E2 waiting for
# memory. PAUSE (fanwatch's heat pause; 19:17 Mon stopped the first try 2 min in) and a soft_end ramp only hold the start:
# a running map waits them out at its next pass (dm_common imports BigMoE's pausewait) and fanwatch freezes it if still hot.
# The wait then starts again and --skip-done keeps the finished names. When every name's out/<name>.json exists, it runs
# the read-out and exits (the task notification is the report cue).
cd ~/FableGod/decision_map || exit 1
LOG=out/runner.log
NAMES=$1; READ=$2
OUTS=$(for n in $NAMES; do echo out/$n.json; done)
Q="$HOME/.local/share/quiet"
LAST=$(date -j -f "%H:%M" "${3:-21:30}" +%s)
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
  fi
}
big_others() {   # the day rule (pacer RULES.md, Ultra section, Mon 20:50): ONE job over ~5 GB at a time across all
  # sessions; such jobs say "MEM <n> GB" in their share lock's reason (peer ls). Later ones queue behind a running map.
  peer ls 2>/dev/null | grep -v "^You:" | grep -v "Breakthrough decision map" | grep -oE "shared run.*MEM [0-9]+ GB" | grep -oE "MEM [0-9]+" | awk '$2 >= 5 {f = 1} END {exit !f}'
}
soft() {         # reasons not to start; a running map waits them out (pausewait holds each pass while PAUSE exists)
  if [ -e "$Q/PAUSE" ]; then echo "PAUSE (heat or a window)"
  elif pgrep -f "quiet/soft_end.sh" >/dev/null; then echo "soft_end ramp"
  elif big_others; then echo "another job over 5 GB (day rule)"
  fi
}
why_not() { local w; w=$(hard); [ -z "$w" ] && w=$(soft); echo "$w"; }   # empty = clear to start beside the others
say "watching (NO-SWAP $(noswap) GB; no start after $(date -r $LAST '+%H:%M')): $NAMES"
ok=0; last=; fails=0
while ! ls $OUTS > /dev/null 2>&1; do
  [ $(date +%s) -ge $LAST ] && { say "gave up: no start after ${3:-21:30}"; exit 3; }
  w=$(why_not); f=$(noswap)
  [ -z "$w" ] && [ "$f" -lt 25 ] && w="NO-SWAP < 25 GB"
  if [ -n "$w" ]; then
    ok=0; [ "$w" != "$last" ] && say "waiting: $w (NO-SWAP $f GB)"; last=$w; sleep 60; continue
  fi
  ok=$((ok + 1)); last=
  [ $ok -lt 3 ] && { sleep 60; continue; }
  if ! peer lock share 60m --cpu 200 --gpu 50 "Breakthrough decision map: MEM 17 GB: $NAMES (flips, no timing)" >> $LOG 2>&1; then
    say "share lock not granted"; ok=0; sleep 60; continue
  fi
  say "go: NO-SWAP $f GB"
  DAY9_NAMES="gate $NAMES" bash run_day9.sh > /dev/null & R=$!
  bad=0; stopped=0; pz=0
  while kill -0 $R 2>/dev/null; do
    sleep 30
    w=$(hard)
    # a heat PAUSE over 5 min stops the map too (Mon 22:3x: a map paused in pausewait still spins one core in
    # bigmoe_ext, so it heats a Mac that is cooling; short pauses are ridden out as before)
    if [ -e "$Q/PAUSE" ]; then pz=$((pz + 1)); else pz=0; fi
    if [ -z "$w" ] && [ $pz -ge 10 ]; then w="PAUSE over 5 min (a paused map spins a core)"; bad=2; fi
    if [ -z "$w" ] && big_others; then w="another job over 5 GB (day rule)"; bad=2; fi   # stop at once (orchestrator 21:3x)
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
