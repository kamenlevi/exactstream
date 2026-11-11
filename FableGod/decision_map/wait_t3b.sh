#!/bin/bash
# M-18 3-bit sure-row arm, second try (Breakthrough, Mon 11-02 13:5x). The first watcher (wait_t3.sh) read "E2's step is
# running" from its shell script alone and started 15 s after E2's capture took its lock, before E2's Python had loaded:
# the map then held the memory E2's capture was checking for (NO-SWAP 21 < 25). Stopped at 13:48:56 (rc 143); E2 started
# at C 96 at 13:49. The orchestrator's gap order (13:4x): E2 and BigMoE's A6 first, the 3-bit arm only beside them if
# memory allows. This one starts only when all of these hold for 3 checks in a row (60 s apart):
#  (a) NO-SWAP >= 26 GB (free + speculative + purgeable + file-backed, the measure E2 and A6 gate on; map peak ~17 + 8);
#  (b) no other session holds or waits for a quiet window; no PAUSE, the lid open, no soft_end ramp;
#  (c) BigMoE's A6 waiter (turn45_a6.sh) has exited;
#  (d) E2 has its memory: its chain has ended, or one of its programs (capture / train / walk / read) runs under it.
# While the map runs, a guard checks (b)-(d) every 30 s and stops it after 3 failing checks in a row (TERM to run_map.py;
# run_day9.sh then unlocks), and the wait starts again; run_map.py --skip-done keeps the arms already written. (a) is not
# checked under the run (the map's own pages count against it); run_day9.sh's swap guard (+2 GB) is.
# Run ATTACHED (Bash run_in_background). No new start after T3_LAST (default 21:30); the night belongs to others.
cd ~/FableGod/decision_map || exit 1
LOG=out/runner.log
NAMES="gate gate_top2t3_cf099 gate_top2t4_cf099 gate_top2t3_cf0999"
OUTS="out/gate_top2t3_cf099.json out/gate_top2t4_cf099.json out/gate_top2t3_cf0999.json"
Q="$HOME/.local/share/quiet"
LAST=$(date -j -f "%H:%M" "${T3_LAST:-21:30}" +%s)
say() { echo "[dm-t3b] $* $(date '+%a %H:%M:%S')" >> $LOG; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
quiet_others() { peer ls 2>/dev/null | grep -v "^You:" | grep -qE "quiet lock until|benchmarking: quiet lock|waiting for a quiet window"; }
blocked() { [ -e "$Q/PAUSE" ] || ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes' || pgrep -f "quiet/soft_end.sh" >/dev/null; }
desc() { local p; for p in $(pgrep -P $1); do echo $p; desc $p; done; }
e2_prog_up() {   # one of E2's programs runs under its chain
  local c p; c=$(pgrep -f "turns/e2_chain" | head -1); [ -z "$c" ] && return 1
  for p in $(desc $c); do ps -o command= -p $p 2>/dev/null | grep -qE "dlog_capture.py|d09_train.py|d02b_walk.py|e2_read.py" && return 0; done
  return 1
}
why_not() {      # empty = clear to run beside the others
  if quiet_others; then echo "a quiet window (others)"
  elif blocked; then echo "PAUSE / lid / soft_end"
  elif pgrep -f "turn45_a6.sh" >/dev/null; then echo "A6 first"
  elif pgrep -f "turns/e2_chain" >/dev/null && ! e2_prog_up; then echo "E2 between steps or waiting (first)"
  fi
}
say "watching (NO-SWAP $(noswap) GB; no start after $(date -r $LAST '+%H:%M')): $NAMES"
ok=0; last=; fails=0
while ! ls $OUTS > /dev/null 2>&1; do
  [ $(date +%s) -ge $LAST ] && { say "gave up: no start after ${T3_LAST:-21:30}"; exit 3; }
  w=$(why_not); f=$(noswap)
  [ -z "$w" ] && [ "$f" -lt 26 ] && w="NO-SWAP < 26 GB"
  if [ -n "$w" ]; then
    ok=0; [ "$w" != "$last" ] && say "waiting: $w (NO-SWAP $f GB)"; last=$w; sleep 60; continue
  fi
  ok=$((ok + 1)); last=
  [ $ok -lt 3 ] && { sleep 60; continue; }
  if ! peer lock share 60m --cpu 200 --gpu 50 "Breakthrough M-18 3-bit sure-row arm: decision map flips (~17 GB, no timing)" >> $LOG 2>&1; then
    say "share lock not granted"; ok=0; sleep 60; continue
  fi
  say "go: NO-SWAP $f GB; E2 program up: $(e2_prog_up && echo yes || echo no)"
  DAY9_NAMES="$NAMES" bash run_day9.sh > /dev/null & R=$!
  bad=0; stopped=0
  while kill -0 $R 2>/dev/null; do
    sleep 30
    w=$(why_not)
    if [ -n "$w" ]; then bad=$((bad + 1)); else bad=0; fi
    if [ $bad -ge 3 ]; then
      P=$(pgrep -P $R -f run_map.py | head -1)
      [ -n "$P" ] && { say "stopping the map for: $w (pid $P)"; kill -TERM $P; stopped=1; }
      bad=0
    fi
  done
  wait $R; ok=0
  if [ $stopped -eq 0 ] && ! ls $OUTS > /dev/null 2>&1; then
    fails=$((fails + 1)); say "run ended without all arms and not stopped by the guard ($fails)"
    [ $fails -ge 2 ] && { say "giving up after 2 failed runs (see the log above)"; exit 5; }
  fi
done
say "all three arms written; read-out"
~/BigMoE/.venv/bin/python dm_day9_read.py > out/DAY9_read.log 2>&1
say "read-out rc $? (out/DAY9_read.txt)"
