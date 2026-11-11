#!/bin/bash
# One-gap map runner for extra arms (Breakthrough, Wed 2026-11-04 04:5x; screens_slot.sh's rules, the arms as arguments).
# Run ATTACHED (Bash run_in_background):
#   [INSTALL=<file of dm_ablations.py lines>] [READOUT="<shell commands, cwd decision_map>"] [S_LAST=HH:MM]
#   [WAIT_PID=<pid>: no start while it lives, e.g. FLOOR night's runner] [PRE="<command run once in the gap with no map
#   process, after INSTALL: 0 = go, 2 = a map process appeared (wait again), else not run>"] (both Wed 06:3x)
#   bash map_slot.sh <arm> [<arm> ...]
# All arms run in ONE map process (one model load), 36 windows (--max 12), --skip-done. It takes the next gap: 10-s
# checks, 3 clear in a row (FLOOR needs 3 clear checks 60 s apart, then waits with "a map process is running"). Clear =
# no run_map.py; no runner of mine mid-run (the last line of FLOOR's / screens_slot's / this log is not a "go"; a runner
# unlocks up to 15 s after its map process ends); no other session's quiet window held or waited for; lid open; no PAUSE;
# NO-SWAP >= 25 GB. INSTALL lines go into dm_ablations.py in one step: import-tested on a scratch copy at arming, moved in
# (atomic mv) in the gap with no map process (skipped if its first line is already there). Share lock MEM 17 GB; guards
# every 15 s (swap +2 GB, quiet windows 3 strikes, lid, PAUSE over 5 min); up to 6 tries; no start after S_LAST (default
# 09:00: FLOOR's D + B + C + R must still end by 13:00), hard stop 12:45.
DM=~/FableGod/decision_map; PY=~/BigMoE/.venv/bin/python; LOG=~/Breakthrough/data/map_slot.log; Q=$HOME/.local/share/quiet
FL=~/Breakthrough/data/floor1/night.log; SL=~/Breakthrough/data/screens_slot.log; SP=~/Breakthrough/data/abl_scratch
ARMS="$*"; [ -z "$ARMS" ] && { echo "usage: map_slot.sh <arm> ..."; exit 2; }
LASTGO=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-04 ${S_LAST:-09:00}" +%s)
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-04 12:45:00" +%s)
say() { echo "[slot] $* $(date '+%a %H:%M:%S')" >> $LOG; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
quiet_others() { peer ls 2>/dev/null | grep -v "^You:" | grep -qE "quiet lock until|benchmarking: quiet lock|waiting for a quiet window"; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
kill_tree() { local p=$1 c; for c in $(pgrep -P $p); do kill_tree $c; done; kill -TERM $p 2>/dev/null; }
mid_run() {   # $1 log, $2 tag: its last tagged line is a running step
  [ -f $1 ] && grep -E "^\[$2\]" $1 | tail -1 | grep -qE " go \(|stopping"; }
why_not() {
  if [ -n "$WAIT_PID" ] && kill -0 $WAIT_PID 2>/dev/null; then echo "pid $WAIT_PID is still running"
  elif pgrep -f "run_map.py" > /dev/null; then echo "a map process is running"
  elif mid_run $FL floor || mid_run $SL screens || mid_run $LOG slot; then echo "a runner of mine is still closing"
  elif quiet_others; then echo "a quiet window (others)"; elif lid_closed; then echo "lid closed"
  elif [ -e "$Q/PAUSE" ]; then echo "PAUSE"; elif [ "$(noswap)" -lt 25 ]; then echo "NO-SWAP < 25 GB"; fi
}
prep() {   # scratch copy = today's dm_ablations.py + INSTALL's lines; import test from the scratch folder
  mkdir -p $SP/test; cp $DM/dm_ablations.py $SP/dm_ablations.py.slot
  grep -qF "$(head -1 $INSTALL)" $SP/dm_ablations.py.slot || cat $INSTALL >> $SP/dm_ablations.py.slot
  cp $SP/dm_ablations.py.slot $SP/test/dm_ablations.py
  (cd $SP/test && PYTHONPATH=$DM $PY -c "import sys, dm_ablations as A; assert A.__file__.startswith('$SP'), A.__file__; m = [a for a in sys.argv[1:] if a not in A.ABLATIONS]; assert not m, m; print('import OK')" $ARMS qom_0_live hyb_cold4_len_t3) >> $LOG 2>&1
}
install() {   # 0 = installed or nothing to install, 1 = failed, 2 = a map process appeared
  [ -z "$INSTALL" ] && return 0
  grep -qF "$(head -1 $INSTALL)" $DM/dm_ablations.py && return 0
  cmp -s <(head -c $(wc -c < $DM/dm_ablations.py) $SP/dm_ablations.py.slot) $DM/dm_ablations.py \
    || { say "install: dm_ablations.py changed since the import test; testing again"; prep || return 1; }
  pgrep -f run_map.py > /dev/null && return 2
  cp $SP/dm_ablations.py.slot $DM/.dm_ablations.py.new && mv $DM/.dm_ablations.py.new $DM/dm_ablations.py \
    && { say "install: $(wc -l < $INSTALL | tr -d ' ') lines into dm_ablations.py"; return 0; } || return 1
}
nwin() { [ -f $DM/out/$1.json ] && $PY -c "import json; d = json.load(open('$DM/out/$1.json')); print(sum(1 for v in d.values() if isinstance(v, dict) and 'tk' in v))" 2>/dev/null || echo 0; }
todo() { local a; for a in $ARMS; do [ "$(nwin $a)" = 36 ] || echo -n "$a "; done; }
say "armed: $ARMS in the next gap; no start after ${S_LAST:-09:00}"
if [ -n "$INSTALL" ]; then prep || { say "the import test failed; dm_ablations.py unchanged; not run"; exit 1; }
else (cd $DM && $PY -c "import sys, dm_ablations as A; m = [a for a in sys.argv[1:] if a not in A.ABLATIONS]; assert not m, m" $ARMS) >> $LOG 2>&1 || { say "unknown names; not run"; exit 1; }; fi
cd $DM || exit 1
tries=0
while :; do
  tries=$((tries + 1)); [ $tries -gt 6 ] && { say "6 tries used; left: $(todo)"; break; }
  ok=0; last=
  while :; do
    [ $(date +%s) -ge $LASTGO ] && { say "no start by ${S_LAST:-09:00}; left: $(todo)"; break 2; }
    w=$(why_not)
    if [ -n "$w" ]; then ok=0; [ "$w" != "$last" ] && say "waiting: $w"; last=$w; sleep 10; continue; fi
    ok=$((ok + 1)); last=; [ $ok -ge 3 ] && break; sleep 10
  done
  names=$(todo); [ -z "$names" ] && break
  install; rc=$?
  [ $rc -eq 1 ] && { say "install failed; dm_ablations.py unchanged; not run"; exit 1; }
  [ $rc -eq 2 ] && { tries=$((tries - 1)); continue; }
  if [ -n "$PRE" ]; then
    (cd $DM && eval "$PRE") >> $LOG 2>&1; rc=$?
    [ $rc -eq 2 ] && { tries=$((tries - 1)); continue; }
    [ $rc -ne 0 ] && { say "PRE failed (rc $rc); not run"; exit 1; }
    say "PRE done"; PRE=
  fi
  peer lock share 40m --cpu 200 --gpu 50 "Breakthrough map arms (the orchestrator): MEM 17 GB (flips, no timing): $names" >> $LOG 2>&1 \
    || { say "share lock not granted"; sleep 30; continue; }
  S0=$(swap_mb); say "go (try $tries: $names; NO-SWAP $(noswap) GB, swap $S0 MB)"
  $PY run_map.py $names --max 12 --skip-done >> $LOG 2>&1 & P=$!
  bad=0; pz=0
  while kill -0 $P 2>/dev/null; do
    sleep 15; w=
    [ $(date +%s) -ge $DEADLINE ] && { say "12:45, stopping the tree"; kill_tree $P; break; }
    [ $(( $(swap_mb) - S0 )) -gt 2048 ] && { say "swap grew > 2 GB, stopping the tree"; kill_tree $P; break; }
    if [ -e "$Q/PAUSE" ]; then pz=$((pz + 1)); else pz=0; fi
    if quiet_others; then w="a quiet window (others)"; elif lid_closed; then w="lid closed"; elif [ $pz -ge 20 ]; then w="PAUSE over 5 min"; fi
    if [ -n "$w" ]; then bad=$((bad + 1)); else bad=0; fi
    [ $bad -ge 3 ] && { say "stopping for: $w"; kill_tree $P; break; }
  done
  wait $P 2>/dev/null; rc=$?; peer unlock >> $LOG 2>&1; say "run ended (rc $rc); left: $(todo); unlocked"
  [ -z "$(todo)" ] && break
  [ $(date +%s) -ge $DEADLINE ] && break
done
[ -n "$READOUT" ] && { (eval "$READOUT") >> $LOG 2>&1; say "read-out: $(eval "$READOUT" 2>/dev/null | grep '^VERDICT' | head -3)"; }
say "done; left: $(todo)"
