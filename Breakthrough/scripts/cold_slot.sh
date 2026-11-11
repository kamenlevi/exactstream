#!/bin/bash
# COLD-CUT screen (Breakthrough, Tue 2026-11-03; Fable C's ~/FableGod/fablec/UNION_STACK.md section 5; the orchestrator
# 18:5x: "run small breakthrough tests now ... start after ~19:40, when the three quiet tests are done; CPU jobs block quiet
# grants"; pre-registered in SCOREBOARD). Run ATTACHED (Bash run_in_background): it takes its own lock.
#   M  after 19:40 and once no other session holds or waits for a quiet window: install the two arms in dm_ablations.py
#      (scratch copy, import test, atomic mv), self-check the mask code against dm_q1u_masks.py (u8_t3 at live lengths, bit
#      for bit; those stay as the CONTROL's masks), build the masks (1 core, ~2 min) and check their bytes (<= x0.55 of
#      EXACT's on code, else stop). The chat form cold6_all's masks are built too, for a follow-up only if COLD PASSes.
#   R  the flips need the 125B map run (one process: hyb_cold4_t3 + the control hyb_0.9_u8_live_t3 = the arm it would
#      replace at the same live checks, ~295 s each, MEM 17 GB, GPU share 50), so
#      it waits for the memory as the QOM arms do: NO-SWAP >= 25 GB, no other session's quiet window or job over 5 GB (day
#      rule), lid open, no PAUSE, no map process; 3 clear checks 20 s apart. No start after 23:45 (tonight's chain starts
#      00:30); guard every 15 s as wed_day.sh, stop at 00:05 at the latest.
#   then dm_cold_read.py -> out/COLD_read.txt.
DM=~/FableGod/decision_map; PY=~/BigMoE/.venv/bin/python; LOG=~/Breakthrough/data/cold_slot.log; Q=$HOME/.local/share/quiet
SP=/private/tmp/claude-501/-Users-kamen-OjoX/2c7bb7fe-3ba1-4c03-91ba-f6be33886b82/scratchpad
FIRST=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-03 19:40:00" +%s)
LASTGO=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-03 23:45:00" +%s)
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-04 00:05:00" +%s)
cd $DM || exit 1
say() { echo "[cold] $* $(date '+%a %H:%M:%S')" >> $LOG; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
quiet_others() { peer ls 2>/dev/null | grep -v "^You:" | grep -qE "quiet lock until|benchmarking: quiet lock|waiting for a quiet window"; }
big_others() { peer ls 2>/dev/null | grep -v "^You:" | grep -oE "shared run.*MEM [0-9]+ GB" | grep -oE "MEM [0-9]+" | awk '$2 >= 5 {f = 1} END {exit !f}'; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
why_not() {   # $1 = mem: the map run's extra gates
  if quiet_others; then echo "a quiet window (others)"; elif lid_closed; then echo "lid closed"
  elif [ -e "$Q/PAUSE" ]; then echo "PAUSE"
  elif [ "$1" = mem ] && pgrep -f "run_map.py" > /dev/null; then echo "a map process is running"
  elif [ "$1" = mem ] && big_others; then echo "another job over 5 GB (day rule)"
  elif [ "$1" = mem ] && [ "$(noswap)" -lt 25 ]; then echo "NO-SWAP < 25 GB"; fi
}
wait_clear() {   # $1 = cpu | mem, $2 = the last start time; 0 = clear, 1 = past it
  local ok=0 last= w
  while [ $(date +%s) -lt $FIRST ]; do sleep 30; done
  while :; do
    [ $(date +%s) -ge $2 ] && return 1
    w=$(why_not $1)
    if [ -n "$w" ]; then ok=0; [ "$w" != "$last" ] && say "waiting ($1): $w"; last=$w; sleep 20; continue; fi
    ok=$((ok + 1)); last=; [ $ok -ge 3 ] && return 0; sleep 20
  done
}
kill_tree() { local p=$1 c; for c in $(pgrep -P $p); do kill_tree $c; done; kill -TERM $p 2>/dev/null; }
say "armed: masks after 19:40 (no quiet window), the map run when NO-SWAP >= 25 GB, no start after 23:45"
if [ "$SKIP_M" != 1 ]; then   # SKIP_M=1 (Tue 19:4x re-arm): the masks are built and checked
wait_clear cpu $LASTGO || { say "M: no clear moment by 23:45"; exit 1; }
say "M go"
if ! grep -q '"hyb_cold4_t3"' dm_ablations.py; then          # install: scratch copy, import test, atomic mv
  mkdir -p $SP/abl_test && cp $SP/dm_ablations.py.cold $SP/abl_test/dm_ablations.py || { say "M: scratch copy missing"; exit 1; }
  if (cd $SP/abl_test && PYTHONPATH=$DM $PY -c "import dm_ablations as A; assert A.__file__.startswith('$SP'), A.__file__; assert all(a in A.ABLATIONS for a in ('hyb_cold4_t3', 'cold6_all', 'hyb_0.9_u8_live_t3')); print('import OK', A.__file__)") >> $LOG 2>&1; then
    pgrep -f run_map.py > /dev/null && { say "M: a map process is running; dm_ablations.py not touched"; exit 1; }
    cp $SP/dm_ablations.py.cold $DM/.dm_ablations.py.new && mv $DM/.dm_ablations.py.new $DM/dm_ablations.py \
      && say "M: dm_ablations.py installed (+ hyb_cold4_t3, cold6_all, hyb_0.9_u8_live_t3)" || { say "M: install failed"; exit 1; }
  else say "M: the import test failed; dm_ablations.py unchanged"; exit 1; fi
fi
out=$($PY dm_cold_masks.py selfcheck 2>&1); echo "$out" >> $LOG
echo "$out" | grep -q "(live): ALL EQUAL (36)" || { say "M: the self-check failed; stopping"; exit 1; }
out=$($PY dm_cold_masks.py hyb_cold4_t3 cold6_all 2>&1); echo "$out" >> $LOG
echo "$out" | grep -qE "hyb_cold4_t3 x[0-9.]+ -> OK$" || { say "M: the bytes check failed (or the build did); no map run"; exit 1; }
say "M done: masks built, bytes OK"
fi
ARMS=${ARMS:-hyb_cold4_t3 hyb_0.9_u8_live_t3}   # ARMS (Tue 19:5x): the follow-up runs hyb_cold4_lowo_t3 cold6_all
tries=0                         # R retries (Tue 19:4x: a 45 s quiet blip stopped the first try): up to 3 tries by 23:45
while :; do
  todo=; for n in $ARMS; do [ -f $DM/out/$n.json ] || todo="$todo $n"; done
  [ -z "$todo" ] && break
  tries=$((tries + 1)); [ $tries -gt 3 ] && { say "R: 3 tries used; the 00:30 prefix runs$todo"; break; }
  wait_clear mem $LASTGO || { say "R: no memory window by 23:45 (NO-SWAP $(noswap) GB); the 00:30 prefix runs$todo"; break; }
  peer lock share 20m --cpu 200 --gpu 50 "Breakthrough COLD-CUT screen: MEM 17 GB: 2 map arms (flips, no timing), ~10 min" >> $LOG 2>&1 \
    || { say "R: share lock not granted"; sleep 60; continue; }
  S0=$(swap_mb); say "R go (try $tries:$todo; NO-SWAP $(noswap) GB, swap $S0 MB)"
  $PY run_map.py $todo --max 12 --skip-done >> $LOG 2>&1 & P=$!
  bad=0; pz=0
  while kill -0 $P 2>/dev/null; do
    sleep 15; w=
    [ $(date +%s) -ge $DEADLINE ] && { say "R: 00:05, stopping the tree"; kill_tree $P; break; }
    [ $(( $(swap_mb) - S0 )) -gt 2048 ] && { say "R: swap grew > 2 GB, stopping the tree"; kill_tree $P; break; }
    big_others && { say "R: another job over 5 GB (day rule), stopping the tree"; kill_tree $P; break; }
    if [ -e "$Q/PAUSE" ]; then pz=$((pz + 1)); else pz=0; fi
    if quiet_others; then w="a quiet window (others)"; elif lid_closed; then w="lid closed"; elif [ $pz -ge 20 ]; then w="PAUSE over 5 min"; fi
    if [ -n "$w" ]; then bad=$((bad + 1)); else bad=0; fi
    [ $bad -ge 3 ] && { say "R: stopping for: $w"; kill_tree $P; break; }
  done
  wait $P 2>/dev/null; rc=$?; peer unlock >> $LOG 2>&1; say "R: ended (rc $rc); unlocked"
done
$PY dm_cold_read.py >> $LOG 2>&1; say "read-out: $(grep -E '^(VERDICT|FOLLOW-UP|COST)' $DM/out/COLD_read.txt 2>/dev/null | tr '\n' ' ')"
say "cold slot done"
