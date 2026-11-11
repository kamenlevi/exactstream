#!/bin/bash
# Night prefix Wed 2026-11-04 00:30 (Breakthrough; the orchestrator Tue 19:1x: "Yes to a 00:30 prefix tonight, ahead of
# BigMoE's turns: COLD (hyb_cold4_t3 + the u8+t3 control) + NewIdeas' three QOM arms + the 3-min layout check, one share
# lock, ~25 min. It needs your >= 25 GB NO-SWAP gate at 00:30. No start after 00:40; if it can't start, everything stays
# for Wed 15:00 as planned."). Run ATTACHED (Bash run_in_background): it takes its own lock.
# One share lock (MEM 17 GB) and three map processes back to back, so FLOOR's runner (which waits for any map process)
# finds no gap to start in:
#   P   hyb_cold4_t3 hyb_0.9_u8_live_t3 (COLD + its control; only if cold_slot.sh built and registered them, and not
#       already run by it) then qom_0.9_live qom_0.5_live qom_0_live (NewIdeas, QUEUE 554dfe0); decision_map/out (~21 min)
#   Lc / Lt  the records3 layout check as wed_day.sh: baseline + t3_top2_all on 3 windows; ctrl = today's ext + records3,
#       test = BIGMOE_EXT_BUILD=build_nat + DM_R3=r3_overlay; dm_layout_check.py -> out_layout/LAYOUT_check.txt (~3 min)
# Start 00:30-00:40 only, 3 clear checks 20 s apart: no map process, NO-SWAP >= 25 GB, lid open, no PAUSE, no other
# session's quiet window held or waited for (BigMoE's waiting ticket excepted: the prefix goes ahead of its turns).
# Guard every 15 s: 01:15 or swap growth > 2 GB stop the tree at once; another session's held quiet window, a waiting
# ticket other than BigMoE's, or the lid after 3 strikes; PAUSE after 5 min. A guard stop ends the prefix; what is left
# stays for Wed 15:00 (wed_day.sh, still armed, skips what is done).
DM=~/FableGod/decision_map; PY=~/BigMoE/.venv/bin/python; LOG=~/Breakthrough/data/night_prefix.log; Q=$HOME/.local/share/quiet
FIRST=${PREFIX_FIRST:-$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-04 00:30:00" +%s)}       # Tue 22:2x (moved to main, Kamen:
LASTGO=${PREFIX_LASTGO:-$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-04 00:40:00" +%s)}     # research NOW): epoch overrides,
DEADLINE=${PREFIX_DEADLINE:-$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-04 01:15:00" +%s)} # e.g. start now
cd $DM || exit 1
say() { echo "[prefix] $* $(date '+%a %H:%M:%S')" >> $LOG; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
BM="larger-than-RAM MoE"                                    # BigMoE's session name
others() { peer ls 2>/dev/null | grep -v "^You:"; }
held_quiet() { others | grep -qE "quiet lock until|benchmarking: quiet lock"; }
wait_quiet() { others | grep -v "$BM" | grep -q "waiting for a quiet window"; }
big_others() { others | grep -oE "shared run.*MEM [0-9]+ GB" | grep -oE "MEM [0-9]+" | awk '$2 >= 5 {f = 1} END {exit !f}'; }
why_not() {
  if pgrep -f "run_map.py" > /dev/null; then echo "a map process is running"
  elif [ "$(date +%H)" -ge 15 ] && big_others; then echo "another job over 5 GB (day rule, to 00:00)"
  elif held_quiet; then echo "a quiet window (others)"; elif wait_quiet; then echo "a queued quiet ticket (others)"
  elif lid_closed; then echo "lid closed"; elif [ -e "$Q/PAUSE" ]; then echo "PAUSE"
  elif [ "$(noswap)" -lt 25 ]; then echo "NO-SWAP < 25 GB"; fi
}
kill_tree() { local p=$1 c; for c in $(pgrep -P $p); do kill_tree $c; done; kill -TERM $p 2>/dev/null; }
stop=0
run() {          # run <tag> <env assignments or -> <max windows> <names...>; skips names whose JSON exists
  local tag=$1 envs=$2 max=$3; shift 3; local dir=$DM/out todo= n e P bad=0 pz=0 w rc
  for e in $envs; do case $e in DM_OUT=*) dir=${e#DM_OUT=};; esac; done
  for n in "$@"; do [ -f $dir/$n.json ] || todo="$todo $n"; done
  [ -z "$todo" ] && { say "$tag: all written already"; return 0; }
  say "$tag go ($todo; $envs; NO-SWAP $(noswap) GB)"
  if [ "$envs" = "-" ]; then $PY run_map.py $todo --max $max --skip-done >> $LOG 2>&1 & P=$!
  else env $envs $PY run_map.py $todo --max $max --skip-done >> $LOG 2>&1 & P=$!; fi
  while kill -0 $P 2>/dev/null; do
    sleep 15; w=
    [ $(date +%s) -ge $DEADLINE ] && { say "$tag: the hard stop $(date -r $DEADLINE '+%H:%M'), stopping the tree"; kill_tree $P; stop=1; break; }
    [ $(( $(swap_mb) - S0 )) -gt 2048 ] && { say "$tag: swap grew > 2 GB, stopping the tree"; kill_tree $P; stop=1; break; }
    if [ -e "$Q/PAUSE" ]; then pz=$((pz + 1)); else pz=0; fi
    if held_quiet; then w="a quiet window (others)"; elif wait_quiet; then w="a queued quiet ticket (others)"
    elif lid_closed; then w="lid closed"; elif [ $pz -ge 20 ]; then w="PAUSE over 5 min"; fi
    if [ -n "$w" ]; then bad=$((bad + 1)); else bad=0; fi
    [ $bad -ge 3 ] && { say "$tag: stopping for: $w"; kill_tree $P; stop=1; break; }
  done
  wait $P 2>/dev/null; rc=$?; say "$tag: ended (rc $rc)"; return $rc
}
say "armed: start $(date -r $FIRST '+%a %H:%M')-$(date -r $LASTGO '+%H:%M') (NO-SWAP >= 25 GB), stop by $(date -r $DEADLINE '+%H:%M')"
while [ $(date +%s) -lt $FIRST ]; do sleep 30; done
tries=0                                     # Tue 22:3x: a guard stop (e.g. a 1-min quiet window) no longer ends the
while :; do                                 # prefix; it waits for a clear moment and resumes what is left (up to 4 tries)
  tries=$((tries + 1)); ok=0; last=
  while :; do
    if [ $(date +%s) -ge $LASTGO ]; then
      [ $tries -eq 1 ] && { say "no start by $(date -r $LASTGO '+%H:%M') (last: ${last:-?}; NO-SWAP $(noswap) GB): everything stays for Wed 15:00"; exit 2; }
      say "try $tries: no clear moment by $(date -r $LASTGO '+%H:%M'); the rest stays for Wed 15:00"; break 2
    fi
    w=$(why_not)
    if [ -n "$w" ]; then ok=0; [ "$w" != "$last" ] && say "waiting: $w (NO-SWAP $(noswap) GB)"; last=$w; sleep 20; continue; fi
    ok=$((ok + 1)); last=; [ $ok -ge 3 ] && break; sleep 20
  done
  COLD="hyb_cold4_t3 hyb_0.9_u8_live_t3"
  for n in $COLD; do
    grep -q "\"$n\"" dm_ablations.py && [ "$(ls out/$n 2>/dev/null | wc -l | tr -d ' ')" = 36 ] || { COLD=; break; }
  done
  [ -z "$COLD" ] && say "COLD not ready (masks or names missing; see cold_slot.log): QOM + the layout check only"
  peer lock share 45m --cpu 200 --gpu 50 "Breakthrough night prefix: MEM 17 GB: COLD + QOM + layout check (flips, no timing), ~25 min" >> $LOG 2>&1 \
    || { [ $tries -eq 1 ] && { say "share lock not granted: everything stays for Wed 15:00"; exit 1; }; say "try $tries: share lock not granted"; break; }
  S0=$(swap_mb); say "go (try $tries; NO-SWAP $(noswap) GB, swap $S0 MB)"; stop=0
  run P - 12 $COLD qom_0.9_live qom_0.5_live qom_0_live
  if [ $stop -eq 0 ]; then
    mkdir -p $DM/out_layout/ctrl $DM/out_layout/test
    for d in ctrl test; do cp $DM/out/gate_dfl2.json $DM/out_layout/$d/; done
    run Lc "DM_OUT=$DM/out_layout/ctrl" 1 baseline t3_top2_all
    [ $stop -eq 0 ] && run Lt "DM_OUT=$DM/out_layout/test DM_R3=$DM/r3_overlay BIGMOE_EXT_BUILD=build_nat" 1 baseline t3_top2_all
  fi
  peer unlock >> $LOG 2>&1; say "unlocked (try $tries)"
  { [ $stop -eq 0 ] || [ $tries -ge 4 ] || [ $(date +%s) -ge $DEADLINE ]; } && break
  say "try $tries stopped by a guard; resuming the rest after a clear moment"
done
if [ -f $DM/out_layout/ctrl/t3_top2_all.json ] && [ -f $DM/out_layout/test/t3_top2_all.json ]; then
  $PY dm_layout_check.py >> $LOG 2>&1; say "L: $(tail -1 $DM/out_layout/LAYOUT_check.txt 2>/dev/null)"
fi
[ -f $DM/out/hyb_cold4_t3.json ] && { $PY dm_cold_read.py >> $LOG 2>&1; say "COLD: $(grep -E '^(VERDICT|COST)' $DM/out/COLD_read.txt | tr '\n' ' ')"; }
left=
for n in hyb_cold4_t3 hyb_0.9_u8_live_t3 qom_0.9_live qom_0.5_live qom_0_live; do [ -f $DM/out/$n.json ] || left="$left $n"; done
grep -q "LAYOUT CHECK: PASS" $DM/out_layout/LAYOUT_check.txt 2>/dev/null || left="$left layout-check"
say "prefix done; left for Wed 15:00:${left:- nothing}"
