#!/bin/bash
# Wed 2026-11-04 day slot (Breakthrough; the orchestrator Tue 17:07 / 17:4x / 17:5x): after FLOOR night 1's 15:00 deadline
# and before EvalSuite's T0 at 15:30 (one GPU job by day). Run ATTACHED (Bash run_in_background): it takes its own locks.
#   Q  NewIdeas' three Q1-ON-MISS arms on the 36 windows, decision_map/out (QUEUE 554dfe0): qom_0.9_live qom_0.5_live
#      qom_0_live (~225 s each); NewIdeas reads them (~/NewIdeas/scripts/qom_read.py).
#   L  the records3 layout check, only if Q is done by 15:20: ctrl = baseline + t3_top2_all on 3 windows with today's ext
#      and records3; test = the same with BIGMOE_EXT_BUILD=build_nat and DM_R3=r3_overlay (BigMoE's 3 converted scratch
#      layers, mk_r3_overlay.py); dm_layout_check.py compares them bitwise (-> out_layout/LAYOUT_check.txt).
# Start: after 15:00 and once no run_map.py process is left (FLOOR's tree), 3 clear checks 20 s apart (no other session's
# quiet window or MEM job over 5 GB, no PAUSE, lid open, NO-SWAP >= 25 GB). Guard every 15 s: 15:27 or swap growth > 2 GB
# or another MEM job stop the tree at once; another quiet window or the lid after 3 strikes; PAUSE after 5 min.
DM=~/FableGod/decision_map; PY=~/BigMoE/.venv/bin/python; LOG=~/Breakthrough/data/wed_day.log; Q=$HOME/.local/share/quiet
FIRST=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-04 15:00:00" +%s)
LCUT=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-04 15:20:00" +%s)
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-04 15:27:00" +%s)
cd $DM || exit 1
say() { echo "[wed] $* $(date '+%a %H:%M:%S')" >> $LOG; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
quiet_others() { peer ls 2>/dev/null | grep -v "^You:" | grep -qE "quiet lock until|benchmarking: quiet lock|waiting for a quiet window"; }
big_others() { peer ls 2>/dev/null | grep -v "^You:" | grep -oE "shared run.*MEM [0-9]+ GB" | grep -oE "MEM [0-9]+" | awk '$2 >= 5 {f = 1} END {exit !f}'; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
why_not() {
  if pgrep -f "run_map.py" > /dev/null; then echo "a map process is still running"
  elif quiet_others; then echo "a quiet window (others)"; elif big_others; then echo "another job over 5 GB (day rule)"
  elif lid_closed; then echo "lid closed"; elif [ -e "$Q/PAUSE" ]; then echo "PAUSE"
  elif [ "$(noswap)" -lt 25 ]; then echo "NO-SWAP < 25 GB"; fi
}
wait_clear() {   # 0 = clear; 1 = the deadline passed
  local ok=0 last= w
  while [ $(date +%s) -lt $FIRST ]; do sleep 30; done
  while :; do
    [ $(date +%s) -ge $DEADLINE ] && return 1
    w=$(why_not)
    if [ -n "$w" ]; then ok=0; [ "$w" != "$last" ] && say "waiting: $w"; last=$w; sleep 20; continue; fi
    ok=$((ok + 1)); last=; [ $ok -ge 3 ] && return 0; sleep 20
  done
}
kill_tree() { local p=$1 c; for c in $(pgrep -P $p); do kill_tree $c; done; kill -TERM $p 2>/dev/null; }
run() {          # run <tag> <env assignments or -> <max windows> <names...>; 0 = ran to its end
  local tag=$1 envs=$2 max=$3; shift 3; local names="$*" P S0 bad=0 pz=0 w rc
  wait_clear || { say "$tag: deadline before its start"; return 1; }
  peer lock share 30m --cpu 200 --gpu 50 "Breakthrough day slot: MEM 17 GB: $tag (flips, no timing)" >> $LOG 2>&1 \
    || { say "$tag: share lock not granted"; return 1; }
  S0=$(swap_mb); say "$tag go ($names; $envs; NO-SWAP $(noswap) GB, swap $S0 MB)"
  if [ "$envs" = "-" ]; then $PY run_map.py $names --max $max --skip-done >> $LOG 2>&1 & P=$!
  else env $envs $PY run_map.py $names --max $max --skip-done >> $LOG 2>&1 & P=$!; fi
  while kill -0 $P 2>/dev/null; do
    sleep 15; w=
    [ $(date +%s) -ge $DEADLINE ] && { say "$tag: 15:27, stopping the tree (T0 at 15:30)"; kill_tree $P; break; }
    [ $(( $(swap_mb) - S0 )) -gt 2048 ] && { say "$tag: swap grew > 2 GB, stopping the tree"; kill_tree $P; break; }
    big_others && { say "$tag: another job over 5 GB (day rule), stopping the tree"; kill_tree $P; break; }
    if [ -e "$Q/PAUSE" ]; then pz=$((pz + 1)); else pz=0; fi
    if quiet_others; then w="a quiet window (others)"; elif lid_closed; then w="lid closed"; elif [ $pz -ge 20 ]; then w="PAUSE over 5 min"; fi
    if [ -n "$w" ]; then bad=$((bad + 1)); else bad=0; fi
    [ $bad -ge 3 ] && { say "$tag: stopping for: $w"; kill_tree $P; break; }
  done
  wait $P 2>/dev/null; rc=$?; peer unlock >> $LOG 2>&1; say "$tag: ended (rc $rc); unlocked"; return $rc
}
say "armed: Q from Wed 15:00 (after FLOOR's tree), L only if Q ends by 15:20, everything stops at 15:27"
run Q - 12 qom_0.9_live qom_0.5_live qom_0_live
for n in qom_0.9_live qom_0.5_live qom_0_live; do [ -f $DM/out/$n.json ] || say "Q: $n.json missing"; done
if [ $(date +%s) -lt $LCUT ]; then
  mkdir -p $DM/out_layout/ctrl $DM/out_layout/test
  for d in ctrl test; do cp $DM/out/gate_dfl2.json $DM/out_layout/$d/; done
  run Lc "DM_OUT=$DM/out_layout/ctrl" 1 baseline t3_top2_all \
    && run Lt "DM_OUT=$DM/out_layout/test DM_R3=$DM/r3_overlay BIGMOE_EXT_BUILD=build_nat" 1 baseline t3_top2_all
  $PY dm_layout_check.py >> $LOG 2>&1; say "L: $(tail -1 $DM/out_layout/LAYOUT_check.txt 2>/dev/null)"
else
  say "L skipped: Q ended after 15:20 (the orchestrator: skip it if QOM runs late)"
fi
say "day slot done"
