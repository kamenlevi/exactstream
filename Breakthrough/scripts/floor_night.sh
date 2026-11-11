#!/bin/bash
# FLOOR night 1 (Breakthrough; mission 13 narrowed, Fable's FLOOR_125B.md section 5; the orchestrator Tue 2026-11-03 07:0x;
# pre-registered SCOREBOARD a277380). The decision map on the 78 windows with HPD-0 cf walks, then the extras on the 36.
# Run ATTACHED (Bash run_in_background) from the session that owns it: it takes and drops its own share locks.
#   usage: floor_night.sh [FIRST HH:MM, default 00:30] [DEADLINE HH:MM, default 15:00]     (DRY=1: no model runs)
# Steps (each step = one run_map.py process, one model load; --skip-done resumes a step after any stop):
#   S0 p1_oea_k1_ku4_t3 top_k8 (the 36, decision_map/out)   the 12 GB pass-1 stack + top_k8 (Tue 14:5x)
#   A1 capture (78)            the baseline + routing; then baseline.json := capture.json
#   A2 gate + section 5's runs 1-4 (78): gate gate_bands gate_top2_cf0999 gate_top5_cf099 gate_bands+hc8w
#   D  extras 5-7 (the 36, data/floor1_36): gate_bands+moe_reuse_chain1_L08_35_dcf0999 gate_top0_cf0999_L08_35 gate_top4_cf099
#   B  CPU: Q1-UNION 0.9 masks for the 78 (dm_q1u_masks.py), + the masks of any arm listed in data/floor1/extra_arms.txt
#   C  q1u_cf09 [+ the listed arms] (78)
#   R  read-outs (DM_OUT each): dm_day9_read.py (bands by class), dm_floor_read.py (section 5's lines), dm_q1u_read.py
# Start conditions per step (3 clear checks 60 s apart): after FIRST, no map process (Tue 19:2x: the 00:30 prefix,
# night_prefix.sh, runs first), BigMoE's turns seen once (or 02:00), no other session's quiet window (held or waited for),
# no PAUSE, the lid open, NO-SWAP >= 25 GB. Guard every 30 s under a run: the DEADLINE stops the tree; swap growth > 2 GB
# over the step's start stops the tree; another session's quiet window, the lid, or PAUSE > 5 min stop it (3 strikes);
# then the step waits and resumes. A step that ends with names missing and no guard stop counts as a failure; two
# failures of one step skip it (logged). Nothing starts after the deadline.
DM=~/FableGod/decision_map; F78=~/Breakthrough/data/floor1; F36=~/Breakthrough/data/floor1_36
PY=~/BigMoE/.venv/bin/python; LOG=$F78/night.log; Q=$HOME/.local/share/quiet
cd $DM || exit 1
NOW=$(date +%s)
FIRST=$(date -j -f "%H:%M" "${1:-00:30}" +%s); [ "$FIRST" -lt "$NOW" ] && FIRST=$((FIRST + 86400))
DEADLINE=$(date -j -f "%H:%M" "${2:-15:00}" +%s); [ "$DEADLINE" -lt "$FIRST" ] && DEADLINE=$((DEADLINE + 86400))
say() { echo "[floor] $* $(date '+%a %H:%M:%S')" >> $LOG; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
quiet_others() { peer ls 2>/dev/null | grep -v "^You:" | grep -qE "quiet lock until|benchmarking: quiet lock|waiting for a quiet window"; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
big_others() { peer ls 2>/dev/null | grep -v "^You:" | grep -oE "shared run.*MEM [0-9]+ GB" | grep -oE "MEM [0-9]+" | awk '$2 >= 5 {f = 1} END {exit !f}'; }
BMLATE=${FLOOR_BMLATE:-$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-04 02:00:00" +%s)}; BMFLAG=$F78/.bigmoe_seen   # Tue 22:2x:
# FLOOR_BMLATE=0 drops the gate (BigMoE now queues first and waits for memory inside its lock; FLOOR yields to its ticket)
bigmoe_first() { # Tue 19:2x (the orchestrator's order prefix -> BigMoE's turns -> FLOOR; BigMoE needs NO-SWAP >= 33 GB to
  # queue, so a FLOOR step started right after the prefix would hold it off): wait until BigMoE has held or queued a quiet
  # window once tonight, or 02:00 at the latest; from then on the usual gates
  [ -f $BMFLAG ] && return 0
  if peer ls 2>/dev/null | grep "larger-than-RAM MoE" | grep -qE "quiet lock until|benchmarking: quiet lock|waiting for a quiet window" \
     || [ $(date +%s) -ge $BMLATE ]; then touch $BMFLAG; return 0; fi
  return 1
}
why_not() {      # a map process = another runner's (tonight's 00:30 prefix, night_prefix.sh): never two at once
  if pgrep -f "run_map.py" > /dev/null; then echo "a map process is running"
  elif ! bigmoe_first; then echo "BigMoE's turns go first (not seen yet; 02:00 at the latest)"
  elif [ "$(date +%H)" -ge 15 ] && big_others; then echo "another job over 5 GB (day rule, to 00:00)"
  elif quiet_others; then echo "a quiet window (others)"; elif lid_closed; then echo "lid closed"
  elif [ -e "$Q/PAUSE" ]; then echo "PAUSE"; elif [ "$(noswap)" -lt 25 ]; then echo "NO-SWAP < 25 GB"; fi
}
wait_clear() {   # 0 = clear to start; 1 = the deadline passed
  local ok=0 last= w
  while [ $(date +%s) -lt $FIRST ]; do sleep 60; done
  while :; do
    [ $(date +%s) -ge $DEADLINE ] && return 1
    w=$(why_not)
    if [ -n "$w" ]; then ok=0; [ "$w" != "$last" ] && say "waiting: $w (NO-SWAP $(noswap) GB)"; last=$w; sleep 60; continue; fi
    ok=$((ok + 1)); last=; [ $ok -ge 3 ] && return 0; sleep 60
  done
}
done_all() { local d=$1; shift; for n in "$@"; do [ "$n" = gate ] && continue; [ -f "$d/$n.json" ] || return 1; done; return 0; }
kill_tree() { local p=$1 c; for c in $(pgrep -P $p); do kill_tree $c; done; kill -TERM $p 2>/dev/null; }
run_step() {     # run_step <tag> <DM_OUT dir> <max windows> <names...>
  local tag=$1 dir=$2 max=$3; shift 3; local names="$*" fails=0 rc P S0 bad pz stopped w
  while ! done_all $dir $names; do
    wait_clear || { say "$tag: deadline passed before its start; left: $names"; return 1; }
    peer lock share 180m --cpu 200 --gpu 50 "Breakthrough FLOOR night: MEM 17 GB: $tag (flips, no timing)" >> $LOG 2>&1 || { say "$tag: share lock not granted"; sleep 60; continue; }
    S0=$(swap_mb); say "$tag go ($names; NO-SWAP $(noswap) GB, swap $S0 MB)"
    if [ -n "$DRY" ]; then (sleep 5; for n in $names; do [ "$n" = gate ] || echo '{}' > $dir/$n.json; done) & P=$!
    else DM_OUT=$dir $PY run_map.py $names --max $max --skip-done >> $LOG 2>&1 & P=$!; fi
    bad=0; pz=0; stopped=0
    while kill -0 $P 2>/dev/null; do
      sleep 30; w=
      [ $(date +%s) -ge $DEADLINE ] && { say "$tag: the deadline: stopping the tree"; kill_tree $P; stopped=1; break; }
      [ $(( $(swap_mb) - S0 )) -gt 2048 ] && { say "$tag: swap grew > 2 GB: stopping the tree"; kill_tree $P; stopped=1; break; }
      if [ -e "$Q/PAUSE" ]; then pz=$((pz + 1)); else pz=0; fi
      if quiet_others; then w="a quiet window (others)"; elif lid_closed; then w="lid closed"; elif [ $pz -ge 10 ]; then w="PAUSE over 5 min"; fi
      if [ -n "$w" ]; then bad=$((bad + 1)); else bad=0; fi
      [ $bad -ge 3 ] && { say "$tag: stopping for: $w"; kill_tree $P; stopped=1; break; }
    done
    wait $P 2>/dev/null; rc=$?; peer unlock >> $LOG 2>&1; say "$tag: run ended (rc $rc); unlocked"
    [ $(date +%s) -ge $DEADLINE ] && { say "$tag: deadline; left: $names"; return 1; }
    if [ $stopped -eq 0 ] && ! done_all $dir $names; then
      fails=$((fails + 1)); say "$tag: ended with names missing, no guard stop ($fails)"
      [ $fails -ge 2 ] && { say "$tag: SKIPPED after 2 failures"; return 1; }
    fi
  done
  say "$tag: all names written"; return 0
}
rm -f $BMFLAG; say "armed: first start $(date -r $FIRST '+%a %H:%M'), deadline $(date -r $DEADLINE '+%a %H:%M')${DRY:+ (DRY)}"
if [ -z "$DRY" ]; then $PY test_reuse.py > $F78/test_reuse.txt 2>&1; tail -1 $F78/test_reuse.txt | grep -q "ALL OK" || { say "unit test FAILED ($F78/test_reuse.txt); nothing runs"; exit 4; }; fi
# S0 (the orchestrator Tue 14:5x, NewIdeas 14:54): the 12 GB pass-1 stack (P1-K1 capped to 4 kept ranks, at 3 bits) and
# top_k8, on the 36 windows in the map's own out dir; NewIdeas reads them (p1_cert_read.py / live_read.py)
run_step S0 $DM/out 12 p1_oea_k1_ku4_t3 top_k8
run_step A1 $F78 30 capture && [ -f $F78/capture.json ] && [ ! -f $F78/baseline.json ] && cp $F78/capture.json $F78/baseline.json && say "baseline.json := capture.json"
[ -f $F78/baseline.json ] && run_step A2 $F78 30 gate gate_bands gate_top2_cf0999 gate_top5_cf099 gate_bands+hc8w
run_step D $F36 12 gate_bands+moe_reuse_chain1_L08_35_dcf0999 gate_top0_cf0999_L08_35 gate_top4_cf099
EXTRA=; [ -f $F78/extra_arms.txt ] && EXTRA=$(grep -v "^#" $F78/extra_arms.txt | awk '{print $1}' | tr '\n' ' ')
if [ -f $F78/baseline.json ] && [ -d $F78/routing ] && [ $(date +%s) -lt $DEADLINE ]; then
  say "B: masks (q1u 0.9${EXTRA:+, $EXTRA})"
  if [ -z "$DRY" ]; then
    DM_OUT=$F78 nice -n 10 $PY dm_q1u_masks.py 0.9 >> $LOG 2>&1
    [ -f $F78/extra_arms.txt ] && grep -v "^#" $F78/extra_arms.txt | while read arm cmd; do [ -n "$cmd" ] && (DM_OUT=$F78 eval "$cmd") >> $LOG 2>&1; done
  fi
  run_step C $F78 30 q1u_cf09 $EXTRA
fi
say "R: read-outs"
if [ -z "$DRY" ]; then
  for d in $F78 $F36; do
    for s in dm_day9_read.py dm_floor_read.py dm_q1u_read.py dm_oea_read.py; do [ -f $DM/$s ] && DM_OUT=$d $PY $s > $d/$(basename $s .py).log 2>&1; done
  done
  (cd ~/Breakthrough/scripts/floor1 && python3 -I floor_rows.py > $F78/floor_rows.out 2>&1 && python3 -I floor_bytes.py > $F78/floor_bytes.out 2>&1) \
    || say "R: floor_rows / floor_bytes failed (see $F78/floor_rows.out, floor_bytes.out)"
fi
say "night done"
