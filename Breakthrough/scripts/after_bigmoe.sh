#!/bin/bash
# After BigMoE's night ticket (Breakthrough, Tue 2026-11-03 23:0x). Arm it as `peer wait; bash after_bigmoe.sh` (its
# quiet lock blocks starting programs). Run ATTACHED (Bash run_in_background).
#   1. Start only after BigMoE has had no ticket for 3 min (a chain that exits at 01:00 for memory re-queues a
#      continuation within ~60 s; BigMoE 23:0x), or 05:30 at the latest.
#   2. The prefix's leftovers: night_prefix.sh now (qom_0_live + the layout check; it skips written JSONs).
#   3. NewIdeas' QOM t3 arms (its QUEUE.md; Tue 23:02): qom_0.5_live_t3 and, if NewIdeas builds it, qom_0_live_t3. Their
#      masks sit in ~/NewIdeas/data/dm_masks/<arm>, marked READY by NewIdeas after its bit-check (NO_<arm> = not built).
#      Install both names once (scratch copy, import test, atomic mv, with no map process), then run each arm as soon as
#      it is READY, in a gap of FLOOR night 1 (20-s start checks beat FLOOR's 60-s ones; FLOOR waits for any map process).
#      No start after 12:15; hard stop 12:45 (FLOOR ends 13:00, Fit12 13:05). An undecided arm is waited for to 07:30.
#      NewIdeas 23:21 (QUEUE fe52587): qom_0.5_live_c164 (QOM at C 160's pools, no t3) after the t3 arms, same markers.
DM=~/FableGod/decision_map; PY=~/BigMoE/.venv/bin/python; LOG=~/Breakthrough/data/night_prefix.log; Q=$HOME/.local/share/quiet
NI=~/NewIdeas/data/dm_masks; SP=~/Breakthrough/data/abl_scratch
LATE=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-04 05:30:00" +%s)
UNDECIDED=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-04 07:30:00" +%s)
LASTGO=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-04 12:15:00" +%s)
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-04 12:45:00" +%s)
say() { echo "[after] $* $(date '+%a %H:%M:%S')" >> $LOG; }
bq() { peer ls 2>/dev/null | grep "larger-than-RAM MoE" | grep -qE "quiet lock until|benchmarking: quiet lock|waiting for a quiet window"; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
quiet_others() { peer ls 2>/dev/null | grep -v "^You:" | grep -qE "quiet lock until|benchmarking: quiet lock|waiting for a quiet window"; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
kill_tree() { local p=$1 c; for c in $(pgrep -P $p); do kill_tree $c; done; kill -TERM $p 2>/dev/null; }
why_not() {
  if pgrep -f "run_map.py" > /dev/null; then echo "a map process is running"
  elif quiet_others; then echo "a quiet window (others)"; elif lid_closed; then echo "lid closed"
  elif [ -e "$Q/PAUSE" ]; then echo "PAUSE"; elif [ "$(noswap)" -lt 25 ]; then echo "NO-SWAP < 25 GB"; fi
}
install_qomt3() {   # scratch copy, import test, atomic mv (only with no map process)
  mkdir -p $SP/test; cp $DM/dm_ablations.py $SP/dm_ablations.py.qomt3
  if ! grep -q 'ABLATIONS\[_arm + "_t3"\]' $SP/dm_ablations.py.qomt3; then
    printf '%s\n' \
      'for _arm in ("qom_0.5_live", "qom_0_live"):   # NewIdeas Tue 23:02 (QUEUE.md): QOM with ranks 0-1 of sure rows at 3 bits' \
      '    ABLATIONS[_arm + "_t3"] = hyb_t3(os.path.expanduser(f"~/NewIdeas/data/dm_masks/{_arm}"), 0.9)   # rmask <arm>_t3 (its masks)' \
      'ABLATIONS["qom_0.5_live_c164"] = q1_union(os.path.expanduser("~/NewIdeas/data/dm_masks/qom_0.5_live_c164"))   # NewIdeas 23:21 (fe52587): QOM at C 160 pools' \
      >> $SP/dm_ablations.py.qomt3
  fi
  cp $SP/dm_ablations.py.qomt3 $SP/test/dm_ablations.py
  (cd $SP/test && PYTHONPATH=$DM $PY -c "import dm_ablations as A; assert A.__file__.startswith('$SP'), A.__file__; assert all(a in A.ABLATIONS for a in ('qom_0.5_live_t3', 'qom_0_live_t3', 'qom_0.5_live_c164', 'hyb_cold4_len_t3')); print('import OK')") >> $LOG 2>&1 \
    || { say "QOM t3: the import test failed; dm_ablations.py unchanged"; return 1; }
  while pgrep -f run_map.py > /dev/null; do sleep 20; done
  cp $SP/dm_ablations.py.qomt3 $DM/.dm_ablations.py.new && mv $DM/.dm_ablations.py.new $DM/dm_ablations.py \
    && say "QOM t3: dm_ablations.py installed" || { say "QOM t3: install failed"; return 1; }
}
# --- 1. BigMoE's ticket gone for 3 min ---
echo "[prefix] prefix done stays BigMoE's signal; the leftovers wait for its ticket to end (after_bigmoe.sh) $(date '+%a %H:%M:%S')" >> $LOG
gone=0
while :; do
  if bq; then gone=0; else gone=$((gone + 1)); [ $gone -ge 6 ] && break; fi
  [ $(date +%s) -ge $LATE ] && break
  sleep 30
done
say "BigMoE's ticket gone for 3 min (or 05:30): the leftovers start"
# --- 2. the prefix's leftovers ---
N=$(date +%s); PREFIX_FIRST=$N PREFIX_LASTGO=$((N + 5400)) PREFIX_DEADLINE=$((N + 7200)) bash ~/Breakthrough/scripts/night_prefix.sh
say "prefix leftovers: $(grep '^\[prefix\] prefix done' $LOG | tail -1 | sed -E 's/^\[prefix\] //')"
# --- 3. NewIdeas' QOM t3 arms, each as soon as READY ---
installed=0; tries=0
cd $DM || exit 1
while [ $(date +%s) -lt $LASTGO ]; do
  ready=; pend=0
  for a in qom_0.5_live_t3 qom_0_live_t3 qom_0.5_live_c164; do   # priority order (c164: NewIdeas 23:21, below the t3 arms)
    [ -f $DM/out/$a.json ] && continue
    if [ -f $NI/$a/READY ] && [ "$(ls $NI/$a/*.npz 2>/dev/null | wc -l | tr -d ' ')" = 36 ]; then ready="$ready $a"
    elif [ ! -f $NI/NO_$a ] && [ $(date +%s) -lt $UNDECIDED ]; then pend=1; fi
  done
  if [ -z "$ready" ]; then [ $pend -eq 0 ] && break; sleep 60; continue; fi
  [ $installed -eq 0 ] && { install_qomt3 || exit 1; installed=1; }
  tries=$((tries + 1)); [ $tries -gt 8 ] && { say "QOM t3: 8 tries used; left$ready"; break; }
  ok=0; last=
  while :; do
    [ $(date +%s) -ge $LASTGO ] && { say "QOM t3: no start by 12:15; left$ready"; break 2; }
    w=$(why_not)
    if [ -n "$w" ]; then ok=0; [ "$w" != "$last" ] && say "QOM t3 waiting: $w"; last=$w; sleep 20; continue; fi
    ok=$((ok + 1)); last=; [ $ok -ge 3 ] && break; sleep 20
  done
  peer lock share 20m --cpu 200 --gpu 50 "Breakthrough QOM t3 arms (NewIdeas): MEM 17 GB (flips, no timing), ~5 min each" >> $LOG 2>&1 || { sleep 60; continue; }
  S0=$(swap_mb); say "QOM t3 go (try $tries:$ready; NO-SWAP $(noswap) GB)"
  $PY run_map.py $ready --max 12 --skip-done >> $LOG 2>&1 & P=$!
  bad=0; pz=0
  while kill -0 $P 2>/dev/null; do
    sleep 15; w=
    [ $(date +%s) -ge $DEADLINE ] && { say "QOM t3: 12:45, stopping the tree"; kill_tree $P; break; }
    [ $(( $(swap_mb) - S0 )) -gt 2048 ] && { say "QOM t3: swap grew > 2 GB, stopping the tree"; kill_tree $P; break; }
    if [ -e "$Q/PAUSE" ]; then pz=$((pz + 1)); else pz=0; fi
    if quiet_others; then w="a quiet window (others)"; elif lid_closed; then w="lid closed"; elif [ $pz -ge 20 ]; then w="PAUSE over 5 min"; fi
    if [ -n "$w" ]; then bad=$((bad + 1)); else bad=0; fi
    [ $bad -ge 3 ] && { say "QOM t3: stopping for: $w"; kill_tree $P; break; }
  done
  wait $P 2>/dev/null; peer unlock >> $LOG 2>&1; say "QOM t3: run ended; unlocked"
  [ $(date +%s) -ge $DEADLINE ] && break
done
left=; for a in qom_0.5_live_t3 qom_0_live_t3 qom_0.5_live_c164; do [ -f $NI/$a/READY ] && [ ! -f $DM/out/$a.json ] && left="$left $a"; done
say "QOM t3 done; left:${left:- nothing}"
