#!/bin/bash
# Wed night map screens in one FLOOR gap (Breakthrough, Wed 2026-11-04 04:1x; lines copied into SCOREBOARD before the run).
# Run ATTACHED (Bash run_in_background).
#   - Screen 1(a), Fable C's BYTES10 (the orchestrator 04:0x; ~/FableGod/fablec/BYTES10.md:217-219): qom_0_live_bonus.
#   - Screen 2, NewIdeas (the orchestrator 04:08; ~/NewIdeas/QUEUE.md:1181-1185): hyb_0.9_k1_u10_live_t3 + its twin
#     q1u_0.9_live_t3. They join only if the gap starts before 09:00: FLOOR's D + B + C + R take ~2.5 h after this run,
#     and must end by 13:00 (otherwise screen 2 waits for 13:45).
# All three run in ONE map process (one model load), in that order. The next gap between FLOOR night 1's steps is taken
# without touching floor_night.sh: 10-s checks, 3 clear in a row (FLOOR needs 3 clear checks 60 s apart, then waits with
# "a map process is running"). Clear = no run_map.py, FLOOR's last line is not a running step (FLOOR unlocks up to 15 s
# after its map process ends), no other session's quiet window held or waited for, lid open, no PAUSE, NO-SWAP >= 25 GB.
# Never two map processes. The names go into dm_ablations.py in one step: import-tested on a scratch copy at arming, moved
# in (atomic mv) in the gap with no map process. Share lock MEM 17 GB; guards every 15 s (swap +2 GB, quiet windows 3
# strikes, lid, PAUSE over 5 min); up to 6 tries with --skip-done (run_map writes an arm's JSON once all its windows are
# done); no start after 12:15, hard stop 12:45 (FLOOR ends 13:00). Read-outs: dm_screen_read.py -> out/SCREEN_read_<arm>.txt.
DM=~/FableGod/decision_map; PY=~/BigMoE/.venv/bin/python; LOG=~/Breakthrough/data/screens_slot.log; Q=$HOME/.local/share/quiet
FL=~/Breakthrough/data/floor1/night.log; SP=~/Breakthrough/data/abl_scratch
S1=qom_0_live_bonus; S2="hyb_0.9_k1_u10_live_t3 q1u_0.9_live_t3"
S2LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-04 09:00:00" +%s)
LASTGO=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-04 12:15:00" +%s)
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-04 12:45:00" +%s)
say() { echo "[screens] $* $(date '+%a %H:%M:%S')" >> $LOG; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
quiet_others() { peer ls 2>/dev/null | grep -v "^You:" | grep -qE "quiet lock until|benchmarking: quiet lock|waiting for a quiet window"; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
kill_tree() { local p=$1 c; for c in $(pgrep -P $p); do kill_tree $c; done; kill -TERM $p 2>/dev/null; }
floor_running() { grep -E "^\[floor\]" $FL | tail -1 | grep -qE " go \(|stopping"; }
why_not() {
  if pgrep -f "run_map.py" > /dev/null; then echo "a map process is running"
  elif floor_running; then echo "FLOOR's step is still closing"
  elif quiet_others; then echo "a quiet window (others)"; elif lid_closed; then echo "lid closed"
  elif [ -e "$Q/PAUSE" ]; then echo "PAUSE"; elif [ "$(noswap)" -lt 25 ]; then echo "NO-SWAP < 25 GB"; fi
}
MARK='ABLATIONS["qom_0_live_bonus"]'
prep() {   # scratch copy = today's dm_ablations.py + the lines; import test from the scratch folder
  mkdir -p $SP/test; cp $DM/dm_ablations.py $SP/dm_ablations.py.screens
  grep -qF "$MARK" $SP/dm_ablations.py.screens || printf '%s\n' \
    'ABLATIONS["qom_0_live_bonus"] = q1_union(os.path.expanduser("~/FableGod/fablec/masks/qom_0_live_bonus"))   # Fable C BYTES10 screen 1(a) (the orchestrator Wed 04:0x): qom_0_live, every bonus row gated' \
    'def t3k_mask(path, k):   # NewIdeas screen 2 (QUEUE.md:1181-1185, the orchestrator Wed 04:08): ranks < k of the gated rows at 3 bits (dm_t3), the rest from the masks' \
    '    def setup(): PERT.update(rmask=path, gate=("t3mask",), t3=3, t3k=k, renorm=False)' \
    '    return setup' \
    'ABLATIONS["hyb_0.9_k1_u10_live_t3"] = t3k_mask(os.path.expanduser("~/NewIdeas/data/dm_masks/hyb_0.9_k1_u10_live_t3"), 1)   # sure rows: rank 0 only' \
    'ABLATIONS["q1u_0.9_live_t3"] = t3k_mask(os.path.expanduser("~/NewIdeas/data/dm_masks/q1u_0.9_live_t3"), 2)   # its top-2 twin' \
    >> $SP/dm_ablations.py.screens
  cp $SP/dm_ablations.py.screens $SP/test/dm_ablations.py
  (cd $SP/test && PYTHONPATH=$DM $PY -c "import dm_ablations as A; assert A.__file__.startswith('$SP'), A.__file__; assert all(a in A.ABLATIONS for a in ('qom_0_live_bonus', 'hyb_0.9_k1_u10_live_t3', 'q1u_0.9_live_t3', 'qom_0_live', 'qom_0_live_t3', 'hyb_cold4_len_t3')); print('import OK')") >> $LOG 2>&1
}
install() {   # 0 = installed (or already in), 1 = failed, 2 = a map process appeared
  grep -qF "$MARK" $DM/dm_ablations.py && return 0
  cmp -s <(head -c $(wc -c < $DM/dm_ablations.py) $SP/dm_ablations.py.screens) $DM/dm_ablations.py \
    || { say "install: dm_ablations.py changed since the import test; testing again"; prep || return 1; }
  pgrep -f run_map.py > /dev/null && return 2
  cp $SP/dm_ablations.py.screens $DM/.dm_ablations.py.new && mv $DM/.dm_ablations.py.new $DM/dm_ablations.py \
    && { say "install: the three names are in dm_ablations.py"; return 0; } || return 1
}
nwin() { [ -f $DM/out/$1.json ] && $PY -c "import json; d = json.load(open('$DM/out/$1.json')); print(sum(1 for v in d.values() if isinstance(v, dict) and 'tk' in v))" 2>/dev/null || echo 0; }
todo() { local a; for a in $S1 $([ $(date +%s) -lt $S2LAST ] && echo $S2); do [ "$(nwin $a)" = 36 ] || echo -n "$a "; done; }
say "armed (the orchestrator 04:0x / 04:08): $S1, then $S2 (if the gap starts before 09:00), in the next FLOOR gap"
prep || { say "the import test failed; dm_ablations.py unchanged; not run"; exit 1; }
cd $DM || exit 1
tries=0
while :; do
  tries=$((tries + 1)); [ $tries -gt 6 ] && { say "6 tries used; left: $(todo)"; break; }
  ok=0; last=
  while :; do
    [ $(date +%s) -ge $LASTGO ] && { say "no start by 12:15; left: $(todo)"; break 2; }
    w=$(why_not)
    if [ -n "$w" ]; then ok=0; [ "$w" != "$last" ] && say "waiting: $w"; last=$w; sleep 10; continue; fi
    ok=$((ok + 1)); last=; [ $ok -ge 3 ] && break; sleep 10
  done
  names=$(todo); [ -z "$names" ] && break
  install; rc=$?
  [ $rc -eq 1 ] && { say "install failed; dm_ablations.py unchanged; not run"; exit 1; }
  [ $rc -eq 2 ] && { tries=$((tries - 1)); continue; }
  peer lock share 40m --cpu 200 --gpu 50 "Breakthrough map screens (the orchestrator 04:0x): MEM 17 GB (flips, no timing): $names" >> $LOG 2>&1 \
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
$PY $DM/dm_screen_read.py $S1 qom_0_live 0.3 >> $LOG 2>&1
[ -f $DM/out/hyb_0.9_k1_u10_live_t3.json ] && $PY $DM/dm_screen_read.py hyb_0.9_k1_u10_live_t3 q1u_0.9_live_t3 >> $LOG 2>&1
for a in $S1 hyb_0.9_k1_u10_live_t3; do say "read-out: $(grep '^VERDICT' $DM/out/SCREEN_read_$a.txt 2>/dev/null || echo "$a none")"; done
