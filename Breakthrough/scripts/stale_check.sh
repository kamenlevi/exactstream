#!/bin/bash
# Stale-slot check of the map's C 51 spin path (Breakthrough, Tue 2026-11-03 22:4x; Leap 50de28c via the orchestrator;
# pre-registered SCOREBOARD 310a0ca). Run ATTACHED (Bash run_in_background): it takes its own share lock.
# The baseline on 9 windows (--max 3) with BIGMOE_GATE=event (a command buffer per gate call) into out_stale/event, then
# tk and lp bitwise against the stored out/baseline.json -> out_stale/STALE_check.txt (PASS / FAIL).
# Start: after BigMoE's night ticket (seen held or queued once, then gone; or 05:30 at the latest), before FLOOR's next
# step (3 clear checks 20 s apart vs FLOOR's 60 s): no map process, no other session's quiet window held or waited for,
# lid open, no PAUSE, NO-SWAP >= 25 GB. No start after 12:15; guard every 15 s; hard stop 12:45 (FLOOR ends 13:00).
DM=~/FableGod/decision_map; PY=~/BigMoE/.venv/bin/python; LOG=~/Breakthrough/data/stale_check.log; Q=$HOME/.local/share/quiet
LATE=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-04 05:30:00" +%s)
LASTGO=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-04 12:15:00" +%s)
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-04 12:45:00" +%s)
OUTD=$DM/out_stale/event; FLAG=$DM/out_stale/.bigmoe_seen
cd $DM || exit 1; mkdir -p $OUTD; rm -f $FLAG
say() { echo "[stale] $* $(date '+%a %H:%M:%S')" >> $LOG; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
quiet_others() { peer ls 2>/dev/null | grep -v "^You:" | grep -qE "quiet lock until|benchmarking: quiet lock|waiting for a quiet window"; }
bigmoe_quiet() { peer ls 2>/dev/null | grep "larger-than-RAM MoE" | grep -qE "quiet lock until|benchmarking: quiet lock|waiting for a quiet window"; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
kill_tree() { local p=$1 c; for c in $(pgrep -P $p); do kill_tree $c; done; kill -TERM $p 2>/dev/null; }
why_not() {
  if [ ! -f $FLAG ]; then
    if bigmoe_quiet; then touch $FLAG; echo "BigMoE's ticket is on"; return; fi
    [ $(date +%s) -lt $LATE ] && { echo "BigMoE's ticket not seen yet"; return; }
    touch $FLAG
  fi
  if pgrep -f "run_map.py" > /dev/null; then echo "a map process is running"
  elif quiet_others; then echo "a quiet window (others)"; elif lid_closed; then echo "lid closed"
  elif [ -e "$Q/PAUSE" ]; then echo "PAUSE"; elif [ "$(noswap)" -lt 25 ]; then echo "NO-SWAP < 25 GB"; fi
}
say "armed: after BigMoE's ticket (or 05:30), no start after 12:15"
ok=0; last=
while :; do
  [ $(date +%s) -ge $LASTGO ] && { say "no clear moment by 12:15; not run"; exit 2; }
  w=$(why_not)
  if [ -n "$w" ]; then ok=0; [ "$w" != "$last" ] && say "waiting: $w"; last=$w; sleep 20; continue; fi
  ok=$((ok + 1)); last=; [ $ok -ge 3 ] && break; sleep 20
done
cp $DM/out/gate_dfl2.json $OUTD/
peer lock share 20m --cpu 200 --gpu 50 "Breakthrough stale-slot check: MEM 17 GB: baseline on 9 windows, BIGMOE_GATE=event (flips, no timing)" >> $LOG 2>&1 \
  || { say "share lock not granted"; exit 1; }
S0=$(swap_mb); say "go (NO-SWAP $(noswap) GB, swap $S0 MB)"
BIGMOE_GATE=event DM_OUT=$OUTD $PY run_map.py baseline --max 3 >> $LOG 2>&1 & P=$!
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
wait $P 2>/dev/null; rc=$?; peer unlock >> $LOG 2>&1; say "run ended (rc $rc); unlocked"
$PY - >> $LOG 2>&1 <<'EOF'
import json, os, numpy as np
D = os.path.expanduser("~/FableGod/decision_map"); p = os.path.join(D, "out_stale/event/baseline.json")
lines = []
if not os.path.exists(p): lines.append("STALE CHECK: INCOMPLETE (no event-mode baseline)")
else:
    E, B = json.load(open(p)), json.load(open(os.path.join(D, "out/baseline.json"))); bad = []; rows = 0
    for nm, e in E.items():
        if not isinstance(e, dict) or "tk" not in e: continue
        b = B[nm]; m = len(e["tk"]); rows += m
        ok = np.array_equal(np.array(e["tk"]), np.array(b["tk"][:m])) and \
             np.array_equal(np.array(e["lp"], np.float32), np.array(b["lp"][:m], np.float32))
        if not ok: bad.append(nm)
        lines.append(f"  {nm}: {'bitwise equal' if ok else 'DIFFERS'} ({m} rows)")
    n = len(lines)
    lines.append(f"STALE CHECK: {'PASS' if not bad and n else 'FAIL'} ({n} windows, {rows} rows; event-mode baseline vs the stored spin-mode baseline"
                 + (f"; differ: {bad}" if bad else "") + ")")
open(os.path.join(D, "out_stale/STALE_check.txt"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
EOF
say "result: $(tail -1 $DM/out_stale/STALE_check.txt 2>/dev/null)"
