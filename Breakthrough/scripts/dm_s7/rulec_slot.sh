#!/bin/bash
# Rule C passes (Fable FULL R2; the orchestrator 21:4x / 21:45): rulec_pass.py run (teacher-forced top-8 baseline passes,
# HumanEval+ 12 + MBPP 100) after tonight's s67 map runner has ended, if it can start by 22:50; hard stop 23:00 (BigMoE's
# ticket). The s67 runner's guards: no map process, no quiet window of others, no other > 5 GB shared run, lid open, no
# PAUSE, NO-SWAP >= 25 GB, 3 clear checks 10 s apart; share lock MEM 16 GB; every 15 s: hard stop, swap +2 GB, quiet
# window / lid (3 strikes), a non-fanwatch PAUSE over 5 min. cwd ~/Breakthrough (fanwatch freezes it with the others).
# A stopped run keeps a finished set's JSON; the next run skips it. Override: RC_LASTGO / RC_DEADLINE "YYYY-mm-dd HH:MM:SS".
DM=~/FableGod/decision_map; PY=~/BigMoE/.venv/bin/python; LOG=~/Breakthrough/data/rulec/rulec_slot.log; Q=$HOME/.local/share/quiet
ts() { date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
quiet_others() { peer ls 2>/dev/null | grep -v "^You:" | grep -qE "quiet lock until|benchmarking: quiet lock|waiting for a quiet window"; }
big_others() { peer ls 2>/dev/null | grep -v "^You:" | grep "shared run" | grep -oE "MEM [0-9]+ GB" | awk '$2 > 5 {f = 1} END {exit !f}'; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
kill_tree() { local p=$1 c; for c in $(pgrep -P $p); do kill_tree $c; done; kill -TERM $p 2>/dev/null; }
stop_run() { kill_tree $1; sleep 5; ps -o stat= -p $1 2>/dev/null | grep -q T && kill -KILL $1 2>/dev/null; }   # a frozen run holds SIGTERM until thawed; never SIGCONT
why_not() {
  if pgrep -f "run_ma[p].py" > /dev/null; then echo "a map process is running"
  elif quiet_others; then echo "a quiet window (others)"; elif big_others; then echo "another session's shared run > 5 GB"
  elif lid_closed; then echo "lid closed"
  elif [ -e "$Q/PAUSE" ]; then echo "PAUSE"; elif [ "$(noswap)" -lt 25 ]; then echo "NO-SWAP < 25 GB"; fi
}
LASTGO=$(ts "${RC_LASTGO:-2026-11-04 22:50:00}"); DEADLINE=$(ts "${RC_DEADLINE:-2026-11-04 23:00:00}")
say() { echo "[rulec] $* $(date '+%a %H:%M:%S')" >> $LOG; }
done_all() { [ -f ~/Breakthrough/data/rulec/rulec_he12.json ] && [ -f ~/Breakthrough/data/rulec/rulec_mbpp.json ]; }
say "armed: after the s67 runner ends; last start $(date -r $LASTGO +%H:%M), hard stop $(date -r $DEADLINE +%H:%M)"
while pgrep -f "s67_slo[t].sh" > /dev/null; do [ $(date +%s) -ge $LASTGO ] && { say "the s67 runner is still on at the last start; Thursday"; exit 0; }; sleep 20; done
need=3; last=; runs=0
while ! done_all; do
  [ $(date +%s) -ge $LASTGO ] && { say "last start passed; Thursday"; break; }
  w=$(why_not)
  if [ -n "$w" ]; then need=3; [ "$w" != "$last" ] && say "waiting: $w"; last=$w; sleep 10; continue; fi
  need=$((need - 1)); [ $need -gt 0 ] && { sleep 10; continue; }
  runs=$((runs + 1)); [ $runs -gt 3 ] && { say "3 runs used"; break; }
  peer lock share 30m --cpu 200 --gpu 100 "Breakthrough Rule C passes (Fable FULL R2; the orchestrator 21:4x): MEM 16 GB (top-8 baseline passes, HumanEval+ 12 + MBPP 100)" >> $LOG 2>&1 \
    || { say "share lock not granted"; need=3; sleep 30; continue; }
  S0=$(swap_mb); say "go (NO-SWAP $(noswap) GB, swap $S0 MB)"
  (cd ~/Breakthrough && exec $PY ~/Breakthrough/scripts/dm_s7/rulec_pass.py run) >> $LOG 2>&1 & P=$!
  bad=0; pz=0; stopped=
  while kill -0 $P 2>/dev/null; do
    sleep 15; w=
    [ $(date +%s) -ge $DEADLINE ] && { stopped="the hard stop"; stop_run $P; break; }
    [ $(( $(swap_mb) - S0 )) -gt 2048 ] && { stopped="swap grew > 2 GB"; stop_run $P; break; }
    if [ -e "$Q/PAUSE" ] && [ ! -e "$Q/PAUSE.fanwatch" ]; then pz=$((pz + 1)); else pz=0; fi
    if quiet_others; then w="a quiet window (others)"; elif lid_closed; then w="lid closed"; elif [ $pz -ge 20 ]; then w="PAUSE over 5 min"; fi
    if [ -n "$w" ]; then bad=$((bad + 1)); else bad=0; fi
    [ $bad -ge 3 ] && { stopped="$w"; stop_run $P; break; }
  done
  wait $P 2>/dev/null; rc=$?; peer unlock >> $LOG 2>&1
  [ -n "$stopped" ] && { need=3; say "stopped for: $stopped"; }
  say "run ended (rc $rc); he12 $([ -f ~/Breakthrough/data/rulec/rulec_he12.json ] && echo done || echo missing), mbpp $([ -f ~/Breakthrough/data/rulec/rulec_mbpp.json ] && echo done || echo missing); unlocked"
  [ $(date +%s) -ge $DEADLINE ] && break
done
say "over: he12 $([ -f ~/Breakthrough/data/rulec/rulec_he12.json ] && echo done || echo missing), mbpp $([ -f ~/Breakthrough/data/rulec/rulec_mbpp.json ] && echo done || echo missing)"
