#!/bin/bash
# Tonight's 12 GB -> 90 jobs for Breakthrough (Mon 9 Nov 2026 21:4x; v2 22:3x for NewIdeas' 22:28 order): my CPU jobs, then
# the chain's step 2 (NewIdeas QUEUE.md "MON 22:28 CHAIN REORDERED": Fit12's S1u -> this map arm -> Leap's tests, which start
# on this runner's done-file). v3 (23:1x, NewIdeas 23:12, Kamen's yes via the orchestrator): the w24 gate moved to 02:10;
# W24c goes first, then this arm (and Fit12's S1p), then Leap's 32 GB tests, all freed by 02:05; clock limits moved:
# CPU jobs by 01:25 (was 00:55), arm start by 01:30 (was 01:00), hard stop 01:45 (was 01:13).
# Run ATTACHED (Bash run_in_background) from ~/Breakthrough:  bash scripts/m1/s1u_map_night.sh
# DRY=1: print every gate once and exit (no job, no lock, no done-file).
# A (from 23:00:00; CPU, one core, taskpolicy -b): while Kamen is idle (HIDIdleTime >= 600 s and no python chat.py above
#   2 GB) and no other session holds or waits for a quiet window: build the q1u_0_live masks with NewIdeas' builder,
#   bit-check them against q1u_0.9_live (q1u0_mask_check.py), then the eviction-policy replay (code12_policy.py). The replay
#   yields to the chain: once S1u has ended it is stopped (or not started) and runs beside the map arm instead (one
#   background core, < 1 GB, its own log data/m1/code12_policy.log), stopped when the arm ends (deferred). A quiet window
#   held or waited for, or Kamen back, stops the running CPU job; it restarts later (a quiet ticket waits while any session
#   uses >= 80 % CPU).
# B (the map arm; never before 23:00, as it follows A): after ~/Fit12/logs/s1u.done (or s1u.sh's own end file
#   ~/Fit12/data/scr/s1u/DONE) and a PASSed mask check, in a clear gap (3 checks 10 s apart: Kamen idle,
#   no python run_map.py or fit_onecall.py process, W24c's turn45 runner gone (W24c first; this wait ends at 01:15),
#   no other session's big shared run (MEM > 5 GB in its reason or >= 50 % GPU), no quiet window held / waited for by
#   others, lid open, no PAUSE, NO-SWAP >= 25 GB = peak 17 + 8), start by 01:30:00 Tue; share lock with 'MEM 17 GB' first in the reason; run
#   q1u0_run_map.py (the arm registered in memory; result ~/FableGod/decision_map/out/q1u_0_live.json, written at the arm's
#   end). Guards every 15 s: hard stop 01:45:00 (memory freed by 01:46; all freed by 02:05, the gate at 02:10), swap +2 GB, a quiet
#   window of others (3 strikes), lid closed, PAUSE over 5 min, a chat process of Kamen's above 2 GB. A stop is TERM then
#   KILL after 20 s (a heat-frozen process frees its memory only on KILL; this script never sends CONT). Then unlock.
# DONE-FILE (NewIdeas 22:28; Leap's tests start on it): ~/Breakthrough/data/m1/q1u_0_live.done, written once, with the
#   reason, when the arm ends (any rc, any stop, the 01:45 hard stop) or is skipped (mask check FAIL, no s1u.done or no clear
#   gap by 01:30, 4 tries used). A skip before s1u.done waits for s1u.done (or 01:30) first, so Leap never starts beside
#   S1u. The read-out (dm_screen_read.py, SCREEN_OUT = data/m1; seconds, one core) runs after it. An EXIT trap writes it if
#   the runner ends any other way, a TERM included: stopping this runner releases Leap.
# Every job starts with cwd ~/Breakthrough (never inside ~/FableGod).
cd ~/Breakthrough || exit 1
PY=~/BigMoE/.venv/bin/python; M1=~/Breakthrough/scripts/m1; DM=~/FableGod/decision_map; Q=$HOME/.local/share/quiet
POLOG=~/Breakthrough/data/m1/code12_policy.log; LOG=~/Breakthrough/data/m1/s1u_map_night.log; MASKS=~/Breakthrough/data/dm_masks; DONE_S1U=~/Fit12/logs/s1u.done
MK=~/Breakthrough/data/m1/q1u_0_live.done
S1U_DONE2=~/Fit12/data/scr/s1u/DONE   # s1u.sh's own end file (s1u_go.sh turns it into s1u.done)
A_AT=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-09 23:00:00" +%s)
A_LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-10 01:25:00" +%s)
B_LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-10 01:30:00" +%s)
B_STOP=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-10 01:45:00" +%s)
W24C_UNTIL=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-10 01:15:00" +%s)   # wait for W24c (first tonight) at most until then
LA=01:25; LB=01:30; LS=01:45                                              # labels of A_LAST, B_LAST, B_STOP
say() { echo "[night] $* ($(date '+%a %H:%M:%S'))" | tee -a $LOG; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
idle_s() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF / 1000000000); exit}'; }
chat_big() { ps -axo rss=,comm=,args= | awk '$2 ~ /[Pp]ython/ && /chat\.py/ && $1 > 2097152 {n++} END {print n + 0}'; }
map_procs() { ps -axo comm=,args= | awk '$1 ~ /[Pp]ython/ && /run_map\.py/ {n++} END {print n + 0}'; }
s1u_procs() { ps -axo comm=,args= | awk '$1 ~ /[Pp]ython/ && /fit_onecall\.py/ {n++} END {print n + 0}'; }
w24c_first() {   # W24c goes first tonight: its turn45 runner (BigMoE) is alive and it is before 01:15
  [ $(date +%s) -lt $W24C_UNTIL ] && ps -axo command= | grep -qE '^bash scripts/turn45\.sh [0-9]+ [0-9]+ BigMoE W24c'
}
s1u_seen() { [ -e $DONE_S1U ] || [ -e $S1U_DONE2 ]; }   # S1u ended (or gave up): either marker
big_others() {   # other sessions' big shared runs: MEM > 5 GB in the reason, or >= 50 % GPU (peer ls cuts long reasons)
  peer ls 2>/dev/null | grep "^- " | grep -oE "· shared run \(others may work\)[^·]*" | awk '{ big = 0
    if (match($0, /MEM [~]?[0-9]+ GB/)) { m = substr($0, RSTART + 4, RLENGTH - 7); gsub("~", "", m); if (m + 0 > 5) big = 1 }
    if (match($0, /[0-9]+% GPU/)) { if (substr($0, RSTART, RLENGTH - 5) + 0 >= 50) big = 1 }
    n += big } END {print n + 0}'
}
quiet_others() { peer ls 2>/dev/null | grep -v "^You:" | grep -qE "quiet lock until|benchmarking: quiet lock|waiting for a quiet window"; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
kamen_busy() {   # prints a reason when Kamen is on the Mac
  local i; i=$(idle_s); [ -z "$i" ] && i=0
> (a quote from the project owner, kept private)
> (a quote from the project owner, kept private)
}
cpu_why_not() {
  local k; k=$(kamen_busy); [ -n "$k" ] && { echo "$k"; return; }
  quiet_others && { echo "a quiet window (others)"; return; }
  lid_closed && { echo "lid closed"; return; }
  [ -e "$Q/PAUSE" ] && echo "PAUSE"
}
map_why_not() {
  local k; k=$(cpu_why_not); [ -n "$k" ] && { echo "$k"; return; }
  [ "$(map_procs)" -gt 0 ] && { echo "a map process is running"; return; }
  [ "$(s1u_procs)" -gt 0 ] && { echo "an S1u process is running"; return; }
  w24c_first && { echo "W24c goes first (its turn45 runner is alive; this wait ends at 01:15)"; return; }
  [ "$(big_others)" -gt 0 ] && { echo "another session's big shared run (MEM > 5 GB or >= 50 % GPU)"; return; }
  [ "$(noswap)" -lt 25 ] && echo "NO-SWAP $(noswap) GB < 25"
}
stop_tree() {   # TERM the tree, KILL after 20 s (works on a stopped process too); never CONT
  local p=$1 c; local all="$p $(pgrep -P $p)"
  for c in $(pgrep -P $p); do all="$all $(pgrep -P $c)"; done
  kill -TERM $all 2>/dev/null; sleep 20; kill -KILL $all 2>/dev/null
}
mark() {   # the done-file Leap's tests start on: written once, with the reason
  [ -e $MK ] && return 0
  echo "$1 ($(date '+%a %H:%M:%S'))" > $MK; say "done-file $MK written: $1"
}
if [ -n "$DRY" ]; then
  echo "DRY $(date '+%H:%M:%S'): idle $(idle_s) s; chat.py > 2 GB: $(chat_big); map processes: $(map_procs); S1u processes: $(s1u_procs); W24c first: $(w24c_first && echo yes || echo no); big others: $(big_others);" \
       "NO-SWAP $(noswap) GB; swap $(swap_mb) MB; quiet others: $(quiet_others && echo yes || echo no); lid closed: $(lid_closed && echo yes || echo no);" \
       "PAUSE: $([ -e $Q/PAUSE ] && echo yes || echo no); s1u ended (s1u.done or DONE): $(s1u_seen && echo yes || echo no); done-file: $([ -e $MK ] && echo yes || echo no);" \
       "cpu_why_not: '$(cpu_why_not)'; map_why_not: '$(map_why_not)'"
  echo "DRY deadlines: A from $(date -r $A_AT '+%a %H:%M:%S'), A last $(date -r $A_LAST '+%a %H:%M:%S'), B start-by $(date -r $B_LAST '+%a %H:%M:%S'), B stop $(date -r $B_STOP '+%a %H:%M:%S')"
  exit 0
fi

# run_cpu <name> <command...>: one CPU job, restarted while the gate is closed, until it exits; returns its rc, 9 when not
# done by $LA (01:25), 8 when $PREEMPT (a test, e.g. s1u_seen) turns true: the job is stopped or not started ($PREEMPT_WHY)
PREEMPT=; PREEMPT_WHY=
run_cpu() {
  local name=$1; shift; local last= w rc
  while :; do
    [ $(date +%s) -ge $A_LAST ] && { say "$name: not done by $LA; given up"; return 9; }
    [ -n "$PREEMPT" ] && $PREEMPT && { say "$name: not started: $PREEMPT_WHY"; return 8; }
    w=$(cpu_why_not)
    if [ -n "$w" ]; then [ "$w" != "$last" ] && say "$name waits: $w"; last=$w; sleep 20; continue; fi
    last=; say "$name: go"
    taskpolicy -b nice -n 15 "$@" >> $LOG 2>&1 & local P=$!
    local stopped=
    while kill -0 $P 2>/dev/null; do
      sleep 5
      if [ -n "$PREEMPT" ] && $PREEMPT; then say "$name: stopping: $PREEMPT_WHY"; stop_tree $P; wait $P 2>/dev/null; return 8; fi
      w=$(cpu_why_not)
      if [ -n "$w" ]; then say "$name: stopping for: $w"; stop_tree $P; stopped=1; break; fi
    done
    wait $P 2>/dev/null; rc=$?
    [ -n "$stopped" ] && continue
    say "$name: ended rc $rc"; return $rc
  done
}

LOCKED=0
on_exit() {   # any exit, a TERM included: stop my own running job (TERM, KILL after 3 s) and drop my lock, then the done-file
  local c k=
  for c in $(pgrep -P $$); do kill -0 $c 2>/dev/null && k="$k $c $(pgrep -P $c)"; done   # live children only
  [ -n "${k// /}" ] && { kill -TERM $k 2>/dev/null; sleep 3; kill -KILL $k 2>/dev/null; say "exit: stopped my running job ($k )"; }
  [ "$LOCKED" = 1 ] && { peer unlock >> $LOG 2>&1; LOCKED=0; say "exit: lock dropped"; }
  mark "runner ended before a result line (an exit or a stop)"
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
rm -f $MK
say "armed v3 (NewIdeas 23:12 order: W24c -> this arm (+ Fit12 S1p) -> Leap on $MK): A (masks, check, policy replay) from 23:00 until $LA; B (map arm q1u_0_live) after S1u's end and W24c, start by $LB, stop $LS"
while [ $(date +%s) -lt $A_AT ]; do sleep 30; done
MASK_OK=0
if [ "$(ls $MASKS/q1u_0_live/*.npz 2>/dev/null | wc -l | tr -d ' ')" = 36 ]; then say "masks already built (36 files)"
else
  rm -rf $MASKS/q1u_0_live
  run_cpu masks $PY -I -B ~/NewIdeas/scripts/dm_oea_masks.py $MASKS q1u_0_live
fi
run_cpu maskcheck $PY -I -B $M1/q1u0_mask_check.py && MASK_OK=1
say "mask check: $([ $MASK_OK = 1 ] && echo PASS || echo 'FAIL or not run: the map arm will not run')"
PREEMPT=s1u_seen; PREEMPT_WHY="S1u has ended: the map arm goes first (the policy replay runs beside it)"
POLICY_DONE=0; run_cpu policy $PY -I -B $M1/code12_policy.py && POLICY_DONE=1
PREEMPT=

if [ $MASK_OK != 1 ]; then
  while ! s1u_seen && [ $(date +%s) -lt $B_LAST ]; do sleep 20; done
  mark "skipped: mask check FAIL or not run (s1u.done $(s1u_seen && echo seen || echo "not seen by $LB"))"
  say "done (no map arm: masks not checked)"; exit 1
fi
last=
while ! s1u_seen; do
  [ $(date +%s) -ge $B_LAST ] && { mark "skipped: no s1u.done by $LB"; say "no s1u.done by $LB: the map arm does not start tonight"; exit 0; }
  [ "$last" != s1u ] && say "B waits for $DONE_S1U"; last=s1u; sleep 20
done
say "B: s1u.done seen"
tries=0; endwhy=
while :; do
  tries=$((tries + 1)); [ $tries -gt 4 ] && { endwhy="4 tries used"; say "4 tries used; stopping"; break; }
  ok=0; last=
  while :; do
    [ $(date +%s) -ge $B_LAST ] && { endwhy="no clear gap by $LB (try $tries)"; say "no clear gap by $LB: the map arm does not start tonight"; break 2; }
    w=$(map_why_not)
    if [ -n "$w" ]; then ok=0; [ "$w" != "$last" ] && say "B waits: $w"; last=$w; sleep 10; continue; fi
    ok=$((ok + 1)); last=; [ $ok -ge 3 ] && break; sleep 10
  done
  peer lock share 20m --cpu 200 --gpu 50 "MEM 17 GB: Breakthrough map arm q1u_0_live (NewIdeas' chain step 2; flips only, no timing)" >> $LOG 2>&1 \
    || { say "share lock not granted"; sleep 30; continue; }
  LOCKED=1; S0=$(swap_mb); say "B go (try $tries; NO-SWAP $(noswap) GB, swap $S0 MB)"
  $PY -I -B $M1/q1u0_run_map.py >> $LOG 2>&1 & P=$!
  PP=
  if [ "$POLICY_DONE" != 1 ] && [ $MASK_OK = 1 ]; then
    taskpolicy -b nice -n 15 $PY -I -B $M1/code12_policy.py > $POLOG 2>&1 & PP=$!
    say "policy replay beside the arm (pid $PP; one background core, < 1 GB; log $POLOG; stopped when the arm ends)"
  fi
  bad=0; pz=0; why=
  while kill -0 $P 2>/dev/null; do
    sleep 15; w=
    [ $(date +%s) -ge $B_STOP ] && { why="$LS hard stop"; say "stopping: $why"; stop_tree $P; break; }
    [ $(( $(swap_mb) - S0 )) -gt 2048 ] && { why="swap grew > 2 GB"; say "stopping: $why"; stop_tree $P; break; }
> (a quote from the project owner, kept private)
    if [ -e "$Q/PAUSE" ]; then pz=$((pz + 1)); else pz=0; fi
    if quiet_others; then w="a quiet window (others)"; elif lid_closed; then w="lid closed"; elif [ $pz -ge 20 ]; then w="PAUSE over 5 min"; fi
    if [ -n "$w" ]; then bad=$((bad + 1)); else bad=0; fi
    [ $bad -ge 3 ] && { why=$w; say "stopping for: $w"; stop_tree $P; break; }
  done
  wait $P 2>/dev/null; rc=$?
  if [ -n "$PP" ]; then   # the replay never outlives the arm (Leap's tests come next)
    if kill -0 $PP 2>/dev/null; then kill -TERM $PP 2>/dev/null; sleep 2; kill -KILL $PP 2>/dev/null; wait $PP 2>/dev/null
      say "policy replay stopped with the arm (deferred)"
    else wait $PP 2>/dev/null; prc=$?; say "policy replay ended rc $prc (beside the arm)"; [ $prc = 0 ] && POLICY_DONE=1; fi
    PP=
  fi
  peer unlock >> $LOG 2>&1; LOCKED=0
  endwhy="run ended rc $rc${why:+, stopped: $why} (try $tries)"
  say "B run ended (rc $rc${why:+, stopped: $why}); unlocked; free memory: NO-SWAP $(noswap) GB"
  [ -f $DM/out/q1u_0_live.json ] && break
  [ -n "$why" ] && [ "$why" = "$LS hard stop" ] && break
done
mark "arm $([ -f $DM/out/q1u_0_live.json ] && echo 'result written' || echo 'no result'): $endwhy"
if [ -f $DM/out/q1u_0_live.json ]; then
  SCREEN_OUT=~/Breakthrough/data/m1 $PY -I -B $DM/dm_screen_read.py q1u_0_live qom_0_live_c51 --also q1u_0.9_live >> $LOG 2>&1
  say "read-out: $(grep '^VERDICT' ~/Breakthrough/data/m1/SCREEN_read_q1u_0_live.txt 2>/dev/null | head -3 | tr '\n' ' ')"
fi
say "done"
