#!/bin/bash
# qom_0_live_c47 in Tuesday's day queue (Breakthrough, Tue 10 Nov 2026 03:4x). NewIdeas' TUESDAY ORDER (~/NewIdeas/QUEUE.md,
# TUE 02:02), AFTER THE GATE, one > 5 GB job at a time, in this order as each is ready: 1. BigMoE's W24e (~15 min, quiet;
# after EvalSuite's 'runner 61 (w24) ended'); 2. this arm (~16 GB, ~4 min, flips only); 4. Fit12's IFBench arm continues,
# "yielding at chunk ends to any quiet ticket". Pre-registered 8a5a382 (docs/PRICE_12GB_90_CODE.md), amended before any data:
# the day slot, DM_C 51 (the harness's own cache: the same flips as at any DM_C, only speed and memory change; c51 ran at 51),
# self-guard 20 GB (the arm measured 15.96 GB at DM_C 51 on Mon night). Kamen's weekday schedule: big tests 00:30 to 15:00.
# Run ATTACHED (Bash run_in_background) from ~/Breakthrough:  bash scripts/m1/c47_day.sh       DRY=1: print the gates, exit.
# 1. WAIT for the w24 card gate's end: EvalSuite's card_w24_start.log gets 'runner 61 (w24) ended' after this arming, or
#    neither card_w24_start.sh nor day_runner61.sh is alive (polled every 30 s).
# 2. W24e FIRST: wait while a BigMoE W24e process runs (turn45.sh ... BigMoE W24e, or scripts/w24e.sh). Go on once its
#    report data/w24/w24e_report.txt is newer than the gate's end, or BigMoE's w24.log says 'W24e not run', or 30 min after
#    the gate's end with no W24e process (nobody started it: "as each is ready"; it then waits ~5 min behind this arm).
# 3. A CLEAR GAP, 3 checks 10 s apart: no PAUSE / HOLD, lid open, no chat.py of Kamen's above 2 GB, no card runner, no W24e,
#    no python run_map.py process, no quiet window held or queued by others, no other session's big shared run (MEM > 5 GB
#    in its reason or >= 50 % GPU), NO-SWAP >= 25 GB (peak 16 + 8 + 1). Start by 14:50 Tue; else no slot today.
#    The masks must be as committed (36 files, git clean) or nothing runs.
# LOCK: a share lock 'MEM 17 GB: ...'. If only another session's big shared run is in the way for 2 min (Fit12's chunks),
#    queue a quiet ticket instead: shared runs cannot start while it waits, and it is granted when no shared run is left.
#    20 s after any lock the gap is checked again (gated waiters start together); not clear = unlock and wait again.
# ARM: DM_C=51 BG_PEAK_MAX=20 bg_run_map.py qom_0_live_c47 data/m1/masks/qom_0_live_c47 (in-memory registration as
#    dm_ablations.q1_union(<abs dir>), like qom_0_live_c51's; result ~/FableGod/decision_map/out/qom_0_live_c47.json).
# GUARDS every 15 s: hard stop 14:58; swap +2 GB over the start; a chat.py of Kamen's above 2 GB; a quiet window of others,
#    lid closed or PAUSE over 5 min (3 strikes). A stop = TERM, KILL after 20 s (never CONT). Unlock after every try; 3 tries.
# READ-OUT (seconds, one core): SCREEN_OUT=data/m1 dm_screen_read.py qom_0_live_c47 qom_0_live_c51 --also q1u_0_live.
# EXIT trap: stop my own running job (TERM, KILL after 3 s) and drop my lock. Every job starts with cwd ~/Breakthrough.
cd ~/Breakthrough || exit 1
PY=~/BigMoE/.venv/bin/python; M1=~/Breakthrough/scripts/m1; DM=~/FableGod/decision_map; Q=$HOME/.local/share/quiet
ARM=qom_0_live_c47; MDIR=data/m1/masks/qom_0_live_c47; OUT=$DM/out/$ARM.json; LOG=~/Breakthrough/data/m1/c47_day.log
GLOG=~/EvalSuite/results/card_w24_start.log; W24LOG=~/BigMoE/data/w24/w24.log; W24REP=~/BigMoE/data/w24/w24e_report.txt
START_BY=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-10 14:50:00" +%s)
HARD_STOP=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-10 14:58:00" +%s)
LSB=14:50; LHS=14:58; FALLBACK_S=1800
LOCKR="MEM 17 GB: Breakthrough qom_0_live_c47 flips (NewIdeas TUE order item 2), ~5 min"
G0=$(wc -l < $GLOG 2>/dev/null | tr -d ' '); G0=${G0:-0}   # gate log lines at arming: only later lines count

say() { echo "[c47] $* ($(date '+%a %H:%M:%S'))" | tee -a $LOG; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
idle_s() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF / 1000000000); exit}'; }
chat_big() { ps -axo rss=,comm=,args= | awk '$2 ~ /[Pp]ython/ && /chat\.py/ && $1 > 2097152 {n++} END {print n + 0}'; }
map_procs() { ps -axo comm=,args= | awk '$1 ~ /[Pp]ython/ && /run_map\.py/ {n++} END {print n + 0}'; }
big_others() {   # other sessions' big shared runs: MEM > 5 GB in the reason, or >= 50 % GPU (peer ls cuts long reasons)
  peer ls 2>/dev/null | grep "^- " | grep -oE "· shared run \(others may work\)[^·]*" | awk '{ big = 0
    if (match($0, /MEM [~]?[0-9]+ GB/)) { m = substr($0, RSTART + 4, RLENGTH - 7); gsub("~", "", m); if (m + 0 > 5) big = 1 }
    if (match($0, /[0-9]+% GPU/)) { if (substr($0, RSTART, RLENGTH - 5) + 0 >= 50) big = 1 }
    n += big } END {print n + 0}'
}
quiet_others() { peer ls 2>/dev/null | grep -v "^You:" | grep -qE "quiet lock until|benchmarking: quiet lock|waiting for a quiet window"; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
gate_running() { pgrep -f 'tools/day_runner6[1]\.sh' > /dev/null || pgrep -f 'tools/card_w24_star[t]\.sh' > /dev/null; }
gate_line() { tail -n +$((G0 + 1)) $GLOG 2>/dev/null | grep 'runner 61 (w24) ended' | tail -1; }
gate_ended() { [ -n "$(gate_line)" ] || ! gate_running; }
w24e_running() { ps -axo command= | grep -qE '^(/bin/)?bash ([^ ]*/)?scripts/(turn45\.sh [0-9]+ [0-9]+ BigMoE W24e|w24e\.sh)'; }
w24e_state() {   # running | done | declined | late | wait   (uses T_GE and W0, set when the gate's end is seen)
  w24e_running && { echo running; return; }
  [ -f $W24REP ] && [ "$(stat -f %m $W24REP)" -gt "$T_GE" ] && { echo done; return; }
  tail -n +$((W0 + 1)) $W24LOG 2>/dev/null | grep -q 'W24e not run' && { echo declined; return; }
  [ $(( $(date +%s) - T_GE )) -ge $FALLBACK_S ] && { echo late; return; }
  echo wait
}
masks_ok() {
  [ "$(ls $MDIR/*.npz 2>/dev/null | wc -l | tr -d ' ')" = 36 ] && git ls-files --error-unmatch $MDIR > /dev/null 2>&1 \
    && [ -z "$(git status --porcelain -- $MDIR)" ]
}
gap_why_not() {   # empty = clear; a reason starting 'BIG:' = only another session's big shared run is in the way
  [ -e "$Q/PAUSE" ] && { echo "PAUSE"; return; }
  [ -e "$Q/HOLD" ] && { echo "HOLD"; return; }
  lid_closed && { echo "lid closed"; return; }
> (a quote from the project owner, kept private)
  gate_running && { echo "an EvalSuite card runner is alive"; return; }
  w24e_running && { echo "BigMoE's W24e runs (it goes first)"; return; }
  [ "$(map_procs)" -gt 0 ] && { echo "a map process is running"; return; }
  quiet_others && { echo "a quiet window (others)"; return; }
  [ "$(big_others)" -gt 0 ] && { echo "BIG: another session's big shared run (MEM > 5 GB or >= 50 % GPU)"; return; }
  [ "$(noswap)" -lt 25 ] && echo "NO-SWAP $(noswap) GB < 25"
}
stop_tree() {   # TERM the tree, KILL after 20 s (works on a stopped process too); never CONT
  local p=$1 c; local all="$p $(pgrep -P $p)"
  for c in $(pgrep -P $p); do all="$all $(pgrep -P $c)"; done
  kill -TERM $all 2>/dev/null; sleep 20; kill -KILL $all 2>/dev/null
}
if [ -n "$DRY" ]; then
  T_GE=$(date +%s); W0=$(wc -l < $W24LOG 2>/dev/null | tr -d ' '); W0=${W0:-0}
  echo "DRY $(date '+%H:%M:%S'): gate running: $(gate_running && echo yes || echo no); gate end line: '$(gate_line)'; gate ended: $(gate_ended && echo yes || echo no);" \
       "W24e running: $(w24e_running && echo yes || echo no); W24e state (as if the gate ended now): $(w24e_state); masks ok: $(masks_ok && echo yes || echo no);" \
       "idle $(idle_s) s; chat.py > 2 GB: $(chat_big); map processes: $(map_procs); big others: $(big_others); quiet others: $(quiet_others && echo yes || echo no);" \
       "lid closed: $(lid_closed && echo yes || echo no); PAUSE: $([ -e $Q/PAUSE ] && echo yes || echo no); HOLD: $([ -e $Q/HOLD ] && echo yes || echo no);" \
       "NO-SWAP $(noswap) GB; swap $(swap_mb) MB; out file: $([ -f $OUT ] && echo EXISTS || echo none); gap_why_not: '$(gap_why_not)'"
  echo "DRY deadlines: start by $(date -r $START_BY '+%a %H:%M:%S'), hard stop $(date -r $HARD_STOP '+%a %H:%M:%S'); gate log lines at arming $G0"
  exit 0
fi

LOCKED=0
on_exit() {   # any exit, a TERM included: stop my own running job (TERM, KILL after 3 s) and drop my lock
  local c k=
  for c in $(pgrep -P $$); do kill -0 $c 2>/dev/null && k="$k $c $(pgrep -P $c)"; done   # live children only
  [ -n "${k// /}" ] && { kill -TERM $k 2>/dev/null; sleep 3; kill -KILL $k 2>/dev/null; say "exit: stopped my running job ($k )"; }
  [ "$LOCKED" = 1 ] && { peer unlock >> $LOG 2>&1; LOCKED=0; say "exit: lock dropped"; }
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
[ -f $OUT ] && { say "$OUT already exists: nothing to run (run_map.py --skip-done would skip it)"; exit 0; }
masks_ok || { say "masks $MDIR are not the committed 36 files: nothing runs"; exit 2; }
say "armed (pid $$; NewIdeas TUE 02:02 order item 2): waits for the w24 gate's end, then W24e first; start by $LSB, hard stop $LHS"

last=
until gate_ended; do
  [ $(date +%s) -ge $START_BY ] && { say "the w24 gate still runs at $LSB: no slot today"; exit 0; }
  [ -z "$last" ] && say "waits for the w24 card gate's end"; last=1; sleep 30
done
T_GE=$(date +%s); W0=$(wc -l < $W24LOG 2>/dev/null | tr -d ' '); W0=${W0:-0}
gl=$(gate_line | cut -c1-120); say "the w24 card gate has ended: ${gl:-its runners are gone (no end line)}"

last=
while :; do
  s=$(w24e_state)
  case $s in
    done) say "W24e done ($W24REP newer than the gate's end)"; break;;
    declined) say "W24e not run (BigMoE's w24.log): this arm goes"; break;;
    late) say "W24e not started 30 min after the gate's end: this arm goes (W24e would wait ~5 min behind it)"; break;;
  esac
  [ $(date +%s) -ge $START_BY ] && { say "W24e still $s at $LSB: no slot today"; exit 0; }
  [ "$s" != "$last" ] && say "W24e first: $s"; last=$s; sleep 30
done

tries=0; endwhy=
while :; do
  tries=$((tries + 1)); [ $tries -gt 3 ] && { endwhy="3 tries used"; say "3 tries used; stopping"; break; }
  ok=0; last=; bigsince=; mode=share
  while :; do
    [ $(date +%s) -ge $START_BY ] && { endwhy="no clear gap by $LSB (try $tries)"; say "no clear gap by $LSB: no slot today"; break 2; }
    masks_ok || { endwhy="masks changed"; say "masks $MDIR changed: nothing runs"; break 2; }
    w=$(gap_why_not)
    case $w in
      BIG:*) [ -z "$bigsince" ] && bigsince=$(date +%s)
             if [ $(( $(date +%s) - bigsince )) -ge 120 ]; then mode=quiet; break; fi;;
      *) bigsince=;;
    esac
    if [ -n "$w" ]; then ok=0; [ "$w" != "$last" ] && say "waits: $w"; last=$w; sleep 10; continue; fi
    ok=$((ok + 1)); last=; [ $ok -ge 3 ] && break; sleep 10
  done
  if [ $mode = quiet ]; then
    qt=$(( START_BY - $(date +%s) )); [ $qt -gt 5400 ] && qt=5400
    say "only another session's big shared run is in the way for 2 min: queue a quiet ticket (waits up to ${qt} s)"
    PEER_QUEUE_TIMEOUT=$qt peer lock quiet --queue 20m "$LOCKR" >> $LOG 2>&1 || { say "quiet ticket not granted"; continue; }
  else
    peer lock share 20m --cpu 200 --gpu 50 "$LOCKR" >> $LOG 2>&1 || { say "share lock not granted"; sleep 30; continue; }
  fi
  LOCKED=1; sleep 20; w=$(gap_why_not)
  if [ -n "$w" ]; then say "20 s after the $mode lock the gap is not clear ($w): unlock, wait again"; peer unlock >> $LOG 2>&1; LOCKED=0; continue; fi
> (a quote from the project owner, kept private)
  DM_C=51 BG_PEAK_MAX=20 $PY -I -B $M1/bg_run_map.py $ARM $MDIR >> $LOG 2>&1 & P=$!
  bad=0; pz=0; why=
  while kill -0 $P 2>/dev/null; do
    sleep 15; w=
    [ $(date +%s) -ge $HARD_STOP ] && { why="$LHS hard stop"; say "stopping: $why"; stop_tree $P; break; }
    [ $(( $(swap_mb) - S0 )) -gt 2048 ] && { why="swap grew > 2 GB"; say "stopping: $why"; stop_tree $P; break; }
> (a quote from the project owner, kept private)
    if [ -e "$Q/PAUSE" ]; then pz=$((pz + 1)); else pz=0; fi
    if quiet_others; then w="a quiet window (others)"; elif lid_closed; then w="lid closed"; elif [ $pz -ge 20 ]; then w="PAUSE over 5 min"; fi
    if [ -n "$w" ]; then bad=$((bad + 1)); else bad=0; fi
    [ $bad -ge 3 ] && { why=$w; say "stopping for: $w"; stop_tree $P; break; }
  done
  wait $P 2>/dev/null; rc=$?
  peer unlock >> $LOG 2>&1; LOCKED=0
  endwhy="run ended rc $rc${why:+, stopped: $why} (try $tries)"
  say "run ended (rc $rc${why:+, stopped: $why}); unlocked; NO-SWAP $(noswap) GB, swap $(swap_mb) MB"
  [ -f $OUT ] && break
  [ "$why" = "$LHS hard stop" ] && break
done
say "arm $([ -f $OUT ] && echo 'result written' || echo 'no result'): $endwhy"
if [ -f $OUT ]; then
  SCREEN_OUT=~/Breakthrough/data/m1 $PY -I -B $DM/dm_screen_read.py $ARM qom_0_live_c51 --also q1u_0_live >> $LOG 2>&1
  say "read-out: $(grep '^VERDICT' ~/Breakthrough/data/m1/SCREEN_read_$ARM.txt 2>/dev/null | head -3 | tr '\n' ' ')"
fi
say "done"
