#!/bin/bash
# Breakthrough's 12 GB neighbour beside EvalSuite's w24 card gate (Tue 10 Nov 2026 from ~02:35; NewIdeas 23:30: EvalSuite
# confirmed margin for one <= 12 GB neighbour; my flip-only arms go first; Fit12's 12 GB IFBench arm starts on my done-file).
# Run ATTACHED (Bash run_in_background) from ~/Breakthrough:  bash scripts/m1/beside_gate.sh      DRY=1: print the gates, exit.
# Order (NewIdeas): qom_0_live_c47 (QOM0 at Fit12's call pool, pre-registered in docs/PRICE_12GB_90_CODE.md), then q1u_0_live
# if tonight's chain did not run it. Flip-only map arms, one at a time, each <= 12.0 GB lifetime footprint MEASURED.
# 0. Waits until tonight's runner (s1u_map_night.sh) has ended, so my jobs never overlap.
# 1. CPU prep (one core, taskpolicy -b; Kamen idle, no quiet window, lid open, no PAUSE): q1u_0_live masks if missing,
#    qom_0_live_c47 masks (NewIdeas' builder; _c47 = 47 base slots per layer), q1u0_mask_check.py and c47_mask_check.py.
# 2. DM_C from tonight's measured peak (the night log's '[q1u0] PEAK FOOTPRINT', the largest if several): <= 11.5 GB -> 51;
#    above -> 51 minus ceil((peak - 11.3) / 0.104) slots (48 layers x 2.16 MB per slot), floor 24; no print -> 32. The
#    wrapper (bg_run_map.py) ends itself if its lifetime footprint passes 11.9 GB (a 0.5 s self-guard).
# 3. An arm starts when ~/EvalSuite/results/w24_gate_eta.txt (<= 30 min old) says 'neighbour may start: yes' and not
#    'passes 13:30: yes'; Kamen idle; no map or S1u process; no quiet window; no other shared run with MEM > 5 GB or >= 50 %
#    GPU except the gate's own (EvalSuite); lid open; no PAUSE; NO-SWAP >= peak + 8 (peak: the measured one at DM_C 51,
#    else 12); by 02:45 and only if its expected length (tonight's wall x 1.5, else 15 min) ends by 02:58. Share lock
#    'MEM 12 GB: Breakthrough <arm> flips beside the gate' (the gate yields only to a shared MEM run above 12 GB, or to two
#    above 5). Guards every 5 s: swap +0.5 GB over its value at the start (the gate stops at +2 GB), 'passes 13:30: yes', an
#    ETA file older than 30 min, the 02:58 hard stop, a chat process of Kamen's above 2 GB, and (3 strikes) a quiet window
#    of others, lid closed, PAUSE over 5 min. A stop is TERM then KILL after 20 s; never CONT. Then unlock and read out
#    (dm_screen_read.py, SCREEN_OUT = data/m1).
# 4. ~/Breakthrough/data/m1/beside_gate.done: written once, with the reason, when my arms end or none can start, by 03:00 at
#    the latest (every wait loop checks 03:00; an EXIT trap stops my job, drops my lock and writes it too).
# 5. Then code12_policy.py (CPU, one core, < 1 GB) if it has not finished, whenever no quiet window is out (NewIdeas 23:30),
#    until 07:00.
# Every job starts with cwd ~/Breakthrough (never inside ~/FableGod).
cd ~/Breakthrough || exit 1
PY=~/BigMoE/.venv/bin/python; M1=~/Breakthrough/scripts/m1; D1=~/Breakthrough/data/m1; DM=~/FableGod/decision_map
Q=$HOME/.local/share/quiet; ETA=~/EvalSuite/results/w24_gate_eta.txt
LOG=$D1/beside_gate.log; NLOG=$D1/s1u_map_night.log; MK=$D1/beside_gate.done
C47=$D1/masks/qom_0_live_c47; Q1U=~/Breakthrough/data/dm_masks/q1u_0_live
START_LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-10 02:45:00" +%s)
STOP=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-10 02:58:00" +%s)
DONE_BY=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-10 03:00:00" +%s)
CPU_LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-10 07:00:00" +%s)
say() { echo "[bg] $* ($(date '+%a %H:%M:%S'))" | tee -a $LOG; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
idle_s() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF / 1000000000); exit}'; }
chat_big() { ps -axo rss=,comm=,args= | awk '$2 ~ /[Pp]ython/ && /chat\.py/ && $1 > 2097152 {n++} END {print n + 0}'; }
map_procs() { ps -axo comm=,args= | awk '$1 ~ /[Pp]ython/ && /run_map\.py/ {n++} END {print n + 0}'; }
s1u_procs() { ps -axo comm=,args= | awk '$1 ~ /[Pp]ython/ && /fit_onecall\.py/ {n++} END {print n + 0}'; }
night_alive() { ps -axo command= | grep -qE '^bash scripts/m1/s1u_map_night\.sh'; }
quiet_others() { peer ls 2>/dev/null | grep -v "^You:" | grep -qE "quiet lock until|benchmarking: quiet lock|waiting for a quiet window"; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
big_others() {   # other sessions' big shared runs (MEM > 5 GB in the reason, or >= 50 % GPU), the gate's own (EvalSuite) excluded
  peer ls 2>/dev/null | grep "^- " | grep -v " ~/EvalSuite: " | grep -oE "· shared run \(others may work\)[^·]*" | awk '{ big = 0
    if (match($0, /MEM [~]?[0-9]+ GB/)) { m = substr($0, RSTART + 4, RLENGTH - 7); gsub("~", "", m); if (m + 0 > 5) big = 1 }
    if (match($0, /[0-9]+% GPU/)) { if (substr($0, RSTART, RLENGTH - 5) + 0 >= 50) big = 1 }
    n += big } END {print n + 0}'
}
eta_age() { [ -f $ETA ] && echo $(( $(date +%s) - $(stat -f %m $ETA) )) || echo 99999; }
eta_may() { [ "$(eta_age)" -le 1800 ] && grep -q "^neighbour may start: yes" $ETA && ! grep -q "^passes 13:30: yes" $ETA; }
eta_stop() { [ -f $ETA ] && grep -q "^passes 13:30: yes" $ETA; }
kamen_busy() {
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
arm_why_not() {   # $1 = NO-SWAP needed (GB)
  local k; k=$(cpu_why_not); [ -n "$k" ] && { echo "$k"; return; }
  eta_may || { echo "the gate's ETA file does not say 'neighbour may start: yes' (age $(eta_age) s)"; return; }
  [ "$(map_procs)" -gt 0 ] && { echo "a map process is running"; return; }
  [ "$(s1u_procs)" -gt 0 ] && { echo "an S1u process is running"; return; }
  [ "$(big_others)" -gt 0 ] && { echo "another session's big shared run (not the gate's)"; return; }
  [ "$(noswap)" -lt "$1" ] && echo "NO-SWAP $(noswap) GB < $1"
}
stop_tree() {   # TERM the tree, KILL after 20 s (works on a stopped process too); never CONT
  local p=$1 c; local all="$p $(pgrep -P $p)"
  for c in $(pgrep -P $p); do all="$all $(pgrep -P $c)"; done
  kill -TERM $all 2>/dev/null; sleep 20; kill -KILL $all 2>/dev/null
}
mark() {   # the done-file Fit12's IFBench arm starts on: written once, with the reason
  [ -e $MK ] && return 0
  echo "$1 ($(date '+%a %H:%M:%S'))" > $MK; say "done-file $MK written: $1"
}
past() { [ $(date +%s) -ge $1 ]; }
# tonight's measured peak and wall (the night wrapper's print), the largest peak if several tries printed
NPEAK=$(grep -h "^\[q1u0\] PEAK FOOTPRINT" $NLOG 2>/dev/null | sed -E 's/.*PEAK FOOTPRINT ([0-9.]+) GB.*/\1/' | sort -n | tail -1)
NWALL=$(grep -h "^\[q1u0\] PEAK FOOTPRINT" $NLOG 2>/dev/null | tail -1 | sed -E 's/.*wall ([0-9]+) s.*/\1/')
plan() {   # sets DMC, NEED (NO-SWAP GB), EXP (expected arm seconds beside the gate) from tonight's print
  if [ -z "$NPEAK" ]; then DMC=32; NEED=20; why="no measured peak tonight: DM_C 32 (self-guard 11.9 GB)"
  else
    local cut; cut=$(awk -v p=$NPEAK 'BEGIN { c = (p - 11.3) / 0.104; c = (c == int(c)) ? c : int(c) + 1; print (p <= 11.5) ? 0 : c }')
    DMC=$((51 - cut)); [ $DMC -lt 24 ] && DMC=24
    if [ $DMC = 51 ]; then NEED=$(awk -v p=$NPEAK 'BEGIN { c = int(p); if (c < p) c++; print c + 8 }'); else NEED=20; fi
    why="tonight's peak $NPEAK GB at DM_C 51 -> DM_C $DMC"
  fi
  if [ -n "$NWALL" ]; then EXP=$(( NWALL * 3 / 2 )); else EXP=900; fi
}
plan
if [ -n "$DRY" ]; then
  echo "DRY $(date '+%H:%M:%S'): night runner alive: $(night_alive && echo yes || echo no); $why; NO-SWAP needed $NEED; expected arm $EXP s"
  echo "  ETA age $(eta_age) s, may start: $(eta_may && echo yes || echo no), passes 13:30: $(eta_stop && echo yes || echo no)"
  echo "  idle $(idle_s) s; chat.py > 2 GB: $(chat_big); map: $(map_procs); S1u: $(s1u_procs); big others (gate excluded): $(big_others);" \
       "NO-SWAP $(noswap) GB; swap $(swap_mb) MB; quiet others: $(quiet_others && echo yes || echo no); lid closed: $(lid_closed && echo yes || echo no);" \
       "PAUSE: $([ -e $Q/PAUSE ] && echo yes || echo no); done-file: $([ -e $MK ] && echo yes || echo no)"
  echo "  cpu_why_not: '$(cpu_why_not)'; arm_why_not: '$(arm_why_not $NEED)'"
  echo "  masks: q1u_0_live $(ls $Q1U/*.npz 2>/dev/null | wc -l | tr -d ' '), qom_0_live_c47 $(ls $C47/*.npz 2>/dev/null | wc -l | tr -d ' ');" \
       "results: q1u_0_live $([ -f $DM/out/q1u_0_live.json ] && echo yes || echo no), qom_0_live_c47 $([ -f $DM/out/qom_0_live_c47.json ] && echo yes || echo no)"
  echo "  deadlines: arm start by $(date -r $START_LAST '+%a %H:%M:%S'), stop $(date -r $STOP '+%a %H:%M:%S'), done-file by $(date -r $DONE_BY '+%a %H:%M:%S'), CPU replay until $(date -r $CPU_LAST '+%a %H:%M:%S')"
  exit 0
fi

LOCKED=0
on_exit() {   # any exit, a TERM included: stop my running job (TERM, KILL after 3 s), drop my lock, then the done-file
  local c k=
  for c in $(pgrep -P $$); do kill -0 $c 2>/dev/null && k="$k $c $(pgrep -P $c)"; done   # live children only
  [ -n "${k// /}" ] && { kill -TERM $k 2>/dev/null; sleep 3; kill -KILL $k 2>/dev/null; say "exit: stopped my running job ($k )"; }
  [ "$LOCKED" = 1 ] && { peer unlock >> $LOG 2>&1; LOCKED=0; say "exit: lock dropped"; }
  mark "runner ended before a result line (an exit or a stop)"
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# run_cpu <name> <deadline epoch> <command...>: one CPU job, restarted while the gate is closed, until it exits or the deadline
run_cpu() {
  local name=$1 dl=$2; shift 2; local last= w rc
  while :; do
    past $dl && { say "$name: not done by $(date -r $dl '+%H:%M'); given up"; return 9; }
    w=$(cpu_why_not)
    if [ -n "$w" ]; then [ "$w" != "$last" ] && say "$name waits: $w"; last=$w; sleep 20; continue; fi
    last=; say "$name: go"
    taskpolicy -b nice -n 15 "$@" >> $LOG 2>&1 & local P=$!
    local stopped=
    while kill -0 $P 2>/dev/null; do
      sleep 5; w=$(cpu_why_not)
      if [ -n "$w" ]; then say "$name: stopping for: $w"; stop_tree $P; stopped=1; break; fi
      past $dl && { say "$name: stopping: past $(date -r $dl '+%H:%M')"; stop_tree $P; wait $P 2>/dev/null; return 9; }
    done
    wait $P 2>/dev/null; rc=$?
    [ -n "$stopped" ] && continue
    say "$name: ended rc $rc"; return $rc
  done
}

rm -f $MK
say "armed: after tonight's runner; prep (masks, checks); arms qom_0_live_c47 then q1u_0_live (if not run tonight) beside the gate when its ETA file says 'neighbour may start: yes', start by 02:45, stop 02:58; done-file by 03:00"
last=
while night_alive; do
  past $START_LAST && { mark "none could start: tonight's runner still alive at 02:45"; exit 0; }
  [ "$last" != night ] && say "waits for tonight's runner to end"; last=night; sleep 30
done
say "tonight's runner has ended; $why; NO-SWAP needed $NEED GB; expected arm $EXP s"

# 1. prep
if [ "$(ls $Q1U/*.npz 2>/dev/null | wc -l | tr -d ' ')" != 36 ]; then
  rm -rf $Q1U
  run_cpu masks_q1u $START_LAST $PY -I -B ~/NewIdeas/scripts/dm_oea_masks.py ~/Breakthrough/data/dm_masks q1u_0_live
fi
if [ "$(ls $C47/*.npz 2>/dev/null | wc -l | tr -d ' ')" != 36 ]; then
  rm -rf $C47
  run_cpu masks_c47 $START_LAST $PY -I -B ~/NewIdeas/scripts/dm_oea_masks.py $D1/masks qom_0_live_c47
fi
Q1U_OK=0; C47_OK=0
run_cpu check_q1u $START_LAST $PY -I -B $M1/q1u0_mask_check.py && Q1U_OK=1
run_cpu check_c47 $START_LAST $PY -I -B $M1/c47_mask_check.py && C47_OK=1
say "mask checks: q1u_0_live $([ $Q1U_OK = 1 ] && echo PASS || echo FAIL), qom_0_live_c47 $([ $C47_OK = 1 ] && echo PASS || echo FAIL)"

# 3. the arms, one at a time
ARMS=
[ $C47_OK = 1 ] && [ ! -f $DM/out/qom_0_live_c47.json ] && ARMS="qom_0_live_c47"
[ $Q1U_OK = 1 ] && [ ! -f $DM/out/q1u_0_live.json ] && ARMS="$ARMS q1u_0_live"
[ -z "${ARMS// /}" ] && { mark "none could start: no arm left with a PASSed mask check"; ARMS=; }
ran=; endwhy=
for ARM in $ARMS; do
  if [ $ARM = qom_0_live_c47 ]; then MD=$C47; REF="qom_0_live_c51 --also q1u_0_live"; else MD=$Q1U; REF="qom_0_live_c51 --also q1u_0.9_live"; fi
  tries=0
  while :; do
    tries=$((tries + 1)); [ $tries -gt 3 ] && { endwhy="$endwhy $ARM: 3 tries used;"; break; }
    ok=0; last=; go=0
    while :; do
      past $DONE_BY && { endwhy="$endwhy $ARM: 03:00 passed;"; break 2; }
      if past $START_LAST || [ $(( $(date +%s) + EXP )) -gt $STOP ]; then
        endwhy="$endwhy $ARM: no start (by 02:45 and ending by 02:58 with ~$EXP s);"; break 2
      fi
      w=$(arm_why_not $NEED)
      if [ -n "$w" ]; then ok=0; [ "$w" != "$last" ] && say "$ARM waits: $w"; last=$w; sleep 10; continue; fi
      ok=$((ok + 1)); last=; [ $ok -ge 3 ] && { go=1; break; }; sleep 10
    done
    [ $go = 1 ] || break
    peer lock share 20m --cpu 200 --gpu 50 "MEM 12 GB: Breakthrough $ARM flips beside the gate" >> $LOG 2>&1 \
      || { say "share lock not granted"; sleep 30; continue; }
    LOCKED=1; S0=$(swap_mb); say "$ARM go (try $tries; DM_C $DMC; NO-SWAP $(noswap) GB, swap $S0 MB)"
    DM_C=$DMC BG_PEAK_MAX=11.9 $PY -I -B $M1/bg_run_map.py $ARM $MD >> $LOG 2>&1 & P=$!
    bad=0; pz=0; why=
    while kill -0 $P 2>/dev/null; do
      sleep 5; w=
      past $STOP && { why="02:58 hard stop"; say "$ARM stopping: $why"; stop_tree $P; break; }
      [ $(( $(swap_mb) - S0 )) -gt 512 ] && { why="swap grew > 0.5 GB"; say "$ARM stopping: $why"; stop_tree $P; break; }
      eta_stop && { why="the gate's ETA says 'passes 13:30: yes'"; say "$ARM stopping: $why"; stop_tree $P; break; }
      [ "$(eta_age)" -gt 1800 ] && { why="the gate's ETA file is older than 30 min"; say "$ARM stopping: $why"; stop_tree $P; break; }
> (a quote from the project owner, kept private)
      if [ -e "$Q/PAUSE" ]; then pz=$((pz + 1)); else pz=0; fi
      if quiet_others; then w="a quiet window (others)"; elif lid_closed; then w="lid closed"; elif [ $pz -ge 60 ]; then w="PAUSE over 5 min"; fi
      if [ -n "$w" ]; then bad=$((bad + 1)); else bad=0; fi
      [ $bad -ge 3 ] && { why=$w; say "$ARM stopping for: $w"; stop_tree $P; break; }
    done
    wait $P 2>/dev/null; rc=$?; peer unlock >> $LOG 2>&1; LOCKED=0
    say "$ARM run ended (rc $rc${why:+, stopped: $why}); unlocked; NO-SWAP $(noswap) GB, swap $(swap_mb) MB"
    if [ -f $DM/out/$ARM.json ]; then
      endwhy="$endwhy $ARM: result written;"; ran="$ran $ARM"
      SCREEN_OUT=$D1 $PY -I -B $DM/dm_screen_read.py $ARM $REF >> $LOG 2>&1
      say "$ARM read-out: $(grep '^VERDICT' $D1/SCREEN_read_$ARM.txt 2>/dev/null | head -3 | tr '\n' ' ')"
      break
    fi
    if [ "$rc" = 7 ]; then   # the self-guard fired (> 11.9 GB): 12 slots fewer (-1.25 GB) for every later try, if time allows
      endwhy="$endwhy $ARM: self-guard at DM_C $DMC;"; DMC=$((DMC - 12)); [ $DMC -lt 24 ] && DMC=24; NEED=20
      say "$ARM: self-guard fired; DM_C -> $DMC for the next try"; continue
    fi
    if [ -n "$why" ] && [ "$why" != "a quiet window (others)" ] && [ "$why" != "lid closed" ] && [ "$why" != "PAUSE over 5 min" ]; then
      endwhy="$endwhy $ARM: stopped ($why);"; break 2
    fi
    [ -z "$why" ] && { endwhy="$endwhy $ARM: rc $rc without a result;"; break; }
  done
done
mark "beside-gate arms done:${endwhy:- none}"

# 5. the CPU replay, if it has not finished
if ! grep -q "^done in" $D1/code12_policy.out 2>/dev/null; then
  run_cpu policy $CPU_LAST $PY -I -B $M1/code12_policy.py
else say "policy replay already finished"; fi
say "done"
