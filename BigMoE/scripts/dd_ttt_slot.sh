#!/bin/bash
# TTT30C arms in BigMoE's slot (docs/PREREG_DEEPDRAFT_TTT30C_2026-11-13.md; the orchestrator 10:4x: "running it beside the
# capture is fine if both fit in memory ... no timing claims"). For each arm in order (t30x2, t30f, t30x4):
#  1. smoke: dd_ttt.py --smoke (the PREREG's equality checks on the pass's first slice + 3 steps on a throwaway name, which
#     also shows the MLX peak). The arm runs only if every check prints EQUAL and the run ends under its guard.
#  2. train: t30b's arguments + the arm's one change, yielding to other sessions' quiet locks, a closed lid and a HEALTH
#     ACT (SIGTERM: dd_ttt saves the weights, both Adam groups and the step, exits 3; it resumes when the gates clear).
#     Swap guard +2 GB (SIGTERM, then wait). Peak guard = the arm's MEM (dd_ttt saves and exits 4: the arm stops).
#  3. eval: b5, t30b, the arm's copies and the arm -> data/deepdraft/ttt30c/<arm>/eval.json.
# Memory: a phase starts only if NO-SWAP (the lowest of 3 reads over 40 s; one read beside a trainer can be > 10 GB high)
# >= its MEM + 8 GB. The session's one peer lock is shared with dd_capture_slot.sh (scripts/dd_slot_lock.sh).
# ~/FableGod/deepdraft/STOP_SLOT ends it (after a clean SIGTERM stop of the running phase).
# usage (a harness background task, not nohup): AFTER_PID=5975 [STAGGER=300] bash scripts/dd_ttt_slot.sh [arms]
#        DRY=1: print the plan and the gates, run nothing
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
source scripts/dd_slot_lock.sh                           # STOP, SLOT_STATE, slot_lock
D=data/deepdraft/ttt30c; mkdir -p $D
LOG="$PWD/$D/slot.log"; RLOG="$PWD/$D/run.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; [ -n "$DRY" ] && echo "$*"; }
PY=$HOME/BigMoE/.venv/bin/python
Q="$HOME/.local/share/quiet"
CKD=$HOME/datasets/deepdraft/ckpt
BASE="--manifest data/deepdraft/ttt30b/split.json --pass-seed 1113 --save-at 600,1500,3000"
ARMS=${*:-"t30x2 t30f t30x4"}
arm_flags() { case $1 in t30x2) echo "--xb 10240";; t30f) echo "--feat";; t30x4) echo "--xb 30720";; esac; }
arm_mem() { case $1 in t30x2) echo 20;; t30f) echo 18;; t30x4) echo 24;; esac; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
avail_min() { local a b c; a=$(avail_gb); sleep 20; b=$(avail_gb); sleep 20; c=$(avail_gb)
  [ $b -lt $a ] && a=$b; [ $c -lt $a ] && a=$c; echo $a; }
swap_mb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}'; }
others_quiet() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (benchmarking|waiting for a quiet window)" | grep -c .; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
health_act() { grep -q "Verdict: ACT" "$Q/HEALTH.md" 2>/dev/null; }
gates_clear() { [ "$(others_quiet)" -eq 0 ] && ! lid_closed && ! health_act; }
gate_state() { echo "quiet locks of others $(others_quiet), lid $(lid_closed && echo closed || echo open), health $(health_act && echo ACT || echo ok), NO-SWAP $(avail_gb) GB"; }

run_job() {   # label MEM cmd...: under the shared lock, with the yield / swap / STOP guards; sets RC and Y (why it was stopped)
  local lab=$1 mem=$2; shift 2
  echo "$$ $mem $lab" > "$SLOT_STATE/arm.mem"; slot_lock
  local s0; s0=$(swap_mb); Y=""
  taskpolicy -b "$@" >> "$RLOG" 2>&1 &
  local tp=$!
  while kill -0 $tp 2>/dev/null; do
    sleep 20
    if [ -e "$STOP" ]; then Y=STOP
    elif ! gates_clear; then Y="yield: $(gate_state)"
    elif [ "$(swap_mb)" -gt $((s0 + 2048)) ]; then Y="swap $(swap_mb) MB > $s0 + 2048"; fi
    if [ -n "$Y" ]; then say "$lab: $Y -> SIGTERM $tp"; kill -TERM $tp; break; fi
  done
  wait $tp; RC=$?
  rm -f "$SLOT_STATE/arm.mem"; slot_lock
}

say "slot start (pid $$; arms $ARMS; AFTER_PID ${AFTER_PID:-none}; stop file $STOP)"
if [ -n "$DRY" ]; then
  for x in $ARMS; do say "DRY: $x: $(arm_flags $x), MEM $(arm_mem $x); smoke, train, eval -> $D/$x"; done
  say "DRY: gates now: $(gate_state)"; exit 0
fi
if [ -n "$AFTER_PID" ]; then
  while kill -0 "$AFTER_PID" 2>/dev/null; do [ -e "$STOP" ] && { say "STOP_SLOT before the start: done"; exit 0; }; sleep 60; done
  say "pid $AFTER_PID ended; staggering ${STAGGER:-300} s so the capture's memory shows first"; sleep ${STAGGER:-300}
fi
for x in $ARMS; do
  F=$(arm_flags $x); MEM=$(arm_mem $x); fails=0; mkdir -p $D/$x
  [ -e $D/$x/DONE ] && { say "$x: done already"; continue; }
  phase=smoke; [ -e $D/$x/SMOKE_OK ] && phase=train
  grep -q '"status": "done"' $CKD/dd_$x.json 2>/dev/null && phase=eval
  say "$x: from phase $phase ($F, MEM $MEM GB)"
  while :; do
    [ -e "$STOP" ] && { say "STOP_SLOT: done"; exit 0; }
    need=$MEM; [ $phase = eval ] && need=12
    n=0
    until gates_clear && [ "$(avail_min)" -ge $((need + 8)) ]; do
      [ -e "$STOP" ] && { say "STOP_SLOT: done"; exit 0; }
      [ $((n % 30)) -eq 0 ] && say "$x $phase waits: $(gate_state); needs $((need + 8)) GB"
      n=$((n + 1)); sleep 60
    done
    case $phase in
      smoke)
        L0=$(wc -l < "$RLOG" 2>/dev/null || echo 0)
        run_job "TTT30C $x smoke" $MEM $PY scripts/dd_ttt.py train --run $x $BASE $F --peak-max $MEM --smoke
        out=$(tail -n +$((L0 + 1)) "$RLOG")
        [ -n "$Y" ] && { say "$x smoke stopped ($Y): again later"; [ "$Y" = STOP ] && exit 0; sleep 120; continue; }
        chk=$(echo "$out" | grep -E "smoke:|lambda|DONE" | tr '\n' ' ' | cut -c1-600)
        if [ $RC -eq 0 ] && ! echo "$out" | grep -q DIFFERENT && echo "$out" | grep -q "smoke: b5's loss.*EQUAL" \
           && { [ "$F" = "--feat" ] || [ "$F" = "" ] || echo "$out" | grep -q "zero-init.*EQUAL"; }; then
          say "$x smoke OK: $chk"; touch $D/$x/SMOKE_OK; phase=train
        else
          say "$x smoke FAILED rc $RC: $chk $(echo "$out" | tail -2 | tr '\n' ' ' | cut -c1-300): arm skipped"; break
        fi ;;
      train)
        run_job "TTT30C $x: MEM $MEM" $MEM $PY scripts/dd_ttt.py train --run $x $BASE $F --peak-max $MEM
        say "$x train ended rc $RC${Y:+ ($Y)}: $(grep "^\[dd\]" "$RLOG" | tail -1 | cut -c1-160)"
        [ "$Y" = STOP ] && exit 0
        if [ $RC -eq 0 ]; then phase=eval
        elif [ $RC -eq 4 ]; then say "$x: MLX peak over its $MEM GB guard: arm stopped (saved; a re-plan is needed)"; break
        elif [ -n "$Y" ]; then sleep 120   # a yield, whatever the rc: 3 = saved and exited; 143 = the SIGTERM came during
                                           # dd_ttt's start, before its handler (it writes nothing before it: the last save stands)
        else fails=$((fails + 1)); [ $fails -ge 2 ] && { say "$x: 2 failures without a yield: arm stopped"; break; }; sleep 120; fi ;;
      eval)
        run_job "TTT30C $x eval" 12 $PY scripts/dd_ttt.py eval --runs b5,t30b,${x}_s600,${x}_s1500,${x}_s3000,$x --out $D/$x
        say "$x eval ended rc $RC${Y:+ ($Y)}: $(tail -1 "$RLOG" | cut -c1-220)"
        [ "$Y" = STOP ] && exit 0
        if [ $RC -eq 0 ] && [ -z "$Y" ]; then touch $D/$x/DONE; break
        elif [ -n "$Y" ]; then sleep 120
        else fails=$((fails + 1)); [ $fails -ge 2 ] && { say "$x eval: 2 failures: stopped"; break; }; sleep 120; fi ;;
    esac
  done
done
say "slot end"
