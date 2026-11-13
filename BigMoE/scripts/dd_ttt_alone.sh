#!/bin/bash
# TTT30C's branch arms ALONE (the orchestrator, Fri 13 Nov 15:3x: "Pause the capture for t30x2 and t30x4. The capacity
# question is worth more now than more data"). Order: t30f beside the capture; onebuild; Leap's 1.2.1 release checks;
# then this script: stop the capture at a prompt boundary, t30x2 alone, then t30x4 alone, each with a guard from its own
# smoke, then resume the capture. (If t30f ends before onebuild starts, t30x2 may start right away: its yield to a quiet
# ticket ends the process, so it never holds memory while paused.)
# Why alone: t30x2's smoke peaked at 22.4 GB at step 1 (over its 20 GB guard); t30x4's branch is 3x wider (~36 GB projected),
# and NO-SWAP beside the capture (24 GB) is ~30 GB (PREREG run notes, 8a47d16).
# Starts only when $D/GO_ALONE exists (the session creates it once the order allows) and the old launcher (OLD_PID) and
# every dd_ttt.py run have ended. Then:
#  1. stops the capture: its launcher first (no child but the capture), then the capture's own stop file, so it ends at
#     its next prompt boundary (resumable; SIGTERM after 120 s, SIGKILL 60 s later);
#  2. per arm: smoke (the PREREG's checks + 3 steps) under SMOKE_MEM; train guard = ceil(smoke peak) + 2 GB; train and
#     eval as dd_ttt_slot.sh (share lock, yields to other sessions' quiet locks by SIGTERM: the run saves and EXITS 3,
#     resumes when the gates clear; swap guard +2 GB; a phase starts only if NO-SWAP >= its MEM + 8 GB);
#  3. resumes the capture (exec dd_capture_slot.sh at C 100) unless STOP_SLOT.
# Quiet locks of others = the "· benchmarking", "· quiet lock lapsed" and "· waiting for a quiet window" segments on the
# "^- " lines of peer ls (any unexpired quiet lock, held or queued). Sat 14 Nov 01:0x, the orchestrator's catch: peer shows
# a held quiet lock as "quiet lock lapsed" once nothing has run under it for 8+ min and its session is under 10 % CPU (here
# Leap's runner in a 120-s memory wait); the first gate missed that segment and restarted t30x2 at 00:50:57 inside Leap's
# 1.2.1 ticket, which cost Leap its 10K / 30K peak tests. The gates are read again after the 40-s memory read, and every
# job start is logged in slot.log ("<job>: starts ...").
> (a quote from the project owner, kept private)
# usage (a harness background task): OLD_PID=74726 bash scripts/dd_ttt_alone.sh [arms]   DRY=1: plan and gates only
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
source scripts/dd_slot_lock.sh                           # STOP, SLOT_STATE, slot_lock
D=data/deepdraft/ttt30c; mkdir -p $D
LOG="$PWD/$D/slot.log"; RLOG="$PWD/$D/run.log"
say() { echo "$(date '+%m-%d %H:%M:%S') alone: $*" >> "$LOG"; [ -n "$DRY" ] && echo "$*"; return 0; }
PY=$HOME/BigMoE/.venv/bin/python
Q="$HOME/.local/share/quiet"
CKD=$HOME/datasets/deepdraft/ckpt
CAPD=$HOME/FableGod/deepdraft/capture
GO=$D/GO_ALONE
BASE="--manifest data/deepdraft/ttt30b/split.json --pass-seed 1113 --save-at 600,1500,3000"
ARMS=${*:-"t30x2 t30x4"}
arm_flags() { case $1 in t30x2) echo "--xb 10240";; t30x4) echo "--xb 30720";; t30f) echo "--feat";; esac; }
smoke_mem() { case $1 in t30x2) echo ${SMOKE_MEM_t30x2:-30};; t30x4) echo ${SMOKE_MEM_t30x4:-42};; *) echo 20;; esac; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
avail_min() { local a b c; a=$(avail_gb); sleep 20; b=$(avail_gb); sleep 20; c=$(avail_gb)
  [ $b -lt $a ] && a=$b; [ $c -lt $a ] && a=$c; echo $a; }
swap_mb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}'; }
others_quiet() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (benchmarking|quiet lock lapsed|waiting for a quiet window)" | grep -c .; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
health_act() { grep -q "Verdict: ACT" "$Q/HEALTH.md" 2>/dev/null; }
gates_clear() { [ "$(others_quiet)" -eq 0 ] && ! lid_closed && ! health_act; }
gate_state() { echo "quiet locks of others $(others_quiet), lid $(lid_closed && echo closed || echo open), health $(health_act && echo ACT || echo ok), NO-SWAP $(avail_gb) GB"; }
ttt_running() { pgrep -f "scripts/dd_ttt[.]py" >/dev/null; }
cap_running() { pgrep -f "scripts/dd_capture[.]py" >/dev/null; }
saved_step() { local s; s=$(grep -oE '"step": [0-9]+' "$CKD/dd_$1.json" 2>/dev/null | head -1 | grep -oE '[0-9]+$'); echo "${s:-0}"; }

run_job() {   # label MEM cmd...: under the shared lock, with the yield / swap / STOP guards; sets RC and Y (why it was stopped)
  local lab=$1 mem=$2; shift 2
  echo "$$ $mem $lab" > "$SLOT_STATE/arm.mem"; slot_lock
  local s0; s0=$(swap_mb); Y=""
  say "$lab: starts${JOBNOTE:+ $JOBNOTE} ($(gate_state))"; JOBNOTE=""
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

stop_capture() {   # its launcher first (so it cannot start a new try), then the capture's own stop file
  local lp t
  for lp in $(pgrep -f "^bash scripts/dd_capture_slot[.]sh"); do   # found even between tries (capture.mem absent then)
    kill -TERM "$lp"; say "capture launcher pid $lp stopped (the capture is paused for the branch arms; resumes after them)"
    echo "$(date '+%m-%d %H:%M:%S') launcher (pid $lp) stopped by dd_ttt_alone.sh: the capture pauses for TTT30C's t30x2 / t30x4 alone (the orchestrator's order, Fri 13 Nov 15:3x); it resumes after them" >> "$CAPD/slot_n1112.log"
  done
  if cap_running; then
    touch "$CAPD/.yield_n1112"; t=0
    while cap_running && [ $t -lt 120 ]; do sleep 5; t=$((t + 5)); done
    cap_running && { say "capture: no prompt boundary within 120 s: SIGTERM"; pkill -TERM -f "scripts/dd_capture[.]py"; t=0
      while cap_running && [ $t -lt 60 ]; do sleep 5; t=$((t + 5)); done
      cap_running && { say "capture still alive 60 s after SIGTERM: SIGKILL"; pkill -KILL -f "scripts/dd_capture[.]py"; }; }
    say "capture stopped: $(grep -E "stopping|session:" "$CAPD/capture_n1112.log" | tail -2 | tr '\n' ' ' | cut -c1-220)"
  fi
  rm -f "$CAPD/.yield_n1112" "$SLOT_STATE/capture.mem"; slot_lock
}

say "start (pid $$; arms $ARMS; OLD_PID ${OLD_PID:-none}; GO file $GO; stop file $STOP)"
if [ -n "$DRY" ]; then
  for x in $ARMS; do say "DRY: $x: $(arm_flags $x); smoke guard $(smoke_mem $x) GB (needs NO-SWAP $(( $(smoke_mem $x) + 8 ))), train guard = ceil(smoke peak) + 2; saved step $(saved_step $x)"; done
  say "DRY: now: GO $([ -e $GO ] && echo yes || echo no), old launcher $([ -n "$OLD_PID" ] && kill -0 $OLD_PID 2>/dev/null && echo running || echo gone), dd_ttt $(ttt_running && echo running || echo none), capture $(cap_running && echo running || echo none); $(gate_state)"
  say "(the lines above were a DRY=1 check, not a run)"; exit 0
fi
n=0
until [ -e "$GO" ] && { [ -z "$OLD_PID" ] || ! kill -0 "$OLD_PID" 2>/dev/null; } && ! ttt_running; do
  [ -e "$STOP" ] && { say "STOP_SLOT before the start: done"; exit 0; }
  [ $((n % 60)) -eq 0 ] && say "waits for GO_ALONE and the old launcher / runs to end ($(gate_state))"
  n=$((n + 1)); sleep 60
done
say "GO: $(cat $GO 2>/dev/null | head -1)"
stop_capture
for x in $ARMS; do
  F=$(arm_flags $x); fails=0; mkdir -p $D/$x
  [ -e $D/$x/DONE ] && { say "$x: done already"; continue; }
  phase=smoke; MEM=$(smoke_mem $x)
  [ -e $D/$x/SMOKE_OK ] && [ -s $D/$x/MEM ] && { phase=train; MEM=$(cat $D/$x/MEM); }
  grep -q '"status": "done"' $CKD/dd_$x.json 2>/dev/null && phase=eval
  say "$x: from phase $phase ($F, MEM $MEM GB)"
  while :; do
    [ -e "$STOP" ] && { say "STOP_SLOT: done (the capture stays stopped)"; exit 0; }
    need=$MEM; [ $phase = eval ] && need=12
    n=0
    until gates_clear && [ "$(avail_min)" -ge $((need + 8)) ] && gates_clear; do   # the gates again after the 40-s read
      [ -e "$STOP" ] && { say "STOP_SLOT: done (the capture stays stopped)"; exit 0; }
      [ $((n % 30)) -eq 0 ] && say "$x $phase waits: $(gate_state); needs $((need + 8)) GB"
      n=$((n + 1)); sleep 60
    done
    case $phase in
      smoke)
        L0=$(wc -l < "$RLOG" 2>/dev/null || echo 0)
        run_job "TTT30C $x smoke" $MEM $PY scripts/dd_ttt.py train --run $x $BASE $F --peak-max $MEM --smoke
        out=$(tail -n +$((L0 + 1)) "$RLOG")
        [ -n "$Y" ] && { say "$x smoke stopped ($Y): again later"; [ "$Y" = STOP ] && exit 0; sleep 120; continue; }
        chk=$(echo "$out" | grep -E "smoke:|DONE" | tr '\n' ' ' | cut -c1-600)
        pk=$(echo "$out" | grep -oE "DONE smk_$x: .*peak [0-9.]+ GB" | grep -oE "peak [0-9.]+" | awk '{print $2}')
        if [ $RC -eq 0 ] && [ -n "$pk" ] && ! echo "$out" | grep -q DIFFERENT && echo "$out" | grep -q "smoke: b5's loss.*EQUAL" \
           && echo "$out" | grep -q "zero-init.*EQUAL"; then
          MEM=$(awk -v p="$pk" 'BEGIN {c = int(p); if (c < p) c++; print c + 2}')
          echo $MEM > $D/$x/MEM; touch $D/$x/SMOKE_OK; phase=train
          say "$x smoke OK (peak $pk GB -> train guard $MEM GB): $chk"
        else
          say "$x smoke FAILED rc $RC (guard $MEM GB): $chk $(echo "$out" | tail -2 | tr '\n' ' ' | cut -c1-300): arm skipped"; break
        fi ;;
      train)
        JOBNOTE="from saved step $(saved_step $x)"
        run_job "TTT30C $x alone: MEM $MEM" $MEM $PY scripts/dd_ttt.py train --run $x $BASE $F --peak-max $MEM
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
[ -e "$STOP" ] && { say "STOP_SLOT: done (the capture stays stopped)"; exit 0; }
say "arms done: the capture resumes (C 100)"
export DD_C=100
exec bash scripts/dd_capture_slot.sh
