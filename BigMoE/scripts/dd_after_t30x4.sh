#!/bin/bash
> (a quote from the project owner, kept private)
# possible". The orchestrator: score t30b + t30x4 first for the verdict; the TAPSQ quick test runs beside that eval; the
# information-only scores after.
#  1. Waits for "[dd] DONE t30x4:" in run.log. The trainer saves "done" at step 4206 before it prints that line.
#     - Then it touches STOP_SLOT, so the old sequencer (dd_ttt_alone.sh, pid SEQ, old code in memory) exits: no 6-run
#       eval, no capture.
#     - It checks that the checkpoint says done at 4206, then removes STOP_SLOT.
#  2. Side by side, under the session's one share lock (dd_slot_lock.sh):
#     (a) the verdict eval, t30b + t30x4 (the judge re-uses b5 from a855212 when t30b re-scores bit for bit);
#     (b) TAPSQ (docs/PREREG_DEEPDRAFT_TAPSQ_2026-11-14.md): smoke -> C -> T -> eval -> judge; R + its eval only on PASS.
#  3. Then the information-only eval: b5 and t30x4's copies at 600 / 1500 / 3000.
#  4. Leaves the capture stopped: the session restarts it after reading the results.
# Guards per job:
#  - another session's quiet lock (held, lapsed or queued), a closed lid or HEALTH ACT -> SIGTERM, rerun from the start after;
#  - swap over this runner's start + 2048 MB -> stop that job;
#  - STOP_SLOT (Kamen's stop) or data/deepdraft/tapsq/STOP -> stop.
# usage (a harness background task): SEQ=73681 bash scripts/dd_after_t30x4.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
SEQ=${SEQ:-73681}
D=$HOME/BigMoE/data/deepdraft/ttt30c; OQ=$HOME/BigMoE/data/deepdraft/tapsq; mkdir -p "$OQ"
LOG=$D/slot.log; RUNLOG=$D/run.log; QLOG=$OQ/run.log; MYSTOP=$OQ/STOP
PY=$HOME/BigMoE/.venv/bin/python
CKD=$HOME/datasets/deepdraft/ckpt
Q="$HOME/.local/share/quiet"
source scripts/dd_slot_lock.sh                           # STOP (= STOP_SLOT), SLOT_STATE, slot_lock
say() { echo "$(date '+%m-%d %H:%M:%S') after: $*" >> "$LOG"; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
avail_lo() { local a b; a=$(avail_gb); sleep 30; b=$(avail_gb); [ $b -lt $a ] && a=$b; echo $a; }
swap_mb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}'; }
others_quiet() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (benchmarking|quiet lock lapsed|waiting for a quiet window)" | grep -c .; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
health_act() { grep -q "Verdict: ACT" "$Q/HEALTH.md" 2>/dev/null; }
gates_clear() { [ "$(others_quiet)" -eq 0 ] && ! lid_closed && ! health_act; }
gate_state() { echo "quiet locks of others $(others_quiet), lid $(lid_closed && echo closed || echo open), health $(health_act && echo ACT || echo ok), NO-SWAP $(avail_gb) GB, swap $(swap_mb) MB"; }
stopped() { [ -e "$STOP" ] || [ -e "$MYSTOP" ]; }

run_guarded() {   # label need_gb log cmd...: runs cmd with the guards; sets RC and Y (why it was stopped)
  local lab=$1 need=$2 log=$3 n a p; shift 3
  while :; do
    n=0
    while :; do
      stopped && { Y=STOP; RC=99; say "$lab: stop file, not started"; return; }
      if gates_clear; then a=$(avail_lo); [ "$a" -ge "$need" ] && break; fi
      [ $((n % 10)) -eq 0 ] && say "$lab waits: $(gate_state); needs NO-SWAP $need GB"
      n=$((n + 1)); sleep 20
    done
    say "$lab: starts (NO-SWAP $a GB, needs $need; $(gate_state))"
    "$@" >> "$log" 2>&1 &
    p=$!; Y=""
    while kill -0 $p 2>/dev/null; do
      sleep 10
      if stopped; then Y=STOP
      elif ! gates_clear; then Y="yield ($(gate_state))"
      elif [ "$(swap_mb)" -gt $((S0 + 2048)) ]; then Y="swap $(swap_mb) MB > $S0 + 2048"
      fi
      if [ -n "$Y" ]; then say "$lab: $Y -> SIGTERM $p"; kill -TERM $p; break; fi
    done
    wait $p; RC=$?
    say "$lab ended rc $RC${Y:+ ($Y)}"
    case "$Y" in yield*) sleep 60; continue;; esac
    return
  done
}

# ---- 1. t30x4's DONE -> the old sequencer exits before its eval
N0=$(wc -l < "$RUNLOG"); L0=$(wc -l < "$LOG")
END=$(( $(date +%s) + ${SECS:-14400} ))
say "start (pid $$): waits for '[dd] DONE t30x4:' after run.log line $N0; sequencer pid $SEQ"
while :; do
  tail -n +$((N0 + 1)) "$RUNLOG" | grep -qF "[dd] DONE t30x4:" && break
  if tail -n +$((L0 + 1)) "$LOG" | grep -qE "alone: (t30x4: MLX|t30x4: 2 failures|arms done|STOP_SLOT)"; then
    say "the sequencer ended t30x4 another way (see above): nothing done"; exit 2; fi
  kill -0 "$SEQ" 2>/dev/null || { say "the sequencer ended before t30x4's DONE: nothing done"; exit 2; }
  [ "$(date +%s)" -ge $END ] && { say "timeout before t30x4's DONE"; exit 2; }
  sleep 1
done
touch "$STOP"
say "t30x4 DONE seen: STOP_SLOT touched, so the sequencer exits before its eval and the capture"
for i in $(seq 1 240); do kill -0 "$SEQ" 2>/dev/null || break; sleep 1; done
kill -0 "$SEQ" 2>/dev/null && { say "the sequencer still runs 240 s after STOP_SLOT: stop here (STOP_SLOT left)"; exit 2; }
EP=$(pgrep -f "scripts/dd_ttt[.]py eval" | head -1)
[ -n "$EP" ] && { say "an eval (pid $EP) had started: SIGTERM"; kill -TERM "$EP"; sleep 5; }
rm -f "$STOP"
ST=$($PY -c "import json; s = json.load(open('$CKD/dd_t30x4.json')); print(s['status'], s['step'])")
say "sequencer gone, STOP_SLOT removed; t30x4 checkpoint: $ST"
[ "$ST" = "done 4206" ] || { say "the checkpoint is not done at 4206: stop here"; exit 2; }

# ---- 2. the verdict eval and TAPSQ, side by side
echo "$$ 26 t30x4 verdict eval + TAPSQ" > "$SLOT_STATE/after.mem"; slot_lock
S0=$(swap_mb)
say "swap at the start $S0 MB (guard +2048)"

verdict() {
  run_guarded "t30x4 verdict eval (t30b, t30x4)" 20 "$RUNLOG" $PY scripts/dd_ttt.py eval --runs t30b,t30x4 --out $D/t30x4
  say "VERDICT-EVAL ended rc $RC${Y:+ ($Y)}"
}

tapsq() {
  local out v arm
  run_guarded "TAPSQ smoke" 28 "$QLOG" $PY scripts/dd_taps.py smoke
  [ "$RC" -eq 0 ] || { say "TAPSQ smoke failed rc $RC: TAPSQ stops"; return; }
  out=$(grep "\[tq\] smoke" "$QLOG" | tail -6)
  if echo "$out" | grep -q "zero-init tap .*: EQUAL" && echo "$out" | grep -qF "(> 0)" && echo "$out" | grep -qF "(CHANGED)" \
     && echo "$out" | grep -qF "DIFFERENT from T"; then say "TAPSQ smoke OK"
  else say "TAPSQ smoke NOT OK: $(echo "$out" | tr '\n' ' ' | cut -c1-400); TAPSQ stops"; return; fi
  for arm in C T; do
    run_guarded "TAPSQ $arm" 28 "$QLOG" $PY scripts/dd_taps.py train --arm $arm --steps 80
    [ "$RC" -eq 0 ] || { say "TAPSQ $arm failed rc $RC: TAPSQ stops"; return; }
  done
  run_guarded "TAPSQ eval" 20 "$QLOG" $PY scripts/dd_taps.py eval --runs t30b,tq_C,tq_T
  [ "$RC" -eq 0 ] || { say "TAPSQ eval failed rc $RC: TAPSQ stops"; return; }
  v=$($PY scripts/dd_taps_judge.py C T 2>&1 | tee -a "$QLOG" | grep "^VERDICT" | tail -1)
  say "TAPSQ C vs T: $v"
  if echo "$v" | grep -q PASS; then
    run_guarded "TAPSQ R" 28 "$QLOG" $PY scripts/dd_taps.py train --arm R --steps 80
    [ "$RC" -eq 0 ] && run_guarded "TAPSQ eval R" 20 "$QLOG" $PY scripts/dd_taps.py eval --runs tq_R --out-name eval_R.json
    [ "$RC" -eq 0 ] && say "TAPSQ R vs T: $($PY scripts/dd_taps_judge.py R T 2>&1 | tee -a "$QLOG" | grep '^VERDICT' | tail -1)"
  fi
  say "TAPSQ-DONE"
}

verdict & VP=$!
sleep 45                                                 # the eval loads first, so TAPSQ's memory gate sees it
tapsq & TP=$!
wait $VP; wait $TP

# ---- 3. information only: b5 and the copies
run_guarded "t30x4 info eval (b5 + copies)" 20 "$RUNLOG" $PY scripts/dd_ttt.py eval --runs b5,t30x4_s600,t30x4_s1500,t30x4_s3000 --out $D/t30x4_info
say "INFO-EVAL ended rc $RC${Y:+ ($Y)}"
rm -f "$SLOT_STATE/after.mem"; slot_lock
say "done: the capture stays stopped until the session restarts it"
