#!/bin/bash
# TAPSQ rerun (amendment 2, Sat 14 Nov 15:4x; docs/PREREG_DEEPDRAFT_TAPSQ_2026-11-14.md).
# Why a rerun: the first run's control (tq_C) lost 13.9 pt at MBPP 23-30 in 80 steps. A fresh Adam (no bias correction)
# at 10x t30b's final lr moved the head 1.8 % (t30b's whole run moved it 2.9 %), so it is VOID.
# Now:
#  - the head continues on t30b's own Adam state at its final lr (1e-6);
#  - the tap gets a fresh, bias-corrected Adam at 1e-5.
# Order:
#  1. C (80 steps) -> its MBPP eval -> the validity line (within 1 pt of t30b at 23-30), else stop;
#  2. T -> its eval -> the judge; R + its eval only on PASS;
#  3. then exec dd_capq.sh (the clean capacity retest), so the two never share the Mac's memory.
# Guards per job: another session's quiet lock (held, lapsed or queued), a closed lid or HEALTH ACT -> SIGTERM, rerun
# after; swap over the start + 2048 MB -> stop; STOP_SLOT (Kamen) or data/deepdraft/tapsq/STOP -> stop.
# usage (a harness background task): bash scripts/dd_taps2.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
D=$HOME/BigMoE/data/deepdraft/ttt30c; OQ=$HOME/BigMoE/data/deepdraft/tapsq
LOG=$D/slot.log; QLOG=$OQ/run.log; MYSTOP=$OQ/STOP
PY=$HOME/BigMoE/.venv/bin/python
Q="$HOME/.local/share/quiet"
source scripts/dd_slot_lock.sh                           # STOP (= STOP_SLOT), SLOT_STATE, slot_lock
say() { echo "$(date '+%m-%d %H:%M:%S') taps2: $*" >> "$LOG"; }
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

S0=$(swap_mb)
echo "$$ 16 TAPSQ rerun" > "$SLOT_STATE/taps2.mem"; slot_lock
say "start (pid $$): swap $S0 MB (guard +2048)"
finish() { rm -f "$SLOT_STATE/taps2.mem"; slot_lock; say "TAPSQ2-DONE: $1; CAPQ next"; exec bash scripts/dd_capq.sh; }

run_guarded "TAPSQ2 C" 28 "$QLOG" $PY scripts/dd_taps.py train --arm C
[ "$RC" -eq 0 ] || finish "C failed rc $RC"
run_guarded "TAPSQ2 eval C" 20 "$QLOG" $PY scripts/dd_taps.py eval --runs tq2_C --out-name eval2_C.json
[ "$RC" -eq 0 ] || finish "the eval of C failed rc $RC"
dc=$($PY -c "import json; e = json.load(open('$OQ/eval2_C.json'))['tq2_C']; r = json.load(open('$HOME/BigMoE/data/deepdraft/ttt30b/eval.json'))['t30b']['mbpp']; print(f'{100 * (e[\"23-30\"] - r[\"23-30\"]):+.2f} {100 * (e[\"0-0\"] - r[\"0-0\"]):+.2f}')")
say "TAPSQ2 validity: C vs t30b at MBPP 23-30 / 0: $dc pt"
ok=$(echo "$dc" | awk '{print ($1 <= 1.0 && $1 >= -1.0) ? 1 : 0}')
[ "$ok" = 1 ] || finish "VOID-HARNESS: C moved more than 1 pt from t30b at 23-30 ($dc); T not run"
run_guarded "TAPSQ2 T" 28 "$QLOG" $PY scripts/dd_taps.py train --arm T
[ "$RC" -eq 0 ] || finish "T failed rc $RC"
run_guarded "TAPSQ2 eval T" 20 "$QLOG" $PY scripts/dd_taps.py eval --runs tq2_T --out-name eval2_T.json
[ "$RC" -eq 0 ] || finish "the eval of T failed rc $RC"
v=$($PY scripts/dd_taps_judge.py C T tq2 2>&1 | tee -a "$QLOG" | grep "^VERDICT" | tail -1)
say "TAPSQ2 C vs T: $v"
if echo "$v" | grep -q PASS; then
  run_guarded "TAPSQ2 R" 28 "$QLOG" $PY scripts/dd_taps.py train --arm R
  [ "$RC" -eq 0 ] && run_guarded "TAPSQ2 eval R" 20 "$QLOG" $PY scripts/dd_taps.py eval --runs tq2_R --out-name eval2_R.json
  [ "$RC" -eq 0 ] && say "TAPSQ2 R vs T: $($PY scripts/dd_taps_judge.py R T tq2 2>&1 | tee -a "$QLOG" | grep '^VERDICT' | tail -1)"
fi
finish "$v"
