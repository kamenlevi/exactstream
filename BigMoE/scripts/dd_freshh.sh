#!/bin/bash
# FRESH-H (docs/PREREG_DEEPDRAFT_FRESHH_2026-11-14.md): arm F of dd_taps.py (the true h on every drafted row) against
# tq2_C. It waits for CAPQ's end in slot.log, then trains F (80 steps) -> MBPP eval -> the verdict.
# Guards: the same as dd_taps2.sh.
# usage (a harness background task): bash scripts/dd_freshh.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
D=$HOME/BigMoE/data/deepdraft/ttt30c; OQ=$HOME/BigMoE/data/deepdraft/tapsq
LOG=$D/slot.log; QLOG=$OQ/run.log; MYSTOP=$OQ/STOP
PY=$HOME/BigMoE/.venv/bin/python
Q="$HOME/.local/share/quiet"
source scripts/dd_slot_lock.sh
say() { echo "$(date '+%m-%d %H:%M:%S') freshh: $*" >> "$LOG"; }
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

L0=$(wc -l < "$LOG")
say "start (pid $$): waits for CAPQ's end"
until grep -qE "capq: (CAPQ-DONE|.*no verdict|stop file)" "$LOG"; do stopped && { say "stop file before the start"; exit 0; }; sleep 15; done
S0=$(swap_mb)
echo "$$ 16 FRESH-H" > "$SLOT_STATE/freshh.mem"; slot_lock
fin() { rm -f "$SLOT_STATE/freshh.mem"; slot_lock; say "FRESHH-DONE: $1"; exit 0; }
run_guarded "FRESHH F" 28 "$QLOG" $PY scripts/dd_taps.py train --arm F
[ "$RC" -eq 0 ] || fin "F failed rc $RC"
run_guarded "FRESHH eval F" 20 "$QLOG" $PY scripts/dd_taps.py eval --runs tq2_F --out-name eval2_F.json
[ "$RC" -eq 0 ] || fin "the eval of F failed rc $RC"
$PY scripts/dd_taps_judge.py C F tq2 2>&1 | grep -vE "^(VERDICT|lines):" >> "$QLOG"
v=$($PY - <<'EOF'
import json, os
D = os.path.expanduser("~/BigMoE/data/deepdraft/tapsq")
c = json.load(open(f"{D}/eval2_C.json"))["tq2_C"]["23-30"]; f = json.load(open(f"{D}/eval2_F.json"))["tq2_F"]["23-30"]
half = c + 0.5 * (0.935 - c)
v = "WORKS" if f >= half else "PARTIAL" if f >= c + 0.02 else "FLAT"
print(f"FRESH-H VERDICT: {v} (MBPP 23-30: F {100 * f:.2f} vs C {100 * c:.2f}, {100 * (f - c):+.2f} pt; WORKS >= {100 * half:.2f}, PARTIAL >= {100 * (c + 0.02):.2f})")
EOF
)
echo "$v" >> "$QLOG"
fin "$v"
