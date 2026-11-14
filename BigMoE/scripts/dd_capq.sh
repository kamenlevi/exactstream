#!/bin/bash
# CAPQ (docs/PREREG_DEEPDRAFT_CAPQ_2026-11-14.md): the clean capacity retest, t30x2c = t30x2 with the branch at the
# head's lr. It waits until dd_after_t30x4.sh logs "done:" (TAPSQ and the information-only eval), then waits on the
# gates (others' quiet locks, lid, HEALTH ACT, NO-SWAP >= 31, the lower of two reads). Then it trains from b5 on t30b's
# slices with t30b's full-pass schedule, SIGTERMs the trainer at step 200 (it saves), and judges against t30b's saved
# per-step KL. Guards while it runs: the gates (yield -> SIGTERM, the run is judged on the steps it has if >= 200,
# else it reruns from b5), swap +2 GB, STOP_SLOT, data/deepdraft/tapsq/STOP.
# usage (a harness background task): bash scripts/dd_capq.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
D=$HOME/BigMoE/data/deepdraft/ttt30c; OC=$HOME/BigMoE/data/deepdraft/capq; mkdir -p "$OC"
LOG=$D/slot.log; CLOG=$OC/run.log; MYSTOP=$HOME/BigMoE/data/deepdraft/tapsq/STOP
PY=$HOME/BigMoE/.venv/bin/python
CKD=$HOME/datasets/deepdraft/ckpt; RUN=t30x2c; STEPS=${STEPS:-200}
Q="$HOME/.local/share/quiet"
source scripts/dd_slot_lock.sh                           # STOP (= STOP_SLOT), SLOT_STATE, slot_lock
say() { echo "$(date '+%m-%d %H:%M:%S') capq: $*" >> "$LOG"; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
avail_lo() { local a b; a=$(avail_gb); sleep 30; b=$(avail_gb); [ $b -lt $a ] && a=$b; echo $a; }
swap_mb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}'; }
others_quiet() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (benchmarking|quiet lock lapsed|waiting for a quiet window)" | grep -c .; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
health_act() { grep -q "Verdict: ACT" "$Q/HEALTH.md" 2>/dev/null; }
gates_clear() { [ "$(others_quiet)" -eq 0 ] && ! lid_closed && ! health_act; }
gate_state() { echo "quiet locks of others $(others_quiet), lid $(lid_closed && echo closed || echo open), health $(health_act && echo ACT || echo ok), NO-SWAP $(avail_gb) GB, swap $(swap_mb) MB"; }
stopped() { [ -e "$STOP" ] || [ -e "$MYSTOP" ]; }
step_of() { $PY -c "import json; print(json.load(open('$CKD/dd_$RUN.json'))['step'])" 2>/dev/null || echo 0; }

L0=$(wc -l < "$LOG")
say "start (pid $$): waits for dd_after_t30x4.sh's 'done:'"
until tail -n +$((L0 + 1)) "$LOG" | grep -q "after: done:"; do
  grep -q "after: done:" "$LOG" && break
  stopped && { say "stop file before the start"; exit 0; }
  sleep 20
done
while :; do
  n=0
  while :; do
    stopped && { say "stop file: not started"; exit 0; }
    if gates_clear; then a=$(avail_lo); [ "$a" -ge 31 ] && break; fi
    [ $((n % 10)) -eq 0 ] && say "waits: $(gate_state); needs NO-SWAP 31"
    n=$((n + 1)); sleep 20
  done
  rm -f "$CKD/dd_$RUN.json" "$CKD/dd_$RUN.safetensors" "$CKD/dd_${RUN}_opt.safetensors"     # each try starts from b5
  echo "$$ 26 CAPQ t30x2c" > "$SLOT_STATE/capq.mem"; slot_lock
  S0=$(swap_mb)
  say "t30x2c: starts from b5 (NO-SWAP $a GB; $(gate_state))"
  $PY scripts/dd_ttt.py train --run $RUN --manifest data/deepdraft/ttt30b/split.json --pass-seed 1113 --xb 10240 \
    --xb-lr-mult 1 --save-at "" --peak-max 28 >> "$CLOG" 2>&1 &
  P=$!; Y=""
  while kill -0 $P 2>/dev/null; do
    sleep 10
    if [ "$(tail -n 40 "$CLOG" | grep -oE "step [0-9]+/4206" | tail -1 | grep -oE "[0-9]+/" | tr -d /)" -ge "$STEPS" ] 2>/dev/null; then Y=DONE
    elif stopped; then Y=STOP
    elif ! gates_clear; then Y="yield ($(gate_state))"
    elif [ "$(swap_mb)" -gt $((S0 + 2048)) ]; then Y="swap $(swap_mb) MB > $S0 + 2048"
    fi
    if [ -n "$Y" ]; then kill -TERM $P; break; fi
  done
  wait $P; RC=$?
  rm -f "$SLOT_STATE/capq.mem"; slot_lock
  st=$(step_of)
  say "t30x2c ended rc $RC at step $st${Y:+ ($Y)}"
  [ "$st" -ge "$STEPS" ] && break
  case "$Y" in yield*) sleep 60; continue;; esac
  say "stopped before step $STEPS: no verdict"; exit 2
done
$PY - "$CKD" "$RUN" "$STEPS" <<'EOF' 2>&1 | tee -a "$CLOG" | tail -12 >> "$OC/verdict.txt"
import json, sys, numpy as np
ckd, run, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
a = np.array(json.load(open(f"{ckd}/dd_{run}.json"))["loss"][:n], float)
b = np.array(json.load(open(f"{ckd}/dd_t30b.json"))["loss"][:n], float)
print(f"CAPQ: {run} vs t30b, paired by step, {len(a)} steps")
for lo, hi in ((1, 20), (21, 100), (101, 200)):
    x, y = a[lo - 1:hi], b[lo - 1:hi]
    print(f"  steps {lo}-{hi}: {run}/t30b = {x.mean() / y.mean():.3f}; lower on {100 * (x < y).mean():.0f} %")
spike = a[:20].mean() / b[:20].mean()
r, fr = a[100:200].mean() / b[100:200].mean(), (a[100:200] < b[100:200]).mean()
v = "INVALID (spike)" if spike > 1.20 else "REOPENS" if r <= 0.95 and fr >= 0.60 else "CLOSED"
print(f"spike check: steps 1-20 ratio {spike:.3f} (<= 1.20); line: steps 101-200 ratio {r:.3f} (REOPENS <= 0.95, lower on >= 60 %: {100 * fr:.0f} %)")
print(f"CAPQ VERDICT: {v}")
EOF
say "CAPQ-DONE: $(grep 'CAPQ VERDICT' "$OC/verdict.txt" | tail -1)"
