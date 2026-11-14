#!/bin/bash
# Sat 14 Nov night chain, after the bank calibration (CALIBRATION-DONE in bank/calibrate.log). Every GPU step yields to
# another session's quiet lock (Leap's 20:40 screen): it is stopped as a process group, waits, and reruns.
#  1. dd_routes.py: routing log for Leap's 32-row gate Step 0 -> bank/routes_stats.json;
#  2. CAPQ read-out (amendment 1): t30bc = t30b's first 200 steps (b5, same slices, seed 1113, full-pass schedule), then
#     t30bc@200 vs t30x2c@200 on MBPP 100 in one eval -> capq/readout/eval.json, plus the determinism check of t30bc's
#     losses against t30b's.
# usage (a harness background task): bash scripts/dd_night.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
CL=$HOME/BigMoE/data/deepdraft/bank/calibrate.log; LOG=$HOME/BigMoE/data/deepdraft/bank/night.log
PY=$HOME/BigMoE/.venv/bin/python; CKD=$HOME/datasets/deepdraft/ckpt; OC=$HOME/BigMoE/data/deepdraft/capq
say() { echo "$(date '+%H:%M:%S') $*" >> "$LOG"; }
oq() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (benchmarking|quiet lock lapsed|waiting for a quiet window)" | grep -c .; }
set -m
guarded() {   # run a command; while another session's quiet lock is held, lapsed or queued: stop it, wait, rerun
  local p
  while :; do
    while [ "$(oq)" -gt 0 ]; do sleep 30; done
    "$@" & p=$!
    while kill -0 $p 2>/dev/null; do
      sleep 10
      if [ "$(oq)" -gt 0 ]; then say "a quiet lock of another session: SIGTERM to group $p, rerun after"; kill -TERM -- -$p 2>/dev/null; wait $p; p=""; break; fi
    done
    [ -n "$p" ] && { wait $p; return $?; }
  done
}
say "waits for the bank calibration's end"
until grep -q "CALIBRATION-DONE" "$CL" 2>/dev/null; do sleep 30; done
say "1. routing log (Leap's Step 0)"
routes() { BIGMOE_EXT_BUILD=build_qom $PY scripts/dd_routes.py 100 data/deepdraft/bank/routes_list.txt >> "$LOG" 2>&1; }
guarded routes
say "2. CAPQ read-out: t30bc = t30b's first 200 steps"
t30bc() {
  rm -f $CKD/dd_t30bc.json $CKD/dd_t30bc.safetensors $CKD/dd_t30bc_opt.safetensors
  $PY scripts/dd_ttt.py train --run t30bc --manifest data/deepdraft/ttt30b/split.json --pass-seed 1113 --save-at "" --peak-max 24 >> $OC/readout.log 2>&1 &
  local P=$!
  while kill -0 $P 2>/dev/null; do
    sleep 10
    [ "$(tail -n 40 $OC/readout.log | grep -oE "step [0-9]+/4206" | tail -1 | grep -oE "[0-9]+/" | tr -d /)" -ge 200 ] 2>/dev/null && { kill -TERM $P; break; }
  done
  wait $P
  $PY -c "import json; s = json.load(open('$CKD/dd_t30bc.json')); exit(0 if s['step'] >= 200 else 1)"
}
mkdir -p $OC/readout
until guarded t30bc; do say "t30bc ended before step 200: again"; sleep 30; done
say "3. the read-out eval (MBPP 100)"
ro() { $PY scripts/dd_ttt.py eval --runs t30bc,t30x2c --sets mbpp --out $OC/readout >> $OC/readout.log 2>&1; }
guarded ro
$PY - >> "$LOG" 2>&1 <<'EOF'
import json, os, numpy as np
CKD = os.path.expanduser("~/datasets/deepdraft/ckpt"); OC = os.path.expanduser("~/BigMoE/data/deepdraft/capq")
a = np.array(json.load(open(f"{CKD}/dd_t30bc.json"))["loss"][:200]); b = np.array(json.load(open(f"{CKD}/dd_t30b.json"))["loss"][:200])
print(f"CAPQ-RO harness: t30bc's losses vs t30b's, steps 1-200: max abs diff {np.abs(a - b).max():.3e} ({'EQUAL' if np.abs(a - b).max() == 0 else 'not bitwise'})")
e = json.load(open(f"{OC}/readout/eval.json"))
x, c = e["t30x2c"]["mbpp"], e["t30bc"]["mbpp"]
for bnd in ("0-0", "1-2", "3-6", "7-10", "11-14", "15-22", "23-30"):
    print(f"CAPQ-RO MBPP {bnd}: t30bc {100 * c[bnd]:.2f} / t30x2c {100 * x[bnd]:.2f} ({100 * (x[bnd] - c[bnd]):+.2f})")
d, dk = 100 * (x["23-30"] - c["23-30"]), x["kept_per_32"] - c["kept_per_32"]
print(f"CAPQ-RO kept per 32: t30bc {c['kept_per_32']:.3f} / t30x2c {x['kept_per_32']:.3f} ({dk:+.3f})")
print(f"CAPQ-RO VERDICT: {'REOPENS' if d >= 1.0 and dk >= 0.1 else 'CLOSED stands'} (23-30 {d:+.2f} pt; REOPENS needs >= +1.0 and kept >= +0.1)")
EOF
say "NIGHT-CHAIN-DONE"
