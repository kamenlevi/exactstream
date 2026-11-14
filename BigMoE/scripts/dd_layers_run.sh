#!/bin/bash
# LAYERS (docs/PREREG_DEEPDRAFT_LAYERS_2026-11-14.md): after the re-prefill ends -> the ridge fits (dd_layers.py, 4 CPU
# threads) -> t30b on MBPP 100 with the stand-in replaced by the re-prefilled wd (the approximate-oracle gate) and by each
# layer set's map, two evals at a time -> a summary with the share of the depth gap each set recovers.
# usage (a harness background task): bash scripts/dd_layers_run.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
OQ=$HOME/BigMoE/data/deepdraft/tapsq; RPL=$OQ/reprefill.log; LOG=$OQ/layers.log
PY=$HOME/BigMoE/.venv/bin/python; CK=$HOME/datasets/deepdraft/ckpt
say() { echo "$(date '+%H:%M:%S') $*" >> "$LOG"; }
S=$(grep -n "start (C 100, 200 files)" "$RPL" | tail -1 | cut -d: -f1)
say "waits for the re-prefill's end (its start is reprefill.log line $S)"
until tail -n +$S "$RPL" | grep -q "re-prefill ended rc"; do sleep 15; done
tail -n +$S "$RPL" | grep "re-prefill ended rc" | tail -1 | grep -q "rc 0; files 200" || { say "the re-prefill ended without 200 files: stop"; exit 2; }
say "re-prefill complete: the ridge fits"
VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4 $PY scripts/dd_layers.py >> "$LOG" 2>&1 || { say "the fits failed"; exit 2; }
peer lock share 40m --cpu 300 --gpu 80 "BigMoE DEEPDRAFT LAYERS evals (2 x ~11 GB)" >/dev/null 2>&1
ev() { $PY scripts/dd_taps.py evalrep --runs t30b "$@" 2>&1 | grep -E "^\[rep\]|Traceback|rror" >> "$LOG"; }
( ev --replace rawwd --out-name layers_oracle.json; ev --replace rawmap --ridge $CK/ridge_L5.npz --out-name layers_5.json
  ev --replace rawmap --ridge $CK/ridge_L5_15.npz --out-name layers_5_15.json ) &
A=$!
( ev --replace rawmap --ridge $CK/ridge_L5_15_25.npz --out-name layers_5_15_25.json
  ev --replace rawmap --ridge $CK/ridge_Lall.npz --out-name layers_all.json; ev --replace rawmap --ridge $CK/ridge_L45.npz --out-name layers_45.json ) &
B=$!
wait $A; wait $B
peer unlock >/dev/null 2>&1
$PY - >> "$LOG" 2>&1 <<'EOF'
import json, os
D = os.path.expanduser("~/BigMoE/data/deepdraft/tapsq")
own = 0.7213
o = json.load(open(f"{D}/layers_oracle.json"))
ok = [k for k in o if "+rawwd" in k][0]; orc = o[ok]["23-30"]
fit = json.load(open(f"{D}/layers_fit.json"))
print(f"LAYERS gate: the approximate oracle (re-prefilled wd) at 23-30 {100 * orc:.2f} % ({'OK' if orc >= 0.90 else 'NOT OK'}, >= 90)")
for s in ("5", "5_15", "5_15_25", "all", "45"):
    e = json.load(open(f"{D}/layers_{s}.json")); k = [k for k in e if "+rawmap" in k][0]; v = e[k]["23-30"]
    print(f"LAYERS set {s:8s}: 23-30 {100 * v:.2f} %, gap share {100 * (v - own) / (orc - own):.0f} %; kept/32 {e[k]['kept_per_32']:.2f}; "
          f"R2 {fit[s]['r2']:.3f}, normalized error {fit[s]['rel_err_normalized']:.3f}; bands " + " ".join(f"{b}:{100 * x:.1f}" for b, x in e[k].items() if "-" in b))
EOF
say "LAYERS-DONE"
