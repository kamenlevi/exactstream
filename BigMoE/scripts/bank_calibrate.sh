#!/bin/bash
# BANK + PROXY calibration (the orchestrator 18:0x). Runs after the bank build logs its end in bank/bank.log:
#  1. bank_proxy.py cache;
#  2. ridge maps on the bank for {5}, {5,15}, {5,15,25}, {45}, all, with the proxy's own R2 definition (val answer rows);
#  3. real REPLACE evals of t30b on MBPP 100 with bank features: each map, plus the oracle (the bank's wd);
#  4. HE+ on the runtime path: t30b's own chain with the bank's context, the oracle, and the {45} map;
#  5. calibration.json (R2 -> MBPP 23-30) and a summary.
# Every step yields to another session's quiet lock (held, lapsed or queued): stopped, rerun after.
# usage (a harness background task): bash scripts/bank_calibrate.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
BL=$HOME/BigMoE/data/deepdraft/bank/bank.log; LOG=$HOME/BigMoE/data/deepdraft/bank/calibrate.log
PY=$HOME/BigMoE/.venv/bin/python; CK=$HOME/datasets/deepdraft/ckpt; FD="--featdir ~/datasets/deepdraft/bank"
say() { echo "$(date '+%H:%M:%S') $*" >> "$LOG"; }
oq() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (benchmarking|quiet lock lapsed|waiting for a quiet window)" | grep -c .; }
set -m                                                   # each background job gets its own process group, so a stop reaches its children
guarded() {   # run a command; while another session's quiet lock is held, lapsed or queued: stop it, wait, rerun from the start
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
B0=$(wc -l < "$BL" 2>/dev/null || echo 0)                # only the bank log's lines written after this runner started count
say "waits for the bank build's end (bank.log lines after $B0)"
until tail -n +$((B0 + 1)) "$BL" | grep -qE "BANK-DONE|the build stopped"; do sleep 20; done
tail -n +$((B0 + 1)) "$BL" | grep -q "BANK-DONE" || { say "the bank build stopped early: calibration not run"; exit 2; }
say "1. the proxy cache"
guarded env VECLIB_MAXIMUM_THREADS=6 $PY scripts/bank_proxy.py cache >> "$LOG" 2>&1 || { say "cache failed"; exit 2; }
say "2. the calibration maps"
for s in tap5 tap5,tap15 tap5,tap15,tap25 tap45 tap5,tap15,tap25,tap35,tap45; do
  nm=$(echo $s | tr ',' '_'); guarded env VECLIB_MAXIMUM_THREADS=6 $PY scripts/bank_proxy.py ridge --inputs $s --save $CK/bank_ridge_$nm.npz >> "$LOG" 2>&1
done
peer lock share 60m --cpu 300 --gpu 80 "BigMoE DEEPDRAFT bank calibration evals (2 x ~11 GB)" >/dev/null 2>&1
say "3-4. REPLACE evals (MBPP) and HE+ on the runtime path"
ev1() { $PY scripts/dd_taps.py evalrep --runs t30b "$@" 2>&1 | grep -E "^\[rep\]|Traceback|rror" >> "$LOG"; }
ev() { guarded ev1 "$@"; }
( ev --replace rawwd $FD --out-name cal_oracle_mbpp.json
  for nm in tap5 tap5_tap15 tap5_tap15_tap25; do ev --replace rawmap --ridge $CK/bank_ridge_$nm.npz $FD --out-name cal_${nm}_mbpp.json; done
  ev --set code --ctx bank --replace wd --refresh 1000 $FD --out-name cal_own_code_bankctx.json ) &
A=$!
( for nm in tap45 tap5_tap15_tap25_tap35_tap45; do ev --replace rawmap --ridge $CK/bank_ridge_$nm.npz $FD --out-name cal_${nm}_mbpp.json; done
  ev --set code --ctx bank --replace rawwd $FD --out-name cal_oracle_code_bankctx.json
  ev --set code --ctx bank --replace rawmap --ridge $CK/bank_ridge_tap45.npz $FD --out-name cal_tap45_code_bankctx.json
  ev --set code --replace wd --refresh 1000 $FD --out-name cal_own_code_d16w.json ) &
B=$!
wait $A; wait $B
peer unlock >/dev/null 2>&1
$PY - >> "$LOG" 2>&1 <<'EOF'
import json, os, re
D = os.path.expanduser("~/BigMoE/data/deepdraft/tapsq"); B = os.path.expanduser("~/BigMoE/data/deepdraft/bank")
log = open(os.path.join(B, "calibrate.log")).read()
r2 = {m.group(1).replace(",", "_"): float(m.group(2)) for m in re.finditer(r"ridge (\S+) \(lambda [^)]*\): held-out R2 ([-0-9.]+)", log)}
def acc(f):
    e = json.load(open(os.path.join(D, f))); k = [x for x in e if x.startswith("t30b+")][0]; return e[k]
pts = []
for nm in ("tap5", "tap5_tap15", "tap5_tap15_tap25", "tap45", "tap5_tap15_tap25_tap35_tap45"):
    a = acc(f"cal_{nm}_mbpp.json"); pts.append({"set": nm, "r2": r2[nm], "mbpp_23_30": 100 * a["23-30"], "kept_per_32": a["kept_per_32"]})
o = acc("cal_oracle_mbpp.json"); pts.append({"set": "oracle (the bank's wd)", "r2": 1.0, "mbpp_23_30": 100 * o["23-30"], "kept_per_32": o["kept_per_32"]})
json.dump({"points": pts, "line": 88.0, "made": "Sat 14 Nov, bank_calibrate.sh: t30b on MBPP 100, stand-in replaced at every drafted row by the map (no training); R2 = bank_proxy.py's (val answer rows)"},
          open(os.path.join(B, "calibration.json"), "w"), indent=1)
for p in sorted(pts, key=lambda p: p["r2"]):
    print(f"CAL {p['set']:32s} R2 {p['r2']:.4f} -> MBPP 23-30 {p['mbpp_23_30']:.2f}, kept {p['kept_per_32']:.2f}")
for f, lab in (("cal_own_code_d16w.json", "HE+ t30b own chain, d16w context (the eval's)"), ("cal_own_code_bankctx.json", "HE+ t30b own chain, runtime-path context"),
               ("cal_oracle_code_bankctx.json", "HE+ oracle (runtime-path wd at drafted rows)"), ("cal_tap45_code_bankctx.json", "HE+ the {45} map, runtime-path context")):
    a = acc(f); print(f"HEPLUS {lab}: " + " ".join(f"{b}:{100 * v:.2f}" for b, v in a.items() if "-" in b) + f"; kept {a['kept_per_32']:.2f}")
EOF
say "CALIBRATION-DONE"
