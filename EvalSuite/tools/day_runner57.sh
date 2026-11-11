#!/bin/bash
# EvalSuite runner 57, Sat 2026-11-07 (attached to a live evalsuite session): the 12 GB theta 0.95 arm t95ck1u's code same
# scores (docs/PREREG_night_gates_2026-11-07.md gate 1 = PREREG_t80ck1u's lines), Fit12's fit_t3.py (sha 11cadf9ec149; one arm per process,
# F12_IDS_OUT) on the core 60 HumanEval+ prompts:
# 0. The VOID check: arm ex, cap 64, the first 4 forward keys; each row's first 32 tokens must equal REF2-T's.
# 1. t95ck1u forward (the keys sorted) in ONE process, then reverse in ONE process; a crashed process resumes with its
#    unwritten keys in a new process (logged: the history breaks there). tools/r57rows.py -> results/t95_{fwd,rev},
#    tools/rows_text.py, tools/he_d4.py vs REF2-T (+ NAX6, REF as information) -> results/T95_DONE. No fallback arm.
# Shared lock "evalsuite: MEM 12 GB" (the day's single >5 GB job); gates: PAUSE / lid, the day rule, the fit_t3.py sha,
# swap, memory (12 + 8 GB NO-SWAP free); no step starts after 00:45. Stop it by pid only, and its children too; then
# peer unlock.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day57_runner.log
AF=results/t95
mkdir -p "$AF"
PY=.venv/bin/python
BPY=$HOME/BigMoE/.venv/bin/python
FIT=$HOME/Fit12/scripts/fit_t3.py
FITSHA=11cadf9ec149
CUTOFF=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-08 00:45" +%s)   # CERT 01:30 is never delayed
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
lid_shut() {
  ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes || return 1
  (( $(system_profiler SPDisplaysDataType -detailLevel mini 2>/dev/null | grep -c "Resolution:") > 1 )) && return 1
  return 0
}
calm() { while [ -e "$HOME/.local/share/quiet/PAUSE" ] || lid_shut; do sleep 30; done; }
mem_busy() {    # another session's shared run with MEM > 5 GB (our own reasons start "evalsuite: MEM")
  peer ls 2>/dev/null | grep "shared run" | grep -v "evalsuite: MEM" | grep -oE "MEM [0-9]+ GB" | awk '$2 > 5 {f = 1} END {exit !f}'
}
freegb() { .venv/bin/python -c "import sys; sys.path.insert(0, '.'); from evalsuite.cli import free_gb; print(int(free_gb()))"; }
swapgb() { sysctl -n vm.swapusage | awk '{gsub("M", "", $6); print int($6 / 1024 + 0.5)}'; }
SWAP0=$(swapgb)
gate() {        # label: every check before a GPU process
  local w=0
  calm
  while mem_busy; do [ $((w % 10)) = 0 ] && say "$1: another session's MEM job holds a shared lock: waiting (day rule)"; w=$((w + 1)); sleep 60; calm; done
  [ "$(shasum -a 256 "$FIT" | cut -c1-12)" = "$FITSHA" ] || { say "fit_t3.py is not $FITSHA: runner stops"; exit 7; }
  [ "$(swapgb)" -le $(( SWAP0 + 2 )) ] || { say "swap $(swapgb) GB > $SWAP0 + 2 at the start: runner stops"; exit 3; }
  [ "$(date +%s)" -lt "$CUTOFF" ] || { say "$1: after 00:45: runner stops"; exit 6; }
  w=0
  until [ "$(freegb)" -ge 20 ]; do
    [ $((w % 10)) = 0 ] && say "$1: waiting for memory: $(freegb) GB NO-SWAP (needs 20)"; w=$((w + 1)); sleep 60; calm
    [ "$(date +%s)" -lt "$CUTOFF" ] || { say "$1: no memory before 00:45: runner stops"; exit 6; }
  done
}
share_lock() {  # reason, lease minutes: retry while a quiet lock runs
  local r w=0
  while :; do
    r=$(peer lock share "${2}m" --cpu 300 --gpu 60 "$1" 2>&1); [ $? = 0 ] && { say "shared lock: $1"; return 0; }
    [ $((w % 10)) = 0 ] && say "share lock refused: $(echo "$r" | tail -1 | cut -c1-120)"
    w=$((w + 1)); sleep 60; calm
    [ "$w" -lt 240 ] || { say "no share lock in 4 h: runner stops"; exit 3; }
  done
}
run_arm() {     # arm, tag, cap, keys (comma, run order), run (rows go there; "" = none), label: ONE fit_t3 process
  local arm=$1 tag=$2 cap=$3 keys=$4 run=$5 lab=$6 rc t0
  gate "$lab"
  share_lock "evalsuite: MEM 12 GB: the 12 GB t95ck1u code gate, $lab" 45
  t0=$(date +%s)
  env -u F12_RERUN -u F12_WIRED F12_IDS_OUT="$HOME/EvalSuite/$AF/$tag.json" F12_T3ARMS="$arm" F12_CAP="$cap" F12_KEYS="$keys" \
    F12_CMCL=47,32 F12_OUT="$HOME/EvalSuite/$AF/$tag" "$BPY" -B "$FIT" "$tag" > "$AF/$tag.log" 2>&1; rc=$?
  peer unlock > /dev/null 2>&1
  say "$lab: rc=$rc ($(( ($(date +%s) - t0) / 60 )) min)"
  [ -s "$AF/$tag.json" ] || { say "$lab: no ids ($AF/$tag.log)"; return 1; }
  [ -n "$run" ] && $PY tools/r57rows.py "$AF/$tag.json" "$run" "$arm" --append >> "$LOG" 2>&1
  return $rc
}
missing() {     # run, comma list -> the keys not yet in results/<run>/decode.jsonl
  python3 -c 'import json, os, sys
p = os.path.join("results", sys.argv[1], "decode.jsonl")
have = {json.loads(l)["key"] for l in open(p)} if os.path.exists(p) else set()
print(",".join(k for k in sys.argv[2].split(",") if k not in have))' "$1" "$2"
}
order_run() {   # arm, run, label, keys: one process for the whole order; a crash resumes with the keys it did not write
  local arm=$1 run=$2 lab=$3 all=$4 m i=0
  while :; do
    m=$(missing "$run" "$all"); [ -z "$m" ] && break
    i=$((i + 1)); [ "$i" -le 4 ] || { say "$lab: 4 processes without all rows: runner stops"; exit 3; }
    [ "$i" -gt 1 ] && say "$lab: process $i resumes with $(echo "$m" | tr ',' '\n' | grep -c .) keys (the history breaks here)"
    run_arm "$arm" "${run}_p$i" item "$m" "$run" "$lab process $i"
  done
  $PY tools/rows_text.py "$run" >> "$LOG" 2>&1
  say "$lab: $(grep -c . "results/$run/decode.jsonl") rows"
}
gate_arm() {    # arm, prefix: both orders + the verdict -> V
  local arm=$1 p=$2 o
  for r in "${p}_fwd" "${p}_rev"; do [ -e "results/$r" ] && { say "results/$r exists: move it aside first: runner stops"; exit 2; }; done
  order_run "$arm" "${p}_fwd" "$arm fwd" "$KEYS"
  order_run "$arm" "${p}_rev" "$arm rev" "$RKEYS"
  V=""
  for o in fwd rev; do
    $PY tools/he_d4.py "${p}_$o" ref2t ref2tnax ref > "results/${p}_${o}_d4.txt" 2>&1
    say "$arm $o: $(grep '^humaneval vs ref2t ' "results/${p}_${o}_d4.txt" | tail -1)"
    V="$V $o: $(grep '^HE_D4:' "results/${p}_${o}_d4.txt" | tail -1)"
  done
  case "$V" in *FAIL*) W=FAIL ;; *) W=PASS ;; esac
  grep -q "^HE_D4:" "results/${p}_fwd_d4.txt" && grep -q "^HE_D4:" "results/${p}_rev_d4.txt" || W=INCOMPLETE
  python3 - "$AF" "$p" <<'MT' >> "$LOG" 2>&1
import glob, json, math, sys
af, p = sys.argv[1:]
for o in ("fwd", "rev"):
    M = {}
    for f in sorted(glob.glob(f"{af}/{p}_{o}_p*_meta.json")):
        M.update(json.load(open(f)))
    ts = [m["tok_s"] for m in M.values() if m.get("tok_s")]
    geo = math.exp(sum(map(math.log, ts)) / len(ts)) if ts else 0
    fin = {}
    for m in M.values():
        fin[m.get("finish")] = fin.get(m.get("finish"), 0) + 1
    print(f"{p} {o}: {len(M)} prompts, tok/s geo mean {geo:.2f} (shared GPU, rough), finish {fin}")
MT
}
# ------------------------------------------------ start
echo $$ > results/q57.pid; trap 'rm -f results/q57.pid' EXIT
rm -f results/T95_DONE
say "runner 57 (the 12 GB t95ck1u code gate) start (pid $$)"
KEYS=$(python3 -c 'import json
print(",".join(sorted(json.loads(l)["key"] for l in open("data/items/dec_humaneval.jsonl"))))')
RKEYS=$(echo "$KEYS" | tr ',' '\n' | tail -r | paste -sd, -)
say "HumanEval+ $(echo "$KEYS" | tr ',' '\n' | grep -c .) keys per order; fit_t3.py $(shasum -a 256 "$FIT" | cut -c1-12); swap $SWAP0 GB at the start"
# ------------------------------------------------ 0. the VOID check: ex, cap 64, the first 4 forward keys vs REF2-T's first 32
VK=$(echo "$KEYS" | tr ',' '\n' | head -4 | paste -sd, -)
rm -f "$AF/void_ex.json"
run_arm ex void_ex 64 "$VK" "" "VOID check (ex, cap 64)"
python3 - "$AF/void_ex.json" <<'VC' >> "$LOG" 2>&1 || { say "the VOID check failed (ex differs from REF2-T before token 32): VOID: runner stops"; echo "T95: VOID (the ex setup check)" > results/T95_DONE; exit 5; }
import json, sys
ids = json.load(open(sys.argv[1]))
R = {r["key"]: r for r in map(json.loads, open("results/ref2t/decode.jsonl"))}
bad = []
for k, v in ids.items():
    a = list(v.values())[0] if isinstance(v, dict) else v      # {key: {arm: [run 1 ids, ...]}}: the first run (r57rows' format)
    a = a[0] if a and isinstance(a[0], list) else a
    b = R[k]["completion_ids"]
    n = min(32, len(a), len(b))
    d = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
    print(f"VOID check {k}: ex vs REF2-T first divergence {d if d is not None else 'none'} (of {min(len(a), len(b))} compared)")
    if a[:n] != b[:n] or n < min(32, len(b)):
        bad.append(k)
print(f"VOID check: {len(ids) - len(bad)} of {len(ids)} rows match REF2-T's first 32 tokens" + (f"; FAIL {bad}" if bad else ""))
sys.exit(1 if bad or len(ids) < 4 else 0)
VC
say "the VOID check passed (results/day56_runner.log)"
# ------------------------------------------------ 1. t95ck1u, both orders (no fallback)
gate_arm t95ck1u t95
echo "T95: t95ck1u $W ($V )" > results/T95_DONE
say "the verdict: $(cat results/T95_DONE)"
say "runner 57 end"
