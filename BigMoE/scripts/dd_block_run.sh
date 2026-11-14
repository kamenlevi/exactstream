#!/bin/bash
# BLOCK refresh sweep (PREREG_DEEPDRAFT_LAYERS_2026-11-14.md, addendum): after LAYERS-DONE, t30b on MBPP 100 with the
# stand-in replaced every k depths by each LAYERS map, k in {2, 4, 8, 16}, two evals at a time; then the grid against
# Leap's MBPP break-even (45 GB / 24 GB). usage (a harness background task): bash scripts/dd_block_run.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
OQ=$HOME/BigMoE/data/deepdraft/tapsq; LL=$OQ/layers.log; LOG=$OQ/block.log
PY=$HOME/BigMoE/.venv/bin/python; CK=$HOME/datasets/deepdraft/ckpt
say() { echo "$(date '+%H:%M:%S') $*" >> "$LOG"; }
say "waits for LAYERS-DONE"
until grep -q "LAYERS-DONE" "$LL" 2>/dev/null; do sleep 15; done
grep -q "LAYERS gate: .*(OK" "$LL" || { say "the LAYERS gate did not pass: the sweep is not run"; exit 2; }
peer lock share 60m --cpu 300 --gpu 80 "BigMoE DEEPDRAFT BLOCK sweep (2 x ~11 GB)" >/dev/null 2>&1
ev() { $PY scripts/dd_taps.py evalrep --runs t30b --replace rawmap --ridge $CK/ridge_L$1.npz --refresh $2 --out-name block_$1_k$2.json 2>&1 | grep -E "^\[rep\]|Traceback|rror" >> "$LOG"; }
( for s in 5 5_15_25 45; do for k in 2 4 8 16; do ev $s $k; done; done ) &
A=$!
( for s in 5_15 all; do for k in 2 4 8 16; do ev $s $k; done; done ) &
B=$!
wait $A; wait $B
peer unlock >/dev/null 2>&1
$PY - >> "$LOG" 2>&1 <<'PYEOF'
import json, os
D = os.path.expanduser("~/BigMoE/data/deepdraft/tapsq")
L = {"5": 5, "5_15": 15, "5_15_25": 25, "all": 45, "45": 45}
BE45 = {5: {1: 13.9, 2: 10.3, 4: 8.6, 8: 7.7, 16: 7.2}, 15: {1: 25.4, 2: 15.8, 4: 11.3, 8: 8.8, 16: 7.5},
        25: {1: 36.8, 2: 21.2, 4: 14.1, 8: 9.9, 16: 7.8}, 45: {1: 59.8, 2: 32.2, 4: 19.5, 8: 12.2, 16: 8.5}}
BE24 = {5: {1: 11.8, 2: 9.3, 4: 8.2, 8: 7.5, 16: 7.2}, 15: {1: 19.8, 2: 13.2, 4: 10.2, 8: 8.5, 16: 7.6},
        25: {1: 27.7, 2: 17.1, 4: 12.3, 8: 9.4, 16: 8.0}, 45: {1: 43.7, 2: 24.8, 4: 16.4, 8: 11.3, 16: 8.8}}
MS45 = {5: {1: 111, 2: 59, 4: 35, 8: 21, 16: 14}, 15: {1: 297, 2: 158, 4: 94, 8: 57, 16: 39},
        25: {1: 483, 2: 257, 4: 154, 8: 94, 16: 64}, 45: {1: 855, 2: 456, 4: 273, 8: 167, 16: 114}}
rows = []
for s in ("5", "5_15", "5_15_25", "all", "45"):
    for k in (1, 2, 4, 8, 16):
        f = f"{D}/layers_{s}.json" if k == 1 else f"{D}/block_{s}_k{k}.json"
        if not os.path.exists(f):
            continue
        e = json.load(open(f)); key = [x for x in e if "+rawmap" in x][0]
        kept, acc = e[key]["kept_per_32"], e[key]["23-30"]
        l = L[s]
        rows.append((MS45[l][k], s, k, l, acc, kept, BE45[l][k], BE24[l][k]))
print("BLOCK grid (MBPP 100; t30b own chain: 23-30 72.13, kept 6.92):")
print("| set | L | k | ms/draft (45 GB) | 23-30 | kept/32 | break-even 45 GB | 24 GB | clears 45 / 24 |")
for ms, s, k, l, acc, kept, b45, b24 in sorted(rows):
    print(f"| {s} | {l} | {k} | {ms} | {100 * acc:.2f} | {kept:.2f} | {b45} | {b24} | {'yes' if kept >= b45 else 'no'} / {'yes' if kept >= b24 else 'no'} |")
ok45 = [r for r in sorted(rows) if r[5] >= r[6]]; ok24 = [r for r in sorted(rows, key=lambda r: r[0]) if r[5] >= r[7]]
print("BLOCK VERDICT 45 GB:", f"FEASIBLE at set {ok45[0][1]} k {ok45[0][2]} (kept {ok45[0][5]:.2f} >= {ok45[0][6]}, 23-30 {100 * ok45[0][4]:.2f})" if ok45 else "NONE")
print("BLOCK VERDICT 24 GB:", f"FEASIBLE at set {ok24[0][1]} k {ok24[0][2]} (kept {ok24[0][5]:.2f} >= {ok24[0][7]}, 23-30 {100 * ok24[0][4]:.2f})" if ok24 else "NONE")
PYEOF
say "BLOCK-DONE"
