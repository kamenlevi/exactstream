#!/bin/bash
# W24f (BigMoE, Tue 10 Nov 09:0x; the orchestrator 09:0x gave the slot: after Breakthrough's c47 and Leap's package check, one
# > 5 GB job at a time, finished by 14:50): MBPP and IFEval whole decode of w13b (the card-gated 24 GB config: 141 base + 26
# sidecar slots, 13 rows, R57_D7B=all) against w13t (W24b's gated config: 130 / 26, 13 rows, both drafters), same-run ABBA:
# tf1 (w13t), bf1, bf2 (w13b), tf2 (w13t); each arm MBPP then IFEval (12 prompts each, CAP 256). A measurement for the papers
# (SMALL_RAM cannot claim MBPP yet); no decision attached. Pooled per config on the prompts unpaused + uncut in all four arms.
# Env only, snapshot w24 (016cb7f). Needs a held quiet lock (turn45). ~18 min.
cd "$HOME/BigMoE" || exit 1
past() { [ $(( (10#$(date +%H) * 60 + 10#$(date +%M) - 10#${1:0:2} * 60 - 10#${1:2:2} + 1440) % 1440 )) -lt 720 ]; }
if past "${W24F_START_BY:-1425}"; then echo "$(date '+%m-%d %H:%M:%S') W24f not started: past ${W24F_START_BY:-1425} (must end by 14:50)" >> data/w24/w24.log; exit 0; fi
W24_ARMS="tf1 bf1 bf2 tf2" W24_SETS="m i" W24_RUN=w24f W24_DEADLINE=${W24F_DEADLINE:-1442} bash scripts/w24.sh
~/BigMoE/.venv/bin/python - > data/w24/w24f_verdict.txt 2>&1 <<'PY'
import json, os
R57 = os.path.expanduser("~/BigMoE/data/r57")
def wd(rows): return sum(r["n"] for r in rows) / sum(r["n"] / r["tps"] for r in rows) if rows else float("nan")
for s, name in (("m", "MBPP"), ("i", "IFEval")):
    A = {}
    for a in ("tf1", "bf1", "bf2", "tf2"):
        try:
            A[a] = ({r["key"]: r for r in json.load(open(f"{R57}/{a}{s}_100_runs.json"))}, json.load(open(f"{R57}/{a}{s}_100_tokens.json")))
        except (OSError, ValueError):
            print(f"{name}: {a} MISSING")
    if len(A) < 4:
        continue
    keys = set.intersection(*[set(v[0]) for v in A.values()])
    com = sorted(k for k in keys if all("PAUSED" not in A[a][0][k]["bad"] and not A[a][0][k]["cut"] for a in A))
    print(f"{name}: {len(com)} of {len(keys)} prompts unpaused + uncut in all four arms")
    for a in A:
        rows = list(A[a][0].values()); pk = max(max(r["mem"][1], r["mem"][-1]) for r in rows)
        bad = [k for k, r in A[a][0].items() if r["bad"].strip() or r["cut"]]
        print(f"  {a}: whole decode ALL {wd(rows):.2f}, COMMON {wd([A[a][0][k] for k in com]):.2f}; lifetime peak {pk:.3f} GB; bad / cut: {bad or 'none'}")
    t = wd([A[a][0][k] for a in ("tf1", "tf2") for k in com]); b = wd([A[a][0][k] for a in ("bf1", "bf2") for k in com])
    print(f"  POOLED COMMON: w13b {b:.2f} vs w13t {t:.2f} = {100 * (b / t - 1):+.1f} %; controls tf1 / tf2 "
          f"{wd([A['tf1'][0][k] for k in com]):.2f} / {wd([A['tf2'][0][k] for k in com]):.2f}")
    for x, y in (("bf2", "bf1"), ("tf2", "tf1")):
        print(f"  tokens {x} vs {y}: {sum(A[x][1].get(k) == A[y][1].get(k) for k in keys)}/{len(keys)} identical")
PY
echo "$(date '+%m-%d %H:%M:%S') W24f done: data/w24/w24f_verdict.txt" >> data/w24/w24.log
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT"
