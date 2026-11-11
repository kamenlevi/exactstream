"""W24c comparison + the gate-switch rule (BigMoE, Mon 9 Nov 23:1x; docs/W24_DECISION.md, section W24c).

Whole decode is pooled as the W24 reports pool it: sum n / sum (n / tps).
- ALL: every prompt of the arm (NewIdeas' number check pools w14a + w14b over all 12 prompts each).
- COMMON: the prompts that are unpaused ('PAUSED' not in bad) and uncut in EVERY arm given, so arms are compared on the
  same prompts (the W24b method).
Lifetime peak = max over rows of max(mem[1], mem[-1]) (GB = 10^9 bytes). Bad rows = a non-empty 'bad' field.
For set h with the four decision arms (w13c w14a w14b w13d) it prints the switch verdict, every condition with its value:
  pooled ALL over w14a + w14b >= 120.0; no bad HE row in w14a / w14b; peak < 24.000 in w14a and w14b; the w14 pair beats
  the controls' mean by more than 4.6 % (the noise between two identical W24 runs) on COMMON; and, as a flag only,
  whether the two controls agree within 2 % (thermal drift otherwise). MBPP / IFEval validity: run sets m and i.
usage: python3 scripts/w24c_compare.py [set h|m|i] [arm ...]   (default: h w13c w14a w14b w13d)
"""
import json
import os
import sys

R57 = os.environ.get("W24C_R57") or os.path.expanduser("~/BigMoE/data/r57")   # W24C_R57: a test dir
s = sys.argv[1] if len(sys.argv) > 1 else "h"
arms = sys.argv[2:] or ["w13c", "w14a", "w14b", "w13d"]


def load(a):
    t = f"{a}{s}_100"
    try:
        runs = {r["key"]: r for r in json.load(open(f"{R57}/{t}_runs.json"))}
        toks = json.load(open(f"{R57}/{t}_tokens.json"))
    except FileNotFoundError:
        return None
    return runs, toks


D = {a: load(a) for a in arms}
have = [a for a in arms if D[a]]
miss = [a for a in arms if not D[a]]
if miss:
    print(f"missing ({s}): {' '.join(miss)}")
if not have:
    sys.exit(1)
keys = set.intersection(*[set(D[a][0]) for a in have])
common = sorted(k for k in keys if all("PAUSED" not in D[a][0][k]["bad"] and not D[a][0][k]["cut"] for a in have))


def wd(rows):
    n = sum(r["n"] for r in rows)
    t = sum(r["n"] / r["tps"] for r in rows)
    return n / t if t else float("nan")


def peak(a):
    return max(max(r["mem"][1], r["mem"][-1]) for r in D[a][0].values())


def bad(a):
    return [(k, r["bad"].strip()) for k, r in D[a][0].items() if r["bad"].strip() or r["cut"]]


print(f"set {s}: {len(common)} of {len(keys)} prompts unpaused + uncut in all of {' '.join(have)}")
ctl = [a for a in ("w13c", "w13d") if a in have]
cm = sum(wd([D[a][0][k] for k in common]) for a in ctl) / len(ctl) if ctl else None
for a in have:
    al, co = wd(list(D[a][0].values())), wd([D[a][0][k] for k in common])
    rel = f" ({100 * (co / cm - 1):+.1f} % vs the controls' mean {cm:.1f})" if cm else ""
    b = bad(a)
    print(f"  {a}: whole decode ALL {len(D[a][0])} prompts {al:.2f}, COMMON {co:.2f}{rel}; lifetime peak {peak(a):.3f} GB; "
          f"bad rows {len(b)}{': ' + '; '.join(f'{k} {v}' for k, v in b) if b else ''}")
for x, y in [("w13d", "w13c"), ("w14b", "w14a"), ("w13a", "w13c"), ("w13b", "w13a")]:
    if x in have and y in have:
        same = sum(D[x][1].get(k) == D[y][1].get(k) for k in keys)
        print(f"  tokens {x} vs {y}: {same}/{len(keys)} prompts identical")
if s == "h" and all(a in have for a in ("w13c", "w14a", "w14b", "w13d")):
    pair = list(D["w14a"][0].values()) + list(D["w14b"][0].values())
    p_all = wd(pair)
    w14c = sum(wd([D[a][0][k] for k in common]) for a in ("w14a", "w14b")) / 2
    gain = 100 * (w14c / cm - 1)
    c1, c2 = (wd(list(D[a][0].values())) for a in ("w13c", "w13d"))
    drift = 100 * abs(c1 / c2 - 1)
    nb = len(bad("w14a")) + len(bad("w14b"))
    pk = max(peak("w14a"), peak("w14b"))
    conds = [(f"pooled ALL over w14a + w14b {p_all:.2f} >= 120.0", p_all >= 120.0),
             (f"bad HE rows in w14a / w14b: {nb}", nb == 0),
             (f"peak max(w14a, w14b) {pk:.3f} < 24.000 GB", pk < 24.0),
             (f"w14 pair COMMON {w14c:.2f} vs controls {cm:.2f}: {gain:+.1f} % > +4.6 %", gain > 4.6)]
    for txt, ok in conds:
        print(f"  [{'PASS' if ok else 'FAIL'}] {txt}")
    print(f"  [flag] controls w13c {c1:.2f} vs w13d {c2:.2f}: {drift:.1f} % apart ({'within' if drift <= 2 else 'OVER'} 2 %)")
    v = all(ok for _, ok in conds)
    print(f"  VERDICT (HE part): {'SWITCH-ELIGIBLE (still needs MBPP + IFEval validity, NewIdeas check, EvalSuite dry run by 01:20)' if v else 'w13t STAYS'}")
