"""guard_fail.py (GOAL, Fri 2026-11-06 21:1x): the MBPP12 CD16 engine run (BigMoE data/fday, c966c0d), read. CPU, python -I.
The answers of the flipped tasks, and the 125B's rank / margin for every token CD16 kept wholesale (the engine's INERT log)."""
import json, os, sys, collections
import numpy as np
from tokenizers import Tokenizer
H = os.path.expanduser("~")
tk = Tokenizer.from_file(f"{H}/BigMoE/models/bare/tokenizer.json")
dec = lambda ids: "".join(tk.decode([int(t)]) for t in ids)
R = f"{H}/BigMoE/data/r57"
tok = {a: json.load(open(f"{R}/{a}_224_tokens.json")) for a in ("dmq", "dmc", "dmcb", "dmqb")}
ver = json.load(open(f"{H}/BigMoE/data/fday/mbpp12_verdicts.json"))
what = sys.argv[1] if len(sys.argv) > 1 else "answers"
if what == "answers":
    for key in sys.argv[2:] or ["mbpp/16", "mbpp/319", "mbpp/296", "mbpp/268"]:
        for a in ("dmq", "dmc", "dmcb"):
            outs = [o for v in tok[a][key].values() for o in v]
            print(f"\n===== {key} [{a}] verdict {ver[a][key]} ({len(outs[0])} tokens)\n{dec(outs[0])}")
if what == "log":
    for a, f in (("dmc", "mc_inert.jsonl"), ("dmcb", "mcb_inert.jsonl")):
        L = [json.loads(l) for l in open(f"{H}/BigMoE/data/fday/{f}")]
        print(f"\n=== {a}: {len(L)} relaxed accepts; keys {sorted(L[0].keys())}")
        byp = collections.defaultdict(list)
        for e in L:
            byp[e["prompt"]].append(e)
        print(f"  {'prompt':22s} {'verdict':7s} {'n':>4s}  rank 0-1 / 2-3 / 4-7 / 8+   dlm >= -1 / -2.3 / -4.6 / below     region com / doc   mode whole / rule")
        for p, es in sorted(byp.items(), key=lambda kv: int(kv[0].split("_")[1])):
            key = "mbpp/" + p.split("_")[1]
            rk = np.array([e["rank"] for e in es]); dl = np.array([e["dlm"] for e in es])
            print(f"  {p:22s} {str(ver[a][key]):7s} {len(es):4d}  {int((rk <= 1).sum()):4d} / {int(((rk >= 2) & (rk <= 3)).sum()):3d} / {int(((rk >= 4) & (rk <= 7)).sum()):3d} / {int((rk >= 8).sum()):3d}   "
                  f"{int((dl >= -1).sum()):4d} / {int(((dl < -1) & (dl >= -2.3)).sum()):3d} / {int(((dl < -2.3) & (dl >= -4.6)).sum()):3d} / {int((dl < -4.6).sum()):3d}      "
                  f"{sum(e['region'] == 'comment' for e in es):4d} / {sum(e['region'] == 'docstring' for e in es):3d}     {sum(e['mode'] == 'whole' for e in es):4d} / {sum(e['mode'] != 'whole' for e in es):3d}")
        rk = np.array([e["rank"] for e in L]); dl = np.array([e["dlm"] for e in L])
        print(f"  all: rank 0 {np.mean(rk == 0):.2f}, 1 {np.mean(rk == 1):.2f}, 2 {np.mean(rk == 2):.2f}, 3-7 {np.mean((rk >= 3) & (rk <= 7)):.2f}, 8+ {np.mean(rk >= 8):.2f}; "
              f"dlm quartiles {np.percentile(dl, [10, 25, 50, 75, 90]).round(2).tolist()}")
if what == "seq":
    a, f, p = sys.argv[2], sys.argv[3], sys.argv[4]
    L = [json.loads(l) for l in open(f"{H}/BigMoE/data/fday/{f}") if json.loads(l)["prompt"].startswith(p)]
    for e in L[:int(sys.argv[5]) if len(sys.argv) > 5 else 60]:
        print(f"  check {e['check']:3d} row {e['row']:2d} {e['region'][:3]} rank {e['rank']:3d} dlm {e['dlm']:7.2f} cf {e['cf']:.2f}  draft {e['dtext']!r:16s} 125B {e['atext']!r:16s} | {e['ctext'][-50:]!r}")
if what == "guards":
    GUARDS = {"T3 (top-3, p >= 0.1 p1) = rule C's form": lambda e: e["rank"] <= 2 and e["dlm"] >= np.log(0.1),
              "T2 (top-2, p >= 0.2 p1)": lambda e: e["rank"] <= 1 and e["dlm"] >= np.log(0.2),
              "T8 (top-8, p >= 0.01 p1)": lambda e: e["rank"] <= 7 and e["dlm"] >= np.log(0.01),
              "top-8, any p (G16's later lines)": lambda e: e["rank"] <= 7}
    runs = {a: {x["key"]: x for x in json.load(open(f"{R}/{a}_224_runs.json"))} for a in ("dmq", "dmc", "dmcb", "dmqb")}
    chk = {a: {x["key"]: x for x in json.load(open(f"{R}/{a}_224_checks.json"))} for a in ("dmq", "dmc", "dmcb", "dmqb")}
    for a, f, ctl in (("dmc", "mc_inert.jsonl", "dmq"), ("dmcb", "mcb_inert.jsonl", "dmqb")):
        L = [json.loads(l) for l in open(f"{H}/BigMoE/data/fday/{f}")]
        byp = collections.defaultdict(list)
        for e in L:
            byp["mbpp/" + e["prompt"].split("_")[1]].append(e)
        print(f"\n=== {a} (control {ctl})")
        print(f"  {'task':9s} verdict ctl/arm   n ctl/arm   checks ctl/arm  K ctl / CD16   open  relaxed | kept by T3  T2  T8  top-8 | first T3 reject at relaxed #, text there")
        tot = collections.Counter()
        for key in runs[a]:
            ra, rc = runs[a][key], runs[ctl][key]
            es = byp.get(key, [])
            i = ra.get("inert") or {}
            g = {nm: sum(1 for e in es if fn(e)) for nm, fn in GUARDS.items()}
            first = next((j for j, e in enumerate(es) if not GUARDS["T3 (top-3, p >= 0.1 p1) = rule C's form"](e)), None)
            ft = "" if first is None else f"#{first}: {es[first]['ctext'][-40:]!r} + draft {es[first]['dtext']!r} (125B {es[first]['atext']!r}, rank {es[first]['rank']}, {es[first]['dlm']:.1f})"
            print(f"  {key:9s} {ver[ctl][key]!s:5s}/{ver[a][key]!s:5s} {rc['n']:4d}/{ra['n']:4d}   {rc['checks']:3d}/{ra['checks']:3d}   {rc['kept']:5.2f} / {ra['kept']:5.2f}  {i.get('open', 0):5d} {len(es):5d} | "
                  f"{g[list(GUARDS)[0]]:4d} {g[list(GUARDS)[1]]:4d} {g[list(GUARDS)[2]]:4d} {g[list(GUARDS)[3]]:4d} | {ft[:150]}")
            ok = ver[a][key] == ver[ctl][key]
            for nm in GUARDS:
                tot[(ok, nm)] += g[nm]
            tot[(ok, "n")] += len(es); tot[(ok, "open")] += i.get("open", 0); tot[(ok, "tokens")] += ra["n"]; tot[(ok, "checks")] += ra["checks"]
            tot[(ok, "ctl_tokens")] += rc["n"]; tot[(ok, "ctl_checks")] += rc["checks"]
        for ok in (True, False):
            n = max(tot[(ok, "n")], 1)
            print(f"  tasks whose verdict {'held' if ok else 'FLIPPED'}: open rows {tot[(ok, 'open')]}, relaxed {tot[(ok, 'n')]} ({100 * tot[(ok, 'n')] / max(tot[(ok, 'open')], 1):.0f} % of open); "
                  + "; ".join(f"{nm.split(' (')[0]} keeps {100 * tot[(ok, nm)] / n:.0f} %" for nm in GUARDS)
                  + f"; K control {tot[(ok, 'ctl_tokens')] / max(tot[(ok, 'ctl_checks')], 1):.2f}, CD16 {tot[(ok, 'tokens')] / max(tot[(ok, 'checks')], 1):.2f}")
            if ok:
                for nm in GUARDS:   # each relaxed accept the guard refuses ends its check there: one more check
                    F = tot[(ok, "n")] - tot[(ok, nm)]
                    print(f"     {nm:44s}: K ~ {tot[(ok, 'tokens')] / (tot[(ok, 'checks')] + F):.2f}  (CD16's checks + {F} refused accepts)")
