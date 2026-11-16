"""runaway4.py (Fit12, Mon 16 Nov 2026 13:3x; NewIdeas 13:3x ask; desk, CPU, python -I, read-only inputs).
Characterize the 1c-tiers gate's runaway keys (variant hits the ~2,048 cap, base does not) from the gate's own rows:
- tokens: lossy-45 base (lx45t1c1) and 1c-tiers (t1cx45t1c1) tier-0 records, exact-45's own tier-0 rows (x45rpl1,
  ~/EvalSuite/results/card_x45rpl1_t0code/decode.jsonl) and its per-position REPLAY margins (top-1 minus top-2 logit,
  replay_margins_x45rpl1.json; valid on exact's own token sequence);
- per check: kept tokens, misses and the variant's 1c counts [e4, e3, edrop, layers] (runs.json t1c_ck).
Per key: the first divergence base vs variant (and vs exact), the exact margin there (when the base still equals exact
up to that point), the variant's loop (smallest period and its start), the 1c drop share in the divergence check vs the
prompt's mean, and the decoded text around both.
"""
import glob
import json
import os

import numpy as np
from tokenizers import Tokenizer

C = os.path.expanduser("~/EvalSuite/results/card")
TK = Tokenizer.from_file(os.path.expanduser("~/BigMoE/models/bare/tokenizer.json"))
KEYS = ["mbpp/122", "mbpp/147", "mbpp/13", "humaneval/HumanEval/147"]
OUT = "/Users/kamen/Fit12/data/lossy32/runaway4.txt"


def rec(tag, part, kind):
    for f in glob.glob(os.path.join(C, f"es_cd{tag}_*_{part}_{kind}.json")):
        yield f, json.load(open(f))


def tokens(tag):
    t = {}
    for part in ("t01", "t02"):
        for _, d in rec(tag, part, "tokens"):
            for k, v in d.items():
                if k in KEYS:
                    t[k] = list(v.values())[0][0] if isinstance(list(v.values())[0][0], list) else list(v.values())[0]
    return t


def runs(tag):
    """runs.json (t1c_ck per check) joined with checks.json (the per-check kept list; runs.json's 'kept' is a mean)"""
    r = {}
    for part in ("t01", "t02"):
        for _, d in rec(tag, part, "runs"):
            for x in d:
                if x.get("key") in KEYS:
                    r[x["key"]] = dict(x)
        for _, d in rec(tag, part, "checks"):
            for x in d:
                if x.get("key") in KEYS and x["key"] in r:
                    r[x["key"]]["kept_list"] = x["kept"]
    return r


def first_div(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def loop(t, maxp=400, reps=3):
    """the smallest period P (<= maxp) with which the sequence's tail repeats >= reps times, and the loop's start"""
    n = len(t)
    best = None
    for P in range(1, maxp + 1):
        if n < reps * P:
            break
        s = n - P
        while s - 1 >= 0 and t[s - 1] == t[s - 1 + P]:      # extend the periodic region backwards
            s -= 1
        if n - s >= reps * P:
            best = (P, s)
            break
    return best


ex_rows = {}
for ln in open(os.path.expanduser("~/EvalSuite/results/card_x45rpl1_t0code/decode.jsonl")):
    j = json.loads(ln)
    if j["key"] in KEYS:
        ex_rows[j["key"]] = j["completion_ids"]
marg = json.load(open(os.path.join(C, "replay_margins_x45rpl1.json")))
tb, tv = tokens("lx45t1c1"), tokens("t1cx45t1c1")
rv = runs("t1cx45t1c1")
L = []


def p(x=""):
    print(x); L.append(x)


dec = lambda ids: TK.decode(ids).replace("\n", "\\n")
p(f"# runaway4 (Fit12): the 1c-tiers gate's 4 cap-hit keys, from the tier-0 rows")
for k in KEYS:
    b, v, e = tb.get(k), tv.get(k), ex_rows.get(k)
    if b is None or v is None:
        p(f"{k}: tokens missing (base {b is not None}, variant {v is not None})"); continue
    dbv, dbe, dve = first_div(b, v), (first_div(b, e) if e else None), (first_div(v, e) if e else None)
    mk = marg.get(k, {})
    m_at = mk.get(str(dbv)) if (e is not None and dbe is not None and dbe > dbv) else None
    lp = loop(v)
    r = rv.get(k, {})
    kept = r.get("kept_list", [])
    ck = r.get("t1c_ck") or []
    cum = np.cumsum(kept) if kept else np.array([])
    ci = int(np.searchsorted(cum, dbv, side="right")) if len(cum) else None
    share = lambda c: c[2] / max(1, c[0] + c[1] + c[2])
    drop_div = share(ck[ci]) if (ck and ci is not None and ci < len(ck)) else None
    drop_mean = float(np.mean([share(c) for c in ck])) if ck else None
    low = sorted((float(mk[str(i)]), i) for i in range(max(0, dbv - 3), dbv + 4) if str(i) in mk)
    p()
    p(f"## {k}: base {len(b)} tokens, variant {len(v)} (cap), exact {len(e) if e else 'n/a'}")
    p(f"- first divergence base vs variant at token {dbv}; base vs exact at {dbe}; variant vs exact at {dve}")
    p(f"- exact's margin at {dbv}: {m_at if m_at is not None else 'n/a (the base already left exact)'}; margins near it "
      f"(pos: nats): {', '.join(f'{i}: {mm:.2f}' for mm, i in sorted(low, key=lambda x: x[1]))}")
    p(f"- divergence check {ci} of {len(kept)}: 1c drop share {drop_div if drop_div is None else round(drop_div, 3)} vs the "
      f"prompt's mean {drop_mean if drop_mean is None else round(drop_mean, 3)}")
    if lp:
        P, s = lp
        p(f"- variant loop: period {P} tokens from token {s} ({(len(v) - s) / P:.1f} repeats; it starts {s - dbv} tokens after "
          f"the divergence); unit: \"{dec(v[s:s + P])[:160]}\"")
    else:
        p("- variant loop: no exact periodic tail found (P <= 400, >= 3 repeats)")
    p(f"- base at the divergence:    \"{dec(b[max(0, dbv - 12):dbv])[-80:]}\" >>> \"{dec(b[dbv:dbv + 12])[:80]}\"")
    p(f"- variant at the divergence: \"{dec(v[max(0, dbv - 12):dbv])[-80:]}\" >>> \"{dec(v[dbv:dbv + 12])[:80]}\"")
open(OUT, "w").write("\n".join(L) + "\n")
