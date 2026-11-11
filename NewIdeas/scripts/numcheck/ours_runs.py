"""NUMBER CHECK (NewIdeas, Sun 8 Nov): our engine's headline numbers re-derived from BigMoE's per-prompt run rows.
For each group of runs (one arm, its passes pooled): prompts, tokens, pooled whole decode (sum n / sum n / tps),
whole request (sum n / sum (n / tps + pfs)), first token (pfs: prefill + setup + first token) mean / median, lifetime
peak (max of each record's last mem field), and how many prompts' tokens equal the AR reference (when one is given).
usage: /usr/bin/python3 -I ours_runs.py"""
import json, os, statistics as st

R57 = os.path.expanduser("~/BigMoE/data/r57")
GROUPS = [  # (label, [run tags], AR reference tokens file or None)
    ("S5' B HumanEval (cert config, lossy)", ["spb1", "spb2"], None),
    ("S5' X HumanEval (lossy FAST)", ["spx1", "spx2"], None),
    ("S5' B MBPP12 (cert config, lossy)", ["spmb1", "spmb2"], None),
    ("S5' X MBPP12 (lossy FAST)", ["spmx1", "spmx2"], None),
    ("E5 E HumanEval (exact)", ["e5h1", "e5h2"], "e5"),
    ("E5 EB HumanEval (exact + bundle)", ["e5hb1", "e5hb2"], "e5"),
    ("E5 E MBPP12 (exact)", ["e5m1", "e5m2"], "e5"),
    ("E5 EB MBPP12 (exact + bundle)", ["e5mb1", "e5mb2"], "e5"),
    ("N45 E IFEval (exact)", ["nie1", "nie2"], "n45"),
    ("N45 L IFEval (lossy cert)", ["nil1", "nil2"], "n45"),
    ("N45 E math (exact)", ["nme1", "nme2"], "n45"),
    ("N45 L math (lossy cert)", ["nml1", "nml2"], "n45"),
    ("CHS4 y0+y0b IFEval (L = cert on chat)", ["y0", "y0b"], None),
    ("CHS4 yes1 IFEval (E + SB1, exact)", ["yes1"], None),
    ("CHS5 z0+z0b IFEval (M1-B + COH)", ["z0", "z0b"], None),
    ("CHS5 zs IFEval (M1-B + COH + SB1)", ["zs"], None),
    ("CHS5 zsgcr IFEval (best stack, ungated)", ["zsgcr"], None),
    ("C32 e32 IFEval at C 160 (exact)", ["e32"], None),
    ("C32 zs32 IFEval at C 160 (lossy, ungated at 32 GB)", ["zs32"], None),
]


def load(tag):
    for c in (256, 160):
        p = f"{R57}/{tag}_{c}_runs.json"
        if os.path.exists(p):
            return json.load(open(p)), f"{R57}/{tag}_{c}_tokens.json"
    return None, None


def toks_of(entry):
    """tokens.json / ar_ref_tokens.json entry -> the first token list found."""
    v = entry
    while isinstance(v, dict):
        v = next(iter(v.values()))
    while isinstance(v, list) and v and isinstance(v[0], list):
        v = v[0]
    return v


for label, tags, refd in GROUPS:
    rows, eq, tot = [], 0, 0
    ref = None
    if refd:
        rp = os.path.expanduser(f"~/BigMoE/data/{refd}/ar_ref_tokens.json")
        ref = json.load(open(rp)) if os.path.exists(rp) else None
    for t in tags:
        r, tp = load(t)
        if r is None:
            print(f"{label}: run {t} not found"); continue
        rows += r
        if ref is not None and os.path.exists(tp):
            tk = json.load(open(tp))
            for x in r:
                k = x["key"]
                if k in tk and k in ref:
                    a, b = toks_of(tk[k]), toks_of(ref[k])
                    m = min(len(a), len(b), x["n"])
                    tot += 1; eq += a[:m] == b[:m] and m > 0
    if not rows:
        continue
    n = sum(x["n"] for x in rows); dec = sum(x["n"] / x["tps"] for x in rows)
    pf = [x["pfs"] for x in rows if x.get("pfs") is not None]
    req = n / (dec + sum(pf))
    peak = max((x.get("mem") or [0])[-1] for x in rows)
    s = (f"{label}: prompts {len(rows)}, tokens {n}, whole decode {n / dec:.1f}, whole request {req:.1f}, first token mean "
         f"{st.mean(pf):.2f} / median {st.median(pf):.2f} s, lifetime peak {peak:.2f} GB")
    if ref is not None:
        s += f", == AR {eq} / {tot}"
    print(s)
