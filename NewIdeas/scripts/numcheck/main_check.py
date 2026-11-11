"""NUMBER CHECK (NewIdeas, Mon 9 Nov): MAIN's sections 6-7 and abstract re-derived from the raw run records.
Per group: prompts, tokens per prompt, pooled whole decode (sum n / sum n/tps), whole request (sum n / sum (n/tps + pfs)),
first token median (min-max) from pfs, lifetime peak (max of each record's last mem, 3 decimals), first-token share of the
pooled request time, == AR when a reference is given. Then token comparisons, PF-D read volumes, Strata's per-set rows,
its MTP acceptance and pass repeatability, Strata vs our exact reference, and the break-even token counts.
usage: /usr/bin/python3 -I main_check.py"""
import glob, json, os, statistics as st

R57 = os.path.expanduser("~/BigMoE/data/r57")
H = os.path.expanduser("~/StrataEval/h2h")


def load(tag):
    for c in (256, 160, 224, 128):
        p = f"{R57}/{tag}_{c}_runs.json"
        if os.path.exists(p):
            return json.load(open(p)), f"{R57}/{tag}_{c}_tokens.json"
    return None, None


def toks_of(entry):
    v = entry
    while isinstance(v, dict):
        v = next(iter(v.values()))
    while isinstance(v, list) and v and isinstance(v[0], list):
        v = v[0]
    return v


def tokens(tag):
    _, tp = load(tag)
    return {k: toks_of(v) for k, v in json.load(open(tp)).items()} if tp and os.path.exists(tp) else {}


def stats(label, tags, refd=None, sets=None):
    rows, eq, tot = [], 0, 0
    ref = None
    if refd:
        rp = os.path.expanduser(f"~/BigMoE/data/{refd}/ar_ref_tokens.json")
        ref = {k: toks_of(v) for k, v in json.load(open(rp)).items()} if os.path.exists(rp) else None
    for t in tags:
        r, _ = load(t)
        if r is None:
            print(f"{label}: run {t} not found"); continue
        if sets:
            r = [x for x in r if x["key"].split("/")[0] in sets]
        rows += r
        if ref is not None:
            tk = tokens(t)
            for x in r:
                k = x["key"]
                if k in tk and k in ref:
                    a, b = tk[k], ref[k]; m = min(len(a), len(b), x["n"])
                    tot += 1; eq += a[:m] == b[:m] and m > 0
    if not rows:
        return None
    n = sum(x["n"] for x in rows); dec = sum(x["n"] / x["tps"] for x in rows)
    pf = [x["pfs"] for x in rows if x.get("pfs") is not None]
    peak = max((x.get("mem") or [0])[-1] for x in rows)
    s = f"{label}: prompts {len(rows)}, tok/prompt {n / len(rows):.1f}, whole decode {n / dec:.2f}"
    if len(pf) == len(rows):
        req = n / (dec + sum(pf))
        s += (f", whole request {req:.2f}, first token median {st.median(pf):.3f} ({min(pf):.2f}-{max(pf):.2f}) s, "
              f"first-token share {sum(pf) / (dec + sum(pf)) * 100:.0f} %")
    else:
        s += f", first token: {len(pf)} of {len(rows)} records carry pfs"
    s += f", peak {peak:.3f} GB"
    if ref is not None:
        s += f", == AR {eq} / {tot}"
    print(s)
    return dict(dec=n / dec, pf=st.median(pf) if pf else None)


print("== A. our rows (Tables 6.1a-c, 7.1, abstract)")
G = [("S5 HumanEval (certified build, s5v1-2)", ["s5v1", "s5v2"], None),
     ("S5 MBPP (s5mv1-2)", ["s5mv1", "s5mv2"], None),
     ("S5' B HumanEval (spb1-2)", ["spb1", "spb2"], None), ("S5' B MBPP (spmb1-2)", ["spmb1", "spmb2"], None),
     ("S5' X HumanEval (spx1-2)", ["spx1", "spx2"], None), ("S5' X MBPP (spmx1-2)", ["spmx1", "spmx2"], None),
     ("E5 EB HumanEval", ["e5hb1", "e5hb2"], "e5"), ("E5 E HumanEval", ["e5h1", "e5h2"], "e5"),
     ("E5 EB MBPP", ["e5mb1", "e5mb2"], "e5"), ("E5 E MBPP", ["e5m1", "e5m2"], "e5"),
     ("N45 EB IFEval", ["nie1", "nie2"], "n45"), ("N45 L IFEval", ["nil1", "nil2"], "n45"),
     ("N45 EB math", ["nme1", "nme2"], "n45"), ("N45 L math", ["nml1", "nml2"], "n45"),
     ("CHS4 yes1", ["yes1"], None), ("CHS4 y0+y0b", ["y0", "y0b"], None),
     ("CHS4 ycoh", ["ycoh"], None), ("CHS4 ycoh2", ["ycoh2"], None), ("CHS4 yeh", ["yeh"], None),
     ("CHS4 yesh", ["yesh"], None),
     ("CHS5 z0+z0b", ["z0", "z0b"], None), ("CHS5 z0", ["z0"], None), ("CHS5 z0b", ["z0b"], None),
     ("CHS5 zs", ["zs"], None), ("CHS5 zsgcr", ["zsgcr"], None),
     ("C32 e32", ["e32"], None), ("C32 zs32", ["zs32"], None)]
G += [(f"PF-D {t}", [t], None) for t in ("pdz0", "pdz1", "pdz1i", "pdz1n", "pdz1p", "pdz1q", "pdz4a", "pdzfa",
                                         "pde0", "pde1", "pde1k", "pde4k", "pdpa1", "pdb32s", "pdb38s")]
OURS = {}
for label, tags, refd in G:
    OURS[label] = stats(label, tags, refd)

print("\n== B. token comparisons")
def same(a, b, label):
    ta, tb = tokens(a), tokens(b)
    ks = sorted(set(ta) & set(tb))
    print(f"{label}: {sum(ta[k] == tb[k] for k in ks)} / {len(ks)} prompts token-identical (full sequences)")
same("spb1", "spb2", "S5' spb1 vs spb2 (HumanEval)")
same("spmb1", "spmb2", "S5' spmb1 vs spmb2 (MBPP)")
same("zs", "z0", "zs vs z0")
same("zs", "z0b", "zs vs z0b")
same("pdz4a", "pdz0", "pdz4a vs pdz0")
same("pdzfa", "pdz0", "pdzfa vs pdz0")
same("ycoh", "ycoh2", "ycoh vs ycoh2")
same("s5v1", "s5v2", "s5v1 vs s5v2")

print("\n== C. PF-D reads per prompt (pfd.ssd) and staged-but-unused share")
for t in ("pdz1", "pdz1i", "pdz1n", "pdz1p", "pdz1q", "pdz4a", "pdzfa"):
    r, _ = load(t)
    if not r:
        continue
    d = [x.get("pfd") or {} for x in r]
    ssd = [x.get("ssd", 0) / 1e9 for x in d]
    p = sum(x.get("pB", 0) + x.get("pC", 0) for x in d); h = sum(x.get("hB", 0) + x.get("hC", 0) for x in d)
    gb = [((x.get("pfst") or {}).get("gb")) for x in r]
    print(f"{t}: pfd.ssd mean {st.mean(ssd):.2f} GB/prompt; pfst.gb mean "
          f"{st.mean([g for g in gb if g is not None]) if any(g is not None for g in gb) else float('nan'):.2f}; "
          f"staged {p}, used {h}, unused {(p - h) / p * 100 if p else float('nan'):.1f} %")

print("\n== D. Strata per arm / set (pooled decode / whole request / first-token median), acceptance, pass repeat")
SETS = ("humaneval", "mbpp", "ifeval", "math")
def srows(d, arm, passes="[0-9]"):
    out = []
    for p in sorted(glob.glob(f"{H}/{d}/{arm}{passes}.requests.jsonl")):
        out += [json.loads(l) for l in open(p)]
    return [r for r in out if r.get("role") == "measure"]
STR = {}
for label, d in (("default", "results"), ("full33", "results_arena33")):
    for arm in "ABCD":
        R = srows(d, arm)
        if not R:
            continue
        for s in SETS + ("all",):
            x = [r for r in R if s == "all" or r["set"] == s]
            g = sum(r["generated"] for r in x)
            dec = g / (sum(r["decode_ms"] for r in x) / 1e3); wr = g / sum(r["wall_total_s"] for r in x)
            tt = st.median(r["wall_ttft_s"] for r in x)
            acc = sum(r.get("drafts_accepted", 0) for r in x); off = sum(r.get("drafts_offered", 0) for r in x)
            STR[(label, arm, s)] = dict(dec=dec, wr=wr, tt=tt)
            print(f"{label:7s} {arm} {s:9s} n {len(x):3d}: {dec:6.2f} / {wr:6.2f} / {tt:5.3f} s; drafts accepted "
                  f"{(acc / off * 100) if off else float('nan'):.1f} % ({acc}/{off})")
    if label == "default":
        for arm in "ABCD":
            for s in SETS + ("all",):
                v = []
                for ps in "12":
                    x = [r for r in srows(d, arm, ps) if s == "all" or r["set"] == s]
                    if x:
                        v.append(sum(r["generated"] for r in x) / (sum(r["decode_ms"] for r in x) / 1e3))
                if len(v) == 2:
                    print(f"  repeat {arm} {s:9s}: pass1 {v[0]:.2f} pass2 {v[1]:.2f} diff {abs(v[0] - v[1]) / min(v) * 100:.1f} %")

print("\n== E. Strata tokens vs our exact references (full sequences / first min-length prefix)")
REF = {}
for refd in ("e5", "n45"):
    rp = os.path.expanduser(f"~/BigMoE/data/{refd}/ar_ref_tokens.json")
    if os.path.exists(rp):
        REF.update({k: toks_of(v) for k, v in json.load(open(rp)).items()})
for label, d in (("default", "results"), ("full33", "results_arena33")):
    for arm in "ABCD":
        R = srows(d, arm, "1")
        if not R:
            continue
        cells = []
        for s in SETS:
            x = [r for r in R if r["set"] == s and r["key"] in REF]
            full = sum(r["ids"] == REF[r["key"]][:len(r["ids"])] and len(r["ids"]) == len(REF[r["key"]]) for r in x)
            pre = sum(r["ids"][:min(len(r["ids"]), len(REF[r["key"]]))] == REF[r["key"]][:min(len(r["ids"]), len(REF[r["key"]]))] for r in x)
            cells.append(f"{s} {full}/{pre}/{len(x)}")
        print(f"{label:7s} {arm} pass 1 (full / prefix / rows): " + " | ".join(cells))

print("\n== F. break-even answer length N: T_o + N/d_o = T_s + N/d_s (first-token medians, pooled decode)")
def be(lbl, To, do, Ts, ds):
    print(f"{lbl}: N = ({To:.2f} - {Ts:.2f}) / (1/{ds:.2f} - 1/{do:.2f}) = {(To - Ts) / (1 / ds - 1 / do):.0f}")
S = lambda arm, s: STR[("full33", arm, s)]
be("HumanEval lossy S5' B vs Strata B", OURS["S5' B HumanEval (spb1-2)"]["pf"], OURS["S5' B HumanEval (spb1-2)"]["dec"], S("B", "humaneval")["tt"], S("B", "humaneval")["dec"])
be("MBPP lossy S5' B vs Strata B", OURS["S5' B MBPP (spmb1-2)"]["pf"], OURS["S5' B MBPP (spmb1-2)"]["dec"], S("B", "mbpp")["tt"], S("B", "mbpp")["dec"])
be("math lossy N45 L vs Strata B", OURS["N45 L math"]["pf"], OURS["N45 L math"]["dec"], S("B", "math")["tt"], S("B", "math")["dec"])
be("chat certified N45 L vs Strata B", OURS["N45 L IFEval"]["pf"], OURS["N45 L IFEval"]["dec"], S("B", "ifeval")["tt"], S("B", "ifeval")["dec"])
be("chat certified CHS4 y0+y0b vs Strata B", OURS["CHS4 y0+y0b"]["pf"], OURS["CHS4 y0+y0b"]["dec"], S("B", "ifeval")["tt"], S("B", "ifeval")["dec"])
be("chat M1-B z0+z0b vs Strata B", OURS["CHS5 z0+z0b"]["pf"], OURS["CHS5 z0+z0b"]["dec"], S("B", "ifeval")["tt"], S("B", "ifeval")["dec"])
be("chat shipped pdz4a vs Strata B", OURS["PF-D pdz4a"]["pf"], OURS["PF-D pdz4a"]["dec"], S("B", "ifeval")["tt"], S("B", "ifeval")["dec"])
be("HumanEval exact EB vs Strata D", OURS["E5 EB HumanEval"]["pf"], OURS["E5 EB HumanEval"]["dec"], S("D", "humaneval")["tt"], S("D", "humaneval")["dec"])
be("math exact EB vs Strata D", OURS["N45 EB math"]["pf"], OURS["N45 EB math"]["dec"], S("D", "math")["tt"], S("D", "math")["dec"])
