"""NUMBER CHECK part 2 (NewIdeas, Mon 9 Nov): MAIN's remaining checkable claims.
(1) exact-chat arms vs the N45 one-token reference; (2) rounding edges at 4 decimals; (3) PF-D logits hashes pdz4a vs pdz0;
(4) PF-D staged-but-unused per prompt; (5) N45 long agent prompts: prompt length, first token, peak; (6) CT1 fit
first token = a + GB / b; (7) exact chat decode share; (8) Strata pass-2 tokens vs our exact reference.
usage: /usr/bin/python3 -I main_check2.py"""
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


REF = {}
for refd in ("e5", "n45"):
    rp = os.path.expanduser(f"~/BigMoE/data/{refd}/ar_ref_tokens.json")
    if os.path.exists(rp):
        REF.update({k: toks_of(v) for k, v in json.load(open(rp)).items()})

print("== 1. chat arms vs the one-token reference (full / prefix to n)")
for t in ("yes1", "ycoh", "ycoh2", "yeh", "yesh", "y0", "y0b", "e32", "nie1", "nie2", "zs", "z0", "pdz0", "pdz4a"):
    r, _ = load(t)
    if r is None:
        continue
    tk = tokens(t)
    ks = [x["key"] for x in r if x["key"] in tk and x["key"] in REF]
    full = sum(tk[k] == REF[k] for k in ks)
    pre = sum(tk[k][:min(len(tk[k]), len(REF[k]))] == REF[k][:min(len(tk[k]), len(REF[k]))] for k in ks)
    print(f"{t}: full {full} / prefix {pre} / {len(ks)}")

print("\n== 2. rounding edges")
def pooled(tags, sets=None):
    rows = []
    for t in tags:
        rows += load(t)[0]
    n = sum(x["n"] for x in rows); dec = sum(x["n"] / x["tps"] for x in rows)
    pf = [x["pfs"] for x in rows if x.get("pfs") is not None]
    req = n / (dec + sum(pf)) if len(pf) == len(rows) else float("nan")
    return n / dec, req, (st.median(pf) if pf else float("nan")), n / len(rows)
for lbl, tags in (("S5 HE", ["s5v1", "s5v2"]), ("S5' B MBPP", ["spmb1", "spmb2"]), ("N45 L math", ["nml1", "nml2"]),
                  ("z0+z0b", ["z0", "z0b"]), ("zsgcr", ["zsgcr"]), ("pdzfa", ["pdzfa"]), ("pdz1p", ["pdz1p"]),
                  ("N45 EB IFEval", ["nie1", "nie2"]), ("S5' B HE", ["spb1", "spb2"])):
    d, q, f, npp = pooled(tags)
    print(f"{lbl}: decode {d:.4f}, request {q:.4f}, first token median {f:.4f}, tok/prompt {npp:.3f}, "
          f"decode s/prompt {npp / d:.4f}, request s/prompt {npp / q:.4f}")
def srows(d, arm, passes="[0-9]"):
    out = []
    for p in sorted(glob.glob(f"{H}/{d}/{arm}{passes}.requests.jsonl")):
        out += [json.loads(l) for l in open(p)]
    return [r for r in out if r.get("role") == "measure"]
for arm in "BD":
    for s in ("humaneval", "mbpp", "ifeval", "math"):
        x = [r for r in srows("results_arena33", arm) if r["set"] == s]
        g = sum(r["generated"] for r in x)
        print(f"Strata full33 {arm} {s}: decode {g / (sum(r['decode_ms'] for r in x) / 1e3):.4f}, request "
              f"{g / sum(r['wall_total_s'] for r in x):.4f}, ttft median {st.median(r['wall_ttft_s'] for r in x):.4f}")

print("\n== 3. logits hashes pdz4a vs pdz0 (per check)")
a, _ = load("pdz4a"); b, _ = load("pdz0")
A = {x["key"]: x.get("lh") or [] for x in a}; B = {x["key"]: x.get("lh") or [] for x in b}
tot = sum(len(v) for v in A.values()); eq = sum(sum(p == q for p, q in zip(A[k], B.get(k, []))) for k in A)
print(f"pdz4a lh entries {tot}, pdz0 lh entries {sum(len(v) for v in B.values())}, equal position-wise {eq}")

print("\n== 4. PF-D staged-but-unused per prompt (pB+pC vs hB+hC), pdz4a and pdzfa")
for t in ("pdz4a",):
    r, _ = load(t)
    u = []
    for x in r:
        d = x.get("pfd") or {}
        p = d.get("pB", 0) + d.get("pC", 0); h = d.get("hB", 0) + d.get("hC", 0)
        if p:
            u.append((p - h) / p * 100)
    ub = [((x["pfd"]["pB"] - x["pfd"]["hB"]) / x["pfd"]["pB"] * 100) for x in r if x.get("pfd", {}).get("pB")]
    uc = [((x["pfd"]["pC"] - x["pfd"]["hC"]) / x["pfd"]["pC"] * 100) for x in r if x.get("pfd", {}).get("pC")]
    print(f"{t}: per prompt unused {min(u):.1f}-{max(u):.1f} % (mean {st.mean(u):.1f}); B part {min(ub):.1f}-{max(ub):.1f} %; "
          f"C part {min(uc):.1f}-{max(uc):.1f} %")

print("\n== 5. N45 long agent prompts (npe = exact, npl = lossy) and pfx")
for t in ("npe1", "npe2", "npl1", "npl2"):
    r, _ = load(t)
    if r is None:
        continue
    print(t, "; ".join(f"{x['key']} P {x['P']} pfs {x.get('pfs')} peak {(x.get('mem') or [0])[-1]:.3f}" for x in r))

print("\n== 6. CT1 fit: first token (pfs) = a + GB / b over the 12 CT1 rows")
r, _ = load("ct1")
xs = [x["pfst"]["gb"] for x in r]; ys = [x["pfs"] for x in r]
mx, my = st.mean(xs), st.mean(ys)
slope = sum((p - mx) * (q - my) for p, q in zip(xs, ys)) / sum((p - mx) ** 2 for p in xs)
print(f"rows {len(r)}, GB per prompt mean {mx:.2f} (sum {sum(xs):.2f}), a = {my - slope * mx:.3f} s, b = {1 / slope:.2f} GB/s; "
      f"at 9.63 GB -> {my - slope * mx + slope * 9.63:.2f} s")

print("\n== 8. Strata both passes vs our exact reference (default cache) and full-cache pass 1")
for d, ps in (("results", "[12]"), ("results_arena33", "1")):
    for arm in "BD":
        R = srows(d, arm, ps)
        for s in ("humaneval", "mbpp", "ifeval", "math"):
            x = [r for r in R if r["set"] == s and r["key"] in REF]
            full = sum(r["ids"] == REF[r["key"]] for r in x)
            pre = sum(r["ids"][:min(len(r["ids"]), len(REF[r["key"]]))] == REF[r["key"]][:min(len(r["ids"]), len(REF[r["key"]]))] for r in x)
            print(f"{d} {arm} {s}: full {full} / prefix {pre} / {len(x)}")
