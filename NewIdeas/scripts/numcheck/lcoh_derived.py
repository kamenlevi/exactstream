"""NUMBER CHECK (NewIdeas, Mon 9 Nov): the derived numbers MAIN needs once it is re-based on L + COH (REL ROWS) as the lossy
row: per-set triples with first-token ranges, ratios and first-token ratios against Strata's full-cache arm B, break-even
answer lengths (first-token medians + pooled decode), peaks. usage: /usr/bin/python3 -I lcoh_derived.py"""
import glob, json, os, statistics as st
R57 = os.path.expanduser("~/BigMoE/data/r57"); H = os.path.expanduser("~/StrataEval/h2h")
def ours(tags):
    R = sum((json.load(open(f"{R57}/{t}_runs.json")) for t in tags), [])
    n = sum(x["n"] for x in R); d = sum(x["n"] / x["tps"] for x in R); pf = [x["pfs"] for x in R]
    return dict(dec=n / d, req=n / (d + sum(pf)), tt=st.median(pf), lo=min(pf), hi=max(pf), peak=max(x["mem"][-1] for x in R))
def strata(arm, s):
    R = []
    for p in sorted(glob.glob(f"{H}/results_arena33/{arm}1.requests.jsonl")):
        R += [json.loads(l) for l in open(p)]
    x = [r for r in R if r.get("role") == "measure" and r["set"] == s]
    g = sum(r["generated"] for r in x)
    return dict(dec=g / (sum(r["decode_ms"] for r in x) / 1e3), req=g / sum(r["wall_total_s"] for r in x), tt=st.median(r["wall_ttft_s"] for r in x))
SETS = (("HumanEval", ["rlh_256"], "humaneval"), ("MBPP", ["rlm_256"], "mbpp"), ("IFEval", ["rli_256", "rlr_256"], "ifeval"),
        ("math", ["rlt_256"], "math"))
for lbl, tags, s in SETS:
    o = ours(tags); b = strata("B", s); d = strata("D", s)
    be = (o["tt"] - b["tt"]) / (1 / b["dec"] - 1 / o["dec"]) if o["dec"] > b["dec"] else float("inf")
    print(f"{lbl:9s} L+COH {o['dec']:.2f} / {o['req']:.2f} / {o['tt']:.2f} ({o['lo']:.2f}-{o['hi']:.2f}) s, {o['peak']:.2f} GB | "
          f"Strata B {b['dec']:.2f} / {b['req']:.2f} / {b['tt']:.2f} | decode x{o['dec'] / b['dec']:.2f}, request x{o['req'] / b['req']:.2f}, "
          f"first token {o['tt'] / b['tt']:.2f}x slower | break-even {be:.0f} tokens")
