"""NUMBER CHECK (NewIdeas, Mon 9 Nov): the package's clean-clone FINAL against the engine's references and runs.
For each config: tokens of ifeval/1220 + 3280 vs the reference (full-length equality and prefix equality), and pooled whole
decode / whole request / first token vs the engine's own runs of the same two keys.
usage: /usr/bin/python3 -I final_check.py"""
import glob, json, os, statistics as st
OUT = os.path.expanduser("~/Leap/data/release/final_1210/out")
R57 = os.path.expanduser("~/BigMoE/data/r57")
def toks_of(v):
    while isinstance(v, dict):
        v = next(iter(v.values()))
    while isinstance(v, list) and v and isinstance(v[0], list):
        v = v[0]
    return v
AR = {k: toks_of(v) for k, v in json.load(open(os.path.expanduser("~/BigMoE/data/n45/ar_ref_tokens.json"))).items()}
CFG = {"exact-45": ("AR", "nie1_256"), "exact-32": ("AR", "e32_160"), "lossy-45": ("rli_256", "rli_256"),
       "lossy-32": ("rli_160", "rli_160"), "chat-45": ("pdz4a_256", "pdz4a_256"), "chat-32": ("zs32_160", "zs32_160")}
KEYS = ("ifeval/1220", "ifeval/3280")
def agg(rows):
    n = sum(x["n"] for x in rows); d = sum(x["n"] / x["tps"] for x in rows); pf = [x["pfs"] for x in rows]
    return n / d, n / (d + sum(pf)), st.median(pf)
for cfg, (refname, runname) in CFG.items():
    tf = glob.glob(f"{OUT}/{cfg}/cli_*_tokens.json"); rf = glob.glob(f"{OUT}/{cfg}/cli_*_runs.json")
    if not tf:
        print(cfg, "no package output"); continue
    pk = {k: toks_of(v) for k, v in json.load(open(tf[0])).items()}
    ref = AR if refname == "AR" else {k: toks_of(v) for k, v in json.load(open(f"{R57}/{refname}_tokens.json")).items()}
    cells = []
    for k in KEYS:
        a, b = pk.get(k), ref.get(k)
        full = a == b
        m = min(len(a), len(b)); pre = a[:m] == b[:m]
        cells.append(f"{k} n {len(a)} vs ref {len(b)}: full {full}, prefix {pre}")
    prow = [x for x in json.load(open(rf[0])) if x["key"] in KEYS]
    erow = [x for x in json.load(open(f"{R57}/{runname}_runs.json")) if x["key"] in KEYS]
    p = agg(prow); e = agg(erow)
    print(f"{cfg}: " + "; ".join(cells))
    print(f"   package {p[0]:.2f} / {p[1]:.2f} / {p[2]:.2f} s vs engine ({runname}) {e[0]:.2f} / {e[1]:.2f} / {e[2]:.2f} s: "
          f"decode x{p[0] / e[0]:.3f}, request x{p[1] / e[1]:.3f}, first token {p[2] - e[2]:+.2f} s")
