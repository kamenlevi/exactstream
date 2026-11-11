"""NUMBER CHECK of SMALL_RAM's 24 GB fill (NewIdeas, Tue 10 Nov 10:4x): W24d / W24f / W24e rows and the gate's memory line,
re-derived from BigMoE's runs.json and EvalSuite's gate process records. STYLE 3: whole decode = sum n / sum n/tps (pooled),
whole request = sum n / sum (n/tps + pfs), first token = median pfs (min-max), lifetime peak = max mem[-1] (1e9 bytes)."""
import json, os, glob, statistics as st
R = os.path.expanduser("~/BigMoE/data/r57")
def L(t): return json.load(open(f"{R}/{t}_100_runs.json"))
def row(name, tags):
    rs = [r for t in tags for r in L(t)]
    n = sum(r["n"] for r in rs); dec = sum(r["n"] / r["tps"] for r in rs); pf = sum(r["pfs"] for r in rs)
    pfs = sorted(r["pfs"] for r in rs); geo = st.geometric_mean([r["tps"] for r in rs])
    per = [sum(r["n"] for r in L(t)) / sum(r["n"] / r["tps"] for r in L(t)) for t in tags]
    bad = [r["key"] for r in rs if r["bad"].strip()]
    print(f"{name}: pooled {n / dec:.2f} (geo {geo:.1f}; passes {', '.join(f'{p:.1f}' for p in per)}) | request {n / (dec + pf):.1f} | "
          f"first token {st.median(pfs):.2f} ({pfs[0]:.2f}-{pfs[-1]:.2f}) | peak {max(r['mem'][-1] for r in rs):.3f} | bad {bad} | "
          f"prompt tokens {min(r['P'] for r in rs)}-{max(r['P'] for r in rs)}")
    return n / dec
row("W24d HE w13b (w13e+w13f)", ["w13eh", "w13fh"])
b = row("W24f MBPP w13b (bf1m+bf2m)", ["bf1m", "bf2m"]); t = row("W24f MBPP w13t (tf1m+tf2m)", ["tf1m", "tf2m"])
row("W24f IFEval w13b (bf1i+bf2i)", ["bf1i", "bf2i"]); row("W24f IFEval w13t (tf1i+tf2i)", ["tf1i", "tf2i"])
for a, c in (("bf1m", "bf2m"), ("bf1i", "bf2i")):
    ta = json.load(open(f"{R}/{a}_100_tokens.json")); tc = json.load(open(f"{R}/{c}_100_tokens.json"))
    print(f"  tokens {a} vs {c}: {sum(ta[k] == tc[k] for k in ta)} / {len(ta)} equal")
print("== W24e (gate's 6 longest prompts)")
base = {r["key"]: r for r in L("pc1024g")}; bt = json.load(open(f"{R}/pc1024g_100_tokens.json"))
for tg in ("pc1024g", "pc512g", "pc256g", "pc1024bg"):
    rs = L(tg); tk = json.load(open(f"{R}/{tg}_100_tokens.json"))
    ft = [r["pfs"] / base[r["key"]]["pfs"] for r in rs]
    print(f"{tg}: peak {max(r['mem'][-1] for r in rs):.3f} | first token {min(r['pfs'] for r in rs):.1f}-{max(r['pfs'] for r in rs):.1f} s, x{min(ft):.2f}-x{max(ft):.2f} of 1024 | "
          f"tokens = 1024 on {sum(tk[k] == bt[k] for k in tk)} / {len(tk)} | lh = 1024 on {sum(r.get('lh') == base[r['key']].get('lh') for r in rs)} / {len(rs)} | P {sorted(r['P'] for r in rs)}")
print("== the gate's 40 processes (EV:w24runs)")
fs = sorted(glob.glob(os.path.expanduser("~/EvalSuite/results/card/es_cdw24_100136_*_runs.json")))
over = 0; mx = (0, None); first_over = []; under_before = []; ifb_only = []
for f in fs:
    rs = json.load(open(f)); pk = max(r["mem"][-1] for r in rs); over += pk > 24.0
    if pk > mx[0]: mx = (pk, max(rs, key=lambda r: r["mem"][-1]))
    hit = False
    for r in rs:
        if not hit and r["mem"][-1] > 24.0: first_over.append((r["P"], r["key"], round(r["mem"][-1], 3))); hit = True
        elif not hit: under_before.append(r["P"])
    if all(r["key"].startswith("ifbench/") for r in rs): ifb_only.append((pk, max(r["P"] for r in rs)))
print(f"processes {len(fs)}, prompt runs {sum(len(json.load(open(f))) for f in fs)}, over 24.0: {over}, max {mx[0]:.3f} GB at P {mx[1]['P']} ({mx[1]['key']})")
print(f"shortest prompt that first took a process over: {min(first_over)}; prompts <= 560 that ran before their process went over: {sum(p <= 560 for p in under_before)} (all under by construction)")
print(f"IFBench-only processes: {len(ifb_only)}, max peak {max(x[0] for x in ifb_only):.3f}, max prompt {max(x[1] for x in ifb_only)}")
