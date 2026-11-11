"""Re-derive RELEASE's 24 GB numbers from the raw run records (editor, 10 Nov)."""
import json, os, statistics as st
H = os.path.expanduser("~")
R57 = f"{H}/BigMoE/data/r57"
CL = f"{H}/Leap/release/cleanclone/final/clone/out"
T = f"{H}/Leap/data/tue_afternoon_1110"
def runs(p): return json.load(open(p))
def peak(x): m = x["mem"]; return max(m[1], m[-1])
def pool(rs):
    n = sum(x["n"] for x in rs); dec = sum(x["n"] / x["tps"] for x in rs); pf = [x["pfs"] for x in rs]
    return n / dec, n / (dec + sum(pf)), st.median(pf), min(pf), max(pf), max(peak(x) for x in rs), n
print("== 24 GB rows (w13b, two passes each)")
for lab, names in (("HumanEval", ["w13eh_100", "w13fh_100"]), ("MBPP", ["bf1m_100", "bf2m_100"]),
                   ("IFEval", ["bf1i_100", "bf2i_100"]), ("math", ["bg1a_100", "bg2a_100"])):
    rs = [x for nm in names for x in runs(f"{R57}/{nm}_runs.json")]
    wd, wr, fm, f0, f1, pk, n = pool(rs)
    per = [peak(x) for x in rs]; Ps = [x["P"] for x in rs]
    print(f"{lab}: {len(rs)} rows, {n} tok: {wd:.2f} / {wr:.2f} / {fm:.2f} ({f0:.2f}-{f1:.2f}), peak {pk:.3f}; P {min(Ps)}-{max(Ps)}; "
          f"rows over 24.0: {[(x['key'], x['P'], round(peak(x), 3)) for x in rs if peak(x) > 24.0]}")
print("== package check (w24_pkg, cli_1110_100130)")
pr = runs(f"{H}/Leap/data/w24_pkg/cli_1110_100130_runs.json"); wd, wr, fm, f0, f1, pk, n = pool(pr)
print(f"pkg HE12: {wd:.2f} / {wr:.2f} / {fm:.2f} ({f0:.2f}-{f1:.2f}), peak {pk:.3f}, n {n}")
def tok(p):
    d = json.load(open(p)); return {k: (list(v.values())[0][-1] if isinstance(v, dict) else v) for k, v in d.items()}
pt = tok(f"{H}/Leap/data/w24_pkg/cli_1110_100130_tokens.json")
for ref in ("w13eh_100", "w13fh_100"):
    rt = tok(f"{R57}/{ref}_tokens.json"); print(f"  tokens vs {ref}: {sum(pt.get(k) == v for k, v in rt.items())}/{len(rt)}")
print("== 36 GB simulation (clone out)")
for lab, c in (("simA_he12", "134618"), ("simA_long", "134739"), ("simB_he12", "134911"), ("simB_long", "135032")):
    rs = runs(f"{CL}/cli_1110_{c}_runs.json"); wd, wr, fm, f0, f1, pk, n = pool(rs)
    print(f"{lab}: {len(rs)} rows P {[x['P'] for x in rs] if len(rs) == 1 else 'HE12'}: {wd:.2f} / {wr:.2f} / {fm:.2f}, peak {pk:.3f}, n {n}")
    if lab.endswith("he12"):
        t = tok(f"{CL}/cli_1110_{c}_tokens.json")
        for ref in ("w13eh_100", "w13fh_100"):
            rt = tok(f"{R57}/{ref}_tokens.json"); print(f"  tokens vs {ref}: {sum(t.get(k) == v for k, v in rt.items())}/{len(rt)}")
print("== peaks by prompt length (24 GB, default prefill)")
for nm in ("p24_10k_default", "p24_30k_default", "p24_10k_pfa", "p24_30k_pfa"):
    rs = runs(f"{T}/{nm}_runs.json"); print(nm, [(x["P"], x["n"], round(x["pfs"], 2), round(peak(x), 3)) for x in rs])
for nm in ("pfa24_default", "pfa24_pfa"):
    for x in (json.loads(l) for l in open(f"{T}/{nm}.jsonl")):
        print(nm, x["P"], x["n"], x["pfs"], round(x["peak"], 3), x["config"])
