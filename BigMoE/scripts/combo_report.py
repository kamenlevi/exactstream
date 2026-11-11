"""THE COMBINATION's report (scripts/combo_tue.sh): per arm (cumulative switch groups at C 256), tok/s per domain (steady state,
checks >= 2; and the whole run), kept per check, draft ms, lifetime peak, copy stats; every token vs cmb0 (the default,
itself vs chat23's REF2-T where stored); the delta of each arm vs the arm before it (the lever it adds) and vs cmb0.
usage: .venv/bin/python scripts/combo_report.py [ARMS=cmb0,cmb1,...] -> data/cmb/combo_report.txt"""
import json, os, sys
import numpy as np

R = os.path.expanduser("~/BigMoE/data/r57")
ARMS = (sys.argv[1] if len(sys.argv) > 1 else "cmb0,cmbj0,cmbj2,cmb1,cmb1c,cmb1t,cmb2,cmb3,cmb4,cmb5,cmb6,cmb3t,cmb32,cmbsc,fastBE,fastCE,fastBEn,fastBEh,fastBEhw").split(",")
PREV = {"cmbj0": "cmb0", "cmbj2": "cmb0", "cmb1": "cmb0", "cmb1t": "cmb1c", "cmb2": "cmb1c", "cmb3t": "cmb4", "cmb32": "cmb4", "cmbsc": "cmb6", "fastBE": "cmb6", "fastCE": "cmb6", "fastBEn": "fastBE", "fastBEh": "fastBEn", "fastBEhw": "fastBEh"}               # the variants sit on cmb4's stack
REF = json.load(open(os.path.join(R, "chat23_ref_tokens.json")))
DOM = lambda k: "code" if k.startswith("humaneval") else "math" if k.startswith(("math500", "gsm8k")) else \
    "agent" if k.startswith("agent") else "chat"
L = []
def say(s=""):
    print(s); L.append(s)
CC = os.environ.get("CMB_C", "256")                     # the chain's C (11-03: 224 when 256 did not fit the memory)
def J(tag, kind):
    try:
        return json.load(open(os.path.join(R, f"{tag}_{CC}_{kind}.json")))
    except Exception:
        return None
def toks(tag):
    return {k: v["gate+t"] for k, v in (J(tag, "tokens") or {}).items() if "gate+t" in v}
def eq(a, b):
    m = min(len(a), len(b))
    return a[:m] == b[:m]
T0 = toks("cmb0")
A = {}
for t in ARMS:
    rs = [r for r in (J(t, "runs") or []) if not r.get("bad")]
    if not rs:
        continue
    d = {}
    for dom in ("code", "math", "chat", "agent"):
        x = [r for r in rs if DOM(r["key"]) == dom]
        if x:
            d[dom] = dict(tpsw=float(np.mean([r["tpsw"] for r in x])), tps=float(np.mean([r["tps"] for r in x])),
                          mb=float(np.mean([r["mb"] for r in x])),   # SSD MB per kept token (the process's disk bytes)
                          kept=float(np.mean([r["kept"] for r in x])), drf=float(np.mean([r["drf"] for r in x])),
                          cyc=float(np.mean([r["cyc"] for r in x])))
    tk = toks(t); ref = T0 if t != "cmb0" else {k: REF[k]["gate+t"] for k in tk if k in REF}
    n = sum(len(v) for k, v in tk.items() if k in ref)
    ok = sum(eq(o, ref[k][0]) for k, v in tk.items() if k in ref for o in v)
    cp = [r["copy"] for r in rs if r.get("copy")]
    fs = [r["fast"] for r in rs if r.get("fast")]
    A[t] = dict(d=d, ok=ok, n=n, bad=sorted(k for k, v in tk.items() if k in ref and not all(eq(o, ref[k][0]) for o in v)),
                peak=max(r["mem"][6] for r in rs), copy=cp, fast=fs)
say(f"THE COMBINATION at C {CC} (shared GPU by day: rough), steady tok/s per domain (checks >= 2), kept per check, draft ms")
say("| arm | adds | tokens equal | code | math | chat | agent | vs the arm before | vs cmb0 | peak GB |")
say("|---|---|---|---|---|---|---|---|---|---|")
ADDS = {"cmb0": "the default", "cmbj0": "joint sizes, equal RAM (on cmb0)", "cmbj2": "joint sizes +2.07 GB (on cmb0)", "cmb1": "chat switch", "cmb1c": "chat at 8 rows", "cmb1t": "chat t 0.4 (on cmb1c)", "cmb2": "A6 keep", "cmb3": "COPYGATE g2", "cmb4": "D7b code",
        "cmb5": "JOINT", "cmb6": "K-83", "cmb3t": "COPYGATE g2t (on cmb4)", "cmb32": "32 rows (on cmb4)", "cmbsc": "sidecar x1.5 (on cmb6)", "fastBE": "FAST B+E (on cmb6; tokens differ by design)",
        "fastCE": "FAST C+E (on cmb6; tokens differ by design)", "fastBEn": "FAST B+E + NAX6", "fastBEh": "FAST B+E + NAX6 + H-1c", "fastBEhw": "fastBEh + warm split (not a speed arm)"}
def mean_tps(a, doms):
    x = [a["d"][d_]["tpsw"] for d_ in doms if d_ in a["d"]]
    return float(np.mean(x)) if x else float("nan")
order = [t for t in ARMS if t in A]
for i, t in enumerate(order):
    a = A[t]; doms = [d_ for d_ in ("code", "math", "chat", "agent")]
    cell = lambda d_: (f"{a['d'][d_]['tpsw']:.1f} (kept {a['d'][d_]['kept']:.1f}, draft {a['d'][d_]['drf']:.1f})"
                       if d_ in a["d"] else "-")
    p = PREV.get(t, order[i - 1] if i else None)
    vs = lambda o: (" / ".join(f"{100 * (a['d'][d_]['tpsw'] / A[o]['d'][d_]['tpsw'] - 1):+.1f}" for d_ in doms
                               if d_ in a["d"] and d_ in A[o]["d"]) if o in A else "-")
    say(f"| {t} | {ADDS.get(t, '')} | {a['ok']} / {a['n']}{' DIFFER ' + ', '.join(a['bad']) if a['bad'] else ''} | "
        + " | ".join(cell(d_) for d_ in doms) + f" | {vs(p) if p else '-'} % | {vs('cmb0') if t != 'cmb0' else '-'} % | "
        f"{a['peak']:.1f} |")
for t in order:
    if A[t]["copy"]:
        c = A[t]["copy"]; pos = sum(x["pos"] for x in c)
        say(f"{t} copy: used {sum(x['used'] for x in c)} of {pos} drafted positions, changed the draft "
            f"{sum(x['chg'] for x in c)}, accepted {sum(x['ok'] for x in c)}; host {np.mean([x['ms'] for x in c]):.1f} ms per run")
for t in order:
    if A[t]["fast"]:
        f_ = A[t]["fast"]; rows = sum(x["rows"] for x in f_)
        say(f"{t} FAST: {sum(x['gated'] for x in f_)} of {rows} rows gated ({100 * sum(x['gated'] for x in f_) / max(rows, 1):.0f} %), "
            f"kept {sum(x['k'] for x in f_)} vs exact acceptance {sum(x['kx'] for x in f_)} on the same checks")
say("SSD MB per kept token per arm (code / math / chat / agent): " + "; ".join(
    f"{t} " + " / ".join(f"{A[t]['d'][d_]['mb']:.1f}" if d_ in A[t]['d'] else "-" for d_ in ("code", "math", "chat", "agent"))
    for t in order))
for t in order:                                          # the warm split's per-row cost (fastBEhw: Fable C's gated expert term)
    try:
        W = json.load(open(os.path.join(R, f"{t}_{CC}_warmsplit.json")))
    except Exception:
        continue
    cw = [sm for w in W if w["key"].startswith("humaneval") for sm in w["samples"]]
    if cw:
        wm = {n: float(np.median([x[1] for sm in cw for x in sm["warm"] if x[0] == n])) for n in (1, 4, 8, 16)}
        say(f"{t} warm split on code ({len(cw)} samples): warm ms at 1 / 4 / 8 / 16 rows {wm[1]:.1f} / {wm[4]:.1f} / {wm[8]:.1f} / "
            f"{wm[16]:.1f}, slope {(wm[16] - wm[1]) / 15:.2f} ms per row; cold 16-row {np.median([sm['cold'][1] for sm in cw]):.1f} ms")
os.makedirs(os.path.expanduser("~/BigMoE/data/cmb"), exist_ok=True)
open(os.path.expanduser("~/BigMoE/data/cmb/combo_report.txt"), "w").write("\n".join(L) + "\n")
