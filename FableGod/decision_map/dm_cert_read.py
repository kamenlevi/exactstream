"""Mission 20's two-pass check on the decision map (Breakthrough, Mon 2026-11-02; the orchestrator 18:1x, Fable's
CACHE.md section 3, cloud Fable's design): pass 1 = top-k1 on every row (the all-rows runs topk2 / topk5); a row is
certified when its own p1 under that cut is >= theta; pass 2 = exact for the rest. cert_top<k1>_p<theta> run that mix in
one teacher-forced pass (certified rows at top-k1, the model's own weights; the rest exact), so the certified rows' cut
reaches later rows through the context, as in DAY9's gated run. Flip = top-1 differs from the baseline's.
Columns: pass 1 alone = the all-rows-cut run's flips on its certified rows (mission 20's "0 flips"); two-pass = the mix.
"p1 < theta in the mix" = certified rows whose p1 under the cut, in the mixed context, falls below theta (pass 1 here
certifies in an all-cut context; the engine's pass 1 sees the committed mix).
Threshold sweep (the orchestrator 19:2x, cloud Fable's 12 GB v2: pass 2 re-reads the 3.44 GB of dense weights whatever its
row count, so how often it runs is the lever): top-2 at theta 0.99 / 0.95 / 0.9 / 0.8 / 0.7, plus "blocks needing pass 2"
= the share of the exact walk's checks (16-row blocks, dm_gate's MAPW walks) with an uncertified row at or before the
first exact miss (cache_replay.py's rule: rows A .. A + accepted; a first check's prompt row is prefill's, exact). CPU columns need only
topk2 / topk5; the two-pass columns need the mixed runs.
Adaptive arm (cloud Fable's 12 GB v3, the orchestrator 20:2x; dm_acert.py): acert_top2_p<lo>_p<hi>_cf<s> certifies p1 >= hi,
or p1 >= lo where the drafter is sure (cf >= s) and its draft equals pass 1's argmax; the _cf09 variant is CPU only.
Usage (BigMoE venv): python dm_cert_read.py -> out/CERT_read.txt"""
import json, os, math
import numpy as np
import dm_gate, dm_acert
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
base = json.load(open(os.path.join(OUT, "baseline.json")))
DOMS = ("code", "ifeval", "multi")
SCHEMES = tuple((f"cert_top2_p0{str(t)[2:]}", "topk2", t, 2, None) for t in (0.99, 0.95, 0.9, 0.8, 0.7)) + (("cert_top5_p09", "topk5", 0.9, 5, None),
           ("acert_top2_p07_p09_cf099", "topk2", 0.9, 2, (0.7, 0.99)), ("acert_top2_p05_p08_cf099", "topk2", 0.8, 2, (0.5, 0.99)),
           ("acert_top2_p05_p095_cf099", "topk2", 0.95, 2, (0.5, 0.99)),
           ("acert_top2_p07_p09_cf09", "topk2", 0.9, 2, (0.7, 0.9)), ("t3cert_top2_p08", "t3_top2_all", 0.8, 2, None))
PRE = ("cert_top2_p099", "cert_top5_p09")              # pre-registered 18:3x (e00fed4); the sweep's others after
REFS = (("gate_top2_cf099", "DAY9 Q1's gated run (drafter cf >= 0.99, top-2)"), ("hc8w", "HC8 (8-bit HC read)"),
        ("gate_top2t4_cf099", "the t4 control (Q1's top-2 through another summation order)"))
lines = []
def say(s=""): print(s, flush=True); lines.append(s)

def wilson(k, n, z=1.96):
    if not n: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, 100 * (c - h)), 100 * (c + h)

def fmt(k, n, ci=False):
    if not n: return "-"
    s = f"{100 * k / n:.2f} % ({k}/{n})"
    if ci: lo, hi = wilson(k, n); s += f" [{lo:.2f}-{hi:.2f}]"
    return s

def load(name):
    p = os.path.join(OUT, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None

def flips(run, nm, m):
    b = np.array(base[nm]["tk"][:m])[:, 0]; return np.array(run[nm]["tk"][:m])[:, 0] != b

CHECKS = {}                                              # window -> [(anchor A, accepted k)] of the exact walk
for dom, wf in dm_gate.MAPW.items():
    M = np.load(os.path.join(dm_gate.TD, wf))
    for s, nm in enumerate([str(x) for x in M["names"]][:12]):
        CHECKS[nm] = [(int(A), int(k)) for (k, m, A) in M[f"{s}_checks"]]
assert set(CHECKS) >= set(base), "walk windows != map windows"

def certmask(nm, S, thr, ad):
    """certified mask over the measured rows (all of the all-rows run's rows)"""
    p1 = np.exp(np.array(S[nm]["lp"])[:, 0])
    if not ad: return p1 >= thr
    return dm_acert.acert(nm, S[nm]["a"], p1, np.array(S[nm]["tk"])[:, 0], ad[0], thr, ad[1])

def blocks(nm, ok):
    """(checks needing pass 2, checks): an uncertified row among A .. A + k (k accepted + the first miss / bonus row)."""
    a, n = base[nm]["a"], base[nm]["a"] + len(ok) + 1
    cert = np.r_[np.full(a, True), ok, False]          # rows 0 .. n - 1; prompt rows exact from prefill (never pass 2)
    need = sum(bool((~cert[A:min(A + k, n - 2) + 1]).any()) for A, k in CHECKS[nm])
    return need, len(CHECKS[nm])

verdict, summary = {}, {}
say("Two-pass check (mission 20) on the decision map: 36 windows, D-02a configuration, 4-bit experts; flips vs the baseline argmax.")
for run, src, thr, k1, ad in SCHEMES:
    S, R = load(src), load(run)
    if S is None: say(f"\n## {run}: its certificate source {src} is not run yet"); continue
    say(f"\n## {run}: top-{k1} on rows whose own p1 under the top-{k1} cut is >= {thr}"
        + (f" (or >= {ad[0]} where the drafter's cf >= {ad[1]} and its draft = pass 1's argmax)" if ad else "")
        + f" (certificate from {src}), the rest exact"
        + ("" if R else " - the mixed run is NOT DONE yet; CPU columns only"))
    say("| domain | certified share | blocks needing pass 2 | pass 1 alone: certified-row flips | two-pass: certified-row flips [95 % CI] | "
        "certified, oracle sure / tie / mid | uncertified (exact) rows: flips through the context | all rows | windows with a flip | "
        "certified rows with p1 < theta in the mix (their flips) |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    for dom in DOMS:
        A = {k: 0 for k in ("n", "nc", "f1", "fc", "fu", "fa", "w", "nw", "lo", "lof", "bn", "b")}; cls_n = [0] * 3; cls_f = [0] * 3
        for nm, b in base.items():
            if b["dom"] != dom: continue
            m = min(len(b["tk"]), len(S[nm]["tk"]), len(R[nm]["tk"]) if R else 10 ** 9)
            lp = np.array(b["lp"][:m]); p1o, p2o = np.exp(lp[:, 0]), np.exp(lp[:, 1])
            cls = np.where(p1o >= 0.99, 0, np.where(p1o - p2o <= 0.1, 1, 2))
            full = certmask(nm, S, thr, ad); cert = full[:m]
            A["n"] += m; A["nc"] += int(cert.sum()); A["f1"] += int(flips(S, nm, m)[cert].sum())
            bn, bt = blocks(nm, full); A["bn"] += bn; A["b"] += bt
            if R:
                f = flips(R, nm, m); A["fc"] += int(f[cert].sum()); A["fu"] += int(f[~cert].sum()); A["fa"] += int(f.sum())
                A["nw"] += 1; A["w"] += bool(f.any())
                lo = cert & (np.exp(np.array(R[nm]["lp"][:m])[:, 0]) < (ad[0] if ad else thr)); A["lo"] += int(lo.sum()); A["lof"] += int(f[lo].sum())
                for c in range(3): cls_n[c] += int((cert & (cls == c)).sum()); cls_f[c] += int(f[cert & (cls == c)].sum())
        two = (" | ".join([fmt(A["fc"], A["nc"], ci=True), " / ".join(fmt(cls_f[c], cls_n[c]) for c in range(3)),
                           fmt(A["fu"], A["n"] - A["nc"]), fmt(A["fa"], A["n"]), f"{A['w']}/{A['nw']}", f"{A['lo']} ({A['lof']})"])
               if R else " | ".join(["-"] * 6))
        say(f"| {dom} | {100 * A['nc'] / A['n']:.1f} % ({A['nc']}/{A['n']}) | {100 * A['bn'] / A['b']:.0f} % ({A['bn']}/{A['b']}) | "
            f"{fmt(A['f1'], A['nc'], ci=True)} | {two} |")
        summary[(run, dom)] = A
        if dom == "code": verdict[run] = A
say("\n## The sweep at a glance (code; ifeval / multi blocks in brackets)")
say("| scheme | certified share | blocks needing pass 2 (ifeval / multi) | pass 1 alone: certified flips | two-pass: certified flips | all code rows | verdict |")
say("|---|---|---|---|---|---|---|")
for run, src, thr, k1, ad in SCHEMES:
    if run not in verdict: continue
    A = verdict[run]; o = [summary[(run, d)] for d in ("ifeval", "multi")]
    v = ("not run" if not A["nw"] else "PASS" if 100 * A["fc"] / A["nc"] <= 1 else "KILL" if 100 * A["fc"] / A["nc"] > 5 else "BETWEEN")
    say(f"| top-{k1} p1 >= {thr}{f' (>= {ad[0]} if cf >= {ad[1]} & agree)' if ad else ''} | {100 * A['nc'] / A['n']:.1f} % | {100 * A['bn'] / A['b']:.0f} % ({' / '.join(f'{100 * x['bn'] / x['b']:.0f} %' for x in o)}) | "
        f"{fmt(A['f1'], A['nc'])} | {fmt(A['fc'], A['nc']) if A['nw'] else '-'} | {fmt(A['fa'], A['n']) if A['nw'] else '-'} | {v} |")
say("\n## For comparison: all-row flips of earlier map runs (code / ifeval / multi)")
for name, what in REFS:
    R = load(name)
    if not R: say(f"- {name}: not run"); continue
    parts = []
    for dom in DOMS:
        k = n = 0
        for nm, b in base.items():
            if b["dom"] == dom: m = min(len(b["tk"]), len(R[nm]["tk"])); k += int(flips(R, nm, m).sum()); n += m
        parts.append(fmt(k, n))
    say(f"- {name} ({what}): " + " / ".join(parts))
say("\n## Verdicts (pre-registered in ~/Breakthrough/SCOREBOARD.md before the runs; DAY9 Q1's line on the certified rows)")
for run, src, thr, k1, ad in SCHEMES:
    A = verdict.get(run)
    if not A or not A["nw"]: say(f"- {run}: not run"); continue
    p = 100 * A["fc"] / A["nc"]; lo, hi = wilson(A["fc"], A["nc"])
    v = "PASS" if p <= 1 else "KILL" if p > 5 else "BETWEEN"
    say(f"- {run}: code certified rows ({100 * A['nc'] / A['n']:.1f} %) flip {p:.2f} % [{lo:.2f}-{hi:.2f}] ({A['fc']}/{A['nc']}) -> {v} "
        f"(line <= 1 %, kill > 5 %); all code rows {100 * A['fa'] / A['n']:.2f} %, uncertified rows {fmt(A['fu'], A['n'] - A['nc'])} "
        f"through the context; {A['w']}/{A['nw']} code windows have a flip; blocks needing pass 2 {100 * A['bn'] / A['b']:.0f} %")
open(os.path.join(OUT, "CERT_read.txt"), "w").write("\n".join(lines) + "\n")
