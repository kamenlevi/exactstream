"""R-43 DFL24 pricing (b43): per-pass costs of the 24 GB engine's verify vs width S, from the traced width sweep (W4 /
W8 / W9: fixed widths on copy-heavy prompts, long n-gram drafts), extrapolated to S = 16 with a straight-line fit, and
predicted tok/s for DFlash2 16-row checks: K tokens kept per check (D-02b: 9.57 code, 7.00 math) / (drafting 12.5 or
7.9 ms + gate wait + verify GPU + commit + other per pass at 16 rows). Compared with A's tok/s from NewIdeas (b41b: 25.1
code, 19.9 math, 19-22 prose). Lines: PASS (build) >= +25 % on code and math and >= 0 on prose; KILL < +10 % on code.
usage: r43_price.py DIR"""
import json, os, sys
import numpy as np

D = sys.argv[1]
RUNS = ("W4", "W8", "W9")
DOM = {"edit": ("code", 9.57, 25.1), "copy_math": ("math", 7.00, 19.9)}


def per_pass(r, P):
    d = json.load(open(os.path.join(D, r + ".json")))["prompts"][P]
    s, ps = d["stats"], d["passes"]
    n = max(s["passes"], 1)
    m = lambda k: float(np.mean([p[k] for p in ps])) if ps else float("nan")
    wall = 1e3 * s["time"] / n                               # decode ms per pass (spec125's time excludes the prefill)
    span = m("span")
    return dict(S=d["S"], tpp=d["tok_per_pass"], tok_s=d["tok_s"], wall=wall, gate=m("stall"), gpu=m("gpu"), span=span,
                commit=1e3 * s["t_commit"] / n, draft=1e3 * s["t_draft"] / n, other=wall - span - 1e3 * (s["t_commit"] + s["t_draft"]) / n,
                reads=m("reads_demand") + m("reads_pf") + m("reads_joint"), busy=m("ssd_busy"))


out = {}
for P, (dom, K, a_tps) in DOM.items():
    rows = {r: per_pass(r, P) for r in RUNS}
    a = per_pass("A", P)
    print(f"\n{dom} ({P}): per verify pass, by width (A = shipped settings on the same prompt)")
    print("  run   S     tok/pass  tok/s   wall ms  gate ms  GPU ms  commit ms  draft ms  other ms  reads  SSD busy ms")
    for r, q in list(rows.items()) + [("A", a)]:
        print(f"  {r:4s} {q['S']:5.2f}  {q['tpp']:6.2f}  {q['tok_s']:6.1f}  {q['wall']:7.1f}  {q['gate']:7.1f}  {q['gpu']:6.1f}"
              f"  {q['commit']:9.1f}  {q['draft']:8.1f}  {q['other']:8.1f}  {q['reads']:5.0f}  {q['busy']:11.1f}")
    S = np.array([rows[r]["S"] for r in RUNS])
    fit = {k: np.polyfit(S, [rows[r][k] for r in RUNS], 1) for k in ("gate", "gpu", "commit", "other", "reads")}
    at16 = {k: float(np.polyval(f, 16.0)) for k, f in fit.items()}
    print("  straight-line fit vs S -> at 16 rows: " + ", ".join(f"{k} {v:.1f}" + ("" if k == "reads" else " ms")
                                                            for k, v in at16.items()))
    base = at16["gate"] + at16["gpu"] + at16["commit"] + at16["other"]
    for dr in (12.5, 7.9):
        t = dr + base
        tps = 1e3 * K / t
        print(f"  DFlash2 16-row check, drafting {dr} ms: {t:.1f} ms per check, {K} kept -> {tps:.1f} tok/s vs A {a_tps} "
              f"({100 * (tps / a_tps - 1):+.0f} %); break-even K {a_tps * t / 1e3:.2f}")
    out[dom] = dict(rows=rows, A=a, at16=at16, K=K, A_tps=a_tps)
json.dump(out, open(os.path.join(D, "price.json"), "w"), indent=1)
