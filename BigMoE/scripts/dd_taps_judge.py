#!/usr/bin/env python3
"""TAPSQ judge: the lines of docs/PREREG_DEEPDRAFT_TAPSQ_2026-11-14.md, from data/deepdraft/tapsq/tq_<arm>.json (per-step
losses) and eval.json / eval_R.json (MBPP 100 depth bands with per-window counts).
usage: dd_taps_judge.py [BASE CAND [TAG]]   default C T tq; R T after a pass; TAG tq2 = the rerun (amendment 2)"""
import json, math, os, sys
import numpy as np

D = os.path.expanduser("~/BigMoE/data/deepdraft/tapsq")
base, cand = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("C", "T")
tag = sys.argv[3] if len(sys.argv) > 3 else "tq"
suf = tag[2:]                                                        # "" for tq (the void run), "2" for tq2
b, c = (json.load(open(f"{D}/{tag}_{a}.json")) for a in (base, cand))
ev = {}
import glob
for f in sorted(glob.glob(f"{D}/eval{suf}*.json")) if suf else [f"{D}/eval.json", f"{D}/eval_R.json"]:
    if os.path.exists(f) and (suf or not os.path.basename(f).startswith("eval2")):
        ev.update(json.load(open(f)))
if "t30b" not in ev:   # the rerun scores t30b nowhere: its MBPP bands from ttt30b/eval.json (the eval is deterministic)
    ev["t30b"] = json.load(open(os.path.expanduser("~/BigMoE/data/deepdraft/ttt30b/eval.json")))["t30b"]["mbpp"]


def ratio(key, lo, hi):
    x, y = np.array(c[key][lo:hi], dtype=float), np.array(b[key][lo:hi], dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    return x[ok].mean() / y[ok].mean(), float((x[ok] < y[ok]).mean()), int(ok.sum())


print(f"TAPSQ: {cand} against {base}, {len(c['loss'])} steps each (start {c['start']}, seed {c['seed']})")
for key, lab in (("loss", "all depths (the training loss)"), ("d1", "depth 1 (band 0)"), ("d24_30", "depths 24-30 (band 23-29)")):
    for lo, hi in ((20, 40), (40, 80)):
        r, fr, n = ratio(key, lo, hi)
        print(f"  {lab}, steps {lo + 1}-{hi}: {cand}/{base} = {r:.3f}; {cand} lower on {100 * fr:.0f} % of {n} steps")
r_all, fr_all, _ = ratio("loss", 40, 80)

kb, kc = f"{tag}_{base}", f"{tag}_{cand}"
pct = lambda v: 100 * v
mb = None
if kb in ev and kc in ev:
    print(f"MBPP 100 bands (t30b / {base} / {cand}):")
    for bnd in ("0-0", "1-2", "3-6", "7-10", "11-14", "15-22", "23-30"):
        t0 = pct(ev["t30b"][bnd]) if "t30b" in ev else float("nan")
        print(f"  {bnd}: {t0:.2f} / {pct(ev[kb][bnd]):.2f} / {pct(ev[kc][bnd]):.2f} ({pct(ev[kc][bnd] - ev[kb][bnd]):+.2f})")
    print(f"  kept per 32: {ev.get('t30b', {}).get('kept_per_32', float('nan')):.3f} / {ev[kb]['kept_per_32']:.3f} / {ev[kc]['kept_per_32']:.3f}")
    mb = pct(ev[kc]["23-30"] - ev[kb]["23-30"])
    w = l = 0
    for pb, pc in zip(ev[kb]["per_window"], ev[kc]["per_window"]):
        (rb, vb), (rc, vc) = pb["23-30"], pc["23-30"]
        if vb and vc:
            ab, ac = rb / vb, rc / vc
            w += ac > ab; l += ac < ab
    n = w + l
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(w, l) + 1)) / 2 ** n) if n else 1.0
    print(f"  23-30 by window: {cand} better on {w}, worse on {l} (two-sided sign test p {p:.3f})")

valid = None
if mb is not None and base == "C":   # amendment 2: the control must stay within 1 pt of t30b at 23-30
    dc = pct(ev[kb]["23-30"] - ev["t30b"]["23-30"]); valid = abs(dc) <= 1.0
    print(f"validity: {base} vs t30b at MBPP 23-30 {dc:+.2f} pt ({'OK' if valid else 'NOT OK'}, within 1 pt)")
if mb is None:
    print("VERDICT: (no MBPP eval yet)")
elif valid is False:
    print("VERDICT: VOID (the control moved more than 1 pt from t30b at 23-30: the harness, not the taps)")
else:
    v = ("PASS" if r_all <= 0.90 and fr_all >= 0.70 and mb >= 2.0
         else "KILL" if r_all >= 0.97 and mb <= 1.0 else "BETWEEN")
    print(f"lines: ratio {r_all:.3f} (PASS <= 0.90), lower on {100 * fr_all:.0f} % (PASS >= 70 %), MBPP 23-30 {mb:+.2f} pt (PASS >= +2; KILL needs ratio >= 0.97 and <= +1)")
    print(f"VERDICT: {v}")
