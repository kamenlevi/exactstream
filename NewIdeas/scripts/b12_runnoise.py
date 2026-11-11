"""B12's run-to-run noise (CPU): score the Eval suite's two B12 runs (guard on: results/b12; guard off: results/b12_noguard)
and the reference with the suite's own scorers, and count per-row pass/fail discordance between the two B12 runs, on all
shared rows and on rows the guard could not touch (rg_replay.py: no penalty on a chosen token in the no-guard output).
Run with PYTHONDONTWRITEBYTECODE=1 (the suite's tree is read-only for NewIdeas)."""
import json, os, sys
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
from evalsuite import decode_report as D

its = D.items()
ref, g, ng = D.load("ref"), D.load("b12"), D.load("b12_noguard")
rep = json.load(open(os.path.expanduser("~/NewIdeas/data/rg/replay_ref_plan4433333333_floor_w2f_r10_b12_noguard.json")))
touch = {o["key"] for o in rep["b12_noguard"] if o["R0"][1] is not None}
for set_, fn in (("math", D.score_math), ("ifeval", D.score_ifeval)):
    sr, sg, sn = fn(ref, its), fn(g, its), fn(ng, its)
    keys = sorted(set(sr) & set(sg) & set(sn))
    for label, ks in (("all shared", keys), ("guard could not touch", [k for k in keys if k not in touch])):
        n = len(ks)
        acc = lambda s: 100 * sum(s[k][1] for k in ks) / n
        def disc(a, b):
            return sum(a[k][1] > b[k][1] for k in ks), sum(a[k][1] < b[k][1] for k in ks)
        same_text = sum(g[k]["completion_ids"] == ng[k]["completion_ids"] for k in ks)
        print(f"{set_:6s} {label:22s} n={n:3d}  acc ref {acc(sr):5.1f}  B12 guard {acc(sg):5.1f}  B12 no-guard {acc(sn):5.1f} | "
              f"B12g vs B12ng discordant {disc(sg, sn)} (token-identical {same_text}) | B12g vs ref {disc(sg, sr)} | "
              f"B12ng vs ref {disc(sn, sr)}")
