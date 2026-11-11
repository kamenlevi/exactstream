"""dh_w4 decode A/B report (data/defer/gen_PROMPT_ARM_rR.json from run_gpu*.sh): per prompt and arm, tok/s, tok/pass, mean
verify width S, verify ms/pass, demand reads and late reads per pass; then the speed ratio at MATCHED verify width (dh's
texts diverge from v3's after a few tokens, so passes differ in S and acceptance): mean verify ms per S bucket, weighted
by the control's S distribution, and the implied tok/s at the control's acceptance.
usage: gen_report.py [arms, default v3,dh,pipe]"""
import collections, glob, json, os, sys
import numpy as np
D = os.path.expanduser("~/MoonShots/data/defer")
ARMS = (sys.argv[1] if len(sys.argv) > 1 else "v3,dh,d1,pipe").split(",")
BUCKETS = ((1, 2), (3, 4), (5, 6), (7, 8))
runs = collections.defaultdict(list)
for f in sorted(glob.glob(f"{D}/gen_*_r[0-9].json")):
    p, a = os.path.basename(f)[4:-8].split("_", 1)                 # gen_PROMPT_ARM_rR.json
    runs[(p, a)].append(json.load(open(f)))
for p in ("code", "prose", "edit"):
    print(f"== {p}")
    per = {}
    for a in ARMS:
        rs = runs.get((p, a))
        if not rs:
            continue
        P = [x for r in rs for x in r["passes"]]
        S = np.array([x["S"] for x in P]); V = np.array([x["t_verify"] for x in P]) * 1e3
        tok = np.mean([r["summary"]["tok_s"] for r in rs]); tpp = np.mean([r["summary"]["tok_per_pass"] for r in rs])
        mis = np.mean([x["x_misses"] for x in P]); late = np.mean([x.get("x_late", 0) for x in P])
        per[a] = (S, V, tpp)
        each = ", ".join("%.1f" % r["summary"]["tok_s"] for r in rs)
        print(f"  {a:5s} ({len(rs)} runs): {tok:5.1f} tok/s ({each}), "
              f"{tpp:.2f} tok/pass, S {S.mean():.2f}, verify {V.mean():6.1f} ms/pass, reads {mis:5.1f}/pass ({late:.1f} late), "
              f"sha {' '.join(r['sha'] for r in rs)}")
    if "v3" not in per:
        continue
    S0, V0, t0 = per["v3"]
    for a in per:
        if a == "v3":
            continue
        S1, V1, t1 = per[a]
        num = den = 0.0
        cells = []
        for lo, hi in BUCKETS:
            m0, m1 = (S0 >= lo) & (S0 <= hi), (S1 >= lo) & (S1 <= hi)
            if m0.sum() and m1.sum():
                w = m0.sum()
                num += w * V1[m1].mean(); den += w * V0[m0].mean()
                cells.append(f"S{lo}-{hi}: {V0[m0].mean():.1f} -> {V1[m1].mean():.1f} ms ({m0.sum()}/{m1.sum()} passes)")
        if den:
            print(f"  {a} vs v3 at matched S: verify x{den / num:.3f} ({'; '.join(cells)}); acceptance {t0:.2f} -> {t1:.2f} tok/pass")
