"""Check cost by verified rows from an r57 checks.json (first check of each prompt dropped): ck / bld / wt / svc /
misses medians per row count. Usage: python rows_cost.py ARM [ARM ...]  (files ~/BigMoE/data/r57/<ARM>_256_checks.json)"""
import json, os, sys
import numpy as np
F = ["ck", "bld", "wt", "svc", "io", "miss", "drf", "mb"]
for arm in sys.argv[1:]:
    d = json.load(open(os.path.expanduser(f"~/BigMoE/data/r57/{arm}_256_checks.json")))
    A = {f: [] for f in F + ["rows", "kept"]}
    for p in d:
        for i in range(1, len(p["rows"])):
            for f in F + ["rows", "kept"]:
                v = p.get(f); A[f].append(v[i] if isinstance(v, list) and len(v) > i and v[i] is not None else np.nan)
    A = {k: np.array(v, float) for k, v in A.items()}
    print(f"{arm}: {len(A['rows'])} checks, 1-row share {np.mean(A['rows'] == 1):.2f}")
    for k in range(1, 7):
        m = A["rows"] == k
        if m.sum() >= 5:
            print(f"  rows {k}: n {m.sum():3d} " + " ".join(f"{f} {np.nanmedian(A[f][m]):.1f}" for f in F)
                  + f" kept {np.nanmean(A['kept'][m]):.2f}")
