"""Union per 16-row check: Q1-UNION (certainty gate: sure rows k0 2, unsure 10) vs OEA (published, arXiv 2511.02237:
every row keeps top-k0, then adds lower ranks already in the union) vs a hybrid (sure rows k0 2, unsure rows k0 ku).
Same blocks as union_theta.py / dm_q1u_masks.py. Reads only. Also: mean experts kept per row (rows add union members)."""
import json, os, sys
import numpy as np
DM = os.path.expanduser("~/FableGod/decision_map"); sys.path.insert(0, DM)
import dm_gate  # noqa: E402
OUT = os.path.join(DM, "out"); base = json.load(open(os.path.join(OUT, "baseline.json"))); G = dm_gate.load()
ARMS = [("q1u_0.99", 0.99, 2, 10), ("q1u_0.9", 0.9, 2, 10), ("oea_k4", None, 4, 4), ("oea_k6", None, 6, 6),
        ("oea_k8", None, 8, 8), ("hyb_0.9_u8", 0.9, 2, 8), ("hyb_0.9_u6", 0.9, 2, 6), ("hyb_0.99_u8", 0.99, 2, 8)]
res = {}
for dom, wf in dm_gate.WALK.items():
    Z = np.load(os.path.join(dm_gate.TD, wf))
    for i, nm in enumerate(str(x) for x in Z["names"]):
        if nm not in base: continue
        ids = np.load(os.path.join(OUT, "routing", nm))["ids"]; N = len(ids); g = np.nan_to_num(G[nm], nan=-1.0)
        for (k, m, A), dc in zip(Z[f"{i}_checks"].tolist(), Z[f"{i}_dcf"].tolist()):
            last, fm = min(A + m, N - 1), min(A + k, N - 1)
            rows, cfs = [], []
            for p in range(A, last + 1):
                if p <= fm: rows.append(p); cfs.append(g[p])
                else: rows.append((p + 97) % N); cfs.append(dc[p - A] if p - A < m else -1.0)
            R = ids[rows]; cfs = np.array(cfs); b = len(rows)
            d = res.setdefault(dom, {"exact": 0, "rowl": 0})
            d["exact"] += sum(len(np.unique(R[:, l, :])) for l in range(48)); d["rowl"] += b * 48
            for name, thr, ks, ku in ARMS:
                k0 = np.full(b, ku) if thr is None else np.where(cfs >= thr, ks, ku)
                u = kept = 0
                for l in range(48):
                    U = np.unique(np.concatenate([R[j, l, :k0[j]] for j in range(b)]))
                    u += len(U); kept += int(np.isin(R[:, l, :], U).sum())
                d[name] = d.get(name, 0) + u; d["k_" + name] = d.get("k_" + name, 0) + kept
for dom, d in res.items():
    print(f"{dom}: exact union {d['exact'] / d['rowl'] * 0 + d['exact']:.0f} expert-layers")
    for name, *_ in ARMS:
        print(f"  {name:12s} union x{d[name] / d['exact']:.3f}   experts kept per row-layer {d['k_' + name] / d['rowl']:.2f} of 10")
