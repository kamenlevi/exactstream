"""Expert union per 16-row check under Q1 / Q1-UNION at several theta (NewIdeas, Tue 3 Nov 06:5x).
Q1-UNION's union equals Q1's at the same theta by construction (dm_q1u_masks.py), so this prices both.
Blocks exactly as dm_q1u_masks.py: the HPD-0 walks' checks (rows A .. A + m), rows after the first miss use the
wrong-draft stand-in (row p + 97) with their own drafts' cf (dcf). Unsure rows keep 10 ranks, sure rows ranks 0-1.
Reads ~/FableGod/decision_map (read-only); writes nothing. Run: taskpolicy -b ~/BigMoE/.venv/bin/python -I <this>"""
import json, os, sys
import numpy as np
DM = os.path.expanduser("~/FableGod/decision_map")
sys.path.insert(0, DM)
import dm_gate  # noqa: E402  (load() only reads gate_dfl2.json)
OUT = os.path.join(DM, "out")
base = json.load(open(os.path.join(OUT, "baseline.json")))
G = dm_gate.load()
THR = [float(t) for t in sys.argv[1:]] or [0.999, 0.99, 0.95, 0.9, 0.8]
res = {}
for dom, wf in dm_gate.WALK.items():
    Z = np.load(os.path.join(dm_gate.TD, wf))
    for i, nm in enumerate(str(x) for x in Z["names"]):
        if nm not in base:
            continue
        ids = np.load(os.path.join(OUT, "routing", nm))["ids"]
        N = len(ids)
        g = np.nan_to_num(G[nm], nan=-1.0)
        for (k, m, A), dc in zip(Z[f"{i}_checks"].tolist(), Z[f"{i}_dcf"].tolist()):
            last, fm = min(A + m, N - 1), min(A + k, N - 1)
            rows, cfs = [], []
            for p in range(A, last + 1):
                if p <= fm:
                    rows.append(p); cfs.append(g[p])
                else:
                    rows.append((p + 97) % N); cfs.append(dc[p - A] if p - A < m else -1.0)
            R = ids[rows]                      # [b, 48, 10]
            cfs = np.array(cfs)
            ex = sum(len(np.unique(R[:, l, :])) for l in range(48))
            d = res.setdefault(dom, {"exact": 0, "checks": 0, "rows": 0, "kept": 0})
            d["exact"] += ex; d["checks"] += 1; d["rows"] += len(rows); d["kept"] += k + 1
            for t in THR:
                s = cfs >= t
                u = 0
                for l in range(48):
                    parts = [R[~s, l, :].ravel(), R[s, l, :2].ravel()]
                    u += len(np.unique(np.concatenate(parts)))
                d[t] = d.get(t, 0) + u
                d[("g", t)] = d.get(("g", t), 0) + int(s[: fm - A + 1].sum())
for dom, d in res.items():
    line = [f"{dom}: {d['checks']} checks, {d['rows']} rows, {d['kept']} kept; exact union {d['exact'] / d['checks'] / 48:.1f} experts per layer-check"]
    for t in THR:
        line.append(f"theta {t}: union x{d[t] / d['exact']:.3f} ({100 * (d[t] / d['exact'] - 1):+.1f} %), gated walk rows {d[('g', t)]}")
    print("\n  ".join(line), flush=True)
