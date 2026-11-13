"""Predicted engine 'uses' for QOM-KEEP + DRAFT17 with the split (the orchestrator Thu 18:xx, for BigMoE's Friday A/B; Breakthrough).
uses = distinct experts computed per layer call (resident included), summed over the 48 layers of a check: per (block, layer)
the union of every block row's KEPT experts (realized rows and stand-ins alike, as the engine computes them), from the same
replay that reproduces the map masks bit for bit (test_qom_keep.py: live checks, stand-ins, DRAFT17's last row, the LRU pool
warmed by the prompt and carried across a domain's windows in walk order). Code windows only (the 12 HumanEval windows).
Arms: qall (split off), s99 (split 0.99), s90 (split 0.9); pools: the 45 GB per-layer C_m (the masks) and a uniform C 160 (BigMoE's
pre-flight). Also the share of rows with draft cf >= 0.99 / >= 0.9 in the first 32 generated tokens of each answer vs all.
Usage (BigMoE venv, CPU): python -I uses_pred.py"""
import json, os, sys
from collections import OrderedDict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
DM = os.path.expanduser("~/FableGod/decision_map"); sys.path.insert(1, DM)
import qom_keep as K
import dm_gate
from test_qom_keep import live_segments_nxt, touch, OUT

def swap_mb():
    import subprocess
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(s.split("used = ")[1].split("M")[0])

def run(split, CL, dom="code"):
    base = json.load(open(os.path.join(OUT, "baseline.json"))); G = dm_gate.load()
    Z = np.load(os.path.join(dm_gate.TD, dm_gate.WALK[dom])); cache = [OrderedDict() for _ in range(48)]
    uses, first32, allr = [], [], []
    for i, nm in enumerate(str(x) for x in Z["names"]):
        if nm not in base: continue
        ids = np.load(os.path.join(OUT, "routing", nm))["ids"]; N = len(ids); a = base[nm]["a"]
        g = np.nan_to_num(G[nm], nan=-1.0)
        for p in range(a, N): (first32 if p < a + 32 else allr).append(g[p])
        for p in range(a):
            for l in range(48):
                for e in ids[p, l].tolist(): touch(cache[l], e, CL[l])
        for (k_, m_, A_), dc_ in zip(Z[f"{i}_checks"].tolist(), Z[f"{i}_dcf"].tolist()):
            for (k, m, A, dc, nxt) in live_segments_nxt(k_, m_, A_, dc_):
                last, fm = min(A + m, N - 1), min(A + k, N - 1)
                rows, cfs = [], []
                for p in range(A, last + 1):
                    if p <= fm: rows.append(p); c_ = -1.0 if (k == m and p == A + m) else g[p]
                    else: rows.append((p + 97) % N); c_ = dc[p - A] if p - A < m else -1.0
                    if p - A == m: c_ = nxt if nxt >= 0 else 1.0     # DRAFT17 (ball): the last row verifies the 17th draft
                    cfs.append(c_)
                if not rows: continue
                k0 = K.k0_from_cf(cfs); st = K.strict_rows(cfs, k0, split); u = 0
                for l in range(48):
                    idsb = ids[rows, l, :]; U = K.loads(idsb, k0); c = cache[l]
                    for e in U.tolist(): touch(c, e, CL[l])
                    res = np.fromiter(c.keys(), np.int64, len(c))
                    kp = K.keep(idsb, k0, U, res, st)
                    u += len(np.unique(idsb[kp]))
                    for e in K.touch_order(idsb, k0, res, st, U): c.move_to_end(e)
                uses.append(u)
    return np.array(uses), np.array(first32), np.array(allr)

if __name__ == "__main__":
    s0 = swap_mb()
    CM = json.load(open(os.path.expanduser("~/MoonShots/data/laysz45_256.json")))["C_m"]
    for pool, CL in (("45 GB (C_m, the masks)", CM), ("C 160 (BigMoE's pre-flight)", [160] * 48)):
        q = None
        for arm, split in (("qall", None), ("s99", 0.99), ("s90", 0.9)):
            u, f32, rest = run(split, CL)
            if q is None: q = u.mean()
            print(f"[uses] {pool} {arm}: {len(u)} checks; uses per check mean {u.mean():.0f} (x{u.mean() / q:.3f} of qall), "
                  f"median {np.median(u):.0f}, p10 {np.percentile(u, 10):.0f}, p90 {np.percentile(u, 90):.0f}, sd {u.std():.0f}", flush=True)
            if swap_mb() - s0 > 1024: print("[uses] swap grew > 1 GB: stopped", flush=True); sys.exit(1)
    d32, dr = f32[f32 >= 0], rest[rest >= 0]
    print(f"[uses] code rows with a draft: first 32 generated tokens {len(d32)} rows: cf >= 0.99 {np.mean(d32 >= 0.99):.3f}, "
          f">= 0.9 {np.mean(d32 >= 0.9):.3f}; the rest of the answers {len(dr)} rows: >= 0.99 {np.mean(dr >= 0.99):.3f}, "
          f">= 0.9 {np.mean(dr >= 0.9):.3f}; rows without a draft: first 32 {np.mean(f32 < 0):.3f}, rest {np.mean(rest < 0):.3f}", flush=True)
    print(f"[uses] swap {swap_mb() - s0:+.0f} MB", flush=True)
