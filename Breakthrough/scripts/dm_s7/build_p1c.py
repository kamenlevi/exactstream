"""Step 0 of ~/Breakthrough/PLAN_12GB_90.md (Breakthrough, Thu 2026-11-05 19:5x): the 12 GB two-pass check's certificate in
T3R2's own form on the decision map, as rank masks (the harness's rmask, renorm False).

Certified rows keep their all-rows pass-1 mask row (NewIdeas' P1 masks, read-only: oea_k1 / oea_k1_ku4 / oea_k2); every other
row keeps all 10 (pass 2 = exact). A row is certified if its own p1 in the all-rows run src (out/<src>.json) is >= thr: the
cert gate's convention, the same rule as NewIdeas' p1_mixed_masks.py (which writes into NewIdeas' repo, so it is re-done here).
With --cf G, T3R2's cf gate too: a check runs pass 1 only if its first draft's cf >= G; otherwise every realized row of that
check is exact. Checks = the map's walks cut live at 0.3 (keepset/test_qom_keep.live_segments_nxt, NewIdeas' rule).

Out: ~/Breakthrough/data/p1c/<name>/<window>.npz, keep [rows, 48, 10] bool.
Usage (BigMoE venv): python -I build_p1c.py <mask arm> <src run> <thr> <name> [--cf G]
  e.g. oea_k1 p1_oea_k1 0.8 p1c_oea_k1_p08 (must equal NewIdeas' masks of that name bit for bit: --check)
       oea_k1_ku4 p1_oea_k1 0.8 p1c_oea_k1_ku4_p08_cf09 --cf 0.9 (t80ck1u's form)"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/Breakthrough/keepset")); sys.path.insert(1, os.path.expanduser("~/FableGod/decision_map"))
import dm_gate
from test_qom_keep import live_segments_nxt

ap = argparse.ArgumentParser()
ap.add_argument("arm"); ap.add_argument("src"); ap.add_argument("thr", type=float); ap.add_argument("name")
ap.add_argument("--cf", type=float, default=None, help="T3R2's cf gate: pass 1 only on checks whose first draft cf >= G")
ap.add_argument("--check", action="store_true", help="compare with ~/NewIdeas/data/dm_masks/<name> bit for bit")
A = ap.parse_args()
M = os.path.expanduser("~/NewIdeas/data/dm_masks"); OUT = os.path.expanduser("~/FableGod/decision_map/out")
OD = os.path.expanduser(f"~/Breakthrough/data/p1c/{A.name}")
S = json.load(open(os.path.join(OUT, A.src + ".json"))); base = json.load(open(os.path.join(OUT, "baseline.json")))

def gated_rows(nm, N):
    """rows (realized, in checks) that the cf gate sends straight to exact: their check's first draft has cf < G"""
    dom = base[nm]["dom"]; Z = np.load(os.path.join(dm_gate.TD, dm_gate.WALK[dom]))
    names = [str(x) for x in Z["names"]]; i = names.index(nm); g = np.zeros(N, bool)
    for (k_, m_, A_), dc_ in zip(Z[f"{i}_checks"].tolist(), Z[f"{i}_dcf"].tolist()):
        for (k, m, A0, dc, nxt) in live_segments_nxt(k_, m_, A_, dc_):
            fm = min(A0 + k, N - 1)
            if not (len(dc) > 0 and dc[0] >= A.cf): g[A0:fm + 1] = True
    return g

os.makedirs(OD, exist_ok=True); st = {}; bad = 0
for f in sorted(os.listdir(os.path.join(M, A.arm))):
    nm = f; keep = np.load(os.path.join(M, A.arm, f))["keep"]; N = len(keep)
    r = S[nm]; p = np.exp(np.array(r["lp"])[:, 0]); a = r["a"]
    assert a + len(p) == N, (nm, a, len(p), N)
    cert = np.r_[np.zeros(a, bool), p >= A.thr]
    gate = gated_rows(nm, N) if A.cf is not None else np.zeros(N, bool)
    cert &= ~gate
    out = np.ones_like(keep); out[cert] = keep[cert]
    np.savez_compressed(os.path.join(OD, f), keep=out)
    if A.check:
        ref = np.load(os.path.join(M, A.name, f))["keep"]; bad += not (ref.shape == out.shape and np.array_equal(ref, out))
    d = st.setdefault(base[nm]["dom"], [0, 0, 0, 0.0])
    d[0] += int(cert.sum()); d[1] += len(p); d[2] += int(gate[a:].sum()); d[3] += float(out[cert].sum(axis=2).mean() * cert.sum()) if cert.any() else 0.0
for dom, (c, n, gt, ek) in st.items():
    print(f"{A.name} {dom}: certified {c}/{n} measured rows ({100 * c / n:.1f} %); cf-gated to exact {gt}; "
          f"experts per certified row-layer {ek / max(c, 1):.2f}", flush=True)
print(f"wrote {len(os.listdir(OD))} windows to {OD}" + (f"; vs NewIdeas' {A.name}: {36 - bad}/36 windows bit-equal" if A.check else ""), flush=True)
sys.exit(1 if bad else 0)
