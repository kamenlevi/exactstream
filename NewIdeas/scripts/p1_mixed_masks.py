"""P1-OEA mixed-run masks (NewIdeas, Tue 3 Nov 2026): the 12 GB two-pass check end to end with P1-K1 as pass 1, in the
form of Breakthrough's cert gate (dm_common._gate_k ("cert", src, thr, k1): the cut on rows whose own p1 under the cut,
read from the all-rows run src, is >= thr; the rest exact = pass 2), with the cut = my all-rows mask instead of plain
top-k1. Certified rows keep their all-rows mask row (top-1 + the block's union members for oea_k1; prompt rows and rows
outside the walk are exact there already); every other row keeps all 10.
In:  ~/NewIdeas/data/dm_masks/<arm>/<window>.npz (dm_oea_masks.py) and ~/FableGod/decision_map/out/<src>.json (read-only).
Out: ~/NewIdeas/data/dm_masks/<name>/<window>.npz, keep [rows, 48, 10] bool (the harness's rmask, renorm False).
Usage (BigMoE venv): python p1_mixed_masks.py <arm> <src> <thr> <name>   e.g. oea_k1 p1_oea_k1 0.8 p1c_oea_k1_p08"""
import json, os, sys
import numpy as np
arm, src, thr, name = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
M = os.path.expanduser("~/NewIdeas/data/dm_masks"); OUT = os.path.expanduser("~/FableGod/decision_map/out")
S = json.load(open(os.path.join(OUT, src + ".json"))); base = json.load(open(os.path.join(OUT, "baseline.json")))
od = os.path.join(M, name); os.makedirs(od, exist_ok=True)
st = {}
for f in sorted(os.listdir(os.path.join(M, arm))):
    nm = f                                                    # the map keys windows by the routing file name (.npz)
    keep = np.load(os.path.join(M, arm, f))["keep"]; N = len(keep)
    r = S[nm]; p = np.exp(np.array(r["lp"])[:, 0]); a = r["a"]
    assert a + len(p) == N, (nm, a, len(p), N)              # measured row i = position a + i (dm_common._cert); the
    cert = np.r_[np.zeros(a, bool), p >= thr]                 # harness's last position n - 1 = N is past the mask: exact
    out = np.ones_like(keep); out[cert] = keep[cert]
    assert out[~cert].all() and (out[cert] == keep[cert]).all()
    np.savez_compressed(os.path.join(od, f), keep=out)
    d = st.setdefault(base[nm]["dom"], [0, 0, 0, 0])
    d[0] += int(cert.sum()); d[1] += len(p); d[2] += int((~out[cert]).any(axis=(1, 2)).sum()); d[3] += int(out.sum())
for dom, (c, n, cut, k) in st.items():
    print(f"{name} {dom}: certified {c}/{n} measured rows ({100 * c / n:.1f} %), of them {cut} with a cut", flush=True)
print(f"wrote {len(os.listdir(od))} windows to {od}")
