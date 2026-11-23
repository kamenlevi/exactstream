"""Validation input for sticky/samelayer_t.py: cap45 as a fake pr_dump with X = x_{L-1} (L-1's MoE input) and Y = x_L
(L's true MoE input) = LEAD1CORR's lead-1 setup (plain 0.612 -> ridge 0.729 @10, all layers). Layers L = 2, 5, ..., 47."""
import os, glob, sys
import numpy as np
D = os.path.expanduser("~/BigMoE/data/cap45"); OUT = sys.argv[1]; NL = 48
os.makedirs(OUT, exist_ok=True)
def bf16(u):
    return (u.astype(np.uint32) << 16).view(np.float32)
for n_, f in enumerate(sorted(glob.glob(os.path.join(D, "*_gate_0.npz")))):
    z = np.load(f); by = {}
    for j, (s, L) in enumerate(zip(z["seq"].tolist(), z["L"].tolist())):
        by.setdefault(s, {}).setdefault(L, j)
    seqs = [s for s in sorted(by) if all(L in by[s] for L in range(NL)) and len({int(z["rows"][by[s][L]]) for L in range(NL)}) == 1]
    xr = z["x"]; Ls, Xs, Ys = [], [], []
    for s in seqs:
        for L in range(2, NL, 3):
            a, b = by[s][L - 1], by[s][L]; r = int(z["rows"][b])
            Xs.append(bf16(xr[int(z["x_off"][a]):int(z["x_off"][a]) + r])); Ys.append(bf16(xr[int(z["x_off"][b]):int(z["x_off"][b]) + r]))
            Ls.append(np.full(r, L, np.int16))
    nm = os.path.basename(f).replace("_gate_0.npz", "")
    key = ("humaneval/" if nm.startswith("humaneval") else nm.split("_")[0] + "/") + nm
    np.savez(os.path.join(OUT, f"fake_{n_:02d}.npz"), key=key, L=np.concatenate(Ls), X=np.concatenate(Xs), Y=np.concatenate(Ys))
    print(key, len(seqs), "checks", sum(len(x) for x in Xs), "row-layers", flush=True)
