#!/usr/bin/env python3 -I
"""lattice_rescue.py (FableGod, mission 29): at the chain's first miss (teacher-forced DFlash2 lattice, Breakthrough's
data/trees/lattice_{code,chat,rest}.npz: cand [blocks, 15, 16] the drafter's candidates per depth, pick [blocks, 15] the
chain's draft, true [blocks, 15] the 125B's token), how often the true token is the drafter's 2nd / 3rd candidate."""
import numpy as np, zipfile, io, os
def arr(z, name):
    return np.load(io.BytesIO(z.read(name + ".npy")), allow_pickle=False)
for f in ("lattice_code.npz", "lattice_chat.npz", "lattice_rest.npz"):
    p = os.path.expanduser(f"~/Breakthrough/data/trees/{f}")
    z = zipfile.ZipFile(p); cand = arr(z, "cand"); pick = arr(z, "pick"); true = arr(z, "true"); ntrue = arr(z, "ntrue")
    try: dom = arr(z, "dom")
    except Exception: dom = np.array(["?"] * cand.shape[0])
    try: sel = arr(z, "sel_cand")
    except Exception: sel = None
    doms = np.unique(dom.astype(str))
    for d_ in doms:
        m = dom.astype(str) == d_; C, P, T, NT = cand[m], pick[m], true[m], ntrue[m]
        n_miss = r2 = r3 = r16 = 0; s2 = s3 = 0; pos = []
        for b in range(C.shape[0]):
            for d in range(min(15, int(NT[b]))):
                if P[b, d] != T[b, d]:
                    n_miss += 1; row = C[b, d]; pos.append(d)
                    # the candidates ordered by the drafter: index of the true token among them (excluding the picked one)
                    others = [t for t in row.tolist() if t != P[b, d]]
                    if T[b, d] in others[:1]: r2 += 1
                    if T[b, d] in others[:2]: r3 += 1
                    if T[b, d] in row.tolist(): r16 += 1
                    if sel is not None:
                        so = [t for t in sel[m][b, d].tolist() if t != P[b, d]]
                        s2 += T[b, d] in so[:1]; s3 += T[b, d] in so[:2]
                    break
        if n_miss:
            print(f"{f} {d_}: blocks {C.shape[0]}, first misses {n_miss} (mean depth {np.mean(pos):.1f}): true token = the drafter's next candidate {r2 / n_miss:.3f}, within the next two {r3 / n_miss:.3f}, within its top-16 {r16 / n_miss:.3f}" + (f"; by the selector's order: {s2 / n_miss:.3f} / {s3 / n_miss:.3f}" if sel is not None else ""))
