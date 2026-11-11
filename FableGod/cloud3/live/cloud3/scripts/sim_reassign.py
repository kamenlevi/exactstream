"""Lynx / SERE / OEA-style bounds on Breakthrough/data/expA/traces125 (800 single-stream positions per domain, top-20 ids +
gate weights per layer). Pseudo-blocks = 16 consecutive positions (a perfect drafter's check). Per block and layer:
 - union of top-10; singleton pairs (expert used by one row);
 - SWAP bound (Lynx / SERE): a singleton (row r, expert e at weight w) is replaced by a union member that sits in row r's
   ranks 11-20 at weight w' when w - w' <= delta (abs) -- the union loses e; mass displaced = w - w' per swap;
 - DROP bound (OEA-like, no substitution): drop singletons whose weight w <= wmax (kept weights unrenormalised); mass lost = w.
Reports the union ratio and the mean per-row gate mass displaced / lost, per domain."""
import numpy as np
P = "/root/350-tps/Breakthrough/data/expA/traces125/"
for dom in ["code", "chat", "math", "bulgarian"]:
    z = np.load(P + f"{dom}_800_g.npz")
    ids = np.stack([z[f"L{l}"] for l in range(48)], 1).astype(np.int32)   # (800, 48, 20)
    g = np.stack([z[f"G{l}"] for l in range(48)], 1).astype(np.float32)   # (800, 48, 20)
    # sanity: are ids rank-sorted by gate?  check monotonic
    mono = np.mean(np.all(np.diff(g[:, :, :10], axis=-1) <= 1e-6, axis=-1))
    N = ids.shape[0]; nb = N // 16
    res = {}
    deltas = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]
    wmaxs = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
    U0 = 0; single = 0; pairs = 0
    swap_u = {d: 0 for d in deltas}; swap_m = {d: 0.0 for d in deltas}; swap_n = {d: 0 for d in deltas}
    drop_u = {x: 0 for x in wmaxs}; drop_m = {x: 0.0 for x in wmaxs}
    mass10 = 0.0
    for b in range(nb):
        rows = slice(b * 16, b * 16 + 16)
        for l in range(48):
            I = ids[rows, l, :]; G = g[rows, l, :]
            top10 = I[:, :10]
            cnt = {}
            for r in range(16):
                for e in top10[r]: cnt[int(e)] = cnt.get(int(e), 0) + 1
            union = set(cnt); U0 += len(union); pairs += 160
            mass10 += G[:, :10].sum()
            singles = [(r, int(e), float(G[r, k]), k) for r in range(16) for k, e in enumerate(top10[r]) if cnt[int(e)] == 1]
            single += len(singles)
            for d in deltas:
                removed = set(); m = 0.0; n = 0
                for (r, e, w, k) in singles:
                    # best union member among row r's ranks 10-19
                    best = None
                    for kk in range(10, 20):
                        e2 = int(I[r, kk]); w2 = float(G[r, kk])
                        if e2 in union and e2 not in removed and w - w2 <= d:
                            best = w2; break
                    if best is not None: removed.add(e); m += w - best; n += 1
                swap_u[d] += len(union) - len(removed); swap_m[d] += m; swap_n[d] += n
            for x in wmaxs:
                rem = [(r, e, w) for (r, e, w, k) in singles if w <= x]
                drop_u[x] += len(union) - len(rem); drop_m[x] += sum(w for _, _, w in rem)
    nbl = nb * 48
    print(f"\n{dom}: blocks {nb}, ids rank-sorted frac {mono:.2f}; union per layer {U0/nbl:.1f}; singletons {single/pairs:.2f} of pairs = {single/U0:.2f} of the union; mean top-10 mass per row {mass10/(nb*48*16):.3f}")
    print("  SWAP (Lynx/SERE, replace a singleton by a union member within delta of its weight): delta -> union ratio, swaps per layer, mass displaced per row")
    for d in deltas: print(f"    delta {d:<6} x{swap_u[d]/U0:.3f}  swaps {swap_n[d]/nbl:5.1f}  mass {swap_m[d]/(nb*48*16):.4f}")
    print("  DROP (no substitution, drop singletons with weight <= wmax): wmax -> union ratio, mass lost per row")
    for x in wmaxs: print(f"    wmax {x:<6} x{drop_u[x]/U0:.3f}  mass {drop_m[x]/(nb*48*16):.4f}")
