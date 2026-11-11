"""Round 32h: can context matching supply the long-horizon foresight Belady needs? (exact: a cache policy only)
Per layer, before evicting at window t: find the earlier window t' (t' < t - 1) whose expert set overlaps the current
one most (ties -> most recent), predict F = union of the expert sets of windows t'+1 .. t'+W (what followed the similar
context last time), and evict first the cached experts NOT in F (LRU among them), then LRU among the rest.
Variants: match on the layer's own set ("own"), or on the summed overlap over all layers ("all", one match per window).
Trace sets: data/verify_trace125.npz (401 committed verify passes) and MoEPrefetch's traces125 *_800.npz (windows of 4,
stride 3). MB/pass of 4-bit experts vs LRU and OPT (insert_policy800.py / opt_bound125.py numbers).
usage: pmatch_policy.py   (CPU, 1 core; taskpolicy -b)"""
import glob, os
import numpy as np

E, MB = 512, 2.765


def load_sets():
    out = {}
    z = np.load("/Users/kamen/MoonShots/data/verify_trace125.npz")
    ids = z["ids"].astype(np.int64)
    out["verify401"] = [[np.unique(ids[p, l]) for p in range(ids.shape[0])] for l in range(ids.shape[1])]
    per = None
    for f in sorted(glob.glob(os.path.expanduser("~/MoEPrefetch/data/traces125/*_800.npz"))):
        zz = np.load(f)
        tr = [zz[f"L{L}"][:, :10].astype(np.int64) for L in range(len(zz.files))]
        per = per or [[] for _ in tr]
        for L in range(len(tr)):
            per[L] += [np.unique(tr[L][s:s + 4]) for s in range(0, tr[0].shape[0] - 4, 3)]
    out["traces800"] = per
    return out


def run(seqs, C, W, mode):
    L, T = len(seqs), len(seqs[0])
    Mx = np.zeros((L, T, E), np.float32)
    for l in range(L):
        for t, s in enumerate(seqs[l]):
            Mx[l, t, s] = 1
    best_all = {}
    if mode == "all" and W > 0:                          # one match per window over all layers (same for every layer)
        G = Mx.transpose(1, 0, 2).reshape(T, L * E)
        for t in range(3, T):
            ov = G[:t - 1] @ G[t]
            best_all[t] = int(len(ov) - 1 - np.argmax(ov[::-1]))
    miss = 0
    for l in range(L):
        cache = {}                                       # e -> last use
        for t in range(T):
            need = seqs[l][t]
            for e in need.tolist():
                miss += e not in cache
                cache[e] = t
            if len(cache) <= C:
                continue
            F = set()
            if t >= 3 and W > 0:
                if mode == "own":
                    ov = Mx[l, :t - 1] @ Mx[l, t]
                    tp = int(len(ov) - 1 - np.argmax(ov[::-1]))  # most overlapping, most recent on ties
                else:
                    tp = best_all[t]
                for j in range(tp + 1, min(t, tp + 1 + W)):          # only windows already seen (no lookahead)
                    F.update(seqs[l][j].tolist())
            nd = set(need.tolist())
            cand = sorted((e for e in cache if e not in nd), key=lambda e: (e in F, cache[e]))
            for e in cand[:len(cache) - C]:
                del cache[e]
    return miss / T * MB


for name, seqs in load_sets().items():
    print(f"{name}: {len(seqs[0])} windows x {len(seqs)} layers", flush=True)
    for C in (64, 96, 128):
        row = [f"LRU {run(seqs, C, 0, 'own'):6.1f}"]
        for mode in ("own", "all"):
            for W in (8, 16, 32):
                row.append(f"{mode}-W{W} {run(seqs, C, W, mode):6.1f}")
        print(f"  C={C:3d} ({48 * C * MB / 1e3:4.1f} GB) MB/pass: " + " | ".join(row), flush=True)
