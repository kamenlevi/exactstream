#!/usr/bin/env python3
"""TREE-UNION step 1 (NewIdeas, Tue 17 Nov; the time is the commit's): is the 125B's expert routing driven by the token or by
the context? From the 84 route files (~/datasets/deepdraft/routes: routes [T, 48, top-10], generated tokens from 'gen' on),
per layer: the mean shared fraction |E_t & E_s| / 10 for lags 1, 2, 4, 8, 32 (same sequence) and for random pairs across
sequences (the baseline). Siblings in a tree share the whole context and differ in their own token, so lag-1 overlap (own
token differs AND the context shifts by one token) is a conservative stand-in for sibling overlap.
usage: taskpolicy -b ~/BigMoE/.venv/bin/python -I -B tu_overlap.py"""
import glob, json, os
import numpy as np

rng = np.random.default_rng(17)
files = sorted(glob.glob(os.path.expanduser("~/datasets/deepdraft/routes/*.npz")))
R = []
for f in files:
    z = np.load(f, allow_pickle=False)
    R.append(z["routes"][int(z["gen"]):].astype(np.int32))
L = R[0].shape[1]
E = int(max(int(r.max()) for r in R)) + 1
print(f"{len(R)} sequences, {sum(len(r) for r in R)} generated tokens, {L} layers, top-{R[0].shape[2]}, experts per layer >= {E}")

def overlap(a, b):                      # a, b: [n, L, 10] -> [n, L] shared fraction
    n = a.shape[0]
    out = np.zeros((n, L))
    for l in range(L):
        A = np.sort(a[:, l, :], 1); B = np.sort(b[:, l, :], 1)
        # count shared ids per row
        out[:, l] = [len(np.intersect1d(x, y, assume_unique=True)) / 10.0 for x, y in zip(A, B)]
    return out

res = {}
for lag in (1, 2, 4, 8, 32):
    a = np.concatenate([r[:-lag] for r in R if len(r) > lag]); b = np.concatenate([r[lag:] for r in R if len(r) > lag])
    idx = rng.choice(len(a), min(6000, len(a)), replace=False)
    res[f"lag{lag}"] = overlap(a[idx], b[idx]).mean(0)
allr = np.concatenate(R)
ia = rng.integers(0, len(allr), 6000); ib = rng.integers(0, len(allr), 6000)
res["random"] = overlap(allr[ia], allr[ib]).mean(0)
bands = [(0, 4), (4, 12), (12, 24), (24, 36), (36, 48)]
print("mean shared fraction of the top-10 experts, by layer band")
print("layers   " + "  ".join(f"{k:>7s}" for k in res))
for lo, hi in bands:
    print(f"{lo:2d}-{hi - 1:2d}    " + "  ".join(f"{res[k][lo:hi].mean():7.3f}" for k in res))
print("all      " + "  ".join(f"{res[k].mean():7.3f}" for k in res))
# marginal expert frequency per layer (for the step-2 simulation)
freq = np.zeros((L, E))
for l in range(L):
    np.add.at(freq[l], allr[:, l, :].ravel(), 1)
json.dump({"res": {k: v.tolist() for k, v in res.items()}, "E": E, "L": L}, open("tu_overlap.json", "w"))
np.save("tu_freq.npy", freq)
