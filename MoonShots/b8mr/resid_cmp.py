"""Compare two MR_RESID runs (run.py gen): R[snapshot, gate layer, expert] = slot or -1; snapshots = [prefill, after warm-up,
after pass 0, 1, ...]. Prints the first snapshot that differs, where (layers, resident-set vs slot-only differences), and the
first pass whose reads / tokens differ (from the gen JSONs).
usage: resid_cmp.py A.npz B.npz [A.json B.json]"""
import json, sys
import numpy as np
A, B = np.load(sys.argv[1])["R"], np.load(sys.argv[2])["R"]
name = lambda i: "prefill" if i == 0 else "warm-up" if i == 1 else f"pass {i - 2}"
n = min(len(A), len(B))
print(f"snapshots {len(A)} vs {len(B)}, gate layers {A.shape[1]}")
first = next((i for i in range(n) if not np.array_equal(A[i], B[i])), None)
if first is None:
    print("residency identical in every common snapshot")
else:
    a, b = A[first], B[first]
    lay = np.nonzero((a != b).any(-1))[0]
    print(f"first differing snapshot: {first} ({name(first)}); gate layers differing: {lay.tolist()[:20]}{' ...' if len(lay) > 20 else ''}")
    for g in lay[:6]:
        ra, rb = set(np.nonzero(a[g] >= 0)[0]), set(np.nonzero(b[g] >= 0)[0])
        print(f"  layer {g}: resident {len(ra)} vs {len(rb)}; only A {sorted(ra - rb)[:8]}, only B {sorted(rb - ra)[:8]}; "
              f"same set, different slots: {int(((a[g] >= 0) & (b[g] >= 0) & (a[g] != b[g])).sum())}")
    if first > 0:
        print(f"  previous snapshot ({name(first - 1)}) identical: {np.array_equal(A[first - 1], B[first - 1])}")
if len(sys.argv) > 4:
    ja, jb = json.load(open(sys.argv[3])), json.load(open(sys.argv[4]))
    ta, tb = ja["tokens"], jb["tokens"]
    k = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    pa = [r["x_misses"] for r in ja["passes"]]; pb = [r["x_misses"] for r in jb["passes"]]
    dm = next((j for j, (u, v) in enumerate(zip(pa, pb)) if u != v), None)
    da = [r.get("d") for r in ja["passes"]]; db = [r.get("d") for r in jb["passes"]]
    dd = next((j for j, (u, v) in enumerate(zip(da, db)) if u != v), None)
    print(f"tokens first differ at {k}; reads first differ at pass {dm}; drafts first differ at pass {dd}; shas {ja['sha']} {jb['sha']}")
