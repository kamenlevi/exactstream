"""Robustness of the LIP/SRRIP insertion result on a second trace set: MoEPrefetch's 125B traces (*_800.npz, 4 prompts x
800 tokens), verify windows of 4 tokens with stride 3 (as planes3_sim.py), per layer C slots, 4-bit MB per pass, LRU vs
LIP-4/8 vs SRRIP-3 vs Belady OPT. Uses insert_policy125.py's and opt_bound125.py's policy functions (imported lazily).
usage: insert_policy800.py   (CPU, 1 core; run with taskpolicy -b)"""
import glob, os, sys, types
import numpy as np
sys.argv = [sys.argv[0]]
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "insert_policy125.py")).read().split("\nfor C in (64, 96, 128, 192):")[0]
src = src.replace('z = np.load("/Users/kamen/MoonShots/data/verify_trace125.npz")', "z = None").replace(
    "ids = z[\"ids\"].astype(np.int32)\nP, L = ids.shape[:2]\nseqs = [[set(np.unique(ids[p, l]).tolist()) for p in range(P)] for l in range(L)]", "")
I = types.ModuleType("I"); exec(src, I.__dict__)
osrc = open(os.path.join(HERE, "opt_bound125.py")).read()
opt_src = osrc[osrc.index("def opt(seq, C):"):osrc.index("MB = 2.765")]
exec(opt_src, I.__dict__)
files = sorted(glob.glob(os.path.expanduser("~/MoEPrefetch/data/traces125/*_800.npz")))
per_layer = None
P = 0
for f in files:
    z = np.load(f)
    tr = [z[f"L{L}"][:, :10].astype(np.int64) for L in range(len(z.files))]
    if per_layer is None:
        per_layer = [[] for _ in tr]
    T = tr[0].shape[0]
    starts = list(range(0, T - 4, 3))
    P += len(starts)
    for L in range(len(tr)):
        per_layer[L] += [np.unique(tr[L][s:s + 4]) for s in starts]   # prompts concatenated (cold start counted once each)
MB = 2.765
print(f"{len(files)} prompts, {P} windows of 4, {len(per_layer)} layers", flush=True)
sets = [[set(x.tolist()) for x in s] for s in per_layer]
for C in (64, 96, 128, 192):
    lru = sum(I.lip(s, C, 0) for s in sets) / P * MB
    l4 = sum(I.lip(s, C, 4) for s in sets) / P * MB
    l8 = sum(I.lip(s, C, 8) for s in sets) / P * MB
    sr = sum(I.srrip(s, C, 3) for s in sets) / P * MB
    op = sum(I.opt(s, C) for s in per_layer) / P * MB
    print(f"C={C:3d} ({48 * C * MB / 1e3:4.1f} GB) MB/pass: LRU {lru:6.1f} | LIP-4 {l4:6.1f} ({l4 / lru - 1:+.1%}) | LIP-8 {l8:6.1f} "
          f"({l8 / lru - 1:+.1%}) | SRRIP-3 {sr:6.1f} ({sr / lru - 1:+.1%}) | OPT {op:6.1f} ({op / lru - 1:+.1%})", flush=True)
