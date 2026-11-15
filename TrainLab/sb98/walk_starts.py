"""WEEK98 S5 WALKW, step 1 (MTP Head): where the runtime walk starts its checks. For every training window of the base
split, run the reference head s30's teacher-forced chain (Sandbox98's core.chain_table, read-only) and walk it as the
engine does (core.kept_walk: accept depths until the first miss, the next check starts k + 1 anchors later). Saves the
absolute anchor of every check start -> data/sb98/walk_starts_s30.npz (key = window file basename).
Diagnostic (EvalSuite 9480ba4 found HE+ depth 0 = 91.6 % at the walk's check starts vs 97.9 % elsewhere on the 125B):
s30's depth-0 accuracy at check starts vs over all anchors, on the MBPP-50 read-out windows.
Usage: cd ~/Sandbox98 && ~/BigMoE/.venv/bin/python sb98/... (run from ~/Sandbox98 so `sb` imports)"""
import os, sys, glob, json
sys.path.insert(0, os.path.expanduser("~/Sandbox98"))
import numpy as np
import mlx.core as mx
from sb import core as C

C.limit_memory(4.5)                      # Sandbox98's caps: 1 GB buffer cache, 4.5 GB memory (06:06 incident: without them the
                                         # cache grew to ~50 GB over the window loop)

OUT = os.path.expanduser("~/TrainLab/data/sb98")
D = C.load_drafter("s30")
split = json.load(open(os.path.join(C.ROOT, "data", "split.json")))
base = split["base"]


def walk_starts(rt, gen, a0):
    r0 = max(0, gen - 1 - a0)
    sub = rt[r0:]; p = 0; st = []
    while p < sub.shape[0]:
        st.append(p)
        k = 0
        while k < sub.shape[1] and sub[p, k] == 1:
            k += 1
        p += k + 1
    return np.array(st, np.int64) + r0          # rows of rt (anchor = a0 + row)


def path_of(key):
    for d in ("mc", "mbpp", "he"):
        p = os.path.join(C.ROOT, "data", d, key + ".npz")
        if os.path.exists(p):
            return p
    return None


T = {}
n_st = n_all = 0
for i, key in enumerate(base):
    f = path_of(key)
    if f is None:
        continue
    rt, gen, a0 = C.chain_table(D, f, 31)
    mx.clear_cache()
    rows = walk_starts(rt, gen, a0)
    T[os.path.basename(f)] = (a0 + rows).astype(np.int64)
    n_st += len(rows); n_all += int((rt[:, 0] >= 0).sum())
np.savez_compressed(os.path.join(OUT, "walk_starts_s30.npz"), **T)
print(f"base windows {len(T)}; check starts {n_st} of {n_all} anchors ({100 * n_st / max(n_all, 1):.1f} %)")
# diagnostic on the MBPP-50 read-out windows (evalx's first 50, sorted)
mb = sorted(f for f in glob.glob(os.path.join(C.ROOT, "data", "mbpp", "*.npz")) if not f.endswith(".taps.npz"))[:50]
a_st = [0, 0]; a_all = [0, 0]
for f in mb:
    rt, gen, a0 = C.chain_table(D, f, 31)
    mx.clear_cache()
    rows = walk_starts(rt, gen, a0)
    c0 = rt[:, 0]
    v = c0 >= 0
    a_all[0] += int((c0[v] == 1).sum()); a_all[1] += int(v.sum())
    s = c0[rows]; s = s[s >= 0]
    a_st[0] += int((s == 1).sum()); a_st[1] += len(s)
print(f"s30 MBPP-50 depth 0: at check starts {100 * a_st[0] / a_st[1]:.2f} % (n {a_st[1]}) vs all anchors "
      f"{100 * a_all[0] / a_all[1]:.2f} % (n {a_all[1]})")
