"""My lru_late vs pol35's per-token loop (copied verbatim) on tie-free random f32 logits: late/dropped masks, loads."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/TrainLab"))
from collections import OrderedDict
import numpy as np
import train35 as T
def pol35_loop(lg, q, k):
    T_, cache, loads = lg.shape[0], OrderedDict(), 0
    sel = np.zeros((T_, k), np.int64); cat = np.zeros((T_, k), np.int8)
    for t in range(T_):
        b = lg[t] + 0.0
        top = np.argpartition(-b, k)[:k]
        top = top[np.argsort(-lg[t][top])]
        p = np.exp(lg[t][top] - lg[t][top].max()); p /= p.sum(); cp = np.cumsum(p)
        rw = max(q["w"], int(np.searchsorted(cp, q["m"] - 1e-9)) + 1 if q["m"] else 0)
        for r, e in enumerate(top):
            e = int(e)
            if r >= q["k"]:
                cat[t, r] = 2; continue
            if e in cache:
                cache.move_to_end(e); continue
            if r < rw: c = 0
            elif q["l"] != "drop" and cp[r] - p[r] < q["x"] - 1e-9: c = 1
            else: c = 2
            cat[t, r] = c
            if c < 2:
                loads += 1
                if q["C"] > 0:
                    if len(cache) >= q["C"]: cache.popitem(last=False)
                    cache[e] = True
        sel[t] = top
    return sel, cat, loads
rng = np.random.default_rng(0)
lg = rng.normal(size=(400, 64)).astype(np.float32) * 2
k = 8
for C, W, l, x in [(13, 2, "half", 1.0), (13, 2, "drop", 1.0), (13, 2, "half", 0.5), (26, 1, "half", 0.7), (6, 2, "drop", 1.0)]:
    q = {"C": C, "w": W, "m": 0.0, "l": l, "x": x, "k": 99}
    sel, cat, loads = pol35_loop(lg, q, k)
    ps = np.exp(np.take_along_axis(lg, sel, 1) - np.take_along_axis(lg, sel, 1).max(1, keepdims=True)); ps /= ps.sum(1, keepdims=True)
    cpb = np.cumsum(ps, 1) - ps
    late, res, myloads, drop = T.lru_late(sel, C, W, 64, True, l != "drop", cpb, None if x >= 1 else x)
    ok = np.array_equal(late, cat == 1) and np.array_equal(drop, cat == 2) and loads == myloads
    print(f"C{C} w{W} l{l} x{x}: late {late.mean():.3f} drop {drop.mean():.3f} loads {myloads} vs pol35 {loads} -> {'MATCH' if ok else 'DIFF'}")
