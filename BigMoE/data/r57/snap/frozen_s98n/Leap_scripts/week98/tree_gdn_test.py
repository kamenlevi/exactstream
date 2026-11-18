"""tree_gdn_test.py (Leap, Sun 15 Nov 2026 12:2x; TREE16 L1's unit test): tree_gdn.gdn_body_tree vs gdnbody.gdn_body run
on each root-to-leaf path as a chain (same conv state, same recurrent state), bitwise on y, q, k, v for every node.
Trees (16 nodes, DFS preorder, children best-first as in the spec): a pure chain, two bushy trees, a deep narrow one,
and 6 random ones (random parents, relabelled in DFS preorder). Random inputs at the 125B's GDN shapes, 2 seeds each.
usage: [BigMoE venv] tree_gdn_test.py   (MEM < 1 GB; share lock)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98n/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98n/Leap_scripts/week98"))
import gdnbody as GB
import tree_gdn as TG
from one32_gdn_test import inputs

T = 16


def say(s):
    print(s, flush=True)


def dfs_relabel(parents):
    """any parent array (root 0) -> the same tree's parents in DFS preorder, children in their original order"""
    n = len(parents)
    kids = [[] for _ in range(n)]
    for c in range(1, n):
        kids[parents[c]].append(c)
    order, stack = [], [0]
    while stack:
        u = stack.pop()
        order.append(u)
        stack.extend(reversed(kids[u]))
    new = {u: i for i, u in enumerate(order)}
    return [-1] + [new[parents[u]] for u in order[1:]]


def trees(rng):
    out = {"chain": [-1] + list(range(T - 1))}
    out["bushy3"] = dfs_relabel([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    out["bushy-deep"] = dfs_relabel([-1, 0, 1, 2, 3, 4, 5, 6, 7, 0, 9, 1, 11, 2, 13, 14])
    out["deep-narrow"] = dfs_relabel([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 13])
    for i in range(6):
        par = [-1] + [int(rng.integers(max(0, c - 4), c)) for c in range(1, T)]
        out[f"random{i}"] = dfs_relabel(par)
    return out


def paths_of(parents):
    kids = [[] for _ in range(T)]
    for c in range(1, T):
        kids[parents[c]].append(c)
    leaves = [u for u in range(T) if not kids[u]]
    out = []
    for leaf in leaves:
        p, u = [], leaf
        while u >= 0:
            p.append(u); u = parents[u]
        out.append(p[::-1])
    return out


def win_of(parents):
    anc = []
    for n in range(T):
        p, u = [], n
        while u >= 0:
            p.append(u); u = parents[u]
        anc.append(p[::-1])                                   # root .. n
    w = []
    for n in range(T):
        d = len(anc[n]) - 1
        w.append([anc[n][d - 3 + t] if d - 3 + t >= 0 else d - 3 + t for t in range(3)])   # -1 / -2 / -3: conv slots
    return w


t0 = time.time()
say(f"TREE-GDN {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}")
rng = np.random.default_rng(3)
ok_all = True
for name, par in trees(rng).items():
    assert all(par[c] < c for c in range(1, T)), (name, par)
    restore, save, ns = TG.fork_tables(par)
    win = win_of(par)
    res = []
    for seed in (1, 2):
        qkv, z, a, b, cst, cw, alog, dtb, st, nw, eps, sc = inputs(T, seed)
        got = TG.gdn_body_tree(qkv, z, a, b, cst, cw, alog, dtb, st, nw, eps, sc, win, restore, save, ns)
        bad = 0
        for P in paths_of(par):
            idx = mx.array(P, mx.int32)
            ref = GB.gdn_body(qkv[:, idx], z[:, idx], a[:, idx], b[:, idx], cst, cw, alog, dtb, st, nw, eps, sc)
            for k in range(4):
                bad += int((~mx.all(got[k][:, idx] == ref[k], axis=tuple(range(2, ref[k].ndim)))).sum().item())
        res.append("ok" if bad == 0 else f"{bad} DIFF")
        ok_all &= bad == 0
    say(f"  {name:12s} slots {ns} paths {len(paths_of(par)):2d}: " + ", ".join(res))
say(f"TREE-GDN verdict: {'ALL BITWISE (y, q, k, v of every node == its path run as a chain)' if ok_all else 'NOT BITWISE'}"
    f" ({time.time() - t0:.0f} s)")
