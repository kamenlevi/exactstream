"""tree_att_test.py (Leap, Sun 15 Nov 2026 15:0x; TREE16 L2's unit test): tree_att.tree_sdpa vs ref2's exact verify SDPA
run on each node's path as a 16-row chain check (the path's rows, then copies of the last row; K / V = the cache, the
path's nodes in depth order, the padding copies), bitwise for every node, every query head.
Trees from tree_gdn_test (chain, bushy, deep-narrow, random; T = 16 and T = 7), contexts L across ref2's classes
(class 0, the 128-511 band and its edges, the 512-key buckets), 2 seeds. Random q / k / v at the model's shapes.
usage: [BigMoE venv] tree_att_test.py   (MEM < 1 GB; share lock)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98a/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98a/Leap_scripts/week98"))
import mtplx.models.qwen4_exp as Q
import ref2
import tree_att as TA

ref2._ORIG["vsdpa"] = Q._verify_sdpa
ref2._ON[0] = True
HQ, HK, D = 24, 2, 256
SCALE = D ** -0.5
BF = mx.bfloat16


def say(s):
    print(s, flush=True)


def dfs_relabel(parents):
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
    out = {"chain": [-1] + list(range(15)), "bushy3": dfs_relabel([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
           "deep-narrow": dfs_relabel([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 13]),
           "small7": dfs_relabel([-1, 0, 0, 1, 2, 3, 3])}
    for i in range(3):
        out[f"random{i}"] = dfs_relabel([-1] + [int(rng.integers(max(0, c - 4), c)) for c in range(1, 16)])
    return out


def anc_depth(par):
    T = len(par)
    anc = []
    for n in range(T):
        p, u = [], n
        while u >= 0:
            p.append(u); u = par[u]
        anc.append(p[::-1])
    return anc, [len(a) - 1 for a in anc]


def causal(rows, N, L):
    pos = L + np.arange(rows)[:, None]
    return mx.array((np.arange(N)[None, :] <= pos)[None, None])


t0 = time.time()
say(f"TREE-ATT {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; ref2 sdpa_bounds {ref2.CFG['sdpa_bounds']}, "
    f"kb_from {ref2.CFG['kb_from']}")
rng = np.random.default_rng(5)
ok_all = True
TA.BATCH_HEADS[0] = os.environ.get("TREE_ATT_BATCH") == "1"
say(f"  BATCH_HEADS {TA.BATCH_HEADS[0]}")
for name, par in trees(rng).items():
    T = len(par)
    anc, depth = anc_depth(par)
    line = []
    for L in (60, 100, 112, 200, 383, 497, 520, 1000, 1500, 2030):
        bad = 0
        for seed in (1, 2):
            ks = mx.random.split(mx.random.key(100000 * T + 100 * L + seed), 5)
            q = (mx.random.normal((1, HQ, T, D), key=ks[0]) * 1.6).astype(BF)
            kc = (mx.random.normal((1, HK, L, D), key=ks[1]) * 1.6).astype(BF)
            vc = mx.random.normal((1, HK, L, D), key=ks[2]).astype(BF)
            kn = (mx.random.normal((1, HK, T, D), key=ks[3]) * 1.6).astype(BF)
            vn = mx.random.normal((1, HK, T, D), key=ks[4]).astype(BF)
            got = TA.tree_sdpa(q, mx.concatenate([kc, kn], 2), mx.concatenate([vc, vn], 2), SCALE, L, anc, depth, T, ref2)
            for n in range(T):
                P = anc[n]
                pad = [P[-1]] * (16 - len(P))
                rows = P + pad
                qch = q[:, :, mx.array(rows)]
                kch = mx.concatenate([kc, kn[:, :, mx.array(rows)]], 2)
                vch = mx.concatenate([vc, vn[:, :, mx.array(rows)]], 2)
                r = ref2._verify_sdpa(qch, kch, vch, scale=SCALE, mask=causal(16, L + 16, L))
                eq = mx.all(got[:, :, n] == r[:, :, depth[n]])
                if not bool(eq.item()):
                    bad += 1
        line.append(f"L{L} {'ok' if bad == 0 else f'{bad} BAD'}")
        ok_all &= bad == 0
    say(f"  {name:12s} T {T:2d}: " + ", ".join(line))
say(f"TREE-ATT verdict: {'ALL BITWISE (every node, every head == its path chain row)' if ok_all else 'NOT BITWISE'}"
    f" ({time.time() - t0:.0f} s)")
