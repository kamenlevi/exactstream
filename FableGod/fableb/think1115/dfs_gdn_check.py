"""dfs_gdn_check.py (Fable B, THINK_1115 idea 2): the GDN recurrence over a token TREE walked depth-first with a state
stack vs each root-to-leaf path run as its own chain. Same fp32 ops in the same order per path, so bitwise. CPU, tiny."""
import numpy as np
rng = np.random.default_rng(1115)
def step(S, q, k, v, g, b):
    S = S * np.float32(g); d = (v - S @ k) * np.float32(b); S = S + np.outer(d, k).astype(np.float32); return S, S @ q
bad = 0; maxstack = 0
for trial in range(200):
    K = int(rng.integers(8, 129)); parent = [-1] + [int(rng.integers(0, i)) for i in range(1, K)]   # random tree
    q, k = [rng.normal(0, 0.09, 128).astype(np.float32) for _ in range(K)], [rng.normal(0, 0.09, 128).astype(np.float32) for _ in range(K)]
    v = [rng.normal(0, 1, 128).astype(np.float32) for _ in range(K)]; g = rng.uniform(0.5, 1, K).astype(np.float32); b = rng.uniform(0, 1, K).astype(np.float32)
    S0 = rng.normal(0, 0.3, (128, 128)).astype(np.float32)
    children = [[] for _ in range(K)]
    for i in range(1, K): children[parent[i]].append(i)
    # per-path chains (the sandbox's path-batched form)
    ref = {}
    for leaf in [i for i in range(K) if not children[i]]:
        path = []; n = leaf
        while n >= 0: path.append(n); n = parent[n]
        S = S0
        for n in reversed(path):
            S, o = step(S, q[n], k[n], v[n], g[n], b[n]); ref.setdefault(n, o)
    # depth-first walk with a stack (what one simdgroup would do)
    out = {}; stack = [(0, S0)]
    while stack:
        n, Sp = stack.pop(); maxstack = max(maxstack, len(stack) + 1)
        S, o = step(Sp, q[n], k[n], v[n], g[n], b[n]); out[n] = o
        for c in reversed(children[n]): stack.append((c, S))
    bad += sum(not np.array_equal(out[n].view(np.uint32), ref[n].view(np.uint32)) for n in range(K))
print("200 random trees of 8-128 nodes: outputs differing from the per-path chains: %d; deepest stack %d -> %s" % (bad, maxstack, "PASS" if bad == 0 else "FAIL"))
