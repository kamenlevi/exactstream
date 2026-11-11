"""Union growth of routed experts over blocks of k consecutive committed tokens (125B, data/verify_trace125.*): each
pass's rows 0..n-1 are the inputs that became committed tokens (n = tokens committed by that pass), so concatenating them
per prompt gives the true routing of the committed sequence. For k = 1..32: mean distinct experts per layer in a block of
k consecutive tokens, and cold 4-bit MB per token if every block had to stream its union (no cache): the amortization a
k-token exact verify would give with perfect drafts. usage: union_growth125.py   (CPU, seconds)"""
import collections, json
import numpy as np
D = "/Users/kamen/MoonShots/data"
z = np.load(f"{D}/verify_trace125.npz")
cb = json.load(open(f"{D}/verify_trace125.json"))["callbacks"]
ids, seq, pr = z["ids"].astype(np.int32), z["seq"], z["prompt"]
tot = collections.Counter()
for p, s, n in cb:
    tot[(p, s)] += n
streams = collections.defaultdict(list)                  # prompt -> [token][layer][10]
for i in range(len(seq)):
    n = tot.get((int(pr[i]), int(seq[i])), 0)
    for r in range(n):
        streams[int(pr[i])].append(ids[i, :, r])
L = ids.shape[1]
print("committed tokens per prompt:", {p: len(v) for p, v in streams.items()})
MB = 2.765
for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
    us = []
    for v in streams.values():
        a = np.stack(v)                                  # [T, L, 10]
        for t0 in range(0, len(a) - k + 1, k):
            blk = a[t0:t0 + k]
            us.append(np.mean([len(np.unique(blk[:, l])) for l in range(L)]))
    u = float(np.mean(us))
    print(f"k={k:2d}: distinct experts/layer {u:6.1f} (x{u / 10:4.2f} of one token) | cold MB/token {u * L * MB / k:7.1f}", flush=True)
