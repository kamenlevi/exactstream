"""Verify-pass expert compute on the 125B: how efficient is MLX's gather_qmm at verify sizes?
One MoE layer (512 experts, 4-bit, gate/up 640x2560, down 2560x640), S rows (1..4) each routed to top-10 with a realistic overlap
(union ~10 / 16 / 21 / 26 experts for S = 1..4, as on the proxy traces). Times the three gathers (sorted path, as SwitchGLU does)
over 16 layers' distinct weights and reports effective GB/s of the *unique* expert bytes. Ideal: every union expert read once at ~250 GB/s.
Usage: verify_gather.py   (GPU, quiet lock; ~1.4 GB x 4 of weights)"""
import time
import numpy as np
import mlx.core as mx

NE, K, D, F, G = 512, 10, 2560, 640, 64
NLAY = 4                                   # 4 x 1.4 GB of expert weights
rng = np.random.default_rng(0)


def q(o, i):
    return mx.quantize((mx.random.normal((NE, o, i)) * 0.02).astype(mx.bfloat16), group_size=G, bits=4)


layers = [(q(F, D), q(F, D), q(D, F)) for _ in range(NLAY)]
mx.eval(layers)
per_expert = sum(a.nbytes for t in layers[0] for a in t) / NE


def routing(S):
    """S rows, top-10 each; consecutive tokens share ~45 % of experts (proxy-trace statistic)."""
    base = rng.choice(NE, K, replace=False)
    rows = [base]
    for _ in range(S - 1):
        keep = rng.choice(base, int(0.45 * K), replace=False)
        new = rng.choice(np.setdiff1d(np.arange(NE), keep), K - len(keep), replace=False)
        rows.append(np.concatenate([keep, new]))
        base = rows[-1]
    return np.stack(rows)


def moe(x, idx, w):
    gate, up, down = w
    flat = idx.reshape(-1)
    order = mx.argsort(flat)
    xs = x[order // K][:, None, :]
    ids = flat[order]
    kw = dict(transpose=True, group_size=G, bits=4, sorted_indices=True)
    g = mx.gather_qmm(xs, *gate, rhs_indices=ids, **kw)
    u = mx.gather_qmm(xs, *up, rhs_indices=ids, **kw)
    return mx.gather_qmm((g * mx.sigmoid(g)) * u, *down, rhs_indices=ids, **kw)


for S in (1, 2, 3, 4):
    idxs = [mx.array(routing(S).astype(np.int32)) for _ in range(NLAY)]
    uniq = np.mean([len(np.unique(np.array(i))) for i in idxs])
    x = mx.random.normal((S, D)).astype(mx.bfloat16)

    def run():
        y = x
        for w, i in zip(layers, idxs):
            y = moe(y, i, w).reshape(S, K, D).sum(1).astype(mx.bfloat16)   # dependent chain like real layers
        mx.eval(y)
    for _ in range(3):
        run()
    ts = []
    for _ in range(15):
        a = time.perf_counter(); run(); ts.append(time.perf_counter() - a)
    t = sorted(ts)[7] / NLAY
    print(f"S={S}: union {uniq:4.1f} experts/layer, {t*1e3:6.3f} ms/layer -> {uniq*per_expert/t/1e9:5.0f} GB/s of unique "
          f"expert bytes; {S*K} (row, expert) pairs -> {S*K*per_expert/t/1e9:5.0f} GB/s if each pair re-reads", flush=True)
