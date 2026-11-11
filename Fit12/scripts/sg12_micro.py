# SG12 step 0 (PLAN.md 'SG12 short-prompt grouped prefill'): synthetic 4-bit experts in the tier's shapes; the engine's
# _prefill_single call (one scratch of every used expert) vs SG12's groups (<= G experts per call, E padded with unused
# zero rows so B_g < 4 E_g). Bit-equal on every single-regime case = PASS; the unpadded control must differ somewhere.
import sys, time, numpy as np, mlx.core as mx
from mlx_lm.models.switch_layers import SwitchGLU, QuantizedSwitchLinear

D, I, E, K = 2560, 640, 512, 10
FIELDS = ("weight", "scales", "biases")
mx.random.seed(0)
rng = np.random.default_rng(0)


def qparams(out, inn):
    w = mx.array(rng.integers(0, 2 ** 32, (E, out, inn // 8), dtype=np.uint32))
    s = mx.random.uniform(0.004, 0.012, (E, out, inn // 64)).astype(mx.bfloat16)
    return [w, s, (-7.5 * s).astype(mx.bfloat16)]


P = {"gate_proj": qparams(I, D), "up_proj": qparams(I, D), "down_proj": qparams(D, I)}
mx.eval(*[a for v in P.values() for a in v])
sw = SwitchGLU(D, I, 1)
for p, (i_, o_) in {"gate_proj": (D, I), "up_proj": (D, I), "down_proj": (I, D)}.items():
    setattr(sw, p, QuantizedSwitchLinear(i_, o_, 1, bias=False, group_size=64, bits=4))


def set_scratch(arrs):
    for p in P:
        for f, a in zip(FIELDS, arrs[p]):
            setattr(getattr(sw, p), f, a)


def take_experts(sel, extra=0):
    idx = mx.array(np.asarray(sel, np.int32))
    out = {}
    for p in P:
        out[p] = [mx.take(a, idx, axis=0) for a in P[p]]
        if extra > 0:
            out[p] = [mx.concatenate([a, mx.zeros((extra, *a.shape[1:]), a.dtype)]) for a in out[p]]
    mx.eval(*[a for v in out.values() for a in v])
    return out


def single(x, ids):
    uniq = np.unique(ids)
    compact = np.searchsorted(uniq, ids).astype(np.uint32)
    set_scratch(take_experts(uniq))
    t0 = time.perf_counter()
    y = SwitchGLU.__call__(sw, x, mx.array(compact))
    mx.eval(y)
    return y, uniq, time.perf_counter() - t0


def grouped(x, ids, uniq, G, pad=True):
    k = ids.shape[-1]
    flat = ids.reshape(-1)
    x2 = x.reshape(-1, D)
    Y = mx.zeros((flat.size, D), x.dtype)
    rhs, padded, tc = 0, 0, 0.0
    for i in range(0, len(uniq), G):
        grp = uniq[i:i + G]
        pos = np.nonzero(np.isin(flat, grp))[0]
        B, Eg = len(pos), len(grp)
        extra = 0
        if B >= 16 and B >= 4 * Eg:
            if pad:
                extra = B // 4 + 1 - Eg; padded += 1
            else:
                rhs += 1
        set_scratch(take_experts(grp, extra))
        t0 = time.perf_counter()
        xp = mx.take(x2, mx.array((pos // k).astype(np.int32)), axis=0)
        cp = mx.array(np.searchsorted(grp, flat[pos]).astype(np.uint32)).reshape(-1, 1)
        y = SwitchGLU.__call__(sw, xp, cp).reshape(B, -1)
        Y[mx.array(pos.astype(np.int32))] = y
        mx.eval(Y)
        tc += time.perf_counter() - t0
    return Y.reshape(*ids.shape, D), rhs, padded, tc


def routing(n, alpha, hot_low=False):
    pr = 1.0 / (np.arange(1, E + 1) ** alpha)
    perm = np.arange(E) if hot_low else rng.permutation(E)   # popular experts spread over ids (hot_low: ids 0..)
    p = np.empty(E); p[perm] = pr / pr.sum()
    return np.stack([rng.choice(E, K, replace=False, p=p) for _ in range(n)])[None].astype(np.uint32)


def ndiff(a, b):
    return int((a.view(mx.uint16) != b.view(mx.uint16)).sum().item())


cases = [(n, a, False) for n in (8, 20, 40, 69, 100, 150, 200, 300) for a in (0.6, 1.0)] + [(100, 1.2, True), (150, 1.2, True)]
bad = ctrl_seen = ctrl_diff = 0
for n, a, hot in cases:
    ids = routing(n, a, hot)
    x = mx.random.normal((1, n, D)).astype(mx.bfloat16)
    y0, uniq, ts = single(x, ids)
    B, U = ids.size, len(uniq)
    regime = "single" if not (B >= 16 and B >= 4 * U) else "rhs (engine groups it already)"
    line = f"n {n:3d} alpha {a} {'hot-low' if hot else 'spread '} B {B:4d} U {U:3d} ({regime}) single {ts * 1e3:6.1f} ms"
    if regime == "single":
        for G in (64, 32):
            yg, _, padded, tg = grouped(x, ids, uniq, G)
            d = ndiff(y0, yg); bad += d > 0
            line += f" | G{G}: diff {d} padded {padded} {tg * 1e3:6.1f} ms"
        yc, rhs, _, _ = grouped(x, ids, uniq, 64, pad=False)
        if rhs:
            dc = ndiff(y0, yc); ctrl_seen += 1; ctrl_diff += dc > 0
            line += f" | control G64 unpadded: {rhs} rhs groups, diff {dc}"
    print(line, flush=True)
    del y0, x
    mx.clear_cache()
print(f"SG12 step 0: single-regime cases with a bit change: {bad}; controls run {ctrl_seen}, differing {ctrl_diff}; "
      f"MLX peak {mx.get_peak_memory() / 1e9:.2f} GB")
print("VERDICT", "KILL" if bad else ("PASS" if ctrl_diff else "NO VERDICT (control never differed)"))
