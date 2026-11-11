# SGB step 0 (PLAN.md 'SGB balanced prefill groups'): synthetic 4-bit experts in the tier's shapes; the engine's
# _prefill_single call (one scratch of every used expert) vs BALANCED groups of <= G experts (non-consecutive ids, heavy and
# light experts mixed so every group keeps the whole call's MLX kernel: sorted-rhs when B >= 4E, the per-pair kernel
# otherwise). Variants: rhs groups padded to a multiple of 64 rows (the grouped path's BUG1 rule) and unpadded. A BUG1
# probe compares each rhs single call with the same pairs run on the per-pair kernel (E padded so B < 4E, SG12's step-0
# rule): wrong rows would show as large errors, rounding as small ones.
import numpy as np, mlx.core as mx
from mlx_lm.models.switch_layers import SwitchGLU, QuantizedSwitchLinear

D, I, E, K = 2560, 640, 512, 10
FIELDS = ("weight", "scales", "biases")
mx.random.seed(0)
rng = np.random.default_rng(1)


def qparams(out, inn):
    w = mx.array(rng.integers(0, 2 ** 32, (E, out, inn // 8), dtype=np.uint32))
    s = mx.random.uniform(0.004, 0.012, (E, out, inn // 64)).astype(mx.bfloat16)
    return [w, s, (-7.5 * s).astype(mx.bfloat16)]


P = {"gate_proj": qparams(I, D), "up_proj": qparams(I, D), "down_proj": qparams(D, I)}
mx.eval(*[a for v in P.values() for a in v])
sw = SwitchGLU(D, I, 1)
for p, (i_, o_) in {"gate_proj": (D, I), "up_proj": (D, I), "down_proj": (I, D)}.items():
    setattr(sw, p, QuantizedSwitchLinear(i_, o_, 1, bias=False, group_size=64, bits=4))


def set_scratch(sel, extra=0):
    idx = mx.array(np.asarray(sel, np.int32))
    for p in P:
        arrs = [mx.take(a, idx, axis=0) for a in P[p]]
        if extra > 0:
            arrs = [mx.concatenate([a, mx.zeros((extra, *a.shape[1:]), a.dtype)]) for a in arrs]
        for f, a in zip(FIELDS, arrs):
            setattr(getattr(sw, p), f, a)
    mx.eval(*[getattr(getattr(sw, p), f) for p in P for f in FIELDS])


def single(x, ids):
    uniq = np.unique(ids)
    set_scratch(uniq)
    y = SwitchGLU.__call__(sw, x, mx.array(np.searchsorted(uniq, ids).astype(np.uint32)))
    mx.eval(y)
    return y, uniq


def run_groups(x, ids, groups, pad64, padE):
    """each group on its own pairs (ascending pair positions), scratch = its experts in ascending id order"""
    k = ids.shape[-1]
    flat = ids.reshape(-1)
    x2 = x.reshape(-1, D)
    Y = mx.zeros((flat.size, D), x.dtype)
    for grp in groups:
        grp = np.sort(grp)
        pos = np.nonzero(np.isin(flat, grp))[0]
        xp = mx.take(x2, mx.array((pos // k).astype(np.int32)), axis=0)
        cpn = np.searchsorted(grp, flat[pos]).astype(np.uint32)
        pad = (-len(pos)) % 64 if pad64 else 0
        if pad:
            xp = mx.concatenate([xp, mx.zeros((pad, D), xp.dtype)])
            cpn = np.concatenate([cpn, np.zeros(pad, np.uint32)])
        extra = max(0, (len(cpn) // 4 + 1) - len(grp)) if padE else 0
        set_scratch(grp, extra)
        y = SwitchGLU.__call__(sw, xp, mx.array(cpn).reshape(-1, 1)).reshape(len(cpn), -1)[:len(pos)]
        Y[mx.array(pos.astype(np.int32))] = y
        mx.eval(Y)
    return Y.reshape(*ids.shape, D)


def balanced(uniq, cnt, G, rhs):
    """the engine copy's rule (v3r tier._sgb_groups, amended 21:4x): greedy on v = rows - 4 (rhs) / 4 - rows (per-pair),
    largest |v| first, positives to the smallest-sum group, negatives to the largest; m = ceil(U / G) .. + 2"""
    cnt = cnt.astype(np.int64)
    v = (cnt - 4) if rhs else (4 - cnt)
    m0 = -(-len(uniq) // G)
    order = np.argsort(-np.abs(v), kind="stable")
    for m in (m0, m0 + 1, m0 + 2):
        sums, gs, ok = np.zeros(m, np.int64), [[] for _ in range(m)], True
        for e in order:
            free = [g for g in range(m) if len(gs[g]) < G]
            if not free:
                ok = False
                break
            g = min(free, key=lambda j: sums[j]) if v[e] > 0 else max(free, key=lambda j: sums[j])
            gs[g].append(e)
            sums[g] += v[e]
        if ok and all(len(gs[g]) and ((sums[g] >= 0 and cnt[gs[g]].sum() >= 64) if rhs else sums[g] > 0) for g in range(m)):
            return [uniq[np.sort(np.array(gs[g], dtype=np.int64))] for g in range(m)]
    return None


def routing(n, alpha):
    pr = 1.0 / (np.arange(1, E + 1) ** alpha)
    perm = rng.permutation(E)
    p = np.empty(E); p[perm] = pr / pr.sum()
    return np.stack([rng.choice(E, K, replace=False, p=p) for _ in range(n)])[None].astype(np.uint32)


def ndiff(a, b):
    return int((a.view(mx.uint16) != b.view(mx.uint16)).sum().item())


bad = rhs_seen = probes = 0
ctrl = [0, 0]
for n in (79, 100, 146, 150, 200):
    for a in (0.5, 0.7, 0.9, 1.1):
        ids = routing(n, a)
        x = mx.random.normal((1, n, D)).astype(mx.bfloat16)
        y0, uniq = single(x, ids)
        flat = ids.reshape(-1)
        cnt = np.bincount(flat, minlength=E)[uniq]
        B, U = flat.size, len(uniq)
        rhs = B >= 16 and B >= 4 * U
        line = f"n {n:3d} a {a} B {B:4d} U {U:3d} B/U {B / U:.2f} ({'rhs' if rhs else 'per-pair'})"
        for G in (64, 48):
            gs = balanced(uniq, cnt, G, rhs)
            if gs is None:
                line += f" | G{G}: no balanced split"
                continue
            mx_ = max(len(g) for g in gs)
            if rhs:
                d1 = ndiff(y0, run_groups(x, ids, gs, True, False))
                d2 = ndiff(y0, run_groups(x, ids, gs, False, False))
                line += f" | G{G} ({len(gs)} groups, max {mx_}): pad64 diff {d1}, unpadded diff {d2}"
                bad += d1 > 0
            else:
                d = ndiff(y0, run_groups(x, ids, gs, False, False))
                line += f" | G{G} ({len(gs)} groups, max {mx_}): diff {d}"
                bad += d > 0
        if not rhs:                                          # control: the heaviest experts in one group (rhs inside)
            order = np.argsort(-cnt, kind="stable")
            h, acc = [], 0
            for e in order:
                h.append(e); acc += int(cnt[e])
                if len(h) >= 4 and acc >= 4 * len(h) + 64 and acc >= 64:
                    break
            if acc >= 4 * len(h) and acc >= 64:
                rest = np.setdiff1d(np.arange(U), h)
                dc = ndiff(y0, run_groups(x, ids, [uniq[np.sort(h)], uniq[rest]], False, False))
                line += f" | control (heavy group of {len(h)}, {acc} rows: rhs): diff {dc}"
                ctrl[0] += 1; ctrl[1] += dc > 0
        if rhs:
            rhs_seen += 1
            yv = run_groups(x, ids, [uniq], False, True)       # the same pairs on the per-pair kernel (E padded)
            err = float(mx.max(mx.abs(yv.astype(mx.float32) - y0.astype(mx.float32))).item())
            ref = float(mx.max(mx.abs(y0.astype(mx.float32))).item())
            line += f" | BUG1 probe: per-pair vs rhs max |d| {err:.3g} of max |y| {ref:.3g}"
            probes += 1
        print(line, flush=True)
        mx.clear_cache()
print(f"SGB step 0: cases with a bit change (pad64 rule for rhs): {bad}; rhs cases {rhs_seen}; MLX peak {mx.get_peak_memory() / 1e9:.2f} GB")
print(f"controls run {ctrl[0]}, differing {ctrl[1]}")
print("VERDICT", "KILL" if bad else ("PASS" if ctrl[1] else "NO VERDICT (control never differed)"))
