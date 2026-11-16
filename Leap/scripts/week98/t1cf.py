"""t1cf.py (Leap, Mon 16 Nov 2026 12:5x; CHECK25 / the dispatch diet's MoE group, BigMoE's OK in principle 12:0x): the
1c-tiers glue of one MoE layer of a check as ONE dispatch. BigMoE's _t1c (bigmoe.tiers1c.tiers_mx + the per-expert counter
+ the 2-D nolsb) and its km path in _fast_call are ~28 MLX ops per layer (~1,350 per 32-row check: ~9 ms of CPU build +
encode, ~1.5-3 ms of GPU). Here, one threadgroup, one thread per (row, rank) pair:
  v      = (w * w) * rw[row]          w = the fused combine's score (fp32), rw = s^row (tiers_mx's power(s, arange))
  I_e    = sum of v over the pairs of expert e, in row-major pair order (tiers_mx: one scatter-add, its order unfixed)
  tier   = 0 if I_e > T4 (2.775 lam), 1 if > T3 (0.8457 lam), else 2   (fp32 thresholds, as MLX compares)
  km     = tier != 2;  scores' = km ? score : 0;  ids' = km ? id : the row's rank-0 id   (the km path)
  nolsb  = tier[:, :2] != 0             (Leap's 2-D form, BigMoE ADOPTED: MSB-3 or drop)
  counts = distinct experts per tier    (an expert's pairs share I_e, so one tier per expert)
Equal to BigMoE's path wherever the scatter-add's order does not move an I_e across a threshold (fp32 last-bit ties;
check() counts any slot that differs). Opt-in; BigMoE's frozen path is untouched.
  fused(ids [M, K], scores [M, K], s, lam) -> (scores' [M, K], ids' [M, K], nolsb [M, 2] bool, counts [3] int32, tier)
  check()  vs tiers_mx + BigMoE's glue on random routing (M 16 / 32), the slots that differ
usage: [BigMoE venv] t1cf.py check"""
from functools import lru_cache
import os
import sys

import numpy as np
import mlx.core as mx

T4_K, T3_K = 2.775, 0.8457
_RW = {}

_SRC = r"""
    const uint i = thread_position_in_grid.x;
    constexpr int P = P_C, K = K_C;
    threadgroup int id_s[P_C];
    threadgroup float v_s[P_C];
    threadgroup int t_s[P_C];
    const float T4 = as_type<float>((uint)T4B), T3 = as_type<float>((uint)T3B);
    if ((int)i < P) {
        id_s[i] = (int)ids[i];
        const float wf = static_cast<float>(scores[i]);
        v_s[i] = (wf * wf) * rw[i / K];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)i < P) {
        const int e = id_s[i];
        float I = 0.0f;
        for (int j = 0; j < P; ++j) if (id_s[j] == e) I += v_s[j];
        const int t = I > T4 ? 0 : (I > T3 ? 1 : 2);
        t_s[i] = t;
        tier[i] = (uint8_t)t;
        const int r0 = ((int)i / K) * K;
        sc_out[i] = t != 2 ? scores[i] : static_cast<ST>(0.0f);
        ids_out[i] = t != 2 ? ids[i] : ids[r0];
        const int rk = (int)i % K;
        if (rk < 2) nolsb[((int)i / K) * 2 + rk] = t != 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)i < P) {
        const int e = id_s[i];
        bool first = true;
        for (int j = 0; j < (int)i; ++j) if (id_s[j] == e) { first = false; break; }
        if (first) atomic_fetch_add_explicit((device atomic_int*)&counts[t_s[i]], 1, memory_order_relaxed);
    }
"""


@lru_cache(None)
def _k():
    return mx.fast.metal_kernel(name="leap_t1cf", input_names=["ids", "scores", "rw"],
                                output_names=["sc_out", "ids_out", "nolsb", "counts", "tier"], source=_SRC)


def _f32bits(x):
    return int(np.array([x], np.float32).view(np.uint32)[0])


def rw_of(M, s):
    """tiers_mx's row weights, by the same MLX ops (cached per (M, s))"""
    k = (M, float(s))
    if k not in _RW:
        _RW[k] = mx.power(mx.array(s, mx.float32), mx.arange(M, dtype=mx.float32))
        mx.eval(_RW[k])
    return _RW[k]


def fused(ids, scores, s=0.98, lam=0.016):
    M, K = ids.shape[-2], ids.shape[-1]
    P = M * K
    assert P <= 1024, P
    ids2 = ids.reshape(M, K)
    sc2 = scores.reshape(M, K)
    T4, T3 = _f32bits(T4_K * lam), _f32bits(T3_K * lam)       # MLX compares fp32 Ii with the Python float as fp32
    tg = max(32, (P + 31) // 32 * 32)
    sc, idn, nl, cnt, tier = _k()(inputs=[ids2, sc2, rw_of(M, s)],
                                  template=[("P_C", P), ("K_C", K), ("T4B", T4), ("T3B", T3), ("ST", sc2.dtype)],
                                  grid=(tg, 1, 1), threadgroup=(tg, 1, 1),
                                  output_shapes=[(M, K), (M, K), (M, 2), (3,), (M, K)],
                                  output_dtypes=[sc2.dtype, ids2.dtype, mx.bool_, mx.int32, mx.uint8], init_value=0)
    return sc.reshape(scores.shape), idn.reshape(ids.shape), nl, cnt, tier


def reference(ids, scores, s=0.98, lam=0.016, n_experts=512):
    """BigMoE's path, op for op: tiers1c.tiers_mx, _t1c's counter and nolsb, _fast_call's km lines"""
    sys.path.insert(0, os.path.expanduser("~/BigMoE"))
    from bigmoe import tiers1c as T1C
    K = ids.shape[-1]
    ids2 = ids.reshape(-1, K)
    tier = T1C.tiers_mx(ids2, scores.reshape(-1, K), s=s, lam=lam)
    texp = mx.full((T1C.N_EXPERTS,), 3, mx.int32).at[ids2.reshape(-1)].minimum(tier.reshape(-1).astype(mx.int32))
    cnt = mx.stack([(texp == 0).sum(), (texp == 1).sum(), (texp == 2).sum()])
    nolsb = tier[:, :2] != 0
    km = (tier != 2).reshape(ids.shape)
    sc = mx.where(km, scores, mx.zeros_like(scores))
    idn = mx.where(km, ids, ids[..., :1])
    return sc, idn, nolsb, cnt, tier


def check(trials=400, seed=1116):
    rng = np.random.default_rng(seed)
    lines, bad_slots, slots = [], 0, 0
    for M in (16, 32):
        for dt_ids in (mx.uint32, mx.int32):
            nd = 0
            for t in range(trials):
                hot = rng.choice(512, 40, replace=False)               # a realistic union: a hot set + a tail
                ids = np.stack([np.concatenate([rng.choice(hot, 4, replace=False),
                                                rng.choice(np.setdiff1d(np.arange(512), hot), 6, replace=False)])
                                for _ in range(M)])
                lg = rng.standard_normal((M, 10)).astype(np.float32) * rng.uniform(0.5, 3.0)
                w = np.exp(lg - lg.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
                w = -np.sort(-w, axis=1)
                I = mx.array(ids.astype(np.int64)).astype(dt_ids)
                S = mx.array(w).astype(mx.bfloat16)
                a = fused(I, S)
                b = reference(I, S)
                mx.eval(*a, *b)
                eq = [bool(mx.array_equal(x, y)) for x, y in zip(a, b)]
                d = int(mx.sum(a[4] != b[4]).item())
                nd += d
                slots += M * 10
                bad_slots += d
                if not all(eq) and d == 0:
                    lines.append(f"  M {M} {dt_ids} trial {t}: outputs differ with equal tiers: {eq}")
            lines.append(f"  M {M} ids {dt_ids}: {trials} checks, {nd} slots with a different tier")
    hdr = f"T1CF check: {slots - bad_slots} of {slots} slots equal to BigMoE's tiers_mx + glue (all five outputs per check)"
    print(hdr, flush=True)
    for l in lines:
        print(l, flush=True)
    return bad_slots, [hdr] + lines


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        b, lines = check()
        p = os.path.expanduser("~/Leap/data/week98/check25/t1cf_check.txt")
        open(p, "w").write("\n".join(lines) + "\n")
        sys.exit(1 if b else 0)
