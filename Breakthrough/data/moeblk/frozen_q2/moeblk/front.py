"""front.py (Breakthrough, Tue 17 Nov 2026 13:2x; MOE BLOCK candidate F2): the block's FRONT as ONE single-threadgroup
kernel, after the router. Today these are ~6 dispatches on the critical path between router_fused and kernel A:
  1. Leap's t1cf (the 1c tiers, the km path, 2-D nolsb): its arithmetic replicated expression for expression (the same
     fp32 products, the same sequential importance sum in pair order, the same fp32 thresholds);
  2. the slot lookups (the posts' result with every expert resident): pre_m = slot[ids'], pre_l = nolsb ? -2 : side[ids'];
  3. the tier-4 upgrade's grouping (xgrp v2: identity tables, rank by (key, upgraded, index), 16-pair groups), reading
     each pair's slots from its own thread's registers;
  4. rowsum64(x): Leap's fixed sequential sum per (row, 64-value group).
Every output equals the separate kernels' bitwise; front_check() verifies that on real routing.
Template: hidden size, experts, top-k, the plan's 4-bit ranks and the thresholds are parameters (moeblk config).
    front(lay, ids, scores, x) -> dict(sc, idn, nl, pm, pl, grp=(pidx, seg, nseg, P), xs)"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import moeblk as MB                                         # noqa: E402

mx = MB.mx
_K = {}

_SRC = r"""
    const uint i = thread_position_in_grid.x;
    const uint nthr = threads_per_threadgroup.x;
    constexpr int P = P_C, K = K_C, R4 = R4_C;
    const int NOTGT = (int)0x80000000;
    // ---- 1. t1cf (Leap), expression for expression
    threadgroup int tid_s[P_C];
    threadgroup float v_s[P_C];
    const float T4 = as_type<float>((uint)T4B), T3 = as_type<float>((uint)T3B);
    if ((int)i < P) {
        tid_s[i] = (int)ids[i];
        const float wf = static_cast<float>(scores[i]);
        v_s[i] = (wf * wf) * rw[i / K];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int t = 2, pmv = 0, plv = 0;
    uint idv = 0;
    if ((int)i < P) {
        const int e = tid_s[i];
        float I = 0.0f;
        for (int j = 0; j < P; ++j) if (tid_s[j] == e) I += v_s[j];
        t = I > T4 ? 0 : (I > T3 ? 1 : 2);
        const int r0 = ((int)i / K) * K;
        sc_out[i] = t != 2 ? scores[i] : static_cast<ST>(0.0f);
        idv = t != 2 ? ids[i] : ids[r0];
        ids_out[i] = idv;
        const int rk = (int)i % K;
        // ---- 2. the slots
        pmv = sm[idv];
        pm[i] = pmv;
        if (rk < R4) {
            nolsb[((int)i / K) * R4 + rk] = t != 0;
            plv = t != 0 ? -2 : sl[idv];
            pl[((int)i / K) * R4 + rk] = plv;
        }
    }
    // ---- 3. the upgrade's grouping (xgrp v2) on this thread's own (pmv, plv, idv)
    threadgroup int key_s[P_C];
    threadgroup int skey[P_C];
    threadgroup int start_s[P_C];
    threadgroup int id_s[P_C];
    threadgroup int tgt_s[P_C];
    threadgroup int up_s[P_C];
    threadgroup atomic_int rtab[RT_C];
    threadgroup atomic_int ltab[LT_C];
    for (int q = (int)i; q < RT_C; q += (int)nthr) atomic_store_explicit(&rtab[q], P_C, memory_order_relaxed);
    for (int q = (int)i; q < LT_C; q += (int)nthr) atomic_store_explicit(&ltab[q], P_C, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int key = 0, cand = 0, up = 0;
    if ((int)i < P) {
        const int r = (int)i % NR_C;
        const int tier0 = ((MASK4_C >> r) & 1) ? 4 : (((MASK2_C >> r) & 1) ? 2 : 3);
        int el = 0, sd = 0;
        bool late = pmv < 0;
        if (tier0 == 4) {
            el = plv;
            if (el == -1) late = true; else if (el < 0) { el = 0; sd = 1; } else el += 1;
        }
        const int tk = (tier0 == 4 && el == 0) ? 3 : tier0;
        key = late ? ((1 << 30) | ((int)idv << 11) | ((tier0 - 2) << 9) | (sd << 8)) : ((pmv << 11) | ((tk - 2) << 9) | el);
        key_s[i] = key;
        id_s[i] = late ? ((1 << 20) | (int)idv) : pmv;
        const bool qom = pmv == -2;
        tgt_s[i] = (tier0 == 4 && sd == 0 && !qom) ? key : NOTGT;
        cand = (!qom && (tier0 == 3 || (tier0 == 4 && sd == 1))) ? 1 : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)i < P && tgt_s[i] != NOTGT) {
        const int me = id_s[i];
        if (me >= 0 && me < RT_C) atomic_fetch_min_explicit(&rtab[me], (int)i, memory_order_relaxed);
        else if (me >= (1 << 20) && me - (1 << 20) < LT_C) atomic_fetch_min_explicit(&ltab[me - (1 << 20)], (int)i, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)i < P && cand && UP_C) {
        const int me = id_s[i];
        int jf = P_C;
        if (me >= 0 && me < RT_C) jf = atomic_load_explicit(&rtab[me], memory_order_relaxed);
        else if (me >= (1 << 20) && me - (1 << 20) < LT_C) jf = atomic_load_explicit(&ltab[me - (1 << 20)], memory_order_relaxed);
        else { for (int j = 0; j < P; ++j) { if (tgt_s[j] != NOTGT && id_s[j] == me) { jf = j; break; } } }
        if (jf < P) { key = tgt_s[jf]; up = 1; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)i < P) { key_s[i] = key; up_s[i] = up; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)i < P) {
        int rank = 0;
        for (int j = 0; j < P; ++j) {
            const int kj = key_s[j], uj = up_s[j];
            rank += (kj < key) || (kj == key && (uj < up || (uj == up && j < (int)i)));
        }
        pidx[rank] = (int)i;
        skey[rank] = key;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)i < P) {
        int r0 = (int)i;
        while (r0 > 0 && skey[r0 - 1] == skey[i]) --r0;
        start_s[i] = (((int)i - r0) % 16) == 0 ? 1 : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)i < P && start_s[i]) {
        int g = 0;
        for (int j = 0; j < (int)i; ++j) g += start_s[j];
        seg[g] = (int)i;
    }
    if ((int)i == 0) {
        int n = 0;
        for (int j = 0; j < P; ++j) n += start_s[j];
        nseg[0] = n;
        seg[n] = P;
    }
    // ---- 4. rowsum64(x): a fixed sequential sum per (row, 64-value group)
    for (int q = (int)i; q < M_C * (H_C / 64); q += (int)nthr) {
        const int row = q / (H_C / 64), g = q % (H_C / 64);
        const device T* p = x + (size_t)row * H_C + g * 64;
        float s = 0.0f;
        for (int k = 0; k < 64; ++k) s += (float)p[k];
        xs[q] = s;
    }
"""


def _kf():
    if "f" not in _K:
        _K["f"] = mx.fast.metal_kernel(name="moeblk_front", input_names=["ids", "scores", "rw", "sm", "sl", "x"],
                                       output_names=["sc_out", "ids_out", "nolsb", "pm", "pl", "pidx", "seg", "nseg", "xs"],
                                       source=_SRC)
    return _K["f"]


def front(lay, ids, scores, x, up=1, rt=1024, lt=1024):
    import t1cf as T1
    from tierglu import plan_masks
    c = lay.c
    M, K = ids.shape[-2], ids.shape[-1]
    H = x.shape[-1]
    plan = c["routed_store"]["plan"]
    m4, m2, r4 = plan_masks(plan)
    assert len(plan) == K and m2 == 0 and H % 64 == 0
    P = M * K
    tg = max(32, (P + 31) // 32 * 32)
    t = c["tiers_1c"]
    T4, T3 = T1._f32bits(T1.T4_K * t["lam"]), T1._f32bits(T1.T3_K * t["lam"])
    outs = _kf()(inputs=[ids.reshape(M, K), scores.reshape(M, K), T1.rw_of(M, t["s"]), lay.sm_dev, lay.sl_dev, x],
                 template=[("P_C", P), ("K_C", K), ("R4_C", r4), ("NR_C", K), ("MASK4_C", m4), ("MASK2_C", m2),
                           ("T4B", T4), ("T3B", T3), ("ST", scores.dtype), ("UP_C", int(up)), ("RT_C", rt), ("LT_C", lt),
                           ("M_C", M), ("H_C", H), ("T", x.dtype)],
                 grid=(tg, 1, 1), threadgroup=(tg, 1, 1),
                 output_shapes=[(M, K), (M, K), (M, r4), (M, K), (M, r4), (P,), (P + 1,), (1,), (M, H // 64)],
                 output_dtypes=[scores.dtype, ids.dtype, mx.bool_, mx.int32, mx.int32, mx.int32, mx.int32, mx.int32,
                                mx.float32])
    sc, idn, nl, pm, pl, pidx, seg, nseg, xs = outs
    return dict(sc=sc, idn=idn, nl=nl, pm=pm, pl=pl, grp=(pidx, seg, nseg, P), xs=xs)


def front_check(lay, x):
    """front == route's t1cf + slots + xgrp v2 + rowsum64, bitwise (the seg array compared up to nseg)"""
    import xgrp
    from tierglu_nax import rowsum64
    ids, scores = MB.route(lay, x)
    sc, idn, nl = MB.glue(lay, ids, scores)
    pm, pl = MB.slots(lay, idn, nl)
    g = xgrp.gpu_grp_up2(pm, pl, idn.astype(mx.int32), lay.c["routed_store"]["plan"])
    xs = rowsum64(x)
    f = front(lay, ids, scores, x)
    mx.eval(sc, idn, nl, pm, pl, *g[:3], xs, *[v for k, v in f.items() if k != "grp"], *f["grp"][:3])
    eq = lambda a, b: a.shape == b.shape and bool(mx.all(a == b).item()) if a.dtype != mx.bfloat16 else \
        bool(mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())
    n1, n2 = int(np.array(g[2])[0]), int(np.array(f["grp"][2])[0])
    res = dict(sc=eq(sc, f["sc"]), idn=eq(idn, f["idn"]), nl=eq(nl, f["nl"]), pm=eq(pm, f["pm"]), pl=eq(pl, f["pl"]),
               pidx=eq(g[0], f["grp"][0]), nseg=n1 == n2,
               seg=bool((np.array(g[1])[:n1 + 1] == np.array(f["grp"][1])[:n2 + 1]).all()) if n1 == n2 else False,
               xs=eq(xs, f["xs"]))
    return res
