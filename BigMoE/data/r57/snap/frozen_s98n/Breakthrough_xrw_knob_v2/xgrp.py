"""xgrp.py (Breakthrough, Tue 17 Nov 2026 10:4x; EXPERT REWRITE, the TIER-4 UPGRADE): ooo1cq's one-call grouping
(_GRP1CQ) with one change. A pair keyed at tier 3 joins its expert's tier-4 key when that expert has an ELIGIBLE tier-4
pair in the call. A tier-3 pair is rank >= 2, or a 4-bit rank whose sidecar is -2. An eligible pair is at a 4-bit
rank, its sidecar is resident or late (not -2), and it is not a QOM drop. The expert's LSB plane is read for that pair
anyway, so the upgraded pairs cost no bytes and run on the 4-bit weights, which are more accurate than MSB-3. The
expert then forms one segment instead of two. x5_seg_merge.py: 62.7 -> 50.5 segments per 32-row call (-19.4 %), and
137 of 275 tier-3 pairs upgrade.
Within a key the real pairs sort first, so every 16-pair group that holds a real pair has a real head, and the kernels
(which take a group's tier and sidecar from its first pair) run it at tier 4. A group headed by an upgraded pair (only
when > 16 pairs share a key) keeps its pairs' own tier 3. Kernels A / B are unchanged (tgl16e's here: tgl16eu).
    gpu_grp_up(pre_m, pre_l, ids, plan) -> (pidx, seg, nseg, P)              # drop-in for ooo1cq.gpu_grp_1cq
    cpu_grp(pre_m, pre_l, ids, plan, up) -> (pidx, seg list, head tier per pair, upgraded flag per pair)   # numpy replica"""
import os
import sys

import numpy as np
import mlx.core as mx

for _p in ("/Users/kamen/BigMoE/data/r57/snap/frozen_s98n/Leap_scripts", "/Users/kamen/BigMoE/data/r57/snap/frozen_s98n/Leap_third_party/ms_gdn"):
    _p = os.path.expanduser(_p)
    if _p not in sys.path:
        sys.path.append(_p)

_K = {}
_SRC = r"""
    const int P = P_C;
    const uint i = thread_position_in_grid.x;
    const int NOTGT = (int)0x80000000;
    threadgroup int key_s[P_C];
    threadgroup int skey[P_C];
    threadgroup int start_s[P_C];
    threadgroup int id_s[P_C];
    threadgroup int tgt_s[P_C];
    threadgroup int up_s[P_C];
    int key = 0, cand = 0, up = 0;
    if ((int)i < P) {
        const int r = (int)i % NR_C, m = (int)i / NR_C;
        const int tier0 = ((MASK4_C >> r) & 1) ? 4 : (((MASK2_C >> r) & 1) ? 2 : 3);
        int el = 0, sd = 0;
        bool late = msb_slot[i] < 0;                                // -1 pending, -2 a QOM drop (both resolved per group)
        if (tier0 == 4) {
            el = lsb_slot[m * R4_C + popcount((uint)MASK4_C & ((1u << r) - 1u))];
            if (el == -1) late = true; else if (el < 0) { el = 0; sd = 1; } else el += 1;   // -2: final, tier 3's key
        }
        const int tk = (tier0 == 4 && el == 0) ? 3 : tier0;
        key = late ? ((1 << 30) | (ids[i] << 11) | ((tier0 - 2) << 9) | (sd << 8)) : ((msb_slot[i] << 11) | ((tk - 2) << 9) | el);
        key_s[i] = key;
        id_s[i] = late ? ((1 << 20) | ids[i]) : msb_slot[i];         // the expert: its slot, or its id while late
        const bool qom = msb_slot[i] == -2;
        tgt_s[i] = (tier0 == 4 && sd == 0 && !qom) ? key : NOTGT;     // an eligible tier-4 pair: its key
        cand = (!qom && (tier0 == 3 || (tier0 == 4 && sd == 1))) ? 1 : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)i < P && cand && UP_C) {                              // the UPGRADE: adopt the expert's tier-4 key
        const int me = id_s[i];
        for (int j = 0; j < P; ++j) {
            const int t = tgt_s[j];
            if (t != NOTGT && id_s[j] == me) { key = t; up = 1; break; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)i < P) { key_s[i] = key; up_s[i] = up; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)i < P) {                                              // rank by (key, upgraded, index): real pairs first
        int rank = 0;
        for (int j = 0; j < P; ++j) {
            const int kj = key_s[j], uj = up_s[j];
            rank += (kj < key) || (kj == key && (uj < up || (uj == up && j < (int)i)));
        }
        pidx[rank] = (int)i;
        skey[rank] = key;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)i < P) {                                              // a group starts at a new key, and every 16 pairs inside one
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
"""


def _kg():
    if "g" not in _K:
        _K["g"] = mx.fast.metal_kernel(name="xgrp_up", input_names=["msb_slot", "lsb_slot", "ids"],
                                       output_names=["pidx", "seg", "nseg"], source=_SRC)
    return _K["g"]


def gpu_grp_up(pre_m, pre_l, ids, plan, up=1):
    from tierglu import plan_masks
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    ms = pre_m.astype(mx.int32).reshape(-1)
    P = ms.size
    assert P <= 1024 and m2 == 0, (P, m2)
    ls = pre_l.astype(mx.int32).reshape(-1) if r4 else mx.zeros((1,), mx.int32)
    pidx, seg, nseg = _kg()(inputs=[ms, ls, ids.astype(mx.int32).reshape(-1)],
                            template=[("P_C", P), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)),
                                      ("UP_C", int(up))],
                            grid=(max(32, (P + 31) // 32 * 32), 1, 1), threadgroup=(max(32, (P + 31) // 32 * 32), 1, 1),
                            output_shapes=[(P,), (P + 1,), (1,)], output_dtypes=[mx.int32] * 3)
    return pidx, seg, nseg, P


def cpu_grp(pre_m, pre_l, ids, plan, up=True):
    """numpy replica: (pidx, seg starts + [P], head tier per pair (the tier its group runs at, all slots resident),
    upgraded flag per pair). up=False replicates ooo1cq._GRP1CQ."""
    from tierglu import plan_masks
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    ms = np.asarray(pre_m, np.int64).reshape(-1)
    ls = np.asarray(pre_l, np.int64).reshape(-1)
    idv = np.asarray(ids, np.int64).reshape(-1)
    P = ms.size
    key, ident, tgt, cand, tier0s, sds = (np.zeros(P, np.int64) for _ in range(6))
    for i in range(P):
        r, m = i % R, i // R
        tier0 = 4 if (m4 >> r) & 1 else (2 if (m2 >> r) & 1 else 3)
        el, sd = 0, 0
        late = ms[i] < 0
        if tier0 == 4:
            el = int(ls[m * r4 + bin(m4 & ((1 << r) - 1)).count("1")])
            if el == -1:
                late = True
            elif el < 0:
                el, sd = 0, 1
            else:
                el += 1
        tk = 3 if (tier0 == 4 and el == 0) else tier0
        key[i] = ((1 << 30) | (idv[i] << 11) | ((tier0 - 2) << 9) | (sd << 8)) if late else ((ms[i] << 11) | ((tk - 2) << 9) | el)
        ident[i] = ((1 << 20) | idv[i]) if late else ms[i]
        qom = ms[i] == -2
        tgt[i] = key[i] if (tier0 == 4 and sd == 0 and not qom) else -(1 << 40)
        cand[i] = 1 if (not qom and (tier0 == 3 or (tier0 == 4 and sd == 1))) else 0
        tier0s[i], sds[i] = tier0, sd
    upf = np.zeros(P, np.int64)
    if up:
        for i in range(P):
            if cand[i]:
                for j in range(P):
                    if tgt[j] != -(1 << 40) and ident[j] == ident[i]:
                        key[i], upf[i] = tgt[j], 1
                        break
    order = sorted(range(P), key=lambda i: (key[i], upf[i], i))
    pidx = np.array(order, np.int64)
    sk = key[pidx]
    starts = [q for q in range(P) if (q - next(r0 for r0 in range(q, -1, -1) if r0 == 0 or sk[r0 - 1] != sk[q])) % 16 == 0]
    seg = starts + [P]
    head_tier = np.zeros(P, np.int64)
    for g in range(len(starts)):
        pf = pidx[seg[g]]
        t = int(tier0s[pf])
        if t == 4:
            el = int(ls[(pf // R) * r4 + bin(m4 & ((1 << (pf % R)) - 1)).count("1")])
            if el < 0:
                t = 3
        head_tier[pidx[seg[g]:seg[g + 1]]] = t
    return pidx, seg, head_tier, upf


def self_test(n=40, seed=1117):
    """the GPU grouping == the numpy replica (pidx, seg) with and without the upgrade, and up=0 == ooo1cq.gpu_grp_1cq,
    on random calls (resident, late, QOM drops, sidecar -2, popular experts)"""
    import ooo1cq as OQ
    rng = np.random.default_rng(seed)
    plan = "4433333333"
    bad = 0
    for t in range(n):
        M = int(rng.choice([16, 32]))
        ex = rng.integers(0, 60, size=(M, 10))                     # few experts: many shared keys
        if t % 4 == 0:
            ex[:, 0] = 7                                           # a popular expert (> 16 pairs)
        slot = {e: (int(rng.integers(0, 400)) if rng.random() > 0.1 else int(rng.choice([-1, -2]))) for e in range(60)}
        pm = np.vectorize(lambda e: slot[e])(ex).astype(np.int32)
        pl = np.where(rng.random((M, 2)) < 0.15, -2, rng.integers(0, 400, size=(M, 2))).astype(np.int32)
        pl = np.where(pm[:, :2] == -1, -1, pl)
        for up in (0, 1):
            g = gpu_grp_up(mx.array(pm), mx.array(pl), mx.array(ex.astype(np.int32)), plan, up=up)
            mx.eval(g[:3])
            gp, gs, gn = np.array(g[0]), np.array(g[1]), int(np.array(g[2])[0])
            cp, cs, _, _ = cpu_grp(pm, pl, ex, plan, up=bool(up))
            ok = (gp == cp).all() and gn == len(cs) - 1 and (gs[:gn + 1] == np.array(cs)).all()
            if up == 0:
                o = OQ.gpu_grp_1cq(mx.array(pm), mx.array(pl), mx.array(ex.astype(np.int32)), plan)
                mx.eval(o[:3])
                ok &= (np.array(o[0]) == gp).all() and int(np.array(o[2])[0]) == gn
            bad += not ok
    print(f"xgrp self-test: {2 * n - bad} of {2 * n} PASS (GPU == numpy replica; up=0 == ooo1cq.gpu_grp_1cq)")
    return bad == 0


if __name__ == "__main__":
    sys.exit(0 if self_test() else 1)


# ---- v2 (Tue 11:3x): the upgrade's lookup in O(1). v1 lets every candidate scan all P pairs for the FIRST eligible
# tier-4 pair (smallest index) with its identity: x6_grp_cost.py measured 19.9 vs today's 9.0 us at P = 160, 39.9 vs 16.5
# at P = 320, on every call's critical path. v2 has each eligible pair atomic-min its index into a threadgroup table
# keyed by its identity (resident slot < RT_C, or late id < LT_C), then each candidate reads its identity's entry.
# Identical choice to v1 by construction (the first eligible pair with that identity); an identity outside the
# tables falls back to v1's scan, so the output equals v1's for ANY input. The rank / start / segment phases are v1's.
_SRC2 = _SRC.replace(
    """    threadgroup int up_s[P_C];
    int key = 0, cand = 0, up = 0;""",
    """    threadgroup int up_s[P_C];
    threadgroup atomic_int rtab[RT_C];                              // resident slot -> first eligible pair index
    threadgroup atomic_int ltab[LT_C];                              // late expert id -> first eligible pair index
    for (int t = (int)i; t < RT_C; t += (int)threads_per_threadgroup.x) atomic_store_explicit(&rtab[t], P_C, memory_order_relaxed);
    for (int t = (int)i; t < LT_C; t += (int)threads_per_threadgroup.x) atomic_store_explicit(&ltab[t], P_C, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int key = 0, cand = 0, up = 0;""").replace(
    """    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)i < P && cand && UP_C) {                              // the UPGRADE: adopt the expert's tier-4 key
        const int me = id_s[i];
        for (int j = 0; j < P; ++j) {
            const int t = tgt_s[j];
            if (t != NOTGT && id_s[j] == me) { key = t; up = 1; break; }
        }
    }""",
    """    if ((int)i < P && tgt_s[i] != NOTGT) {                         // v2: eligible pairs post their index (first wins)
        const int me = id_s[i];
        if (me >= 0 && me < RT_C) atomic_fetch_min_explicit(&rtab[me], (int)i, memory_order_relaxed);
        else if (me >= (1 << 20) && me - (1 << 20) < LT_C) atomic_fetch_min_explicit(&ltab[me - (1 << 20)], (int)i, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)i < P && cand && UP_C) {                              // the UPGRADE: adopt the expert's tier-4 key
        const int me = id_s[i];
        int jf = P_C;
        if (me >= 0 && me < RT_C) jf = atomic_load_explicit(&rtab[me], memory_order_relaxed);
        else if (me >= (1 << 20) && me - (1 << 20) < LT_C) jf = atomic_load_explicit(&ltab[me - (1 << 20)], memory_order_relaxed);
        else {                                                     // outside the tables: v1's scan
            for (int j = 0; j < P; ++j) { if (tgt_s[j] != NOTGT && id_s[j] == me) { jf = j; break; } }
        }
        if (jf < P) { key = tgt_s[jf]; up = 1; }
    }""")
assert _SRC2 != _SRC and _SRC2.count("atomic_fetch_min_explicit") == 2


def _kg2():
    if "g2" not in _K:
        _K["g2"] = mx.fast.metal_kernel(name="xgrp_up2", input_names=["msb_slot", "lsb_slot", "ids"],
                                        output_names=["pidx", "seg", "nseg"], source=_SRC2)
    return _K["g2"]


def gpu_grp_up2(pre_m, pre_l, ids, plan, up=1, rt=1024, lt=1024):
    from tierglu import plan_masks
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    ms = pre_m.astype(mx.int32).reshape(-1)
    P = ms.size
    assert P <= 1024 and m2 == 0, (P, m2)
    ls = pre_l.astype(mx.int32).reshape(-1) if r4 else mx.zeros((1,), mx.int32)
    pidx, seg, nseg = _kg2()(inputs=[ms, ls, ids.astype(mx.int32).reshape(-1)],
                             template=[("P_C", P), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)),
                                       ("UP_C", int(up)), ("RT_C", rt), ("LT_C", lt)],
                             grid=(max(32, (P + 31) // 32 * 32), 1, 1), threadgroup=(max(32, (P + 31) // 32 * 32), 1, 1),
                             output_shapes=[(P,), (P + 1,), (1,)], output_dtypes=[mx.int32] * 3)
    return pidx, seg, nseg, P


def self_test2(n=120, seed=2117):
    """v2 == v1 (pidx, seg, nseg) on random calls: resident / late / QOM / -2 sidecars, popular experts, sidecar slots
    that are NOT consistent per expert (stress: v1 takes the first eligible index per identity), slots beyond the
    resident table (the fallback scan), M = 1..32"""
    rng = np.random.default_rng(seed)
    plan = "4433333333"
    bad = 0
    for t in range(n):
        M = int(rng.choice([1, 2, 3, 5, 8, 16, 17, 24, 32]))
        ex = rng.integers(0, 60, size=(M, 10))
        if t % 4 == 0 and M >= 4:
            ex[:, 0] = 7
        big = t % 5 == 0                                       # slots beyond the resident table -> v1's scan
        slot = {e: ((int(rng.integers(1000, 1200)) if big else int(rng.integers(0, 400))) if rng.random() > 0.1
                    else int(rng.choice([-1, -2]))) for e in range(60)}
        pm = np.vectorize(lambda e: slot[e])(ex).astype(np.int32)
        pl = np.where(rng.random((M, 2)) < 0.15, -2, rng.integers(0, 400, size=(M, 2))).astype(np.int32)
        pl = np.where(rng.random((M, 2)) < 0.1, -1, pl)
        for up in (0, 1):
            a = gpu_grp_up(mx.array(pm), mx.array(pl), mx.array(ex.astype(np.int32)), plan, up=up)
            b = gpu_grp_up2(mx.array(pm), mx.array(pl), mx.array(ex.astype(np.int32)), plan, up=up)
            mx.eval(a[:3], b[:3])
            na, nb_ = int(np.array(a[2])[0]), int(np.array(b[2])[0])   # seg past nseg is never written (uninitialized)
            ok = na == nb_ and (np.array(a[0]) == np.array(b[0])).all() and \
                (np.array(a[1])[:na + 1] == np.array(b[1])[:nb_ + 1]).all()
            bad += not ok
    print(f"xgrp v2 self-test: {2 * n - bad} of {2 * n} PASS (gpu_grp_up2 == gpu_grp_up: pidx, seg, nseg)")
    return bad == 0
