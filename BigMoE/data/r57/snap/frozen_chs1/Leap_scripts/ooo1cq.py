"""FAST-NAX (Leap, Thu 2026-11-05 19:0x; TABLE_350 'FAST-NAX'): the FAST path's QOM drops and U3 sure rows on NAX6-N's
one-call OOO (native / p11 pools: the converted records3). ooo1c.py (live, unchanged) covers the exact NAX one-call.
On top of it, here:
  - grouping: late pairs keyed by expert id, tier AND sidecar state (bit 8: a 4-bit pair whose sidecar slot is -2, i.e.
    U3's sure row or LP1, is final tier 3), so every late group is uniform. U3's -2 is per row, so without the bit one
    late group could hold a sure row's pair and another row's 4-bit pair of the same expert.
  - kernel A head: a base slot -1 is polled; -2 (QOM: an optional-only expert the loader did not read; ext ensure()
    decides per expert, so per group) drops the group. A writes rs_m = -2 and returns. Each pair writes only ITS OWN
    sidecar entry: ooo1c's head writes rs_l for every pair with the first pair's tier, which in a mixed resident
    group (a -2-sidecar 4-bit pair + rank >= 2 pairs) lands on the next row's rank-0 entry (past the buffer for the
    last row).
  - kernel B: a dropped group writes zeros for its rows (combine_phase then adds nothing for it; QOM's rule is weight 0).
Default off: ref2 CFG nax_1cq / REF2_NAX_1CQ=1, with ooo_1c and tier_nax_ooo + nax_native.
  tier_glu_nax6n_1cq(x, msb_n, lsb_n, pre_m, pre_l, ids, plan, region, offs, status, ...) -> (y [M, R, D], rs_m, rs_l)
  gpu_grp_1cq(pre_m, pre_l, ids, plan)   check_src() -> the source transforms apply (CPU, no kernel built)"""
import os, sys
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs1/Leap_scripts"))
import ooo1c as OC
import tierglu_nax6 as N6
from tierglu import plan_masks, PROJ
from tierglu_nax import rowsum64

_GRP_OLD = """        int tier = ((MASK4_C >> r) & 1) ? 4 : (((MASK2_C >> r) & 1) ? 2 : 3);
        int el = 0;
        if (tier == 4) {
            el = lsb_slot[m * R4_C + popcount((uint)MASK4_C & ((1u << r) - 1u))];
            if (el < 0) { tier = 3; el = 0; } else el += 1;
        }
        key_s[i] = (msb_slot[i] << 11) | ((tier - 2) << 9) | el;"""
_GRP_NEW = """        const int tier0 = ((MASK4_C >> r) & 1) ? 4 : (((MASK2_C >> r) & 1) ? 2 : 3);
        int el = 0, sd = 0;
        bool late = msb_slot[i] < 0;                                // -1 pending, -2 a QOM drop (both resolved per group)
        if (tier0 == 4) {
            el = lsb_slot[m * R4_C + popcount((uint)MASK4_C & ((1u << r) - 1u))];
            if (el == -1) late = true; else if (el < 0) { el = 0; sd = 1; } else el += 1;   // -2: final, tier 3's key
        }
        // resident: NAX6's key (slot, tier, sidecar slot); late: after every resident key, by expert id, tier and the
        // sidecar's -2 state (U3's sure rows are per row), so a late group's pairs resolve alike
        const int tk = (tier0 == 4 && el == 0) ? 3 : tier0;
        key_s[i] = late ? ((1 << 30) | (ids[i] << 11) | ((tier0 - 2) << 9) | (sd << 8)) : ((msb_slot[i] << 11) | ((tk - 2) << 9) | el);"""
_GRP1CQ = N6._GRP.replace(_GRP_OLD, _GRP_NEW)

_HEAD_Q = """    int e_ = msb_slot[pf];
    const uint i4_0 = popcount((uint)MASK4_C & ((1u << r0) - 1u));
    int el_ = tier0 == 4 ? lsb_slot[(pf / NR) * R4_C + i4_0] : 0;
    const bool need = e_ == -1 || (e_ >= 0 && tier0 == 4 && el_ == -1);   // -1 waits; a base -2 is a QOM drop (final)
    threadgroup int rs_s[2];
    if (need) {
        if (thread_index_in_threadgroup == 0) {
            device atomic_int* st_ = (device atomic_int*)status;
            int a = e_, b = el_;
            if (a == -1) a = oc_poll((coherent(system) device int*)(region + offs[0]), (int)pf, offs[2], st_);
            if (a >= 0 && tier0 == 4 && b == -1) b = oc_poll((coherent(system) device int*)(region + offs[1]),
                                                             (int)((pf / NR) * R4_C + i4_0), offs[2], st_);
            rs_s[0] = a; rs_s[1] = b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        e_ = rs_s[0]; el_ = rs_s[1];
    }
    if (threadgroup_position_in_grid.x == 0 && thread_index_in_threadgroup < (uint)nq) {   // the slots for kernel B
        const uint pj_ = (uint)pidx[q0 + thread_index_in_threadgroup];
        rs_m[pj_] = e_;
        if ((MASK4_C >> (pj_ % NR)) & 1)                            // the pair's OWN sidecar entry (4-bit ranks only)
            rs_l[(pj_ / NR) * R4_C + popcount((uint)MASK4_C & ((1u << (pj_ % NR)) - 1u))] = (e_ >= 0 && tier0 == 4) ? el_ : -2;
    }
    if (e_ < 0) return;                                             // a dropped group (uniform): kernel B writes its zeros
    const uint e = (uint)e_;
    int tier = tier0;
    uint el = 0;
    if (tier == 4) {
        if (el_ < 0) tier = 3; else el = (uint)el_;                // -2: final (LP1 / U3's sure rows) -> tier 3
    }"""

_B_E = "    const uint e = (uint)msb_slot[pf];\n"
_B_DROP = """    if (msb_slot[pf] < 0) {                                         // FAST-NAX: a dropped group: zeros for its rows
        if (kc == 0)
            for (int i = (int)lane; i < 16 * 32; i += 32) {
                const int row = i >> 5, col = i & 31;
                if (row < nq) y[(size_t)pidx[q0 + row] * D + n0 + col] = static_cast<T>(0.0f);
            }
        return;
    }
""" + _B_E


def a_one_call_q(src):
    """a NAX6-family kernel A (NAX6's head: _A4, NAX6-N's _A_N / _A_NP) -> the FAST-NAX one-call form"""
    assert src.count(OC._OLD_HEAD) == 1, "kernel A head changed"
    return src.replace(OC._OLD_HEAD, _HEAD_Q)


def b_drop(src):
    """a NAX6-family kernel B -> zeros for a dropped group (its resolved base slot < 0)"""
    assert src.count(_B_E) == 1 and src.count("const uint ti = sg % NSG_C, kc = sg / NSG_C;") == 1, "kernel B head changed"
    return src.replace(_B_E, _B_DROP)


def check_src():
    """CPU: every transform applies once and changes what it should (no kernel is built)"""
    import tierglu_nax6n as NN
    assert _GRP1CQ != N6._GRP and _GRP1CQ.count("(sd << 8)") == 1
    for a in (NN._A_N, NN._A_NP, N6._A4):
        s = a_one_call_q(a)
        assert s.count("if (e_ < 0) return;") == 1 and s.count("(MASK4_C >> (pj_ % NR)) & 1") == 1
    for b in (NN._B_N, NN._B_NP, N6._B4):
        s = b_drop(b)
        assert s.count("FAST-NAX: a dropped group") == 1 and s.index("FAST-NAX") < s.index("threadgroup uint32_t tb_all")
    return True


_K = {}


def _kgrp():
    if "grp" not in _K:
        _K["grp"] = mx.fast.metal_kernel(name="ooo1cq_grp", input_names=["msb_slot", "lsb_slot", "ids"],
                                         output_names=["pidx", "seg", "nseg"], source=_GRP1CQ)
    return _K["grp"]


def _ka(p11):
    import tierglu_nax6n as NN
    import p11 as P11
    key = ("a", p11)
    if key not in _K:
        _K[key] = mx.fast.metal_kernel(
            name=f"ooo1cq_nax6n{'p' if p11 else ''}_a",
            input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                         "msb_slot", "lsb_slot", "pidx", "seg", "nseg", "region", "offs", "status"],
            output_names=["h", "rs_m", "rs_l"], source=a_one_call_q(NN._A_NP if p11 else NN._A_N),
            header=N6._HDR4 + (P11.P11_METAL if p11 else "") + OC.SYS)
    return _K[key]


def _kb(p11):
    import tierglu_nax6n as NN
    import p11 as P11
    key = ("b", p11)
    if key not in _K:
        _K[key] = mx.fast.metal_kernel(
            name=f"ooo1cq_nax6n{'p' if p11 else ''}_b",
            input_names=["h", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
            output_names=["y"], source=b_drop(NN._B_NP if p11 else NN._B_N), header=N6._HDR4 + (P11.P11_METAL if p11 else ""))
    return _K[key]


def gpu_grp_1cq(pre_m, pre_l, ids, plan):
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    ms = pre_m.astype(mx.int32).reshape(-1)
    P = ms.size
    assert P <= 1024 and m2 == 0, (P, m2)
    ls = pre_l.astype(mx.int32).reshape(-1) if r4 else mx.zeros((1,), mx.int32)
    pidx, seg, nseg = _kgrp()(inputs=[ms, ls, ids.astype(mx.int32).reshape(-1)],
                              template=[("P_C", P), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1))],
                              grid=(max(32, (P + 31) // 32 * 32), 1, 1), threadgroup=(max(32, (P + 31) // 32 * 32), 1, 1),
                              output_shapes=[(P,), (P + 1,), (1,)], output_dtypes=[mx.int32] * 3)
    return pidx, seg, nseg, P


def tier_glu_nax6n_1cq(x, msb_n, lsb_n, pre_m, pre_l, ids, plan, region, offs, status, xs=None, grp=None, nsg_b=2, ks=4,
                       ks_b=2):
    """ooo1c.tier_glu_nax6n_1c's call with the FAST-NAX kernels: U3 sure rows (pre_l -2) and QOM drops (a base slot that
    is or resolves to -2: the group's rows are zeros). Returns (y [M, R, D], rs_m, rs_l)"""
    import p11 as P11
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    assert m2 == 0
    p11 = msb_n["gate_proj"]["s3"].dtype == mx.uint32
    if p11:
        N, D = msb_n["gate_proj"]["w3"].shape[1], msb_n["down_proj"]["w3"].shape[1]
        pwm, pwl = msb_n["gate_proj"]["s3"].shape[-1], lsb_n["gate_proj"]["b"].shape[-1]
        assert pwm >= P11.piece_words(N * 40) and pwm % 4 == 0 and pwl >= P11.piece_words(N * 40) and pwl % 4 == 0
    else:
        N, D = msb_n["gate_proj"]["s3"].shape[1], msb_n["down_proj"]["s3"].shape[1]
    if grp is None:
        grp = gpu_grp_1cq(pre_m, pre_l, ids, plan)
    pidx, seg, nsg_, nseg = grp
    bb = {p: msb_n[p]["b3"] for p in PROJ}
    ls = pre_l.astype(mx.int32).reshape(-1) if r4 else mx.zeros((1,), mx.int32)
    ms = pre_m.astype(mx.int32).reshape(-1)
    L = lsb_n
    g, u, d = (msb_n[p] for p in PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0), ("GS_C", 1)]
    if p11:
        tmpl = tmpl + [("PWM_C", pwm), ("PWL_C", pwl)]
    xs = rowsum64(x) if xs is None else xs
    h, rs_m, rs_l = _ka(p11)(
        inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls, pidx, seg, nsg_,
                region, offs, status],
        template=tmpl + [("K_C", K), ("N_C", N), ("KS_C", ks)], grid=(N // 32 * 64 * ks, nseg, 1),
        threadgroup=(64 * ks, 1, 1), output_shapes=[(M * R, N), (M * R,), (max(M * r4, 1),)],
        output_dtypes=[x.dtype, mx.int32, mx.int32])
    hs = rowsum64(h)
    (y,) = _kb(p11)(
        inputs=[h, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"],
                rs_m, rs_l, pidx, seg, nsg_],
        template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)],
        grid=(D // (32 * nsg_b) * 32 * nsg_b * ks_b, nseg, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
        output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D), rs_m, rs_l


if __name__ == "__main__":
    print("ooo1cq source transforms:", "OK" if check_src() else "FAIL")
