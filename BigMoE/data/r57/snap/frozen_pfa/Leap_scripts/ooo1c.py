"""ONE-CALL OOO (Leap, Wed 2026-11-04 15:xx; BUILD_DAY #5; TABLE_350 'OOO-1C'): BigMoE's OOO in ONE expert call per MoE
layer on the NAX path (ref2 C1, the NAX reference Kamen approved 15:0x pending certification). Today (ref2._ooo_nax6):
post -> NAX6 phase 1 (resident pairs) -> two spin kernels (the loader's release) -> NAX6 phase 2 (late pairs) ->
combine_phase. Here: post -> ONE NAX6 call whose groups are ordered resident-first and whose late groups resolve their
slot inside kernel A, polling the release region the way the coherent spin does (line 0's tag == seq, written last
after a release fence; the pair's word stamped with seq), then kernel B (unchanged) reads the slots kernel A resolved,
then the same combine_phase. A pair's bits do not depend on its group (INV), so this == today's two phases, bitwise.
No ext or bigmoe/tier.py change: the release region, its layout and the stamps are BigMoE's (gated.Gate.region / status,
ext spin_wait's coherent mode, BIGMOE_SPIN_COHERENT=1 releases pair 0 only).
  gpu_grp_1c(pre_m, pre_l, ids, plan)  NAX6's grouping with late pairs keyed by expert id and sorted last
  tier_glu_nax6_1c(x, msb, lsb, pre_m, pre_l, ids, plan, region, offs, status, xs=None, grp=None)
      -> y [M, R, D] (== tier_glu_nax6 on the released slots, bitwise), rs_m, rs_l (the resolved slots)
  release_words(slots, seq) / region_offset(...)  the loader's encoding (test harnesses)"""
import os, sys
from functools import lru_cache
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_third_party/ms_gdn"))
import tierglu_nax6 as N6
from tierglu import plan_masks, PROJ
from tierglu_nax import rowsum64

SYS = """
#pragma METAL internals : enable
#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
namespace metal { constexpr constant thread_scope thread_scope_system = static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__); }
#endif
#ifndef OC_CAP
#define OC_CAP 4000000
#endif
// the slot of pair i of a released (pass seq, layer) region (ext spin_wait's coherent mode, BigMoE's encoding):
// line 0's tag (word 0) is written last, after a release fence; word 32 * (i / 31) + 1 + i % 31 = (seq & 0x3FF) << 22 | slot
inline int oc_poll(coherent(system) device int* rg, int i, int seq, device atomic_int* st) {
    const uint STAMP = (((uint)seq) & 0x3FFu) << 22;
    const int idx = 32 * (i / 31) + 1 + i % 31;
    for (int it = 0; it < OC_CAP; it++) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
        if (rg[0] != seq) continue;
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
        const uint w = (uint)rg[idx];
        if ((w & 0xFFC00000u) != STAMP) continue;
        const uint v = w & 0x3FFFFFu;
        return v >= 0x3FFFFEu ? (int)v - 0x400000 : (int)v;
    }
    atomic_fetch_add_explicit(st, 1, memory_order_relaxed);   // timeout: the pass is invalid (gate.check raises)
    return 0;
}
"""

_OLD_HEAD = """    const uint e = (uint)msb_slot[pf];
    int tier = tier0;
    uint el = 0;
    if (tier == 4) {
        const int el_ = lsb_slot[(pf / NR) * R4_C + popcount((uint)MASK4_C & ((1u << r0) - 1u))];
        if (el_ < 0) tier = 3; else el = (uint)el_;                // LP1: no LSB read -> tier 3
    }"""
_NEW_HEAD = """    int e_ = msb_slot[pf];
    const uint i4_0 = popcount((uint)MASK4_C & ((1u << r0) - 1u));
    int el_ = tier0 == 4 ? lsb_slot[(pf / NR) * R4_C + i4_0] : 0;
    const bool need = e_ < 0 || (tier0 == 4 && el_ == -1);   // a late group: resolve its expert's slots (uniform per group);
                                                              // a sidecar slot -2 is final (LP1 / U3's sure rows: tier 3)
    threadgroup int rs_s[2];
    if (need) {
        if (thread_index_in_threadgroup == 0) {
            device atomic_int* st_ = (device atomic_int*)status;
            int a = e_, b = el_;
            if (a < 0) a = oc_poll((coherent(system) device int*)(region + offs[0]), (int)pf, offs[2], st_);
            if (tier0 == 4 && b == -1) b = oc_poll((coherent(system) device int*)(region + offs[1]),
                                                 (int)((pf / NR) * R4_C + i4_0), offs[2], st_);
            rs_s[0] = a; rs_s[1] = b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        e_ = rs_s[0]; el_ = rs_s[1];
    }
    if (threadgroup_position_in_grid.x == 0 && thread_index_in_threadgroup < (uint)nq) {   // the slots for kernel B
        const uint pj_ = (uint)pidx[q0 + thread_index_in_threadgroup];
        rs_m[pj_] = e_;
        if (tier0 == 4) rs_l[(pj_ / NR) * R4_C + popcount((uint)MASK4_C & ((1u << (pj_ % NR)) - 1u))] = el_;
    }
    const uint e = (uint)e_;
    int tier = tier0;
    uint el = 0;
    if (tier == 4) {
        if (el_ < 0) tier = 3; else el = (uint)el_;                // LP1: no LSB read -> tier 3
    }"""


def a_one_call(src):
    """kernel A source -> its one-call form (any NAX6-family A that keeps NAX6's head: _A4, NAX6-N's _A_N / _A_NP)"""
    assert src.count(_OLD_HEAD) == 1, "kernel A head changed"
    return src.replace(_OLD_HEAD, _NEW_HEAD)


_GRP1C = N6._GRP.replace("""        int tier = ((MASK4_C >> r) & 1) ? 4 : (((MASK2_C >> r) & 1) ? 2 : 3);
        int el = 0;
        if (tier == 4) {
            el = lsb_slot[m * R4_C + popcount((uint)MASK4_C & ((1u << r) - 1u))];
            if (el < 0) { tier = 3; el = 0; } else el += 1;
        }
        key_s[i] = (msb_slot[i] << 11) | ((tier - 2) << 9) | el;""", """        const int tier0 = ((MASK4_C >> r) & 1) ? 4 : (((MASK2_C >> r) & 1) ? 2 : 3);
        int el = 0;
        bool late = msb_slot[i] < 0;
        if (tier0 == 4) {
            el = lsb_slot[m * R4_C + popcount((uint)MASK4_C & ((1u << r) - 1u))];
            if (el == -1) late = true; else if (el < 0) el = 0; else el += 1;   // -2: final (LP1), tier 3's key
        }
        // resident: NAX6's key (slot, tier, sidecar slot); late: after every resident key, by expert id (1:1 with the slot
        // the loader will give it), so the late groups are the same pairs as on the released slots, last in the grid
        const int tk = (tier0 == 4 && el == 0) ? 3 : tier0;
        key_s[i] = late ? ((1 << 30) | (ids[i] << 11) | ((tier0 - 2) << 9)) : ((msb_slot[i] << 11) | ((tk - 2) << 9) | el);""")
assert _GRP1C != N6._GRP


@lru_cache(None)
def _kgrp1c():
    return mx.fast.metal_kernel(name="ooo1c_grp", input_names=["msb_slot", "lsb_slot", "ids"],
                                output_names=["pidx", "seg", "nseg"], source=_GRP1C)


def gpu_grp_1c(pre_m, pre_l, ids, plan):
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    ms = pre_m.astype(mx.int32).reshape(-1)
    P = ms.size
    assert P <= 1024 and m2 == 0, (P, m2)
    ls = pre_l.astype(mx.int32).reshape(-1) if r4 else mx.zeros((1,), mx.int32)
    pidx, seg, nseg = _kgrp1c()(inputs=[ms, ls, ids.astype(mx.int32).reshape(-1)],
                                template=[("P_C", P), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1))],
                                grid=(max(32, (P + 31) // 32 * 32), 1, 1), threadgroup=(max(32, (P + 31) // 32 * 32), 1, 1),
                                output_shapes=[(P,), (P + 1,), (1,)], output_dtypes=[mx.int32] * 3)
    return pidx, seg, nseg, P


@lru_cache(None)
def _ka1c():
    return mx.fast.metal_kernel(
        name="ooo1c_nax6_a",
        input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "pidx", "seg", "nseg", "region", "offs", "status"],
        output_names=["h", "rs_m", "rs_l"], source=a_one_call(N6._A4), header=N6._HDR4 + SYS)


def tier_glu_nax6_1c(x, msb, lsb, pre_m, pre_l, ids, plan, region, offs, status, xs=None, grp=None, nsg_b=2, ks=4, ks_b=2):
    """tier_glu_nax6 with the late pairs' slots resolved in kernel A from the release region (offs = [base region
    offset, sidecar region offset, seq] as int32). Returns (y [M, R, D], rs_m [M * R], rs_l [M * R4])"""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    N = msb["gate_proj"]["s3"].shape[1]
    D = msb["down_proj"]["s3"].shape[1]
    assert M <= 64 and K % 64 == 0 and N % 64 == 0 and D % (32 * nsg_b) == 0 and m2 == 0
    if grp is None:
        grp = gpu_grp_1c(pre_m, pre_l, ids, plan)
    pidx, seg, nsg_, nseg = grp
    bb = {p: msb[p]["b3"] for p in PROJ}
    ls = pre_l.astype(mx.int32).reshape(-1) if r4 else mx.zeros((1,), mx.int32)
    L = lsb if r4 else {p: {"l1": msb[p]["w3"], "b": msb[p]["b3"]} for p in PROJ}
    ms = pre_m.astype(mx.int32).reshape(-1)
    g, u, d = (msb[p] for p in PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0), ("GS_C", 1)]
    xs = rowsum64(x) if xs is None else xs
    h, rs_m, rs_l = _ka1c()(inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                                    L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls,
                                    pidx, seg, nsg_, region, offs, status],
                            template=tmpl + [("K_C", K), ("N_C", N), ("KS_C", ks)],
                            grid=(N // 32 * 64 * ks, nseg, 1), threadgroup=(64 * ks, 1, 1),
                            output_shapes=[(M * R, N), (M * R,), (max(M * r4, 1),)],
                            output_dtypes=[x.dtype, mx.int32, mx.int32])
    hs = rowsum64(h)
    (y,) = N6._kb4()(inputs=[h, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"],
                             rs_m, rs_l, pidx, seg, nsg_],
                     template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)],
                     grid=(D // (32 * nsg_b) * 32 * nsg_b * ks_b, nseg, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
                     output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D), rs_m, rs_l


def region_offset(seq, nl, L, nreg, pairs, rs):
    """ints from the region array's start to (seq, L)'s entry (ext: ((seq * nl + L) % reg_n) * pairs * rs)"""
    return ((seq * nl + L) % nreg) * pairs * rs


def release_words(slots, seq):
    """the loader's stamped words for slot ids (ext stamp_enc): (seq & 0x3FF) << 22 | (slot & 0x3FFFFF); -1 / -2 kept"""
    out = []
    for s in slots:
        v = (s + 0x400000) if s < 0 else s
        out.append(((seq & 0x3FF) << 22) | (v & 0x3FFFFF))
    return out


# ---------------------------------------------------------------- the 3c form (today's exact default, REF2-T)
# BigMoE's 3c (MoonShots' tierglu, as bigmoe.tier loads it, LP1 patch included when active): kernel A takes its pair
# from pord (resident pairs first, late ones last), resolves a late pair's slots from the release region, writes them
# for kernel B, and computes as before; kernel B with the fused combine (tierglu._BC, unchanged) reads the resolved
# slots. == tier_glu(..., scores) on the released slots == bigmoe.tier._ooo_call's two phases + combine_phase, bitwise.
_ORD = r"""
    // one threadgroup, thread i = pair i: late pairs (base or 4-bit sidecar not resident) after the resident ones,
    // each class in pair order
    const int P = P_C;
    const uint i = thread_position_in_grid.x;
    threadgroup int late_s[P_C];
    if ((int)i < P) {
        const int r = (int)i % NR_C, m = (int)i / NR_C;
        bool late = msb_slot[i] < 0;
        if ((MASK4_C >> r) & 1) late = late || lsb_slot[m * R4_C + popcount((uint)MASK4_C & ((1u << r) - 1u))] == -1;
        late_s[i] = late ? 1 : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)i < P) {
        int nres = 0, before = 0;
        for (int j = 0; j < P; ++j) { nres += 1 - late_s[j]; before += (j < (int)i && late_s[j] == late_s[i]) ? 1 : 0; }
        pord[late_s[i] ? nres + before : before] = (int)i;
    }
"""


@lru_cache(None)
def _kord():
    return mx.fast.metal_kernel(name="ooo1c_ord", input_names=["msb_slot", "lsb_slot"], output_names=["pord"], source=_ORD)


def gpu_order(pre_m, pre_l, plan):
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    ms = pre_m.astype(mx.int32).reshape(-1)
    P = ms.size
    assert P <= 1024, P
    ls = pre_l.astype(mx.int32).reshape(-1) if r4 else mx.zeros((1,), mx.int32)
    (pord,) = _kord()(inputs=[ms, ls], template=[("P_C", P), ("NR_C", R), ("MASK4_C", m4), ("R4_C", max(r4, 1))],
                      grid=(max(32, (P + 31) // 32 * 32), 1, 1), threadgroup=(max(32, (P + 31) // 32 * 32), 1, 1),
                      output_shapes=[(P,)], output_dtypes=[mx.int32])
    return pord


_QOM_OLD = """    threadgroup int rs_s[2];
    if (e_r < 0 || (t4_r && el_r == -1)) {                       // a late pair: its slots from the loader's release
                                                                  // (a sidecar slot -2 is final: LP1 / U3's sure rows)
        if (thread_index_in_threadgroup == 0) {
            device atomic_int* st_ = (device atomic_int*)status;
            int a = e_r, b = el_r;
            if (a < 0) a = oc_poll((coherent(system) device int*)(region + offs[0]), (int)p, offs[2], st_);
            if (t4_r && b == -1) b = oc_poll((coherent(system) device int*)(region + offs[1]), (int)(m * R4_C + i4_r), offs[2], st_);
            rs_s[0] = a; rs_s[1] = b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        e_r = rs_s[0]; el_r = rs_s[1];
    }
    if (threadgroup_position_in_grid.x == 0 && thread_index_in_threadgroup == 0) {
        rs_m[p] = e_r;
        if (t4_r) rs_l[m * R4_C + i4_r] = el_r;
    }"""
_QOM_NEW = """    threadgroup int rs_s[3];
    int kp_r = 1;
    if (e_r < 0 || (t4_r && el_r == -1)) {                       // a late pair: its slots from the loader's release
                                                                  // (a sidecar slot -2 is final: LP1 / U3's sure rows)
        if (thread_index_in_threadgroup == 0) {
            device atomic_int* st_ = (device atomic_int*)status;
            int a = e_r, b = el_r, k_ = 1;
            if (a == -1) a = oc_poll((coherent(system) device int*)(region + offs[0]), (int)p, offs[2], st_);
            if (a < 0) {                                          // QOM: an optional use whose expert the loader did not
                k_ = 0;                                           // read (-2) is dropped (weight 0); it reads its row's
                int a0 = msb_slot[m * NR];                        // rank-0 slot, as the two phases' where(kq, sm, sm[:, :1])
                if (a0 < 0) a0 = oc_poll((coherent(system) device int*)(region + offs[0]), (int)(m * NR), offs[2], st_);
                a = a0;
            }
            if (t4_r && b == -1) b = oc_poll((coherent(system) device int*)(region + offs[1]), (int)(m * R4_C + i4_r), offs[2], st_);
            rs_s[0] = a; rs_s[1] = b; rs_s[2] = k_;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        e_r = rs_s[0]; el_r = rs_s[1]; kp_r = rs_s[2];
    }
    if (threadgroup_position_in_grid.x == 0 && thread_index_in_threadgroup == 0) {
        rs_m[p] = e_r;
        if (t4_r) rs_l[m * R4_C + i4_r] = el_r;
        kp[p] = kp_r;
    }"""


def a3c_one_call_qom(src):
    """QOM-OOO (Wed 17:1x): a3c_one_call's form for QOM (BigMoE r57 _ooo_call_t3 with t["qom"]): a late base slot that
    resolves to -2 (an optional use the loader did not read) is dropped: the pair reads its row's rank-0 slot and its
    keep flag kp[p] = 0 zeroes its combine weight in kernel B (bc_qom), == the two phases' QOM rule, bitwise"""
    s1 = a3c_one_call(src)
    assert s1.count(_QOM_OLD) == 1, "a3c_one_call's resolution block changed"
    return s1.replace(_QOM_OLD, _QOM_NEW)


def bc_qom(src):
    """3c's fused-combine kernel B (tierglu._BC) with the pairs' keep flags: a dropped pair's score reads 0"""
    old = "scores[m * NR + rr]"
    assert src.count(old) == 1, "3c kernel bc's score read changed"
    return src.replace(old, "(kp[m * NR + rr] != 0 ? scores[m * NR + rr] : static_cast<T>(0))")


def a3c_one_call(src):
    """3c kernel A (tierglu._A, LP1-patched or not) -> its one-call form"""
    p_old = "    const uint p = threadgroup_position_in_grid.y;              // pair = m * NR + r"
    mr = "    const uint m = p / NR, r = p % NR;"
    assert src.count(p_old) == 1 and src.count(mr) == 1 and src.count("(uint)msb_slot[p]") == 1
    src = src.replace(p_old, "    const uint p = (uint)pord[threadgroup_position_in_grid.y];  // resident pairs first, late last")
    src = src.replace(mr, mr + """
    const uint i4_r = popcount((uint)MASK4_C & ((1u << r) - 1u));
    const bool t4_r = ((MASK4_C >> r) & 1) != 0;
    int e_r = msb_slot[p];
    int el_r = t4_r ? lsb_slot[m * R4_C + i4_r] : 0;
    threadgroup int rs_s[2];
    if (e_r < 0 || (t4_r && el_r == -1)) {                       // a late pair: its slots from the loader's release
                                                                  // (a sidecar slot -2 is final: LP1 / U3's sure rows)
        if (thread_index_in_threadgroup == 0) {
            device atomic_int* st_ = (device atomic_int*)status;
            int a = e_r, b = el_r;
            if (a < 0) a = oc_poll((coherent(system) device int*)(region + offs[0]), (int)p, offs[2], st_);
            if (t4_r && b == -1) b = oc_poll((coherent(system) device int*)(region + offs[1]), (int)(m * R4_C + i4_r), offs[2], st_);
            rs_s[0] = a; rs_s[1] = b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        e_r = rs_s[0]; el_r = rs_s[1];
    }
    if (threadgroup_position_in_grid.x == 0 && thread_index_in_threadgroup == 0) {
        rs_m[p] = e_r;
        if (t4_r) rs_l[m * R4_C + i4_r] = el_r;
    }""")
    src = src.replace("(uint)msb_slot[p]", "(uint)e_r")
    lp1 = "lsb_slot[m * R4_C + popcount((uint)MASK4_C & ((1u << r) - 1u))]"       # BigMoE's LP1 patch, if active
    src = src.replace(lp1, "el_r")
    old_el = "const uint el = (uint)lsb_slot[m * R4_C + i4];"
    assert src.count(old_el) == 1, "3c kernel A's sidecar read changed"
    src = src.replace(old_el, "const uint el = (uint)el_r;")
    assert "msb_slot[" not in src.replace("int e_r = msb_slot[p];", "") and \
        "lsb_slot[" not in src.replace("lsb_slot[m * R4_C + i4_r]", ""), "a slot read left unresolved"
    return src


_K3C = {}
def _ka3c_1c():
    import bigmoe.tier as BT
    TG = BT._tier_glu_mod()                                       # applies BigMoE's LP1 patch first when it is on
    key = TG._A
    if key not in _K3C:
        _K3C.clear()
        _K3C[key] = mx.fast.metal_kernel(
            name="ooo1c_3c_a" + ("_lp1" if "el_r" in a3c_one_call(TG._A) and "LP1" in TG._A else ""),
            input_names=["x", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                         "msb_slot", "lsb_slot", "pord", "region", "offs", "status"],
            output_names=["h", "rs_m", "rs_l"], source=a3c_one_call(TG._A), header=TG._HDR + SYS)
    return _K3C[key], TG


_K3Q = {}
def _ka3c_1c_qom():
    """QOM-OOO: kernel A (QOM form) + kernel B (bc_qom), keyed by BigMoE's current (LP1-patched or not) sources"""
    import bigmoe.tier as BT
    TG = BT._tier_glu_mod()
    key = (TG._A, TG._BC)
    if key not in _K3Q:
        _K3Q.clear()
        ka = mx.fast.metal_kernel(
            name="ooo1c_3c_aq", input_names=["x", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg",
                                             "bu", "msb_slot", "lsb_slot", "pord", "region", "offs", "status"],
            output_names=["h", "rs_m", "rs_l", "kp"], source=a3c_one_call_qom(TG._A), header=TG._HDR + SYS)
        kb = mx.fast.metal_kernel(
            name="ooo1c_3c_bcq", input_names=["h", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "scores",
                                              "kp"],
            output_names=["y"], source=bc_qom(TG._BC), header=TG._HDR)
        _K3Q[key] = (ka, kb)
    return _K3Q[key], TG


def tier_glu_3c_1c(x, msb, lsb, pre_m, pre_l, plan, region, offs, status, scores, ra=4, rbc=4, ly=8, ld=0, qom=False):
    """tier_glu(..., scores) with the late pairs' slots resolved in kernel A (BigMoE's 3c, REF2-T's kernel):
    -> (y [M, D] combined, rs_m, rs_l). qom=True (QOM-OOO): a late base slot resolving to -2 is a dropped optional use
    (its row's rank-0 slot, weight 0), as BigMoE's QOM two phases"""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    assert m2 == 0
    N = msb["gate_proj"]["s3"].shape[1]
    D = msb["down_proj"]["s3"].shape[1]
    if qom:
        (ka, kbq), TG = _ka3c_1c_qom()
    else:
        ka, TG = _ka3c_1c()
    bb = {p: msb[p]["b3"] for p in PROJ}
    L = lsb if r4 else {p: {"l1": msb[p]["w3"], "b": msb[p]["b3"]} for p in PROJ}
    ms = pre_m.astype(mx.int32).reshape(-1)
    ls = pre_l.astype(mx.int32).reshape(-1) if r4 else mx.zeros((1,), mx.int32)
    pord = gpu_order(pre_m, pre_l, plan)
    g, u, d = (msb[p] for p in PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1))]
    ins_a = [x, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
             L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls, pord, region, offs, status]
    shp_a = [(M * R, N), (M * R,), (max(M * r4, 1),)]
    if qom:
        h, rs_m, rs_l, kp = ka(inputs=ins_a, template=tmpl + [("K_C", K), ("N_C", N), ("RA_C", ra), ("LD_C", ld)],
                               grid=(N // ra * 64, M * R, 1), threadgroup=(64, 1, 1),
                               output_shapes=shp_a + [(M * R,)], output_dtypes=[x.dtype, mx.int32, mx.int32, mx.int32])
        (y,) = kbq(inputs=[h, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"],
                           rs_m, rs_l, scores.astype(x.dtype).reshape(-1), kp],
                   template=tmpl + [("N_C", N), ("D_C", D), ("RB_C", rbc), ("LY_C", ly)],
                   grid=(D // rbc * 32 * R, M, 1), threadgroup=(32 * R, 1, 1),
                   output_shapes=[(M, D)], output_dtypes=[x.dtype])
        return y, rs_m, rs_l
    h, rs_m, rs_l = ka(inputs=ins_a,
                       template=tmpl + [("K_C", K), ("N_C", N), ("RA_C", ra), ("LD_C", ld)],
                       grid=(N // ra * 64, M * R, 1), threadgroup=(64, 1, 1),
                       output_shapes=shp_a, output_dtypes=[x.dtype, mx.int32, mx.int32])
    (y,) = TG._kbc()(inputs=[h, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"],
                             rs_m, rs_l, scores.astype(x.dtype).reshape(-1)],
                     template=tmpl + [("N_C", N), ("D_C", D), ("RB_C", rbc), ("LY_C", ly)],
                     grid=(D // rbc * 32 * R, M, 1), threadgroup=(32 * R, 1, 1),
                     output_shapes=[(M, D)], output_dtypes=[x.dtype])
    return y, rs_m, rs_l


# ---------------------------------------------------------------- native forms (Friday's converted store: native+p11+pad)
# The same splices on NAX6-N's kernel A (bf16 native / p11 sources) and on 3c-N's kernel A (native / native + p11);
# kernel B is each family's own (NAX6-N's B, 3c-N's fused-combine bc) reading the resolved slots. Bitwise == the native
# two phases, which are == the row-major ones (NATIVE-3C / P11PAD / NATIVE-FRI step 1).
_KN = {}


def _ka_nax6n_1c(p11):
    import tierglu_nax6n as NN
    import p11 as P11
    key = ("nax6n", p11)
    if key not in _KN:
        _KN[key] = mx.fast.metal_kernel(
            name=f"ooo1c_nax6n{'p' if p11 else ''}_a",
            input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                         "msb_slot", "lsb_slot", "pidx", "seg", "nseg", "region", "offs", "status"],
            output_names=["h", "rs_m", "rs_l"], source=a_one_call(NN._A_NP if p11 else NN._A_N),
            header=N6._HDR4 + (P11.P11_METAL if p11 else "") + SYS)
    return _KN[key]


def tier_glu_nax6n_1c(x, msb_n, lsb_n, pre_m, pre_l, ids, plan, region, offs, status, xs=None, grp=None, nsg_b=2, ks=4,
                      ks_b=2):
    """tier_glu_nax6n (native / p11 pools) with the late pairs resolved in kernel A: (y [M, R, D], rs_m, rs_l)"""
    import tierglu_nax6n as NN
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
        grp = gpu_grp_1c(pre_m, pre_l, ids, plan)
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
    h, rs_m, rs_l = _ka_nax6n_1c(p11)(
        inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls, pidx, seg, nsg_,
                region, offs, status],
        template=tmpl + [("K_C", K), ("N_C", N), ("KS_C", ks)], grid=(N // 32 * 64 * ks, nseg, 1),
        threadgroup=(64 * ks, 1, 1), output_shapes=[(M * R, N), (M * R,), (max(M * r4, 1),)],
        output_dtypes=[x.dtype, mx.int32, mx.int32])
    hs = rowsum64(h)
    (y,) = (NN._kbn_p11 if p11 else NN._kbn)()(
        inputs=[h, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"],
                rs_m, rs_l, pidx, seg, nsg_],
        template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)],
        grid=(D // (32 * nsg_b) * 32 * nsg_b * ks_b, nseg, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
        output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D), rs_m, rs_l


def _ka3cn_1c(p11):
    import hashlib
    import tierglu_3cn as C3
    import p11 as P11
    C3._sync()                                                    # follows BigMoE's live hooks (LP1) like 3c-N does
    src = a3c_one_call(C3._SRC["a"][2 if p11 else 1])
    hdr = C3.TG._HDR + (P11.P11_METAL if p11 else "") + SYS
    key = hashlib.sha1((hdr + src).encode()).hexdigest()[:10]
    if key not in _KN:
        _KN[key] = mx.fast.metal_kernel(name=f"ooo1c_3cn{'p' if p11 else ''}_a_{key}",
                                        input_names=C3._IN_A + ["pord", "region", "offs", "status"],
                                        output_names=["h", "rs_m", "rs_l"], source=src, header=hdr)
    return _KN[key]


def _ka3cn_1c_qom(p11):
    """QOM-OOO on 3c-N (Friday's converted store): kernel A in the QOM form + 3c-N's bc with the keep flags"""
    import hashlib
    import tierglu_3cn as C3
    import p11 as P11
    C3._sync()
    src_a = a3c_one_call_qom(C3._SRC["a"][2 if p11 else 1])
    src_b = bc_qom(C3._SRC["bc"][2 if p11 else 1])
    hdr_b = C3.TG._HDR + (P11.P11_METAL if p11 else "")
    hdr_a = hdr_b + SYS
    key = hashlib.sha1((hdr_a + src_a + src_b).encode()).hexdigest()[:10]
    if key not in _KN:
        _KN[key] = (mx.fast.metal_kernel(name=f"ooo1c_3cn{'p' if p11 else ''}_aq_{key}",
                                         input_names=C3._IN_A + ["pord", "region", "offs", "status"],
                                         output_names=["h", "rs_m", "rs_l", "kp"], source=src_a, header=hdr_a),
                    mx.fast.metal_kernel(name=f"ooo1c_3cn{'p' if p11 else ''}_bcq_{key}",
                                         input_names=C3._IN_B + ["scores", "kp"], output_names=["y"], source=src_b,
                                         header=hdr_b))
    return _KN[key]


def tier_glu_3cn_1c(x, msb_n, lsb_n, pre_m, pre_l, plan, region, offs, status, scores, ra=4, rbc=4, ly=8, ld=0, qom=False):
    """3c-N's tier_glu_n(..., scores) with the late pairs resolved in kernel A: (y [M, D] combined, rs_m, rs_l).
    qom=True: QOM-OOO's drops (a late base slot resolving to -2: the row's rank-0 slot, weight 0)"""
    import tierglu_3cn as C3
    M, K = x.shape
    m4, m2, r4 = plan_masks(plan)
    assert m2 == 0
    N, D, p11 = C3._dims(msb_n)
    R, ina, inb, tmpl = C3._common(x, msb_n, lsb_n, pre_m, pre_l if r4 else None, plan, None)
    pord = gpu_order(pre_m, pre_l, plan)
    if qom:
        ka, kb = _ka3cn_1c_qom(p11)
        h, rs_m, rs_l, kp = ka(inputs=ina + [pord, region, offs, status],
                               template=tmpl + [("K_C", K), ("N_C", N), ("RA_C", ra), ("LD_C", ld)],
                               grid=(N // ra * 64, M * R, 1), threadgroup=(64, 1, 1),
                               output_shapes=[(M * R, N), (M * R,), (max(M * r4, 1),), (M * R,)],
                               output_dtypes=[x.dtype, mx.int32, mx.int32, mx.int32])
        (y,) = kb(inputs=[h] + inb[:-2] + [rs_m, rs_l if r4 else inb[-1], scores.astype(x.dtype).reshape(-1), kp],
                  template=tmpl + [("N_C", N), ("D_C", D), ("RB_C", rbc), ("LY_C", ly)],
                  grid=(D // rbc * 32 * R, M, 1), threadgroup=(32 * R, 1, 1),
                  output_shapes=[(M, D)], output_dtypes=[x.dtype])
        return y, rs_m, rs_l
    h, rs_m, rs_l = _ka3cn_1c(p11)(inputs=ina + [pord, region, offs, status],
                                   template=tmpl + [("K_C", K), ("N_C", N), ("RA_C", ra), ("LD_C", ld)],
                                   grid=(N // ra * 64, M * R, 1), threadgroup=(64, 1, 1),
                                   output_shapes=[(M * R, N), (M * R,), (max(M * r4, 1),)],
                                   output_dtypes=[x.dtype, mx.int32, mx.int32])
    (y,) = C3._kern("bc", p11)(inputs=[h] + inb[:-2] + [rs_m, rs_l if r4 else inb[-1], scores.astype(x.dtype).reshape(-1)],
                               template=tmpl + [("N_C", N), ("D_C", D), ("RB_C", rbc), ("LY_C", ly)],
                               grid=(D // rbc * 32 * R, M, 1), threadgroup=(32 * R, 1, 1),
                               output_shapes=[(M, D)], output_dtypes=[x.dtype])
    return y, rs_m, rs_l
