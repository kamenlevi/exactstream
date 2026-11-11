"""Expert-major ("grouped") form of MoonShots' 3c fused GLU (tierglu.tier_glu, the kernels BigMoE v3 runs): one threadgroup
per (output tile, DISTINCT expert) loops over that expert's (row, rank) pairs, so each weight tile is read from DRAM once
per pass and the repeats hit the threadgroup's L1. Each pair's arithmetic is kernel A / B's, unchanged, and the combine is
MLX's LY=8 order (tierglu.combine_dh with every pair in phase 1). So tier_glu_grouped == tier_glu(..., scores=) BIT FOR BIT.
Pairs are grouped on the GPU: one tiny kernel (one threadgroup, P <= 1024 pairs) ranks the pairs by (expert, pair index)
and writes gpairs [P] (pair indices, expert-major) and grp_off [P + 1] (group g = gpairs[grp_off[g] : grp_off[g + 1]];
groups past the number of distinct experts are empty, and their threadgroups return at once).
Dispatches: csr + A' + B' + combine (vs 2 for tier_glu with the fused combine)."""
import os, sys
from functools import lru_cache
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/MoonShots/defer/v3d"))
import tierglu as TG

_AG = TG._A.replace("    const uint p = threadgroup_position_in_grid.y;              // pair = m * NR + r\n",
                    "    const uint g_ = threadgroup_position_in_grid.y;             // distinct-expert group\n"
                    "    threadgroup T gate_v[RA];\n"
                    "    for (int gi_ = grp_off[g_]; gi_ < grp_off[g_ + 1]; ++gi_) {\n"
                    "    const uint p = (uint)gpairs[gi_];                           // pair = m * NR + r\n") \
           .replace("    threadgroup T gate_v[RA];\n    for (int row = 0; row < RA; ++row) result[row] = simd_sum(result[row]);",
                    "    for (int row = 0; row < RA; ++row) result[row] = simd_sum(result[row]);") \
      + "    threadgroup_barrier(mem_flags::mem_threadgroup);\n    }\n"
assert _AG.count("threadgroup T gate_v[RA];") == 1 and "gpairs[gi_]" in _AG
_BG = TG._B.replace("    const uint p = threadgroup_position_in_grid.y;\n", "    const uint g_ = threadgroup_position_in_grid.y;\n") \
           .replace("    if (d0 >= (uint)D) return;\n",
                    "    if (d0 >= (uint)D) return;\n"
                    "    for (int gi_ = grp_off[g_]; gi_ < grp_off[g_ + 1]; ++gi_) {\n"
                    "    const uint p = (uint)gpairs[gi_];\n") \
      + "    }\n"
assert "gpairs[gi_]" in _BG and "const uint p = threadgroup_position_in_grid.y" not in _BG

_CSR = r"""
    // one threadgroup, thread i = pair i: rank by (expert, i); group id = number of distinct experts below e_i
    const uint i = thread_position_in_grid.x;
    const int P = (int)P_C;
    threadgroup int ids_s[P_C];
    threadgroup int first_s[P_C];
    if ((int)i < P) ids_s[i] = ids[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int e = 0;
    bool first = true;
    if ((int)i < P) {
        e = ids_s[i];
        for (int j = 0; j < (int)i; ++j) first = first && (ids_s[j] != e);
        first_s[i] = first ? 1 : 0;
    }
    for (int t = (int)i; t <= P; t += (int)threads_per_threadgroup.x) grp_off[t] = P;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if ((int)i < P) {
        int rank = 0, gid = 0;
        for (int j = 0; j < P; ++j) {
            const int f = ids_s[j];
            rank += (f < e) || (f == e && j < (int)i);
            gid += (f < e) && first_s[j];
        }
        gpairs[rank] = (int)i;
        if (first) grp_off[gid] = rank;
    }
"""


@lru_cache(None)
def _kag():
    return mx.fast.metal_kernel(
        name="tierglu_ag",
        input_names=["x", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "grp_off", "gpairs"],
        output_names=["h"], source=_AG, header=TG._HDR)


@lru_cache(None)
def _kbg():
    return mx.fast.metal_kernel(
        name="tierglu_bg",
        input_names=["h", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "grp_off", "gpairs"],
        output_names=["y"], source=_BG, header=TG._HDR)


@lru_cache(None)
def _kcsr():
    return mx.fast.metal_kernel(name="tierglu_csr", input_names=["ids"], output_names=["gpairs", "grp_off"], source=_CSR)


def csr(ids):
    """ids [P] int32 -> gpairs [P] (pair indices grouped by expert), grp_off [P + 1]"""
    P = ids.size
    assert P <= 1024, P
    gp, go = _kcsr()(inputs=[ids.astype(mx.int32).reshape(-1)], template=[("P_C", P)], grid=(max(P + 1, 32), 1, 1),
                     threadgroup=(max(P + 1, 32), 1, 1), output_shapes=[(P,), (P + 1,)], output_dtypes=[mx.int32, mx.int32])
    return gp, go


def tier_glu_grouped(x, msb, lsb, msb_slot, lsb_slot, scores, plan="4433333333", b2=None, ra=4, rb=4, nsg_b=2, ld=0,
                     groups=None):
    """== tierglu.tier_glu(x, ..., plan, scores=scores) bitwise; expert-major. groups = (gpairs, grp_off) to skip the csr"""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = TG.plan_masks(plan)
    N = msb["gate_proj"]["s3"].shape[1]
    D = msb["down_proj"]["s3"].shape[1]
    bb = b2 if b2 is not None else {p: msb[p]["b3"] for p in TG.PROJ}
    if r4:
        ls = lsb_slot.astype(mx.int32).reshape(-1); L = lsb
    else:
        ls = mx.zeros((1,), mx.int32); L = {p: {"l1": msb[p]["w3"], "b": msb[p]["b3"]} for p in TG.PROJ}
    ms = msb_slot.astype(mx.int32).reshape(-1)
    gp, go = groups if groups is not None else csr(ms)
    P = M * R
    g, u, d = (msb[p] for p in TG.PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1))]
    (h,) = _kag()(inputs=[x, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                          L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls, go, gp],
                  template=tmpl + [("K_C", K), ("N_C", N), ("RA_C", ra), ("LD_C", ld)],
                  grid=(N // ra * 64, P, 1), threadgroup=(64, 1, 1), output_shapes=[(P, N)], output_dtypes=[x.dtype])
    (y,) = _kbg()(inputs=[h, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"], ms, ls,
                          go, gp],
                  template=tmpl + [("N_C", N), ("D_C", D), ("RB_C", rb)],
                  grid=(D // (rb * nsg_b) * 32 * nsg_b, P, 1), threadgroup=(32 * nsg_b, 1, 1),
                  output_shapes=[(P, D)], output_dtypes=[x.dtype])
    return TG.combine_dh(y, y, mx.ones((M, R), mx.int32), scores.reshape(M, R), 0)


def compile_graph():
    """for patches/sol_layer/compile_check.py (offline xcrun metal of every generated kernel; no GPU work)"""
    E, M, R = 4, 3, 10
    SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
    msb = {p: {"w3": mx.zeros((E, o, i * 3 // 32), mx.uint32), "s3": mx.zeros((E, o, i // 64), mx.bfloat16),
               "b3": mx.zeros((E, o, i // 64), mx.bfloat16)} for p, (o, i) in SH.items()}
    lsb = {p: {"l1": mx.zeros((E, o, i // 32), mx.uint32), "b": mx.zeros((E, o, i // 64), mx.bfloat16)} for p, (o, i) in SH.items()}
    x = mx.zeros((M, 2560), mx.bfloat16)
    ids = mx.zeros((M, R), mx.int32)
    sc = mx.zeros((M, R), mx.bfloat16)
    tier_glu_grouped(x, msb, lsb, ids, ids[:, :2], sc, plan="4433333333")
    tier_glu_grouped(x, msb, None, ids, None, sc, plan="3333333333")
