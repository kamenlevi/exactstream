"""msh.py (Breakthrough, Tue 17 Nov 2026 12:5x; MOE BLOCK candidate F1): the routed expert kernels (tgl16eu: the
threadgroup-staged MMA, the analytic epilogue, the tier-4 upgrade's grouping) with the SHARED EXPERT folded in as extra
segments. Threadgroups with grid y >= PR_C (the routed segment capacity) take the shared path:
  - rows: the call's M rows in 16-row tiles (tile t: rows 16 t ..), no head or polling (shared weights are always
    resident);
  - weights: the model's own quantized linear format (config: MLX affine, bits 4, group size 64). The codes are already
    nibbles in order, so a lane copies its output row's 8 words per quant group to threadgroup memory: no spread decode.
    Scales / biases are plain bf16;
  - the same MMA (bf16 x uint4b -> float, 2 per quant group), epilogue (acc += s cg + b xs), K chunks, partial sums and
    swiglu (A) / store (B) as the routed path;
  - h / y gain M rows after the routed M x K rows (shared row m at M K + m), so rowsum64(h) covers them.
The combine is unchanged: combine_phase on the routed rows (exact), then one compiled elementwise op adds
y_shared * sigmoid(shared_gate(x)) (the gate stays MLX's tiny quantized matmul). The shared expert's numerics are FAST
(the accelerator's MMA vs MLX's quantized matmul).
Template constraints: shared inter == routed expert inter (true for this model; lift by giving the shared path its own
n-tile count); group size 64; the shared format MLX affine 4-bit."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
XRW = os.path.expanduser(os.environ.get("MOEBLK_KNOB", "~/Breakthrough/data/xrw/knob_v2"))   # the FROZEN routed kernels
for _p in (HERE, XRW):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import moeblk as MB                                         # noqa: E402
import w4a8 as W                                            # noqa: E402
import tgl16 as G                                           # noqa: E402
import tgl16v as V                                          # noqa: E402

mx = MB.mx
_K = {}
_TG_DECL = "    threadgroup uint32_t tg_all[2 * KS_C * SGW_C];               // per-simdgroup slices; later the chunk partials\n"
_TG_DECL_B = "    threadgroup uint32_t tg_all[NSG_C * KS_C * SGW_C];\n"

# The shared path's core, one simdgroup's K chunk of one projection / column tile: like tgl16e's core, with the code
# tile copied (MLX 4-bit: 8 words per row and quant group) instead of decoded, and bf16 scales / biases.
_SH_CORE = r"""
        constexpr int SKG = SK_ / 64;
        const device uint32_t* swrow = SW_ + (size_t)(sn0 + lane) * (SK_ / 8);
        const device T* ssrow = SS_ + (size_t)(sn0 + lane) * SKG;
        const device T* sbrow = SB_ + (size_t)(sn0 + lane) * SKG;
        threadgroup uint32_t* tbw = tg_all + sg * SGW_C;
        threadgroup T* taw = (threadgroup T*)(tbw + 256);
        threadgroup float* sbw = (threadgroup float*)(tbw + 768);
        threadgroup float* rxw = (threadgroup float*)(tbw + 832);
        constexpr int GPC = SKG / SKS_;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        using TA = tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline>;
        using TB = tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline>;
        TA ta0(taw, dextents<int32_t, 2>(32, 16), array<int32_t, 2>{1, 64});
        TA ta1(taw + 32, dextents<int32_t, 2>(32, 16), array<int32_t, 2>{1, 64});
        TB tb0((threadgroup uchar*)tbw, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
        TB tb1((threadgroup uchar*)tbw + 16, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
        auto cg = op.template get_destination_cooperative_tensor<TA, TB, float>();
        const short qid = lane >> 2;
        const short fm0 = (qid & 4) | ((lane >> 1) & 3);
        const short fn0 = ((qid & 2) | (lane & 1)) * 4;
        float acc[16];
        for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
        const int sr = lane >> 1, shf = lane & 1, rr = lane & 15;
        const uint xr_s = srow0 + (uint)(sr < snr ? sr : 0), xr_r = srow0 + (uint)(rr < snr ? rr : 0);
        const device T* xb_s = SX_ + (size_t)xr_s * SK_ + 32 * shf;
        const device float* rx_src = SXS_ + (size_t)xr_r * SKG;
        for (int g = skc * GPC; g < (skc + 1) * GPC; ++g) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const device uint4* wsrc = (const device uint4*)(swrow + 8 * g);
            const uint4 c0 = wsrc[0], c1 = wsrc[1];
            const device uint4* xsrc = (const device uint4*)(xb_s + 64 * g);
            const uint4 xa = xsrc[0], xb = xsrc[1], xc = xsrc[2], xd4 = xsrc[3];
            const float rxv = rx_src[g];
            threadgroup uint4* dst = (threadgroup uint4*)(tbw + lane * 8);
            dst[0] = c0;
            dst[1] = c1;
            threadgroup uint4* xd = (threadgroup uint4*)(taw + sr * 64 + 32 * shf);
            xd[0] = xa; xd[1] = xb; xd[2] = xc; xd[3] = xd4;
            sbw[lane] = (float)ssrow[g];
            sbw[32 + lane] = (float)sbrow[g];
            if (lane < 16) rxw[lane] = rxv;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = 0; i < 16; ++i) cg[i] = 0.0f;
            op.run(ta0, tb0, cg);
            op.run(ta1, tb1, cg);
            const float4 s0 = *(threadgroup const float4*)(sbw + fn0), s1 = *(threadgroup const float4*)(sbw + 16 + fn0);
            const float4 b0 = *(threadgroup const float4*)(sbw + 32 + fn0), b1 = *(threadgroup const float4*)(sbw + 48 + fn0);
            const float x0 = rxw[fm0], x1 = rxw[fm0 + 8];
            for (int i = 0; i < 16; ++i) {
                const float sv_ = (i >> 3) ? s1[i & 3] : s0[i & 3];
                const float bv_ = (i >> 3) ? b1[i & 3] : b0[i & 3];
                acc[i] += sv_ * cg[i] + bv_ * (((i >> 2) & 1) ? x1 : x0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* part = (threadgroup float*)tg_all;
        for (int i = 0; i < 16; ++i)
            part[(SPI_ * SKS_ + skc) * 512 + (fm0 + 8 * ((i >> 2) & 1)) * 32 + 16 * (i >> 3) + fn0 + (i & 3)] = acc[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
"""

_SH_A = r"""
    {   // ---- the SHARED expert's segments (grid y >= PR_C): gate | up + swiglu into h's rows M R + m
        const uint gq_ = threadgroup_position_in_grid.y;
        if (gq_ >= (uint)PR_C) {
            const ushort lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint spj = sg & 1u, skc = sg >> 1;
            const uint st = gq_ - (uint)PR_C;
            const uint srow0 = st * 16;
            const int snr = min(16, M_C - (int)srow0);
            const uint sn0 = threadgroup_position_in_grid.x * 32;
            const device uint32_t* SW_ = spj == 0 ? swg : swu;
            const device T* SS_ = spj == 0 ? ssg : ssu;
            const device T* SB_ = spj == 0 ? sbg : sbu;
#define SK_ K_C
#define SKS_ KS_C
#define SX_ x
#define SXS_ xs
#define SPI_ spj
""" + _SH_CORE + r"""
#undef SK_
#undef SKS_
#undef SX_
#undef SXS_
#undef SPI_
            for (uint t = thread_index_in_threadgroup; t < 512u; t += 2 * KS_C * 32) {
                const int row = (int)(t >> 5), col = (int)(t & 31);
                float gs = part[row * 32 + col], us = part[KS_C * 512 + row * 32 + col];
                for (int c = 1; c < KS_C; ++c) { gs += part[c * 512 + row * 32 + col]; us += part[(KS_C + c) * 512 + row * 32 + col]; }
                if (row < snr) {
                    const T gv = static_cast<T>(gs);
                    const T ub = static_cast<T>(us);
                    h[(size_t)(MR_C + srow0 + row) * N_C + sn0 + col] = TgMultiply()(TgMultiply()(gv, TgSigmoid()(gv)), ub);
                }
            }
            return;
        }
    }
"""

_SH_B = r"""
    {   // ---- the SHARED expert's segments (grid y >= PR_C): down into y's rows M R + m
        const uint gq_ = threadgroup_position_in_grid.y;
        if (gq_ >= (uint)PR_C) {
            const ushort lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint sti = sg % NSG_C, skc = sg / NSG_C;
            const uint st = gq_ - (uint)PR_C;
            const uint srow0 = st * 16;
            const int snr = min(16, M_C - (int)srow0);
            const uint sn0 = (threadgroup_position_in_grid.x * NSG_C + sti) * 32;
            const device uint32_t* SW_ = swd;
            const device T* SS_ = ssd;
            const device T* SB_ = sbd;
#define SK_ N_C
#define SKS_ KS_C
#define SX_ (h + (size_t)MR_C * N_C)
#define SXS_ (hs + (size_t)MR_C * (N_C / 64))
#define SPI_ sti
""" + _SH_CORE + r"""
#undef SK_
#undef SKS_
#undef SX_
#undef SXS_
#undef SPI_
            for (uint t = thread_index_in_threadgroup; t < (uint)(NSG_C * 512); t += NSG_C * KS_C * 32) {
                const int tt = (int)(t >> 9), row = (int)((t >> 5) & 15), col = (int)(t & 31);
                float ys = part[(tt * KS_C) * 512 + row * 32 + col];
                for (int c = 1; c < KS_C; ++c) ys += part[(tt * KS_C + c) * 512 + row * 32 + col];
                if (row < snr) y[(size_t)(MR_C + srow0 + row) * D_C + (threadgroup_position_in_grid.x * NSG_C + tt) * 32 + col] = static_cast<T>(ys);
            }
            return;
        }
    }
"""


def sources():
    """kernel A / B: the shared block + tgl16eu's routed kernels (head transform, rs_l fix, coherent reads)"""
    import ooo1c as OC
    import ooo1cq as OQ
    import coh1cq as CQ
    a_routed = W._PRE_A1 + OC._OLD_HEAD + "\n" + G._PRE_A2 + V._core_e() + W._POST_A
    assert a_routed.count(_TG_DECL) == 1
    a_routed = a_routed.replace(_TG_DECL, "")
    a = OQ.a_one_call_q(a_routed)
    assert a.count(V._RSL_OLD) == 1
    a = a.replace(V._RSL_OLD, V._RSL_NEW)
    a = _TG_DECL + _SH_A + a
    b_routed = W._PRE_B1.replace(G._PRE_B_STAGE_OLD, G._PRE_B_STAGE_NEW) + V._core_e() + W._POST_B
    assert b_routed.count(_TG_DECL_B) == 1
    b_routed = b_routed.replace(_TG_DECL_B, "")
    b = _TG_DECL_B + _SH_B + OQ.b_drop(b_routed)
    return CQ._ren(CQ._body(a)), CQ._ren(CQ._body(b))


def _ka():
    if "a" not in _K:
        a, _ = sources()
        _K["a"] = mx.fast.metal_kernel(
            name="msh_a",
            input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                         "msb_slot", "lsb_slot", "pidx", "seg", "nseg", "region", "offs", "status",
                         "swg", "swu", "ssg", "ssu", "sbg", "sbu"],
            output_names=["h", "rs_m", "rs_l"], source=a, header=W._hdr())
    return _K["a"]


def _kb():
    if "b" not in _K:
        _, b = sources()
        _K["b"] = mx.fast.metal_kernel(
            name="msh_b",
            input_names=["h", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg", "nseg",
                         "swd", "ssd", "sbd"],
            output_names=["y"], source=b, header=W._hdr())
    return _K["b"]


def experts_and_shared(lay, x, pm, pl, idn, xs=None, grp=None):
    """routed experts (tgl16eu) + the shared expert in the same two kernels -> (y [M K + M, D], rs_m, rs_l)"""
    import xgrp
    from tierglu_nax import rowsum64
    c = lay.c
    assert c["shared_inter"] == c["expert_inter"] and c["shared"]["bits"] == 4 and c["group_size"] == 64
    plan = c["routed_store"]["plan"]
    ids = idn.astype(mx.int32)
    if grp is None:
        grp = xgrp.gpu_grp_up2(pm, pl, ids, plan)
    M, K, R, r4, N, D, grp, bb, ls, ms, tmpl, PROJ = V._common(x, lay.base, lay.side, pm, pl, ids, plan, grp, 4, 2, 2)
    pidx, seg, nsg_, P = grp
    g, u, d = (lay.base[p] for p in PROJ)
    L_ = lay.side
    Wt = lay.W
    gp, up, dp = c["shared"]["tensors"]
    NSH = (M + 15) // 16
    xs = rowsum64(x) if xs is None else xs
    tm = tmpl + [("SGW_C", G.SGW), ("PR_C", P), ("MR_C", M * R), ("M_C", M)]
    h, rs_m, rs_l = _ka()(
        inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                L_["gate_proj"]["l1"], L_["up_proj"]["l1"], L_["gate_proj"]["b"], L_["up_proj"]["b"], ms, ls, pidx, seg, nsg_,
                lay.region, lay.offs, lay.status,
                Wt[gp + ".weight"], Wt[up + ".weight"], Wt[gp + ".scales"], Wt[up + ".scales"], Wt[gp + ".biases"],
                Wt[up + ".biases"]],
        template=tm + [("K_C", x.shape[1]), ("N_C", N), ("KS_C", 4)], grid=(N // 32 * 64 * 4, P + NSH, 1),
        threadgroup=(64 * 4, 1, 1), output_shapes=[(M * R + M, N), (M * R,), (max(M * r4, 1),)],
        output_dtypes=[x.dtype, mx.int32, mx.int32])
    hs = rowsum64(h)
    (y,) = _kb()(
        inputs=[h, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L_["down_proj"]["l1"], L_["down_proj"]["b"],
                rs_m, rs_l, pidx, seg, nsg_, Wt[dp + ".weight"], Wt[dp + ".scales"], Wt[dp + ".biases"]],
        template=tm + [("N_C", N), ("D_C", D), ("NSG_C", 2), ("KS_C", 2)],
        grid=(D // 64 * 32 * 2 * 2, P + NSH, 1), threadgroup=(32 * 2 * 2, 1, 1),
        output_shapes=[(M * R + M, D)], output_dtypes=[x.dtype])
    return y, rs_m, rs_l


@mx.compile
def _gated_add(yc, ysh, sg):
    return yc + ysh * mx.sigmoid(sg)


def block_f1(lay, x):
    """the F1 block: exact router + glue + slots, then the routed experts and the shared expert in two kernels, then
    combine_phase (exact) and one compiled gated add"""
    import bigmoe.tier as BT
    c = lay.c
    M, K = x.shape[0], c["top_k"]
    ids, scores = MB.route(lay, x)
    sc, idn, nl = MB.glue(lay, ids, scores)
    pm, pl = MB.slots(lay, idn, nl)
    y, _, _ = experts_and_shared(lay, x, pm, pl, idn)
    TG = BT._tier_glu_mod()
    yr = y[:M * K]
    yc = TG.combine_phase(yr, yr, mx.ones((M, K), mx.int32), sc.reshape(-1, K))
    sg = lay.qmm(x, c["shared"]["gate_tensor"], c["shared"]["gate_bits"])
    return _gated_add(yc, y[M * K:], sg)


if __name__ == "__main__":
    a, b = sources()
    print(f"msh sources: A {len(a)} chars, B {len(b)} chars; coherent pointers {a.count('coherent(system) const device')} / "
          f"{b.count('coherent(system) const device')}; head polls {a.count('oc_poll(')}; drop {b.count('FAST-NAX: a dropped group')}")
