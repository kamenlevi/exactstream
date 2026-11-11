"""G2-3CN (Leap, Sun 2026-11-08 01:2x; CHAT-150 G2, FAST): 3c-N's expert one-call for L's 1-2-row chat calls. G2-PRICE
(data/g2_bench.txt, quiet) put MoonShots' 3c on the converted store (GEMV per (row, rank) pair) at -12 % / -6 % vs L's
NAX6-N FAST one-call (ooo1cq) at 1 / 2 rows. ooo1c already has 3c-N one-call kernels (tier_glu_3cn_1c, QOM form too), but
L runs U3: a sure row's 4-bit ranks carry sidecar slot -2 (run at tier 3, no LSB read), which 3c-N handles only under
BigMoE's LP1 kernel patch, and LP1's env (BIGMOE_POM_LP1=1) also switches the loader's sidecar mode (set_lp1). Here: the
LP1 kernel line alone ("a 4-bit pair whose sidecar slot is < 0 runs at tier 3"), applied to private copies of tierglu's A
and BC sources, then 3c-N's native / p11 swaps and ooo1c's one-call forms (plain and QOM). TG / C3 stay untouched.
tier_glu_3cn_1c_u3(...) == ooo1c.tier_glu_3cn_1c's signature and returns (y [M, D] combined, rs_m, rs_l)."""
import hashlib

import mlx.core as mx

_LP1_OLD = "const int tier = ((MASK4_C >> r) & 1) ? 4 : (((MASK2_C >> r) & 1) ? 2 : 3);"
_LP1_NEW = ("int tier = ((MASK4_C >> r) & 1) ? 4 : (((MASK2_C >> r) & 1) ? 2 : 3);\n"
            "    if (tier == 4 && lsb_slot[m * R4_C + popcount((uint)MASK4_C & ((1u << r) - 1u))] < 0) tier = 3;   // LP1")
_KN = {}


def _lp1(src):
    if "// LP1" in src:                                  # BigMoE's LP1 patch already on (same line)
        return src
    assert src.count(_LP1_OLD) == 1, "tierglu's tier line changed"
    return src.replace(_LP1_OLD, _LP1_NEW)


def _srcs(p11):
    import tierglu_3cn as C3
    TG = C3.TG
    a = C3._swap(_lp1(TG._A), C3._A_PAIRS)
    bc = C3._swap(_lp1(TG._BC), C3._B_PAIRS)
    if p11:
        a, bc = C3._swap(a, C3._A_P11), C3._swap(bc, C3._B_P11)
    return a, bc


def _kernels(p11, qom):
    import tierglu_3cn as C3
    import p11 as P11
    import ooo1c as OC
    a, bc = _srcs(p11)
    src_a = OC.a3c_one_call_qom(a) if qom else OC.a3c_one_call(a)
    src_b = OC.bc_qom(bc) if qom else bc
    hdr_b = C3.TG._HDR + (P11.P11_METAL if p11 else "")
    hdr_a = hdr_b + OC.SYS
    key = hashlib.sha1((hdr_a + src_a + src_b).encode()).hexdigest()[:10]
    if key not in _KN:
        tag = f"{'p' if p11 else ''}{'q' if qom else ''}_{key}"
        outs_a = ["h", "rs_m", "rs_l"] + (["kp"] if qom else [])
        _KN[key] = (mx.fast.metal_kernel(name=f"g2u3_3cn_a{tag}", input_names=C3._IN_A + ["pord", "region", "offs", "status"],
                                         output_names=outs_a, source=src_a, header=hdr_a),
                    mx.fast.metal_kernel(name=f"g2u3_3cn_bc{tag}", input_names=C3._IN_B + ["scores"] + (["kp"] if qom else []),
                                         output_names=["y"], source=src_b, header=hdr_b))
    return _KN[key]


def tier_glu_3cn_1c_u3(x, msb_n, lsb_n, pre_m, pre_l, plan, region, offs, status, scores, ra=4, rbc=4, ly=8, ld=0,
                       qom=False):
    """ooo1c.tier_glu_3cn_1c with the LP1 tier-3 fallback in both kernels (U3's -2 sidecar slots): (y [M, D], rs_m, rs_l)"""
    import tierglu_3cn as C3
    import ooo1c as OC
    from tierglu import plan_masks
    M, K = x.shape
    m4, m2, r4 = plan_masks(plan)
    assert m2 == 0
    N, D, p11 = C3._dims(msb_n)
    R, ina, inb, tmpl = C3._common(x, msb_n, lsb_n, pre_m, pre_l if r4 else None, plan, None)
    pord = OC.gpu_order(pre_m, pre_l, plan)
    ka, kb = _kernels(p11, qom)
    nouts = 4 if qom else 3
    outs = ka(inputs=ina + [pord, region, offs, status], template=tmpl + [("K_C", K), ("N_C", N), ("RA_C", ra), ("LD_C", ld)],
              grid=(N // ra * 64, M * R, 1), threadgroup=(64, 1, 1),
              output_shapes=[(M * R, N), (M * R,), (max(M * r4, 1),)] + ([(M * R,)] if qom else []),
              output_dtypes=[x.dtype, mx.int32, mx.int32] + ([mx.int32] if qom else []))
    h, rs_m, rs_l = outs[:3]
    assert len(outs) == nouts
    (y,) = kb(inputs=[h] + inb[:-2] + [rs_m, rs_l if r4 else inb[-1], scores.astype(x.dtype).reshape(-1)] + ([outs[3]] if qom else []),
              template=tmpl + [("N_C", N), ("D_C", D), ("RB_C", rbc), ("LY_C", ly)],
              grid=(D // rbc * 32 * R, M, 1), threadgroup=(32 * R, 1, 1), output_shapes=[(M, D)], output_dtypes=[x.dtype])
    return y, rs_m, rs_l
