"""WEEK98 MODULE 2 (MoonShots): candidate forms of ooo1cq's one-call kernels A and B, as source transforms of Leap's
own kernel text (ooo1cq.a_one_call_q(tierglu_nax6n._A_NP) / ooo1cq.b_drop(tierglu_nax6n._B_NP)). Every form keeps each
output's arithmetic and its order, so it must equal ooo1cq bit for bit (m2_bits.py checks).
  C1  register prefetch: each simdgroup loads the NEXT quant group's 6 code words, its LSB words (tier 4) and its
      scale / bias (the 11-bit pieces, read as P11R does) before decoding the current group, so the reads are in flight
      during the decode and the MMA (today a group's loads are issued at the top of its own step and waited on there).
Usage: import; ka(form) / kb(form) return mx.fast.metal_kernel objects with ooo1cq's input / output names."""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/Leap/third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
import mlx.core as mx                                    # noqa: E402
import ooo1cq as OQ                                      # noqa: E402
import tierglu_nax6 as N6                                # noqa: E402
import tierglu_nax6n as NN                               # noqa: E402
import p11 as P11                                        # noqa: E402
import ooo1c as OC                                       # noqa: E402

SRC_A = OQ.a_one_call_q(NN._A_NP)
SRC_B = OQ.b_drop(NN._B_NP)
HDR_A = N6._HDR4 + P11.P11_METAL + OC.SYS
HDR_B = N6._HDR4 + P11.P11_METAL

GPL = "    for (int gp = kc * GPC; gp < (kc + 1) * GPC; ++gp) {"
M2 = "        simdgroup_barrier(mem_flags::mem_threadgroup);"
DECODE_OLD = r"""            const int g = gp * GS_C + j;
            const device uint32_t* c = wrow + 192 * g;
            const uint w0 = c[0], w1 = c[1], w2 = c[2], w3 = c[3], w4 = c[4], w5 = c[5];
            uint ch[8] = {w0 & 0xFFFFFFu, (w0 >> 24) | ((w1 & 0xFFFFu) << 8), (w1 >> 16) | ((w2 & 0xFFu) << 16), w2 >> 8,
                          w3 & 0xFFFFFFu, (w3 >> 24) | ((w4 & 0xFFFFu) << 8), (w4 >> 16) | ((w5 & 0xFFu) << 16), w5 >> 8};
            uint nb[8];
            if (tier == 4) {
                const uint l0 = lrow[64 * g], l1 = lrow[64 * g + 1];
                for (int q = 0; q < 8; ++q) nb[q] = spread8(ch[q], ((q < 4 ? l0 : l1) >> (8 * (q & 3))) & 0xFFu);
            } else if (tier == 2) {
                for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]) & 0x66666666u;
            } else {
                for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]);
            }
            threadgroup uint4* dst = (threadgroup uint4*)(tb + lane * 8 * GS_C + 8 * j);
            dst[0] = uint4(nb[0], nb[1], nb[2], nb[3]);
            dst[1] = uint4(nb[4], nb[5], nb[6], nb[7]);
            sb[64 * j + lane] = (float)srow[32 * g] * smul;
            sb[64 * j + 32 + lane] = (float)brow[32 * g];
        }
"""

C1_LOOP = r"""    static_assert(GS_C == 1, "m2 C1: one quant group per stage");
    const int g_lo = kc * GPC, g_hi = (kc + 1) * GPC;
    uint nw0, nw1, nw2, nw3, nw4, nw5, nl0 = 0u, nl1 = 0u;   // the next group's words, in flight during this group
    float nsv, nbv;
    {
        const device uint32_t* c = wrow + 192 * g_lo;
        nw0 = c[0]; nw1 = c[1]; nw2 = c[2]; nw3 = c[3]; nw4 = c[4]; nw5 = c[5];
        if (tier == 4) { nl0 = lrow[64 * g_lo]; nl1 = lrow[64 * g_lo + 1]; }
        nsv = (float)srow[32 * g_lo]; nbv = (float)brow[32 * g_lo];
    }
    for (int gp = g_lo; gp < g_hi; ++gp) {
        threadgroup uint32_t* tb = tbw + (gp & 1) * (32 * 8 * GS_C);
        threadgroup float* sb = sbw + (gp & 1) * (64 * GS_C);
        const uint w0 = nw0, w1 = nw1, w2 = nw2, w3 = nw3, w4 = nw4, w5 = nw5, l0 = nl0, l1 = nl1;
        const float sv = nsv, bv = nbv;
        if (gp + 1 < g_hi) {
            const device uint32_t* c = wrow + 192 * (gp + 1);
            nw0 = c[0]; nw1 = c[1]; nw2 = c[2]; nw3 = c[3]; nw4 = c[4]; nw5 = c[5];
            if (tier == 4) { nl0 = lrow[64 * (gp + 1)]; nl1 = lrow[64 * (gp + 1) + 1]; }
            nsv = (float)srow[32 * (gp + 1)]; nbv = (float)brow[32 * (gp + 1)];
        }
        {
            uint ch[8] = {w0 & 0xFFFFFFu, (w0 >> 24) | ((w1 & 0xFFFFu) << 8), (w1 >> 16) | ((w2 & 0xFFu) << 16), w2 >> 8,
                          w3 & 0xFFFFFFu, (w3 >> 24) | ((w4 & 0xFFFFu) << 8), (w4 >> 16) | ((w5 & 0xFFu) << 16), w5 >> 8};
            uint nb[8];
            if (tier == 4) {
                for (int q = 0; q < 8; ++q) nb[q] = spread8(ch[q], ((q < 4 ? l0 : l1) >> (8 * (q & 3))) & 0xFFu);
            } else if (tier == 2) {
                for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]) & 0x66666666u;
            } else {
                for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]);
            }
            threadgroup uint4* dst = (threadgroup uint4*)(tb + lane * 8);
            dst[0] = uint4(nb[0], nb[1], nb[2], nb[3]);
            dst[1] = uint4(nb[4], nb[5], nb[6], nb[7]);
            sb[lane] = sv * smul;
            sb[32 + lane] = bv;
        }
"""


C1D2_LOOP = r"""    static_assert(GS_C == 1, "m2 C1d2: one quant group per stage");
    const int g_lo = kc * GPC, g_hi = (kc + 1) * GPC;
    uint aw0 = 0u, aw1 = 0u, aw2 = 0u, aw3 = 0u, aw4 = 0u, aw5 = 0u, al0 = 0u, al1 = 0u;   // group gp + 1 (in flight)
    uint bw0 = 0u, bw1 = 0u, bw2 = 0u, bw3 = 0u, bw4 = 0u, bw5 = 0u, bl0 = 0u, bl1 = 0u;   // group gp + 2 (in flight)
    float asv = 0.0f, abv = 0.0f, bsv = 0.0f, bbv = 0.0f;
    {
        const device uint32_t* c = wrow + 192 * g_lo;
        aw0 = c[0]; aw1 = c[1]; aw2 = c[2]; aw3 = c[3]; aw4 = c[4]; aw5 = c[5];
        if (tier == 4) { al0 = lrow[64 * g_lo]; al1 = lrow[64 * g_lo + 1]; }
        asv = (float)srow[32 * g_lo]; abv = (float)brow[32 * g_lo];
    }
    if (g_lo + 1 < g_hi) {
        const device uint32_t* c = wrow + 192 * (g_lo + 1);
        bw0 = c[0]; bw1 = c[1]; bw2 = c[2]; bw3 = c[3]; bw4 = c[4]; bw5 = c[5];
        if (tier == 4) { bl0 = lrow[64 * (g_lo + 1)]; bl1 = lrow[64 * (g_lo + 1) + 1]; }
        bsv = (float)srow[32 * (g_lo + 1)]; bbv = (float)brow[32 * (g_lo + 1)];
    }
    for (int gp = g_lo; gp < g_hi; ++gp) {
        threadgroup uint32_t* tb = tbw + (gp & 1) * (32 * 8 * GS_C);
        threadgroup float* sb = sbw + (gp & 1) * (64 * GS_C);
        const uint w0 = aw0, w1 = aw1, w2 = aw2, w3 = aw3, w4 = aw4, w5 = aw5, l0 = al0, l1 = al1;
        const float sv = asv, bv = abv;
        aw0 = bw0; aw1 = bw1; aw2 = bw2; aw3 = bw3; aw4 = bw4; aw5 = bw5; al0 = bl0; al1 = bl1; asv = bsv; abv = bbv;
        if (gp + 2 < g_hi) {
            const device uint32_t* c = wrow + 192 * (gp + 2);
            bw0 = c[0]; bw1 = c[1]; bw2 = c[2]; bw3 = c[3]; bw4 = c[4]; bw5 = c[5];
            if (tier == 4) { bl0 = lrow[64 * (gp + 2)]; bl1 = lrow[64 * (gp + 2) + 1]; }
            bsv = (float)srow[32 * (gp + 2)]; bbv = (float)brow[32 * (gp + 2)];
        }
""" + C1_LOOP[C1_LOOP.index("        {\n            uint ch[8]"):]


def _swap_loop(src, loop):
    i0, i1 = src.index(GPL), src.index(M2)
    assert src.count(GPL) == 1 and src.count(M2) == 1 and i0 < i1
    seg = src[i0:i1]
    assert seg.count(DECODE_OLD) == 1, "the decode loop changed"
    return src[:i0] + loop + src[i1:]


def _core_c1(src):
    return _swap_loop(src, C1_LOOP)


def _core_c1d2(src):
    return _swap_loop(src, C1D2_LOOP)


# C2: the grid's y is capped at GY_C rows (template; the host passes min(P, cap)) instead of P = M * R = 320, and each
# threadgroup walks segments gq, gq + GY_C, ... < nseg: today ~260 of the 320 rows are empty threadgroups (x 20 for A,
# x 40 for B) that launch only to read nseg and return. A segment's work is unchanged, so its bits are too. The
# threadgroup arrays move to function scope; early returns become continues (after a barrier where rs_s was shared);
# a barrier closes each segment before the next one reuses threadgroup memory.
GQ_HEAD = "    const uint gq = threadgroup_position_in_grid.y;\n    if ((int)gq >= nseg[0]) return;\n"
TG_A = ["    threadgroup int rs_s[2];\n",
        "    threadgroup uint32_t tb_all[2 * KS_C * 2 * 32 * 8 * GS_C];     // later reused for the chunk accumulators\n",
        "    threadgroup float sb_all[2 * KS_C * 2 * 64 * GS_C];\n"]
TG_B = ["    threadgroup uint32_t tb_all[NSG_C * KS_C * 2 * 32 * 8 * GS_C];  // later reused for the chunk accumulators\n",
        "    threadgroup float sb_all[NSG_C * KS_C * 2 * 64 * GS_C];\n"]
RET_A = ("    if (e_ < 0) return;", "    if (e_ < 0) { threadgroup_barrier(mem_flags::mem_threadgroup); continue; }")
RET_B = ("        return;\n    }\n    const uint e = (uint)msb_slot[pf];", "        continue;\n    }\n    const uint e = (uint)msb_slot[pf];")


# C3 (kernel A only): a threadgroup runs TPT_C consecutive 32-column tiles in turn (grid x = N / (32 TPT_C)), so its
# setup chain (nseg -> seg -> pidx -> slots -> pointers / P11 bases) is paid once per TPT_C tiles; and with the tiles of
# a 64-column group in one threadgroup it writes rowsum64(h) itself: per row, the 64 bf16 h values (as stored) summed in
# fp32 from 0 in column order, rowsum64's own loop, so the separate rowsum64 launch goes. Output "hs" [M R, N / 64].
TPT = int(os.environ.get("M2_TPT", "2"))
N0_A = "    const uint n0 = threadgroup_position_in_grid.x * 32;\n"
WROW = "    const device uint32_t* wrow = "
H_WRITE = "                h[(size_t)pidx[q0 + row] * N + n0 + col] = TgMultiply()(TgMultiply()(gv, TgSigmoid()(gv)), ub);\n"
HB_DECL = "    threadgroup T hb_all[16 * 32 * TPT_C];                          // C3: this tile run's h values (bf16)\n"
C3_TAIL = r"""    threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    static_assert(TPT_C % 2 == 0, "m2 C3: whole 64-column groups per threadgroup");
    if (thread_index_in_threadgroup < 16u * (TPT_C / 2)) {           // rowsum64(h): one thread per (row, 64-col group)
        const int row = (int)(thread_index_in_threadgroup % 16u), gg = (int)(thread_index_in_threadgroup / 16u);
        if (row < nq) {
            float s = 0.0f;
            for (int i = 0; i < 64; ++i) s += (float)hb_all[row * (32 * TPT_C) + gg * 64 + i];
            hs[(size_t)pidx[q0 + row] * (N / 64) + threadgroup_position_in_grid.x * (TPT_C / 2) + gg] = s;
        }
    }
"""


def _c3a(src):
    if "rs_s" not in src:                                # kernel B: unchanged (it reads A's hs)
        return src
    for k in (N0_A, WROW, H_WRITE, TG_A[1]):
        assert src.count(k) == 1, k[:60]
    src = src.replace(N0_A, "").replace(TG_A[1], TG_A[1] + HB_DECL)
    i = src.index(WROW)
    src = (src[:i] + "    for (int t_ = 0; t_ < TPT_C; ++t_) {\n"
           "    const uint n0 = (threadgroup_position_in_grid.x * TPT_C + (uint)t_) * 32;\n" + src[i:])
    src = src.replace(H_WRITE, "                const T hv = TgMultiply()(TgMultiply()(gv, TgSigmoid()(gv)), ub);\n"
                               "                h[(size_t)pidx[q0 + row] * N + n0 + col] = hv;\n"
                               "                hb_all[row * (32 * TPT_C) + t_ * 32 + col] = hv;\n")
    return src.rstrip() + "\n" + C3_TAIL


# C3B (kernel B only): a threadgroup runs TPB_C rounds of its NSG_C column tiles in turn (grid x = D / (32 NSG_C TPB_C)),
# paying the setup chain once per TPB_C rounds; a dropped group writes its zeros for every round. Per tile: unchanged.
TPB = int(os.environ.get("M2_TPB", "4"))
N0_B = "    const uint n0 = (threadgroup_position_in_grid.x * NSG_C + ti) * 32;\n"
DROP_B = """        if (kc == 0)
            for (int i = (int)lane; i < 16 * 32; i += 32) {
                const int row = i >> 5, col = i & 31;
                if (row < nq) y[(size_t)pidx[q0 + row] * D + n0 + col] = static_cast<T>(0.0f);
            }
"""
N0_B_T = "    const uint n0 = ((threadgroup_position_in_grid.x * TPB_C + (uint)t_) * NSG_C + ti) * 32;\n"


def _c3b(src):
    if "rs_s" in src:                                    # kernel A: unchanged
        return src
    for k in (N0_B, DROP_B, WROW):
        assert src.count(k) == 1, k[:60]
    src = src.replace(N0_B, "")
    src = src.replace(DROP_B, "        for (int t_ = 0; t_ < TPB_C; ++t_) {\n    " + N0_B_T + DROP_B + "        // [drop round end]\n        }\n")
    i = src.index(WROW)
    src = src[:i] + "    for (int t_ = 0; t_ < TPB_C; ++t_) {\n" + N0_B_T + src[i:]
    return src.rstrip() + "\n    // [round end]\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    }\n"


def _c2(src):
    is_a = "rs_s" in src
    decl = (TG_A + ([HB_DECL] if HB_DECL in src else [])) if is_a else (TG_B + ([C4_DECL] if C4_DECL in src else []))
    for d in decl:
        assert src.count(d) == 1, d
        src = src.replace(d, "")
    assert src.count(GQ_HEAD) == 1
    loop = ("".join(decl) + "    const int nseg_all = nseg[0];\n"
            "    for (uint gq = threadgroup_position_in_grid.y; (int)gq < nseg_all; gq += GY_C) {\n")
    src = src.replace(GQ_HEAD, loop)
    a, b = RET_A if is_a else RET_B
    assert src.count(a) == 1, a
    src = src.replace(a, b)
    assert "return;" not in src.split("for (uint gq =")[1], "an early return is left inside the segment loop"
    return src.rstrip() + "\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    }\n"


# C1x (on top of C1): the next group's activation fragments (8 x vec4 per lane: rows ar0 / ar1, both K halves) and its
# row sums (as0 / as1) are loaded with its weight words, so the MMA step reads registers instead of waiting on device
# loads after the barrier. Same values, same ops.
def _xl(pfx, G):
    s = ""
    for hh in (0, 1):
        k = f"64 * ({G}) + {32 * hh}"
        s += (f"            {pfx}{hh}_00 = *(const device vec<T, 4>*)(ar0 + {k} + fn0); "
              f"{pfx}{hh}_10 = *(const device vec<T, 4>*)(ar1 + {k} + fn0);\n"
              f"            {pfx}{hh}_01 = *(const device vec<T, 4>*)(ar0 + {k} + 16 + fn0); "
              f"{pfx}{hh}_11 = *(const device vec<T, 4>*)(ar1 + {k} + 16 + fn0);\n")
    return s + f"            {pfx}s0 = as0[{G}]; {pfx}s1 = as1[{G}];\n"


def _c1x(src):
    assert "nw0" in src and "aw0" not in src, "C1x goes on top of C1"
    reps = [
        ("    uint nw0, nw1, nw2, nw3, nw4, nw5, nl0 = 0u, nl1 = 0u;",
         "    vec<T, 4> nx0_00, nx0_10, nx0_01, nx0_11, nx1_00, nx1_10, nx1_01, nx1_11;\n    float nxs0, nxs1;\n"
         "    uint nw0, nw1, nw2, nw3, nw4, nw5, nl0 = 0u, nl1 = 0u;"),
        ("        nsv = (float)srow[32 * g_lo]; nbv = (float)brow[32 * g_lo];\n",
         "        nsv = (float)srow[32 * g_lo]; nbv = (float)brow[32 * g_lo];\n" + _xl("nx", "g_lo").replace("            ", "        ")),
        ("        const float sv = nsv, bv = nbv;\n",
         "        const float sv = nsv, bv = nbv;\n"
         "        const vec<T, 4> cx0_00 = nx0_00, cx0_10 = nx0_10, cx0_01 = nx0_01, cx0_11 = nx0_11;\n"
         "        const vec<T, 4> cx1_00 = nx1_00, cx1_10 = nx1_10, cx1_01 = nx1_01, cx1_11 = nx1_11;\n"
         "        const float cxs0 = nxs0, cxs1 = nxs1;\n"),
        ("            nsv = (float)srow[32 * (gp + 1)]; nbv = (float)brow[32 * (gp + 1)];\n",
         "            nsv = (float)srow[32 * (gp + 1)]; nbv = (float)brow[32 * (gp + 1)];\n" + _xl("nx", "gp + 1")),
        ("                const vec<T, 4> a00 = *(const device vec<T, 4>*)(ar0 + k0 + fn0);",
         "                const vec<T, 4> a00 = hh == 0 ? cx0_00 : cx1_00;"),
        ("                const vec<T, 4> a10 = *(const device vec<T, 4>*)(ar1 + k0 + fn0);",
         "                const vec<T, 4> a10 = hh == 0 ? cx0_10 : cx1_10;"),
        ("                const vec<T, 4> a01 = *(const device vec<T, 4>*)(ar0 + k0 + 16 + fn0);",
         "                const vec<T, 4> a01 = hh == 0 ? cx0_01 : cx1_01;"),
        ("                const vec<T, 4> a11 = *(const device vec<T, 4>*)(ar1 + k0 + 16 + fn0);",
         "                const vec<T, 4> a11 = hh == 0 ? cx0_11 : cx1_11;"),
        ("            const float x0 = as0[g], x1 = as1[g];", "            const float x0 = cxs0, x1 = cxs1;"),
    ]
    for a, b in reps:
        assert src.count(a) == 1, a[:70]
        src = src.replace(a, b)
    return src


# C4 (kernel B): the combine folded into B. After a threadgroup's y writes (per tile round / segment, and in a dropped
# group's zero path), one thread per (pair row, tile) counts it in cnt[m, tile] (zeroed per call); the count that
# completes a (row, tile) (the R-th) makes this threadgroup combine those 32 columns of row m: combine_phase's own loop
# (LY bf16 partials over the R ranks, then summed j = 1..LY-1), reading the R y rows through coherent(device) loads after a
# device fence. Inputs + cnt (int32 [M D / 32], zeros) and scores (bf16 [M R]); outputs + out [M, D] (the combined rows).
C4_DECL = "    threadgroup int c4_item[16 * NSG_C];                          // C4: (row, tile) items to combine\n"
C4_BLOCK = r"""    threadgroup_barrier(mem_flags::mem_device);                     // C4: this threadgroup's y writes are done
    {
        const uint tb0 = n0 / 32 - ti, tt = thread_index_in_threadgroup;
        if (tt < 16u * NSG_C) {
            int item = -1;
            const int row = (int)(tt / NSG_C);
            if (row < nq) {
                const uint key = ((uint)pidx[q0 + row] / NR) * (D / 32) + tb0 + tt % NSG_C;
                atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
                if (atomic_fetch_add_explicit((device atomic_int*)cnt + key, 1, memory_order_relaxed) == NR - 1) item = (int)key;
            }
            c4_item[tt] = item;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint it = sg; it < 16u * NSG_C; it += NSG_C * KS_C) {      // one simdgroup per item; lane = column
        const int key = c4_item[it];
        if (key < 0) continue;
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const uint m = (uint)key / (D / 32), col = ((uint)key % (D / 32)) * 32 + lane;
        T part[LY_C];
        for (int j = 0; j < LY_C; ++j) part[j] = static_cast<T>(0);
        for (int rr = 0; rr < NR; ++rr) {
            const T yv = ((coherent(device) device T*)y)[(size_t)(m * NR + rr) * D + col];
            const T prod = TgMultiply()(yv, scores[m * NR + rr]);
            part[rr % LY_C] = prod + part[rr % LY_C];
        }
        T tot = part[0];
        for (int j = 1; j < LY_C; ++j) tot = part[j] + tot;
        out[(size_t)m * D + col] = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);                // c4_item is reused by the next round / segment
"""
DROP_RET_B = "        return;\n    }\n    const uint e = (uint)msb_slot[pf];"


def _c4b(src):
    """kernel B only (A unchanged). Apply AFTER C3B (the counting sits inside each tile round) and BEFORE C2."""
    if "rs_s" in src:
        return src
    top = "    const uint ti = sg % NSG_C, kc = sg / NSG_C;\n"
    assert src.count(top) == 1, "kernel B text changed"
    src = src.replace(top, top + C4_DECL)               # at the top: the dropped group's path comes before tb_all
    if "        // [drop round end]\n" in src:                # C3B: count after each round's zeros
        src = src.replace("        // [drop round end]\n", C4_BLOCK + "        // [drop round end]\n")
    else:
        assert src.count(DROP_RET_B) == 1
        src = src.replace(DROP_RET_B, C4_BLOCK + DROP_RET_B)
    if "    // [round end]\n" in src:                        # C3B: after each round's epilogue
        src = src.replace("    // [round end]\n", C4_BLOCK + "    // [round end]\n")
    else:
        src = src.rstrip() + "\n" + C4_BLOCK
    return src


# M8T (TIMING SCREEN ONLY, bits wrong for rows 8-15): the MMA as MPP matmul2d at M = 8 (rows 0-7 of the 16-row tile,
# elements placed through ca8's own index map, scalar loads), cg8 copied into cg. Leap 22:5x: time kernel A first.
M8_DECL_OLD = "    auto cg = op.template get_destination_cooperative_tensor<decltype(ca), TB, float>();\n"
M8_DECL_NEW = M8_DECL_OLD + (
    "    constexpr auto desc8 = mpp::tensor_ops::matmul2d_descriptor(8, 32, 32, false, true, true,\n"
    "        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);\n"
    "    mpp::tensor_ops::matmul2d<desc8, metal::execution_simdgroup> op8;\n"
    "    auto ca8 = op8.template get_left_input_cooperative_tensor<T, uint4b_format, float>();\n"
    "    auto cg8 = op8.template get_destination_cooperative_tensor<decltype(ca8), TB, float>();\n")
M8_RUN_OLD = """                for (int q = 0; q < 4; ++q) { ca[q] = a00[q]; ca[4 + q] = a10[q]; ca[8 + q] = a01[q]; ca[12 + q] = a11[q]; }
                TB tbt((threadgroup uchar*)tb + 32 * j + 16 * hh, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64 * GS_C});
                op.run(ca, tbt, cg);"""
M8_RUN_NEW = """                for (int i = 0; i < (int)ca8.get_capacity(); ++i) { auto ix = ca8.get_multidimensional_index(i);
                    ca8[i] = (ix[1] < 4 ? ar0 : ar1)[k0 + ix[0]]; }
                TB tbt((threadgroup uchar*)tb + 32 * j + 16 * hh, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64 * GS_C});
                op8.run(ca8, tbt, cg8);"""


def _m8t(src):
    for a in (M8_DECL_OLD, M8_RUN_OLD, "            for (int i = 0; i < 16; ++i) cg[i] = 0.0f;\n"):
        assert src.count(a) == 1, a[:60]
    src = src.replace(M8_DECL_OLD, M8_DECL_NEW).replace(M8_RUN_OLD, M8_RUN_NEW)
    src = src.replace("            for (int i = 0; i < 16; ++i) cg[i] = 0.0f;\n",
                      "            for (int i = 0; i < 16; ++i) cg[i] = 0.0f;\n"
                      "            for (int i = 0; i < (int)cg8.get_capacity(); ++i) cg8[i] = 0.0f;\n")
    i = src.index("            const float x0 = as0[g], x1 = as1[g];")
    return src[:i] + "            for (int i = 0; i < (int)cg8.get_capacity(); ++i) cg[i] = cg8[i];\n" + src[i:]


def _neg(src):
    """NEGATIVE CONTROL (must FAIL the bit tests): the bias scaled by 1 + 2^-23, a 1-ulp-scale change"""
    a = "sb[64 * j + 32 + lane] = (float)brow[32 * g];"
    assert src.count(a) == 1
    return src.replace(a, "sb[64 * j + 32 + lane] = (float)brow[32 * g] * 1.00000012f;")


FORMS = {"ref": lambda s: s, "C1": _core_c1, "C1d2": _core_c1d2, "C2": _c2, "C1C2": lambda s: _c2(_core_c1(s)),
         "C1d2C2": lambda s: _c2(_core_c1d2(s)), "NEG": _neg, "C3": _c3a, "C1C3": lambda s: _c3a(_core_c1(s)),
         "C1C3C2": lambda s: _c2(_c3a(_core_c1(s))), "C1d2C3C2": lambda s: _c2(_c3a(_core_c1d2(s))),
         "C3B": _c3b, "C1C3C3BC2": lambda s: _c2(_c3b(_c3a(_core_c1(s)))),
         "C1d2C3C3BC2": lambda s: _c2(_c3b(_c3a(_core_c1d2(s)))), "C1x": lambda s: _c1x(_core_c1(s)),
         "C1xC3C3BC2": lambda s: _c2(_c3b(_c3a(_c1x(_core_c1(s))))), "C4": _c4b,
         "C1C3C3BC4C2": lambda s: _c2(_c4b(_c3b(_c3a(_core_c1(s))))),
         "C1xC3C3BC4C2": lambda s: _c2(_c4b(_c3b(_c3a(_c1x(_core_c1(s))))))}
FORMS["M8T"] = _m8t
FORMS["C4NEG"] = lambda s: _c4b(s).replace("part[rr % LY_C] = prod + part[rr % LY_C];",
                                            "part[(rr + 1) % LY_C] = prod + part[(rr + 1) % LY_C];")   # NEGATIVE CONTROL
C4SET = {"C4", "C1C3C3BC4C2", "C1xC3C3BC4C2", "C4NEG"}            # forms whose kernel B also writes the combined rows (out)
CAPPED = {"C1C3C3BC4C2", "C1xC3C3BC4C2", "C1xC3C3BC2", "C2", "C1C2", "C1d2C2", "C1C3C2", "C1d2C3C2", "C1C3C3BC2", "C1d2C3C3BC2"}   # grid y min(P, GY), GY_C
C3BSET = {"C1C3C3BC4C2", "C1xC3C3BC4C2", "C1xC3C3BC2", "C3B", "C1C3C3BC2", "C1d2C3C3BC2"}            # forms whose kernel B runs TPB_C rounds (grid x / TPB_C)
C3SET = {"C1C3C3BC4C2", "C1xC3C3BC4C2", "C1xC3C3BC2", "C3", "C1C3", "C1C3C2", "C1d2C3C2", "C1C3C3BC2", "C1d2C3C3BC2"}   # kernel A also writes hs
GY_CAP = int(os.environ.get("M2_GY", "96"))
_K = {}
IN_A = ["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu", "msb_slot", "lsb_slot",
        "pidx", "seg", "nseg", "region", "offs", "status"]
IN_B = ["h", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"]


def src_a(form, coh=False):
    s = FORMS[form](SRC_A)
    return _coh_src(s) if coh else s


def src_b(form, coh=False):
    s = FORMS[form](SRC_B)
    return _coh_src(s) if coh else s


def _coh_src(s):
    """coh1cq's own transforms (pool reads through coherent(system) pointers; P11 helpers renamed to their coherent
    twins), applied to a candidate's text: the COH twin, to compare with coh1cq"""
    import coh1cq as CQ
    return CQ._ren(CQ._body(s))


def _coh_hdr(h):
    import coh1cq as CQ
    return h + CQ._p11_hdr()


def ka(form, coh=False):
    if ("a", form, coh) not in _K:
        _K[("a", form, coh)] = mx.fast.metal_kernel(
            name=f"m2_{form.lower()}{'_coh' if coh else ''}_a", input_names=IN_A,
            output_names=["h", "rs_m", "rs_l"] + (["hs"] if form in C3SET else []),
            source=src_a(form, coh), header=_coh_hdr(HDR_A) if coh else HDR_A)
    return _K[("a", form, coh)]


def kb(form, coh=False):
    if ("b", form, coh) not in _K:
        c4 = form in C4SET
        _K[("b", form, coh)] = mx.fast.metal_kernel(
            name=f"m2_{form.lower()}{'_coh' if coh else ''}_b", input_names=IN_B + (["cnt", "scores"] if c4 else []),
            output_names=["y"] + (["out"] if c4 else []),
            source=src_b(form, coh), header=_coh_hdr(HDR_A) if coh else HDR_B)   # coh1cq's B header has OC.SYS too
    return _K[("b", form, coh)]


def tier_glu_1cq(form, coh=False):
    """ooo1cq.tier_glu_nax6n_1cq's call (same inputs, templates and outputs; p11 pools) with the candidate's kernels A / B
    (and, for the capped forms, grid y = min(P, GY_CAP) + template GY_C): a drop-in for ref2's f1c"""
    from tierglu import plan_masks, PROJ
    from tierglu_nax import rowsum64

    def f(x, msb_n, lsb_n, pre_m, pre_l, ids, plan, region, offs, status, xs=None, grp=None, nsg_b=2, ks=4, ks_b=2):
        M, K = x.shape
        R = len(plan)
        m4, m2, r4 = plan_masks(plan)
        assert m2 == 0 and msb_n["gate_proj"]["s3"].dtype == mx.uint32
        N, D = msb_n["gate_proj"]["w3"].shape[1], msb_n["down_proj"]["w3"].shape[1]
        pwm, pwl = msb_n["gate_proj"]["s3"].shape[-1], lsb_n["gate_proj"]["b"].shape[-1]
        if grp is None:
            grp = OQ.gpu_grp_1cq(pre_m, pre_l, ids, plan)
        pidx, seg, nsg_, nseg = grp
        ny = min(nseg, GY_CAP) if form in CAPPED else nseg
        gyt = [("GY_C", ny)] if form in CAPPED else []
        bb = {p: msb_n[p]["b3"] for p in PROJ}
        ls = pre_l.astype(mx.int32).reshape(-1) if r4 else mx.zeros((1,), mx.int32)
        ms = pre_m.astype(mx.int32).reshape(-1)
        L = lsb_n
        g, u, d = (msb_n[p] for p in PROJ)
        tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0),
                ("GS_C", 1), ("PWM_C", pwm), ("PWL_C", pwl)]
        xs = rowsum64(x) if xs is None else xs
        c3 = form in C3SET
        outs = ka(form, coh)(
            inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                    L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls, pidx, seg,
                    nsg_, region, offs, status],
            template=tmpl + [("K_C", K), ("N_C", N), ("KS_C", ks)] + gyt + ([("TPT_C", TPT)] if c3 else []),
            grid=(N // (32 * (TPT if c3 else 1)) * 64 * ks, ny, 1),
            threadgroup=(64 * ks, 1, 1),
            output_shapes=[(M * R, N), (M * R,), (max(M * r4, 1),)] + ([(M * R, N // 64)] if c3 else []),
            output_dtypes=[x.dtype, mx.int32, mx.int32] + ([mx.float32] if c3 else []))
        h, rs_m, rs_l = outs[:3]
        hs = outs[3] if c3 else rowsum64(h)
        (y,) = kb(form, coh)(
            inputs=[h, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"],
                    rs_m, rs_l, pidx, seg, nsg_],
            template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)] + gyt
            + ([("TPB_C", TPB)] if form in C3BSET else []),
            grid=(D // (32 * nsg_b * (TPB if form in C3BSET else 1)) * 32 * nsg_b * ks_b, ny, 1),
            threadgroup=(32 * nsg_b * ks_b, 1, 1),
            output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
        return y.reshape(M, R, D), rs_m, rs_l
    return f


def full_call(form, coh=False):
    """the one-call AND its combine on ooo1cq's arguments + the scores [M, R]: for the C4 forms B writes the combined rows
    itself; for the others ooo1cq-style y then tierglu.combine_phase (ref2's _kern). Returns [M, D]."""
    from tierglu import plan_masks, PROJ, combine_phase
    from tierglu_nax import rowsum64

    def f(x, msb_n, lsb_n, pre_m, pre_l, ids, plan, region, offs, status, sc, nsg_b=2, ks=4, ks_b=2):
        M, K = x.shape
        R = len(plan)
        if form not in C4SET:
            y, _, _ = tier_glu_1cq(form, coh)(x, msb_n, lsb_n, pre_m, pre_l, ids, plan, region, offs, status, nsg_b=nsg_b,
                                              ks=ks, ks_b=ks_b)
            return combine_phase(y.reshape(M * R, -1), y.reshape(M * R, -1), mx.ones((M, R), mx.int32), sc.reshape(-1, R))
        m4, m2, r4 = plan_masks(plan)
        N, D = msb_n["gate_proj"]["w3"].shape[1], msb_n["down_proj"]["w3"].shape[1]
        pwm, pwl = msb_n["gate_proj"]["s3"].shape[-1], lsb_n["gate_proj"]["b"].shape[-1]
        pidx, seg, nsg_, nseg = OQ.gpu_grp_1cq(pre_m, pre_l, ids, plan)
        ny = min(nseg, GY_CAP) if form in CAPPED else nseg
        gyt = [("GY_C", ny)] if form in CAPPED else []
        ls = pre_l.astype(mx.int32).reshape(-1) if r4 else mx.zeros((1,), mx.int32)
        ms = pre_m.astype(mx.int32).reshape(-1)
        L = lsb_n
        g, u, d = (msb_n[p] for p in PROJ)
        tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0),
                ("GS_C", 1), ("PWM_C", pwm), ("PWL_C", pwl)]
        xs = rowsum64(x)
        c3 = form in C3SET
        outs = ka(form, coh)(
            inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], g["b3"], u["b3"],
                    L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls, pidx, seg,
                    nsg_, region, offs, status],
            template=tmpl + [("K_C", K), ("N_C", N), ("KS_C", ks)] + gyt + ([("TPT_C", TPT)] if c3 else []),
            grid=(N // (32 * (TPT if c3 else 1)) * 64 * ks, ny, 1), threadgroup=(64 * ks, 1, 1),
            output_shapes=[(M * R, N), (M * R,), (max(M * r4, 1),)] + ([(M * R, N // 64)] if c3 else []),
            output_dtypes=[x.dtype, mx.int32, mx.int32] + ([mx.float32] if c3 else []))
        h, rs_m, rs_l = outs[:3]
        hs = outs[3] if c3 else rowsum64(h)
        cb = form in C3BSET
        _, out = kb(form, coh)(
            inputs=[h, hs, d["w3"], d["s3"], d["b3"], d["b3"], L["down_proj"]["l1"], L["down_proj"]["b"], rs_m, rs_l,
                    pidx, seg, nsg_, mx.zeros((M * (D // 32),), mx.int32), sc.astype(x.dtype).reshape(-1)],
            template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)] + gyt
            + ([("TPB_C", TPB)] if cb else []) + [("LY_C", 8)],
            grid=(D // (32 * nsg_b * (TPB if cb else 1)) * 32 * nsg_b * ks_b, ny, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
            output_shapes=[(M * R, D), (M, D)], output_dtypes=[x.dtype, x.dtype])
        return out
    return f


if __name__ == "__main__":
    for f in FORMS:
        a, b = src_a(f), src_b(f)
        print(f, len(a), len(b), "C1 loop" if "nw0" in a and "nw0" in b else "")
