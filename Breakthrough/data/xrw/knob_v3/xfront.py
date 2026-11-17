"""xfront.py (Breakthrough, Tue 17 Nov 2026 21:2x; MOE BLOCK front fold, STACK FORM; docs/MOEBLOCK.md "FRONT FOLD, STACK
FORM (mode tgl16ef)", pre-registered 1b71659): the two front dispatches that tgl16e runs inside the one-call,
ooo1cq.gpu_grp_1cq and rowsum64(x), as ONE dispatch whose outputs equal theirs bitwise (scripts/xrw/xfront_check.py).
  threadgroup 0     the pair grouping, keyed by ooo1cq's own key code (_GRP_NEW, spliced in as text). The order is
                    _GRP1CQ's rank: (key, pair index); segments start at a new key and every 16 pairs inside one.
                    var "v": _GRP1CQ's text as is; "r": the same rank formula, keys broadcast by simd_shuffle and the
                    segment index by a SIMD prefix count; "s": a bitonic sort by (key, pair index) + the same count.
  threadgroups 1..  rowsum64(x), with tierglu_nax's loop: a sequential fp32 sum per row and 64-value group.
  gpu_front(pre_m, pre_l, ids, plan, x, var) -> ((pidx, seg, nseg, P), xs)
  tier_glu_nax6n_1cq_tgl16ef(...)   tgl16e (tgl16v.py, unchanged) with grp and xs from gpu_front. It falls back to plain
                    tgl16e (today's two dispatches) if P > 1024 or if ooo1cq's grouping / tierglu_nax's rowsum source
                    is not the text this file was checked against (COUNTS)."""
import hashlib
import os
import sys

import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

VAR = ["r"]
COUNTS = {"fold": 0, "plain_big_p": 0, "plain_src": 0}
# the sources this fold was checked against (md5 of the text): tierglu_nax6._GRP, ooo1cq._GRP_OLD / _GRP_NEW, rowsum's _RS
SRC_MD5 = {"grp": "0a3079450f94b62cca7cdba5f688f56b", "old": "10117106cffb7239f94b324f005f14c4",
           "new": "5c1790d4ba3dc7bb5a1d80b79c98254f", "rs": "04709a684eb3da702e8836ac8d5ccb12"}
_K = {}
_SRC_OK = []

_ROWSUM = r"""
    if (threadgroup_position_in_grid.y > 0) {      // rowsum64(x): tierglu_nax's loop, one (row, 64-value group) per thread
        const uint q = (threadgroup_position_in_grid.y - 1) * (uint)TGX_C + thread_position_in_grid.x;
        if (q < (uint)(M_C * (L_C / 64))) {
            const uint r_ = q / (uint)(L_C / 64), g_ = q % (uint)(L_C / 64);
            const device T* p = x + (size_t)r_ * L_C + g_ * 64;
            float s = 0.0f;
            for (int k = 0; k < 64; ++k) s += (float)p[k];
            xs[(size_t)r_ * (L_C / 64) + g_] = s;
        }
        return;
    }
"""

_COUNT = r"""
    // segment starts: a new key, then every 16 pairs inside one (_GRP1CQ's rule, on the sorted keys)
    int st = 0;
    if ((int)i < P) {
        int r0 = (int)i;
        while (r0 > 0 && skey[r0 - 1] == skey[i]) --r0;
        st = (((int)i - r0) % 16) == 0 ? 1 : 0;
    }
    // seg[g] = the g-th start: g = the starts before it (a SIMD prefix count + the SIMD groups before this one)
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const int ex = simd_prefix_exclusive_sum(st);
    const int tot = simd_sum(st);
    if (lane == 0) wsum[sg] = tot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int off = 0;
    for (uint w = 0; w < sg; ++w) off += wsum[w];
    if (st) seg[off + ex] = (int)i;
    if (i == 0) {
        int n = 0;
        for (int w = 0; w < TGX_C / 32; ++w) n += wsum[w];
        nseg[0] = n;
        seg[n] = P;
    }
"""


def _src(var):
    import ooo1cq as OQ
    if var == "v":
        return _ROWSUM + OQ._GRP1CQ
    keys = ("    if ((int)i < P) {\n        const int r = (int)i % NR_C, m = (int)i / NR_C;\n" + OQ._GRP_NEW +
            "\n    }\n")
    if var == "r":
        return _ROWSUM + r"""
    constexpr int P = P_C;
    const uint i = thread_position_in_grid.x;
    threadgroup int key_s[TGX_C];
    threadgroup int skey[P_C];
    threadgroup int wsum[32];
    if ((int)i >= P) key_s[i] = 0x7fffffff;
""" + keys + r"""
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {                                               // _GRP1CQ's rank, the keys broadcast 32 at a time
        const int k = key_s[i];
        const uint ln = thread_index_in_simdgroup;
        int rank = 0;
        for (int c = 0; c < TGX_C; c += 32) {
            const int kr = key_s[c + (int)ln];
            for (ushort t = 0; t < 32; ++t) {
                const int kj = simd_shuffle(kr, t);
                const int j = c + (int)t;
                rank += (j < P) && ((kj < k) || (kj == k && j < (int)i));
            }
        }
        if ((int)i < P) { pidx[rank] = (int)i; skey[rank] = k; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
""" + _COUNT
    if var == "s":
        return _ROWSUM + r"""
    constexpr int P = P_C;
    const uint i = thread_position_in_grid.x;
    threadgroup int key_s[NPX_C];
    threadgroup int si[NPX_C];
    threadgroup int wsum[32];
""" + keys + r"""
    if ((int)i < P) si[i] = (int)i;
    for (int q = P + (int)i; q < NP_C; q += TGX_C) { key_s[q] = 0x7fffffff; si[q] = q; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 2; k <= NP_C; k <<= 1) {           // bitonic sort, ascending by (key, pair index)
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int t = (int)i; t < NP_C / 2; t += TGX_C) {
                const int a = 2 * t - (t & (j - 1)), b = a + j;
                const int ka = key_s[a], kb = key_s[b], ia = si[a], ib = si[b];
                const bool gt = ka > kb || (ka == kb && ia > ib);
                if (gt == ((a & k) == 0)) { key_s[a] = kb; key_s[b] = ka; si[a] = ib; si[b] = ia; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if ((int)i < P) pidx[i] = si[i];
""" + _COUNT.replace("skey", "key_s")
    raise ValueError(var)


def _kf(var):
    if var not in _K:
        _K[var] = mx.fast.metal_kernel(name=f"xfront_{var}", input_names=["msb_slot", "lsb_slot", "ids", "x"],
                                       output_names=["pidx", "seg", "nseg", "xs"], source=_src(var))
    return _K[var]


def src_ok():
    """True iff ooo1cq's grouping and tierglu_nax's rowsum are the texts this fold was checked against"""
    if not _SRC_OK:
        import ooo1cq as OQ
        import tierglu_nax6 as N6
        import tierglu_nax as TN
        h = lambda s: hashlib.md5(s.encode()).hexdigest()
        got = {"grp": h(N6._GRP), "old": h(OQ._GRP_OLD), "new": h(OQ._GRP_NEW), "rs": h(TN._RS)}
        _SRC_OK.append(got == SRC_MD5 and OQ._GRP1CQ == N6._GRP.replace(OQ._GRP_OLD, OQ._GRP_NEW))
    return _SRC_OK[0]


def gpu_front(pre_m, pre_l, ids, plan, x, var=None):
    """== (ooo1cq.gpu_grp_1cq(pre_m, pre_l, ids, plan), rowsum64(x)) in one dispatch"""
    from tierglu import plan_masks
    var = var or VAR[0]
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    ms = pre_m.astype(mx.int32).reshape(-1)
    P = ms.size
    assert P <= 1024 and m2 == 0, (P, m2)
    ls = pre_l.astype(mx.int32).reshape(-1) if r4 else mx.zeros((1,), mx.int32)
    M_, L = x.shape
    assert L % 64 == 0
    tgx = max(32, (P + 31) // 32 * 32)
    nrs = (M_ * (L // 64) + tgx - 1) // tgx
    tm = [("P_C", P), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("TGX_C", tgx),
          ("M_C", M_), ("L_C", L), ("T", x.dtype)]
    if var == "s":
        np2 = 1 << max(1, (P - 1).bit_length())
        tm += [("NP_C", np2), ("NPX_C", max(np2, tgx))]
    pidx, seg, nseg, xs = _kf(var)(inputs=[ms, ls, ids.astype(mx.int32).reshape(-1), x], template=tm,
                                   grid=(tgx, 1 + nrs, 1), threadgroup=(tgx, 1, 1),
                                   output_shapes=[(P,), (P + 1,), (1,), (M_, L // 64)],
                                   output_dtypes=[mx.int32, mx.int32, mx.int32, mx.float32])
    return (pidx, seg, nseg, P), xs


def tier_glu_nax6n_1cq_tgl16ef(x, msb_n, lsb_n, pre_m, pre_l, ids, plan, region, offs, status, xs=None, grp=None,
                               nsg_b=2, ks=4, ks_b=2):
    """tgl16e with its grouping and rowsum64(x) from one dispatch (gpu_front); bitwise == tgl16e"""
    import tgl16v
    if grp is None and xs is None:
        if pre_m.size > 1024:
            COUNTS["plain_big_p"] += 1
        elif not src_ok():
            COUNTS["plain_src"] += 1
        else:
            grp, xs = gpu_front(pre_m, pre_l, ids, plan, x)
            COUNTS["fold"] += 1
    return tgl16v.tier_glu_nax6n_1cq_tgl16e(x, msb_n, lsb_n, pre_m, pre_l, ids, plan, region, offs, status, xs=xs,
                                            grp=grp, nsg_b=nsg_b, ks=ks, ks_b=ks_b)
