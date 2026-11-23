"""tree_att.py (Leap, Sun 15 Nov 2026 14:4x; TREE16 L2, data/week98/tree16_spec.md 'L2 design notes'): attention route
(b) for the exact tree check. Step 1 (this file so far): a replica of MLX's split-K NAX GEMM in the nn layout, the P·V
of the exact path's unfused SDPA (Breakthrough's profile: steel_gemm_splitk_nax_nn bm64 bn64 bk256 wm2 wn2 +
steel_gemm_splitk_accum_bfloat16_float32). Each output element accumulates its partition's k-steps of 32 in order with
NAX 16 x 32 x 32 tiles (gemm_nax.h's per-element arithmetic, independent of the tile shape), the partitions are written
as fp32 partials and summed in order. leapk's S4 is the same scheme in the nt layout, bitwise to MLX for the HC GEMM.
The partition size MLX picks for P·V is pinned by tree_att_probe.py (match vs mx.matmul).
Step 2 (next): per-row V tails in the tail k-steps for the tree's nodes.
NOT RUN YET (written while C7c waits for memory)."""
from functools import lru_cache
import os
import sys
import mlx.core as mx

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98c/Leap_scripts"))

BF = mx.bfloat16

# one simdgroup per (16-row tile, 32-column tile, partition); A [M, K] row-major, V [K, N] row-major (nn)
_NN_SRC = """
    using namespace mlx::steel;
    const int tg = int(threadgroup_position_in_grid.x);
    const int ct = tg % GNT;                     // 32-column tile
    const int rt = (tg / GNT) % GMT;             // 16-row tile
    const int part = tg / (GNT * GMT);           // K partition
    const int c_col = ct * 32;
    const short sgp_sn = short(min(32, GN - c_col));
    const int k0 = part * GPS, k1 = min(k0 + GPS, GK);
    const int iters = (k1 - k0) / 32;
    const device T* Ap = x + size_t(rt) * 16 * GK + k0;
    const device T* Bp = v + size_t(k0) * GN + c_col;
    NAXTile<float, 1, 2> Dt;
    const bool full_n = sgp_sn == 32;
    Dt.clear();
    for (int s = 0; s < iters; s++) {
        NAXTile<T, 1, 2> At;
        NAXTile<T, 2, 2> Bt;
        At.load(Ap + s * 32, GK);
        if (full_n) {
            Bt.load(Bp + size_t(s) * 32 * GN, GN);
        } else {
            Bt.load_safe(Bp + size_t(s) * 32 * GN, GN, short2(sgp_sn, 32));
        }
        tile_matmad_nax(Dt, At, metal::bool_constant<false>{}, Bt, metal::bool_constant<false>{});
    }
    device float* Cp = cs + (size_t(part) * GM + size_t(rt) * 16) * GN + c_col;
    if (full_n) {
        Dt.store(Cp, GN);
    } else {
        Dt.store_safe(Cp, GN, short2(sgp_sn, 16));
    }
"""
_K = {}

_ACC = mx.fast.metal_kernel(
    name="leap_tree_att_accum", input_names=["cs"], output_names=["y"],
    source="""
    const uint i = thread_position_in_grid.x;
    if (i >= GMN) return;
    float out = 0;
    for (int p = 0; p < NP; p++) {
        out += cs[size_t(p) * GMN + i];
    }
    y[i] = static_cast<T>(out);
""")


def nn_partials(a, v, ps):
    """a [M, K] bf16 (M % 16 == 0, K % 32 == 0), v [K, N] bf16 -> fp32 partials [NP, M, N] over K partitions of ps"""
    import leapk as LK
    M, K = a.shape
    N = v.shape[1]
    assert M % 16 == 0 and K % 32 == 0 and ps % 32 == 0
    NP, NT, MT = (K + ps - 1) // ps, (N + 31) // 32, M // 16
    key = (M, N, K, ps)
    if key not in _K:
        _K[key] = mx.fast.metal_kernel(
            name=f"leap_tree_att_nn_{M}_{N}_{K}_{ps}", input_names=["x", "v"], output_names=["cs"], source=_NN_SRC,
            header=LK._nax_header() + "\n" + LK.consts(GM=M, GN=N, GK=K, GPS=ps, GNP=NP, GNT=NT, GMT=MT))
    (cs,) = _K[key](inputs=[a, v], template=[("T", BF)], grid=(32 * NT * MT * NP, 1, 1), threadgroup=(32, 1, 1),
                    output_shapes=[(NP, M, N)], output_dtypes=[mx.float32])
    return cs, NP


def pv_replica(a, v, ps):
    """MLX's split-K NAX nn GEMM a @ v, replicated: partitions of ps, fp32 partials summed in order, cast to bf16"""
    M, N = a.shape[0], v.shape[1]
    cs, NP = nn_partials(a, v, ps)
    (y,) = _ACC(inputs=[cs], template=[("T", BF), ("GMN", M * N), ("NP", NP)], grid=(M * N, 1, 1), threadgroup=(256, 1, 1),
                output_shapes=[(M, N)], output_dtypes=[BF])
    return y


# ---- step 2: P·V with per-row V tails (TREE16 route (b)). Rows are a 16-node tile of one query head (head-major fold).
# k-steps that end at or before TS0 (the first 32-aligned step touching the tail) read the shared V; the at most two
# k-steps from TS0 to TS1 run once per root-to-leaf path p (TREE16-TIME trim; was once per row): A = the probs of the
# rows p owns (others zero, adding exact zeros), B = p's chain-order V rows (cache rows, then p's nodes in depth order,
# zeros past it). A node on p has p's ancestors up to its depth; past its depth its probs are 0.
_NNT_SRC = """
    using namespace mlx::steel;
    const int tg = int(threadgroup_position_in_grid.x);
    const int ct = tg % GNT;
    const int rt = (tg / GNT) % GMT;
    const int part = tg / (GNT * GMT);
    const int c_col = ct * 32;
    const short sgp_sn = short(min(32, GN - c_col));
    const int k0 = part * GPS, k1 = min(k0 + GPS, GK);
    const int iters = (k1 - k0) / 32;
    const device T* Ap = x + size_t(rt) * 16 * GK + k0;
    const device T* Bp = v + size_t(k0) * GN + c_col;
    NAXTile<float, 1, 2> Dt;
    const bool full_n = sgp_sn == 32;
    Dt.clear();
    for (int s = 0; s < iters; s++) {
        const int kb = k0 + s * 32;
        if (kb >= TS0 && kb < TS1) {
            const int kk = kb - TS0;
            for (int p = 0; p < NPATH; p++) {          // one step per root-to-leaf path: its rows only, its V
                NAXTile<T, 1, 2> At;
                NAXTile<T, 2, 2> Bt;
                At.load(am + (size_t)((rt * NPATH + p) * 16) * TW + kk, TW);
                if (full_n) {
                    Bt.load(vt + ((size_t)p * TW + kk) * GN + c_col, GN);
                } else {
                    Bt.load_safe(vt + ((size_t)p * TW + kk) * GN + c_col, GN, short2(sgp_sn, 32));
                }
                tile_matmad_nax(Dt, At, metal::bool_constant<false>{}, Bt, metal::bool_constant<false>{});
            }
        } else {
            NAXTile<T, 1, 2> At;
            NAXTile<T, 2, 2> Bt;
            At.load(Ap + s * 32, GK);
            if (full_n) {
                Bt.load(Bp + size_t(s) * 32 * GN, GN);
            } else {
                Bt.load_safe(Bp + size_t(s) * 32 * GN, GN, short2(sgp_sn, 32));
            }
            tile_matmad_nax(Dt, At, metal::bool_constant<false>{}, Bt, metal::bool_constant<false>{});
        }
    }
    device float* Cp = cs + (size_t(part) * GM + size_t(rt) * 16) * GN + c_col;
    if (full_n) {
        Dt.store(Cp, GN);
    } else {
        Dt.store_safe(Cp, GN, short2(sgp_sn, 16));
    }
"""
_KT = {}


def mlx_pv_parts(K):
    """MLX 0.32.2's split-K NAX nn partition for this P·V (tree_att_probe): none up to 512 keys, else 2 at next_pow2(K)/2"""
    if K <= 512:
        return K
    p = 1
    while p < K:
        p *= 2
    return p // 2


def pv_tree(a, v, L, anc, depth, T, k_rule=None):
    """a [R, K] bf16 probs, each row in its node's chain order (R = heads x 16, head-major; node = row % 16 of the
    tile); v [K, N] bf16 in the tree layout (cache rows 0..L-1, the T nodes at L..L+T-1, zeros after); anc / depth: the
    tree's chain-order ancestor lists. Returns a @ (each row's chain-order V) [R, N], MLX's split-K arithmetic."""
    import numpy as np
    import leapk as LK
    R, K = a.shape
    N = v.shape[1]
    assert R % 16 == 0 and K % 32 == 0 and L + 16 <= K
    ts0 = (L // 32) * 32
    ts1 = -(-(L + 16) // 32) * 32
    TW = ts1 - ts0
    inner_nodes = {u for n in range(T) for u in anc[n][:-1]}
    paths = [anc[n] for n in range(T) if n not in inner_nodes]   # root-to-leaf, in node order of their leaves
    owner = [next(i for i, pth in enumerate(paths) if n in pth) for n in range(T)]
    owner += [owner[T - 1]] * (16 - T)                    # padding rows (copies of the last node, discarded)
    NPATH = len(paths)
    idx = np.full((NPATH, TW), K, np.int32)               # K = the appended zero row
    for p_, pth in enumerate(paths):
        for kk in range(TW):
            kpos = ts0 + kk
            if kpos < L:
                idx[p_, kk] = kpos
            elif kpos - L < len(pth):
                idx[p_, kk] = L + pth[kpos - L]
    vext = mx.concatenate([v, mx.zeros((1, N), v.dtype)], axis=0)
    vt = mx.take(vext, mx.array(idx).reshape(-1), axis=0).reshape(NPATH, TW, N)
    at = a[:, ts0:ts1].reshape(R // 16, 16, TW)
    sel = np.zeros((NPATH, 16), np.float32)
    for i in range(16):
        sel[owner[i], i] = 1.0
    am = (mx.array(sel).astype(a.dtype)[None, :, :, None] * at[:, None, :, :]).reshape(R // 16, NPATH, 16, TW)
    ps = mlx_pv_parts(K if k_rule is None else k_rule)    # MLX's rule reads the call's own K (before 32-padding)
    ps = -(-ps // 32) * 32
    NP, NT, MT = (K + ps - 1) // ps, (N + 31) // 32, R // 16
    key = (R, N, K, ps, ts0, ts1, NPATH)
    if key not in _KT:
        _KT[key] = mx.fast.metal_kernel(
            name=f"leap_tree_att_nnt_{R}_{N}_{K}_{ps}_{ts0}_{ts1}_{NPATH}", input_names=["x", "v", "am", "vt"],
            output_names=["cs"], source=_NNT_SRC,
            header=LK._nax_header() + "\n" + LK.consts(GM=R, GN=N, GK=K, GPS=ps, GNP=NP, GNT=NT, GMT=MT, TS0=ts0,
                                                        TS1=ts1, TW=TW, NPATH=NPATH))
    (cs,) = _KT[key](inputs=[a, v, am, vt], template=[("T", BF)], grid=(32 * NT * MT * NP, 1, 1),
                     threadgroup=(32, 1, 1), output_shapes=[(NP, R, N)], output_dtypes=[mx.float32])
    (y,) = _ACC(inputs=[cs], template=[("T", BF), ("GMN", R * N), ("NP", NP)], grid=(R * N, 1, 1),
                threadgroup=(256, 1, 1), output_shapes=[(R, N)], output_dtypes=[BF])
    return y



# ---- route (b): the tree's attention core, each node == its row in its path's chain check (ref2's exact verify SDPA)
BATCH_HEADS = [True]        # True (default since TREE16-TIME 15:00): one scores matmul / gather / softmax / pv_tree per kv head (bits: tree_att_test)


def _b3_mask(sel, Kr):
    """B3's per-node selection [T, L + 16] (qsa_tree.tree_select) in this msk layout: 16 rows (padding = copies of the
    last node) x Kr columns (False past L + 16). The same ops as Breakthrough's qsa_tree.route_b_mask, inlined."""
    T, W = int(sel.shape[0]), int(sel.shape[1])
    m = sel if T == 16 else mx.concatenate([sel, mx.repeat(sel[T - 1:T], 16 - T, axis=0)], axis=0)
    if Kr > W:
        m = mx.concatenate([m, mx.zeros((16, Kr - W), dtype=mx.bool_)], axis=1)
    return m[:, :Kr]


def tree_sdpa(q, k, v, scale, L, anc, depth, T, ref2, sel=None):
    """q [1, 24, T, 256] (the nodes' queries, RoPE'd at their depths); k / v [1, 2, L + T, 256] (cache, then the T nodes
    in node order). sel: B3's per-node selection (qsa_tree.tree_select: a bool [T, L + 16], each node's mask in its chain
    column order) above ~2K tokens, None when every cache key is selected (the mask is then built here). Per AR class
    (keys L + depth + 16, as ref2 classes a 16-row check's rows): MLX's own scores (q * scale @ k^T, 2-head chunks as
    MTPLX's plan), each node's columns gathered into its chain order, MLX's where + precise softmax at the class's run
    length, then pv_tree. Returns [1, 24, T, 256]."""
    import numpy as np
    CFG = ref2.CFG
    HQ, HK, D = q.shape[1], k.shape[1], q.shape[3]
    gq = HQ // HK
    Nc = L + 16                                           # the chain check's key count (16 rows, padded)
    runN = lambda c, n: c if c >= CFG["kb_from"] else n if c == 0 else max(n, CFG["sdpa_bounds"][c - 1])
    cls = [ref2._klass(L + depth[n] + 16) for n in range(T)]
    out = None
    neg = mx.array(mx.finfo(q.dtype).min, q.dtype)       # MLX's fallback fills masked scores with its dtype's min
    for c in sorted(set(cls)):
        Kr = runN(c, Nc)
        Kp = -(-Kr // 32) * 32
        pad = Kr - (L + T)
        kk = k if pad == 0 else mx.concatenate([k, mx.zeros(k.shape[:2] + (pad, D), k.dtype)], axis=2)
        vv = v if pad == 0 else mx.concatenate([v, mx.zeros(v.shape[:2] + (pad, D), v.dtype)], axis=2)
        idx = np.zeros((16, Kr), np.int32)                # 16 rows always: ref2 pads short checks to 16 rows, and
        msk = np.zeros((16, Kr), bool)                    # MLX's scores GEMM kernel depends on the rows (TREE-ATT T 7)
        for n in range(T):
            idx[n, :] = np.arange(Kr)
            msk[n, :L] = True
            for j in range(16):
                cpos = L + j
                if cpos >= Kr:
                    break
                if j <= depth[n]:
                    idx[n, cpos] = L + anc[n][j]
                    msk[n, cpos] = True
                else:
                    idx[n, cpos] = cpos if cpos < L + T else L       # masked: any valid column
        for n in range(T, 16):                            # padding rows: copies of the last node's row (discarded)
            idx[n], msk[n] = idx[T - 1], msk[T - 1]
        idx_m = mx.array(idx)
        msk_m = mx.array(msk) if sel is None else _b3_mask(sel, Kr)             # B3: [16, Kr] in this msk layout
        q16 = q if T == 16 else mx.concatenate([q, mx.repeat(q[:, :, T - 1:T], 16 - T, axis=2)], axis=2)
        qs = q16 * mx.array(scale, q.dtype)
        if BATCH_HEADS[0]:                                # all of a kv head's query heads in one op each (TREE16-TIME)
            outs = []
            for j in range(HK):
                kj, vj = kk[:, j:j + 1], vv[:, j:j + 1]
                sc = mx.matmul(qs[:, j * gq:(j + 1) * gq], kj.swapaxes(-1, -2))          # [1, gq, 16, Kr]
                sg = mx.take_along_axis(sc, mx.broadcast_to(idx_m[None, None], sc.shape), axis=-1)
                sg = mx.where(msk_m[None, None], sg, neg)
                pr = mx.softmax(sg, axis=-1, precise=True)
                prp = pr if Kp == Kr else mx.concatenate([pr, mx.zeros(pr.shape[:3] + (Kp - Kr,), pr.dtype)], axis=-1)
                vjp = vj[0, 0] if Kp == Kr else mx.concatenate([vj[0, 0], mx.zeros((Kp - Kr, D), vj.dtype)], axis=0)
                y = pv_tree(prp[0].reshape(gq * 16, Kp), vjp, L, anc, depth, T, k_rule=Kr)    # [gq * 16, D]
                outs.append(y.reshape(gq, 16, D)[:, :T])
            oc = mx.concatenate(outs, axis=0).reshape(1, HQ, T, D)
            sel_c = mx.array([cls[n] == c for n in range(T)])[None, None, :, None]
            out = oc if out is None else mx.where(sel_c, oc, out)
            continue
        heads = []
        for j in range(HK):
            kj, vj = kk[:, j:j + 1], vv[:, j:j + 1]
            for h0 in range(0, gq, 2):
                qc = qs[:, j * gq + h0: j * gq + h0 + 2]                    # [1, 2, 16, D]
                sc = mx.matmul(qc, kj.swapaxes(-1, -2))                     # [1, 2, 16, Kr], MLX's own scores
                sg = mx.take_along_axis(sc, mx.broadcast_to(idx_m[None, None], sc.shape), axis=-1)
                sg = mx.where(msk_m[None, None], sg, neg)
                pr = mx.softmax(sg, axis=-1, precise=True)                  # [1, 2, 16, Kr]
                prp = pr if Kp == Kr else mx.concatenate([pr, mx.zeros(pr.shape[:3] + (Kp - Kr,), pr.dtype)], axis=-1)
                rows = prp[0]                                               # [2, 16, Kp]
                vjp = vj[0, 0] if Kp == Kr else mx.concatenate([vj[0, 0], mx.zeros((Kp - Kr, D), vj.dtype)], axis=0)
                y = pv_tree(rows.reshape(2 * 16, Kp), vjp, L, anc, depth, T, k_rule=Kr)   # [32, D]
                heads.append(y.reshape(2, 16, D)[:, :T])
        oc = mx.stack(heads, axis=0).reshape(1, HQ, T, D)
        sel_c = mx.array([cls[n] == c for n in range(T)])[None, None, :, None]
        out = oc if out is None else mx.where(sel_c, oc, out)
    return out
