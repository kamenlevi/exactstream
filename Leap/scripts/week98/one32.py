"""one32.py (Leap, Sun 15 Nov 2026 11:3x; CHECK55 step 1, data/week98/check55_plan.md): one dispatch per op for the
exact 17-32-row check. v32 runs every row-local op at 17-32 rows as two 16-row calls. The second call costs a full
latency-bound chunk (G32T: the step at 17 rows), even though the weights come from cache. Here each piece covers both
16-row tiles in ONE dispatch, with each tile computed by the 16-row form's own instructions, so each row is bitwise
the certified 16-row form by construction. Pieces (each bit-tested by one32_test.py before it is wired into v32):
  qmm16c2  REF2's qmm16c (4-bit g64, N >= 1024, N % 64 == 0: the dense projections, the shared down, the head) with a
           second row tile on the grid's y axis;
  mlx_b16  MLX's own 16-row call for the shapes REF2 leaves to MLX (small N, the 8-bit router, the K-73 shared cat),
           as a batch of 2 x 16 rows in one call, if MLX keeps its 16-row kernel per batch (one32_test decides).
Not imported by any live loop."""
from functools import lru_cache
import os
import sys
import mlx.core as mx

sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
import qmm16 as Q16

HI = 32


def _src2():
    s = Q16._SRC3
    subs = [("    const short qid = lane >> 2;\n",
             "    const uint mt = threadgroup_position_in_grid.y;            // one32: the 16-row tile (0 or 1)\n"
             "    const short qid = lane >> 2;\n"),
            ("const device T* xr0 = x + (size_t)fm * K_C + 16 * fq;",
             "const device T* xr0 = x + (size_t)(mt * 16 + fm) * K_C + 16 * fq;"),
            ("const device T* xr1 = x + (size_t)(fm + 8) * K_C + 16 * fq;",
             "const device T* xr1 = x + (size_t)(mt * 16 + fm + 8) * K_C + 16 * fq;"),
            ("        y[(size_t)m * N_C + n] = static_cast<T>(acc[i]);",
             "        y[(size_t)(mt * 16 + m) * N_C + n] = static_cast<T>(acc[i]);")]
    for old, new in subs:
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    return s


@lru_cache(None)
def _k2():
    return mx.fast.metal_kernel(name="leap_one32_qmm16c2", input_names=["x", "w", "scales", "biases"], output_names=["y"],
                                source=_src2(), header=Q16._HDR)


def qmm16c2(x, w, scales, biases, ks=1, nt=1):
    """qmm16c for 17-32 rows in one dispatch: rows [0, 16) and [16, 32) each as qmm16c's 16-row tile (zero-padded)"""
    M, K = x.shape
    N = w.shape[0]
    assert 16 < M <= HI and N % (32 * nt) == 0 and K % (64 * ks) == 0
    xp = x if M == HI else mx.concatenate([x, mx.zeros((HI - M, K), x.dtype)], 0)
    (y,) = _k2()(inputs=[xp, w, scales, biases], template=[("T", x.dtype), ("K_C", K), ("N_C", N), ("KS", ks), ("NT", nt)],
                 grid=(N // 32 * 32 * ks, 2, 1), threadgroup=(32 * ks * nt, 1, 1), output_shapes=[(HI, N)],
                 output_dtypes=[x.dtype])
    return y[:M]


def mlx_b16(x, w, scales, biases, group_size, bits, mode="affine"):
    """MLX's 16-row quantized call for 17-32 rows as ONE call on a [2, 16, K] batch (zero-padded rows)"""
    M, K = x.shape
    xp = x if M == HI else mx.concatenate([x, mx.zeros((HI - M, K), x.dtype)], 0)
    y = mx.quantized_matmul(xp.reshape(2, 16, K), w, scales, biases, transpose=True, group_size=group_size, bits=bits,
                            mode=mode)
    return y.reshape(HI, -1)[:M]


# ---- qmv32: MLX's own vector kernel (qmv_fast_impl / qmv_impl, from the installed quantized.h through leapk's D2
# header) launched over 32 rows. MLX 0.32.2 serves an M-row 4-bit / 8-bit x @ w.T with qmv below its batch limit (leapk
# routes_splitk: 25 rows at K <= 4096, N <= 4096; 13 past that), so REF2's padded 16-row calls on the small shapes are
# qmv; the kernel takes its x row from the threadgroup's x index, so each row's arithmetic does not depend on M.
# qmv_fast when N % 8 == 0 and K % 512 == 0 (MLX's choice), else qmv; one32_test checks both against MLX's 16-row call.
_QV_SRC = """
    const uint3 tid = threadgroup_position_in_grid;
    QFN<T, 64, BITS>(w, scales, biases, x, y, GK, GN, tid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""
_QVK = {}


def qmv32(x, w, scales, biases, bits, fast=None):
    """x [M <= 32, K] -> [M, N]: MLX's qmv per row, all 32 (zero-padded) rows in one dispatch"""
    import leapk as LK
    M, K = x.shape
    N = int(w.shape[0])
    if fast is None:
        fast = N % 8 == 0 and K % 512 == 0
    xp = x if M == HI else mx.concatenate([x, mx.zeros((HI - M, K), x.dtype)], 0)
    key = (N, K, bits, fast)
    if key not in _QVK:
        src = _QV_SRC.replace("QFN", "qmv_fast_impl" if fast else "qmv_impl").replace("BITS", str(bits))
        _QVK[key] = mx.fast.metal_kernel(name=f"leap_one32_qmv_{N}_{K}_{bits}_{int(fast)}",
                                         input_names=["w", "scales", "biases", "x"], output_names=["y"],
                                         header=LK._q_header() + LK.consts(GN=N, GK=K), source=src)
    (y,) = _QVK[key](inputs=[w, scales, biases, xp], template=[("T", x.dtype)],
                     grid=(32 * HI, 2 * ((N + 7) // 8), 1), threadgroup=(32, 2, 1), output_shapes=[(HI, N)],
                     output_dtypes=[x.dtype])
    return y[:M]


# ---- qmvw32: MLX's qmv_wide_impl (the kernel MLX 0.32.2 runs for M-row calls on the small shapes: its 16-row output is
# not qmv's, one32's probe) over 32 rows. Per vector the arithmetic depends only on k_lanes (each lane's strided
# groups, decoded in 8-value sub-chunks, then a fixed shuffle ladder), not on vecs_per_tg or M, so a 32-row launch with
# MLX's k_lanes reproduces its 16-row bits per row. k_lanes per shape: found by matching MLX's 16-row output (KL below).
_QW_SRC = """
    const uint3 tid = threadgroup_position_in_grid;
    qmv_wide_impl<T, 64, BITS, VPT, KL>(w, scales, biases, x, y, GK, GN, GM, tid, simdgroup_index_in_threadgroup,
                                        thread_index_in_simdgroup);
"""
_QWK = {}
KL = {}                                                  # (N, K, bits) -> MLX's k_lanes (filled by one32_test / probe)


def qmvw32(x, w, scales, biases, bits, k_lanes, vecs_per_tg=4, rows=HI):
    """x [M <= rows, K] -> [M, N]: MLX's qmv_wide per row (k_lanes as MLX's 16-row call), all rows in one dispatch"""
    import leapk as LK
    M, K = x.shape
    N = int(w.shape[0])
    xp = x if M == rows else mx.concatenate([x, mx.zeros((rows - M, K), x.dtype)], 0)
    key = (N, K, bits, k_lanes, vecs_per_tg, rows)
    if key not in _QWK:
        src = _QW_SRC.replace("BITS", str(bits)).replace("VPT", str(vecs_per_tg)).replace("KL", str(k_lanes))
        _QWK[key] = mx.fast.metal_kernel(name=f"leap_one32_qmvw_{N}_{K}_{bits}_{k_lanes}_{vecs_per_tg}_{rows}",
                                         input_names=["w", "scales", "biases", "x"], output_names=["y"],
                                         header=LK._q_header() + LK.consts(GN=N, GK=K, GM=rows), source=src)
    rps = 32 // k_lanes
    (y,) = _QWK[key](inputs=[w, scales, biases, xp], template=[("T", x.dtype)],
                     grid=(32 * ((rows + vecs_per_tg - 1) // vecs_per_tg), 2 * ((N + 2 * rps - 1) // (2 * rps)), 1),
                     threadgroup=(32, 2, 1), output_shapes=[(rows, N)], output_dtypes=[x.dtype])
    return y[:M]


# ---- gdn_body32: scripts/gdnbody.py's one-dispatch GDN body (non-G3, gates in-kernel) for 17-32 rows. The 16-row body
# keeps a chunk's q / k / v / out rows in threadgroup memory (4 x 16 x 128 bf16); 32 rows would pass the 32 KB limit, so
# this one loops over 16-row tiles inside the dispatch: every row's prework, recurrence step and gated norm are the 16-row
# body's expressions verbatim, the fp32 state stays in registers across the tiles, and the conv window, the gates and
# every input / output are indexed by the row's global position. Same bits as chunk 1 then chunk 2 by construction
# (one32_gdn_test.py checks it).
_GDN32_SRC = r"""
    constexpr int C = 10240, QK = 2048, HD = 128, VW = 6144, NHV = 48;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32, lane = tid % 32;
    const uint hv = threadgroup_position_in_grid.z;
    const uint hk = hv / 3;
    const bool own_qk = (hv % 3) == 0;                       // one writer per q / k channel
    threadgroup T qs[16 * HD];
    threadgroup T ks[16 * HD];
    threadgroup T vs[16 * HD];
    threadgroup T os[16 * HD];
    threadgroup float gt[16];
    threadgroup T bt[16];

    // the new conv state: rows S, S + 1, S + 2 of [state; qkv] for this head's channels (inputs only)
    for (uint i = tid; i < 3u * 384u; i += 32 * NSG) {
        const uint j = i / 384, cc = i % 384, part = cc / HD, off = cc % HD;
        if (part < 2 && !own_qk) continue;
        const uint c = (part == 0 ? hk * HD : (part == 1 ? QK + hk * HD : 2 * QK + hv * HD)) + off;
        const int st = S_C + (int)j;
        state_out[j * C + c] = st < 3 ? state[st * C + c] : qkv[(st - 3) * C + c];
    }

    constexpr int RPS = HD / NSG;                             // state rows per simdgroup: dv = sg + NSG j
    float sr[RPS][4];
    for (int j = 0; j < RPS; ++j) {
        const uint base = (hv * HD + sg + NSG * j) * HD + lane * 4;
        for (int i = 0; i < 4; ++i) sr[j][i] = static_cast<float>(dstate[base + i]);
    }

    for (int t0 = 0; t0 < S_C; t0 += 16) {
        const int ST = min(16, S_C - t0);
        if (tid < (uint)ST) {                                 // compute_g + sigmoid, once per (row, head)
            const uint s = t0 + tid;
            const T sx = a[s * NHV + hv] + dtb[hv];
            const T sp = GbLogAddExp()(sx, static_cast<T>(0));
            const float e = GbExp()(static_cast<float>(alog[hv]));
            const float m = (-e) * static_cast<float>(sp);
            gt[tid] = GbExp()(m);
            bt[tid] = GbSigmoid()(b[s * NHV + hv]);
        }

        // ---- A: prework of this tile's rows ----
        for (uint task = sg; task < (uint)(ST * 3); task += NSG) {
            const uint sl = task / 3, part = task % 3;
            const uint s = t0 + sl;
            const uint gch = part == 0 ? hk * HD : (part == 1 ? QK + hk * HD : 2 * QK + hv * HD);
            const uint c0 = gch + lane * 4;
            T sv[4];
            for (int i = 0; i < 4; ++i) {
                const uint c = c0 + i;
                float acc = 0.0f;
                for (int t = 0; t < 4; ++t) {
                    const int st = (int)s + t;
                    const float xv = st < 3 ? (float)state[st * C + c] : (float)qkv[(st - 3) * C + c];
                    acc = acc + (float)cw[c * 4 + t] * xv;
                }
                const T co = static_cast<T>(acc);
                sv[i] = TgMultiply()(co, TgSigmoid()(co));
            }
            if (part == 2) {
                for (int i = 0; i < 4; ++i) {
                    vs[sl * HD + lane * 4 + i] = sv[i];
                    v_out[s * VW + hv * HD + lane * 4 + i] = sv[i];
                }
            } else {
                float xf[4], p = 0.0f;
                for (int i = 0; i < 4; ++i) xf[i] = (float)sv[i];
                for (int i = 0; i < 4; ++i) { const float sq = xf[i] * xf[i]; p += sq; }
                const float tot = simd_sum(p);
                const float r = metal::precise::rsqrt(tot + 1e-6f);
                for (int i = 0; i < 4; ++i) {
                    const T nb = static_cast<T>(xf[i] * r);
                    if (part == 0) {
                        const T qv = TgMultiply()(scale[0], nb);
                        qs[sl * HD + lane * 4 + i] = qv;
                        if (own_qk) q_out[s * QK + hk * HD + lane * 4 + i] = qv;
                    } else {
                        ks[sl * HD + lane * 4 + i] = nb;
                        if (own_qk) k_out[s * QK + hk * HD + lane * 4 + i] = nb;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- B: the recurrence over this tile's rows in order (mlx_lm gated_delta_step's loop body, verbatim) ----
        for (int sl = 0; sl < ST; ++sl) {
            const threadgroup T* k_ = ks + sl * HD;
            const threadgroup T* q_ = qs + sl * HD;
            const float gv = gt[sl];
            const T bv = bt[sl];
            for (int j = 0; j < RPS; ++j) {
                const uint dv_idx = sg + NSG * j;
                float kv_mem = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    auto s_idx = 4 * lane + i;
                    sr[j][i] = sr[j][i] * gv;
                    kv_mem += sr[j][i] * k_[s_idx];
                }
                kv_mem = simd_sum(kv_mem);

                auto delta = (vs[sl * HD + dv_idx] - kv_mem) * bv;

                float out = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    auto s_idx = 4 * lane + i;
                    sr[j][i] = sr[j][i] + k_[s_idx] * delta;
                    out += sr[j][i] * q_[s_idx];
                }
                out = simd_sum(out);
                if (lane == 0) {
                    os[sl * HD + dv_idx] = static_cast<T>(out);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- C: the gated norm of this tile's rows ----
        for (uint sl = sg; sl < (uint)ST; sl += NSG) {
            const uint s = t0 + sl;
            float acc = 0.0f;
            float hv4[4];
            for (int i = 0; i < 4; ++i) { hv4[i] = (float)os[sl * HD + lane * 4 + i]; acc += hv4[i] * hv4[i]; }
            acc = simd_sum(acc);
            const float inv = metal::precise::rsqrt(acc / 128.0f + epsv[0]);
            for (int i = 0; i < 4; ++i) {
                const T xn = static_cast<T>(hv4[i] * inv);
                const T xw = TgMultiply()(w[lane * 4 + i], xn);
                const float gt_ = TgSigmoid()((float)z[s * VW + hv * HD + lane * 4 + i]);
                y[s * VW + hv * HD + lane * 4 + i] = static_cast<T>(gt_ * (float)xw);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int j = 0; j < RPS; ++j) {
        const uint base = (hv * HD + sg + NSG * j) * HD + lane * 4;
        for (int i = 0; i < 4; ++i) dstate_out[base + i] = static_cast<float>(sr[j][i]);
    }
"""


@lru_cache(None)
def _kgdn32():
    import gdnbody as GB
    return mx.fast.metal_kernel(
        name="leap_one32_gdn_body32", input_names=["qkv", "state", "cw", "scale", "a", "b", "dtb", "alog", "dstate", "z",
                                                   "w", "epsv"],
        output_names=["y", "state_out", "dstate_out", "q_out", "k_out", "v_out"], source=_GDN32_SRC, header=GB._HDR_G)


def gdn_body32(qkv, z, a, b, conv_state, conv_w, A_log, dt_bias, state, norm_w, eps, scale_bf16, nsg=8):
    """gdnbody.gdn_body (gates in-kernel, non-G3) for 1-32 rows in ONE dispatch; same interface and outputs"""
    S = qkv.shape[-2]
    assert qkv.shape[-1] == 10240 and 1 <= S <= HI and state.dtype == mx.float32 and qkv.dtype == mx.bfloat16
    assert a.dtype == b.dtype == A_log.dtype == dt_bias.dtype == mx.bfloat16 and nsg in (8, 16, 32)
    y, cs, ds, q, k, v = _kgdn32()(
        inputs=[qkv.reshape(-1), conv_state.reshape(-1), conv_w.reshape(-1), scale_bf16.reshape(-1), a.reshape(-1),
                b.reshape(-1), dt_bias, A_log, state.reshape(-1), z.reshape(-1), norm_w.astype(qkv.dtype),
                mx.array([eps], mx.float32)],
        template=[("T", qkv.dtype), ("S_C", S), ("NSG", nsg)], grid=(32 * nsg, 1, 48), threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(S * 6144,), (3 * 10240,), (48 * 128 * 128,), (S * 2048,), (S * 2048,), (S * 6144,)],
        output_dtypes=[qkv.dtype, qkv.dtype, mx.float32, qkv.dtype, qkv.dtype, qkv.dtype])
    return (y.reshape(1, S, 48, 128), q.reshape(1, S, 16, 128), k.reshape(1, S, 16, 128), v.reshape(1, S, 48, 128),
            cs.reshape(1, 3, 10240), ds.reshape(1, 48, 128, 128))


# ---- hc_read32b: the HC read (GatedResidual's verify read on ref2's leapk path, fuse_gate) for 17-32 rows with ONE
# dispatch per kernel. hc32.py's S4x2 (two accumulators per simdgroup) was bitwise but slower than two 16-row reads
# (C6); here leapk's own 16-row S4 tile code runs with a second row tile on the grid's y axis (its own threadgroups,
# the weight tiles re-read from cache), so each row is the 16-row S4's arithmetic by construction. The norm, hc32's
# _ACCUM_GATE32 (the same per-element partition order at a 32-row stride), MLX's up GEMM (C2: the 32-row call is
# bitwise to 2 x 16 for this shape) and the tail are row-generic, as in hc32 (bitwise in C3d).
def _s4_src2():
    import leapk as LK
    s = LK._S4_SRC
    subs = [("    const int tg = int(threadgroup_position_in_grid.x);\n",
             "    const int tg = int(threadgroup_position_in_grid.x);\n"
             "    const int mt = int(threadgroup_position_in_grid.y);          // one32: the 16-row tile\n"),
            ("    const device T* Ap = x + k0;\n", "    const device T* Ap = x + size_t(mt) * 16 * GK + k0;\n"),
            ("    device float* Cp = cs + size_t(part) * 16 * GN + c_col;\n",
             "    device float* Cp = cs + (size_t(part) * 32 + size_t(mt) * 16) * GN + c_col;\n")]
    for old, new in subs:
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    return s


_S42K = {}


def s4_partials2(a, w, pf=4):
    """[1, 32, K] x [N, K]^T -> split-K partials cs [NP, 32, N] (fp32): leapk.s4_partials per 16-row tile, one dispatch"""
    import leapk as LK
    N, K = w.shape
    PS = 4096
    NP, NT = (K + PS - 1) // PS, (N + 31) // 32
    key = (N, K, pf)
    if key not in _S42K:
        _S42K[key] = mx.fast.metal_kernel(
            name=f"leap_one32_s4x2t_{N}_{K}_{pf}", input_names=["x", "w"], output_names=["cs"], source=_s4_src2(),
            header=LK._nax_header() + "\n" + LK.consts(GN=N, GK=K, GPS=PS, GNP=NP, GNT=NT, GPF=pf))
    (cs,) = _S42K[key](inputs=[a, w], template=[("T", mx.bfloat16)], grid=(32 * NT * NP, 2, 1), threadgroup=(32, 1, 1),
                       output_shapes=[(NP, 32, N)], output_dtypes=[mx.float32])
    return cs, NP


def hc_read32b(g, wcat, hyper, pending=None):
    """hc32.hc_read32 with s4_partials2 in place of S4x2: hyper [1, 32, HC x D] (zero-padded rows) ->
    (mixed [1, 32, D], written, inject [1, 32, HC])"""
    import leapk as LK
    import hc32
    rows = HI
    hc, width = g.hc_count, g.hidden_size
    lowrank = int(g.input_mix_weight_down.weight.shape[0])
    k = LK._hc_kernels(hc, width, lowrank, float(g.hc_norm.eps))
    lead = hyper.shape[:-1]
    hcd = hc * width
    x = hyper.reshape(rows, hcd)
    BF = mx.bfloat16
    if pending is not None:
        bo, inj = pending[0].reshape(rows, width), pending[1].reshape(rows, hc)
        written, normed = k["norm_p"](inputs=[x, bo, inj, g.hc_norm.weight], template=[("T", BF)],
                                      grid=(rows * hc * 256, 1, 1), threadgroup=(256, 1, 1),
                                      output_shapes=[(rows, hcd), (rows, hcd)], output_dtypes=[BF, BF])
        written = written.reshape(hyper.shape)
    else:
        (normed,) = k["norm_n"](inputs=[x, x, x, g.hc_norm.weight], template=[("T", BF)],
                                grid=(rows * hc * 256, 1, 1), threadgroup=(256, 1, 1),
                                output_shapes=[(rows, hcd)], output_dtypes=[BF])
        written = hyper
    cs, NP = s4_partials2(normed.reshape(1, rows, hcd), wcat)
    N = int(wcat.shape[0])
    assert N == lowrank + hc, (N, lowrank, hc)
    mix, inject = hc32._ACCUM_GATE32(inputs=[cs, k["tab"]], template=[("T", BF), ("N", N), ("NP", NP), ("ROWS", rows),
                                     ("LR", lowrank), ("HC", hc)], grid=(32 * N, 1, 1), threadgroup=(256, 1, 1),
                                     output_shapes=[(rows, lowrank), (rows, hc)], output_dtypes=[BF, BF])
    up = mix.reshape(1, rows, lowrank) @ g.input_mix_weight_up.weight.T
    (mixed,) = k["tail"](inputs=[up, normed, k["tab"]], template=[("T", BF), ("D", width), ("HC", hc),
                         ("NOUT", rows * width)], grid=(rows * width, 1, 1), threadgroup=(256, 1, 1),
                         output_shapes=[(rows, width)], output_dtypes=[BF])
    return mixed.reshape(*lead, width), written, inject.reshape(*lead, hc)
