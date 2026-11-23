"""GDNB (Leap, Mon 2026-11-02 19:3x; cloud Fable's dense-pass plan, ~/FableGod/CLOUD_ANSWER_DENSE.md "tomorrow's kernel"):
the GDN body of one 125B layer for 1-16 rows as ONE dispatch, written to keep REF2-T's bits: every value is computed by
the same expression as the reference chain, only the threading and the phase order change.
Reference chain (REF2-T, ref2 CFG gdn_exact): MoonShots' exact prework (conv + silu + q / k l2norm + scale + conv state;
~/Leap/third_party/ms_gdn/gdn_exact.py) -> mlx_lm's gated_delta_update (beta = sigmoid(b), g = compute_g(A_log, a,
dt_bias), then its recurrence kernel) -> MoonShots' exact gated norm (SigmoidRMSNormGated).
This kernel: one threadgroup per value head (48 x 32 NSG threads; NSG simdgroups, 8 / 16 / 32):
  A  the prework of every row (one simdgroup per (row, q | k | v part): the same lane layout and reduction as the exact
     kernel; the head's q / k are recomputed by its 3 value heads, written out once by the first);
  B  the recurrence row by row, the fp32 state in registers (simdgroup sg holds rows dv = sg + NSG j, lane l holds
     columns 4 l .. 4 l + 3, as mlx_lm's kernel), mlx_lm's loop body verbatim;
  C  the gated norm of every row (one simdgroup per row), the exact kernel's body verbatim.
g / beta: in-kernel (gates_in, default) with MLX's functors verbatim: compute_g's chain (Add, LogAddExp with utils.h
log1p, AsType, Exp = metal::precise::exp, Negative, Multiply) and Sigmoid with metal::precise::exp (MLX's own build; a
custom kernel's metal::exp is the fast one: 1 / 65,536 bf16 sigmoids differ). Proof by exhaustion
(gdnb_gates_exhaust.py): 0 / 113,246,208 g (every bf16 a x 48 heads x the 36 GDN layers' A_log / dt_bias), 0 / 65,536
beta. Each (row, head) is computed once (R-67's per-thread recomputation was the slow part). gates_in=False keeps MLX's
two ops outside (the A / B control).
q / k / v are still written out (capture-commit replays the kept rows from them).
gdn_body(qkv [1, S, 10240], z [1, S, 48, 128], a, b [1, S, 48], conv_state [1, 3, 10240], conv_w [10240, 4, 1],
         A_log, dt_bias [48], state [1, 48, 128, 128] fp32, norm_w [128], eps, scale_bf16 [1])
  -> y [1, S, 48, 128] (gated, normed; pre out_proj), q, k [1, S, 16, 128], v [1, S, 48, 128], conv_state', state'"""
import os, sys
from functools import lru_cache
import mlx.core as mx

sys.path.insert(0, os.path.expanduser("~/Leap/third_party/ms_gdn"))
from tierglu import _HDR                                     # TgSigmoid, TgMultiply: MLX's ops, verbatim

_HDR_G = _HDR + r"""
struct GbLogAddExp {                                        // mlx binary_ops.h LogAddExp, verbatim (real part)
  template <typename T>
  T operator()(T x, T y) thread {
    if (metal::isnan(x) || metal::isnan(y)) {
      return metal::numeric_limits<T>::quiet_NaN();
    }
    constexpr T inf = metal::numeric_limits<T>::infinity();
    T maxval = metal::max(x, y);
    T minval = metal::min(x, y);
    return (minval == -inf || maxval == inf)
        ? maxval
        : (maxval + log1p(metal::exp(minval - maxval)));
  };
};
struct GbExp { template <typename T> T operator()(T x) thread { return metal::precise::exp(x); }; };
struct GbSigmoid {                                          // mlx unary_ops.h Sigmoid; MLX's own build = precise exp
  template <typename T>
  T operator()(T x) thread {
    auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};
"""

_SRC = r"""
    constexpr int C = 10240, QK = 2048, HD = 128, VW = 6144, NHV = 48;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32, lane = tid % 32;
    const uint hv = threadgroup_position_in_grid.z;
    const uint hk = hv / 3;
    const bool own_qk = (hv % 3) == 0;                       // one writer per q / k channel
    threadgroup T qs[S_C * HD];
    threadgroup T ks[S_C * HD];
    threadgroup T vs[S_C * HD];
    threadgroup T os[S_C * HD];
#if GATES_IN
    threadgroup float gt[S_C];
    threadgroup T bt[S_C];
    if (tid < (uint)S_C) {                                    // compute_g + sigmoid, once per (row, head)
        const uint s = tid;
        const T sx = a[s * NHV + hv] + dtb[hv];
        const T sp = GbLogAddExp()(sx, static_cast<T>(0));
        const float e = GbExp()(static_cast<float>(alog[hv]));
        const float m = (-e) * static_cast<float>(sp);
        gt[s] = GbExp()(m);
        bt[s] = GbSigmoid()(b[s * NHV + hv]);
    }
#endif

    // ---- A: prework, every row (ms_gdn gdn_prework_exact's per-simdgroup body) ----
    for (uint task = sg; task < (uint)(S_C * 3); task += NSG) {
        const uint s = task / 3, part = task % 3;
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
                vs[s * HD + lane * 4 + i] = sv[i];
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
                    qs[s * HD + lane * 4 + i] = qv;
                    if (own_qk) q_out[s * QK + hk * HD + lane * 4 + i] = qv;
                } else {
                    ks[s * HD + lane * 4 + i] = nb;
                    if (own_qk) k_out[s * QK + hk * HD + lane * 4 + i] = nb;
                }
            }
        }
    }
    // the new conv state: rows S, S + 1, S + 2 of [state; qkv] for this head's channels
    for (uint i = tid; i < 3u * 384u; i += 32 * NSG) {
        const uint j = i / 384, cc = i % 384, part = cc / HD, off = cc % HD;
        if (part < 2 && !own_qk) continue;
        const uint c = (part == 0 ? hk * HD : (part == 1 ? QK + hk * HD : 2 * QK + hv * HD)) + off;
        const int st = S_C + (int)j;
        state_out[j * C + c] = st < 3 ? state[st * C + c] : qkv[(st - 3) * C + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- B: the recurrence, rows in order (mlx_lm gated_delta_step's loop body, verbatim per state row) ----
    constexpr int RPS = HD / NSG;                             // state rows per simdgroup: dv = sg + NSG j
    float sr[RPS][4];
    for (int j = 0; j < RPS; ++j) {
        const uint base = (hv * HD + sg + NSG * j) * HD + lane * 4;
        for (int i = 0; i < 4; ++i) sr[j][i] = static_cast<float>(dstate[base + i]);
    }
    for (int s = 0; s < S_C; ++s) {
        const threadgroup T* k_ = ks + s * HD;
        const threadgroup T* q_ = qs + s * HD;
#if GATES_IN
        const float gv = gt[s];
        const T bv = bt[s];
#else
        const float gv = g[s * NHV + hv];
        const T bv = beta[s * NHV + hv];
#endif
#if G3_C
        // G3: the same per-element operations, the RPS rows' chains interleaved (decay + dot, reductions, deltas,
        // update + dot, reductions), so RPS independent chains are in flight per step
        float kvm[RPS], dl[RPS], ov[RPS];
        for (int j = 0; j < RPS; ++j) {
            kvm[j] = 0.0f;
            for (int i = 0; i < 4; ++i) {
                auto s_idx = 4 * lane + i;
                sr[j][i] = sr[j][i] * gv;
                kvm[j] += sr[j][i] * k_[s_idx];
            }
        }
        for (int j = 0; j < RPS; ++j) kvm[j] = simd_sum(kvm[j]);
        for (int j = 0; j < RPS; ++j) {
            auto delta = (vs[s * HD + sg + NSG * j] - kvm[j]) * bv;
            dl[j] = delta;
        }
        for (int j = 0; j < RPS; ++j) {
            ov[j] = 0.0f;
            for (int i = 0; i < 4; ++i) {
                auto s_idx = 4 * lane + i;
                sr[j][i] = sr[j][i] + k_[s_idx] * dl[j];
                ov[j] += sr[j][i] * q_[s_idx];
            }
        }
        for (int j = 0; j < RPS; ++j) ov[j] = simd_sum(ov[j]);
        if (lane == 0) {
            for (int j = 0; j < RPS; ++j) os[s * HD + sg + NSG * j] = static_cast<T>(ov[j]);
        }
#else
        for (int j = 0; j < RPS; ++j) {
            const uint dv_idx = sg + NSG * j;
            float kv_mem = 0.0f;
            for (int i = 0; i < 4; ++i) {
                auto s_idx = 4 * lane + i;
                sr[j][i] = sr[j][i] * gv;
                kv_mem += sr[j][i] * k_[s_idx];
            }
            kv_mem = simd_sum(kv_mem);

            auto delta = (vs[s * HD + dv_idx] - kv_mem) * bv;

            float out = 0.0f;
            for (int i = 0; i < 4; ++i) {
                auto s_idx = 4 * lane + i;
                sr[j][i] = sr[j][i] + k_[s_idx] * delta;
                out += sr[j][i] * q_[s_idx];
            }
            out = simd_sum(out);
            if (lane == 0) {
                os[s * HD + dv_idx] = static_cast<T>(out);
            }
        }
#endif
    }
    for (int j = 0; j < RPS; ++j) {
        const uint base = (hv * HD + sg + NSG * j) * HD + lane * 4;
        for (int i = 0; i < 4; ++i) dstate_out[base + i] = static_cast<float>(sr[j][i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- C: the gated norm, every row (ms_gdn gated_norm_exact's body) ----
    for (uint s = sg; s < (uint)S_C; s += NSG) {
        float acc = 0.0f;
        float hv4[4];
        for (int i = 0; i < 4; ++i) { hv4[i] = (float)os[s * HD + lane * 4 + i]; acc += hv4[i] * hv4[i]; }
        acc = simd_sum(acc);
        const float inv = metal::precise::rsqrt(acc / 128.0f + epsv[0]);
        for (int i = 0; i < 4; ++i) {
            const T xn = static_cast<T>(hv4[i] * inv);
            const T xw = TgMultiply()(w[lane * 4 + i], xn);
            const float gt = TgSigmoid()((float)z[s * VW + hv * HD + lane * 4 + i]);
            y[s * VW + hv * HD + lane * 4 + i] = static_cast<T>(gt * (float)xw);
        }
    }
"""


NSG_DEFAULT = [int(os.environ.get("GDNB_NSG", "8"))]       # simdgroups per head's threadgroup (8 / 16 / 32)
G3_DEFAULT = [os.environ.get("GDNB_G3") == "1"]             # G3 (row-interleaved recurrence), off unless asked


@lru_cache(None)
def _k(gates_in, g3=False):
    gin = ["a", "b", "dtb", "alog"] if gates_in else ["g", "beta"]
    return mx.fast.metal_kernel(
        name=f"leap_gdn_body{'_gi' if gates_in else ''}{'_g3' if g3 else ''}",
        input_names=["qkv", "state", "cw", "scale"] + gin + ["dstate", "z", "w", "epsv"],
        output_names=["y", "state_out", "dstate_out", "q_out", "k_out", "v_out"],
        source=f"#define GATES_IN {int(gates_in)}\n#define G3_C {int(g3)}\n" + _SRC, header=_HDR_G)


def gdn_body(qkv, z, a, b, conv_state, conv_w, A_log, dt_bias, state, norm_w, eps, scale_bf16, gates_in=True, nsg=None,
             g3=None):
    S = qkv.shape[-2]
    assert qkv.shape[-1] == 10240 and 1 <= S <= 16 and state.dtype == mx.float32 and qkv.dtype == mx.bfloat16
    nsg = nsg or NSG_DEFAULT[0]
    g3 = G3_DEFAULT[0] if g3 is None else bool(g3)
    assert nsg in (8, 16, 32)
    if gates_in:
        assert a.dtype == b.dtype == A_log.dtype == dt_bias.dtype == mx.bfloat16
        gin = [a.reshape(-1), b.reshape(-1), dt_bias, A_log]
    else:
        from mlx_lm.models.gated_delta import compute_g
        gin = [compute_g(A_log, a, dt_bias).reshape(-1), mx.sigmoid(b).reshape(-1)]   # mlx_lm's own two ops
    y, cs, ds, q, k, v = _k(bool(gates_in), g3)(
        inputs=[qkv.reshape(-1), conv_state.reshape(-1), conv_w.reshape(-1), scale_bf16.reshape(-1)] + gin
               + [state.reshape(-1), z.reshape(-1), norm_w.astype(qkv.dtype), mx.array([eps], mx.float32)],
        template=[("T", qkv.dtype), ("S_C", S), ("NSG", nsg)], grid=(32 * nsg, 1, 48), threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(S * 6144,), (3 * 10240,), (48 * 128 * 128,), (S * 2048,), (S * 2048,), (S * 6144,)],
        output_dtypes=[qkv.dtype, qkv.dtype, mx.float32, qkv.dtype, qkv.dtype, qkv.dtype])
    return (y.reshape(1, S, 48, 128), q.reshape(1, S, 16, 128), k.reshape(1, S, 16, 128), v.reshape(1, S, 48, 128),
            cs.reshape(1, 3, 10240), ds.reshape(1, 48, 128, 128))
