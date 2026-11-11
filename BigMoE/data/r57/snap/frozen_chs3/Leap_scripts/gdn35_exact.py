"""L1-15 (Leap, Sat 2026-10-24): exact fused pieces of the 35B's GatedDeltaNet mixer (mlx_lm qwen3_5), adapted from
MoonShots' patches/planes_engine/gdn_exact.py (125B / MTPLX shapes; their method: replicate every rounding point,
reduction order and MLX elementwise struct, see ~/MoonShots/FINDINGS.md 'Method').
- prework35: concat [conv_state; qkv] -> depthwise conv1d k=4 (fp32 accumulate, one bf16 rounding) -> nn.silu in bf16 ->
  split -> q, k: mx.fast.rms_norm(x, None, 1e-6) (lane l squares its 4 contiguous values, simd_sum, precise rsqrt of
  sum / 128 + eps, bf16(x * inv)) -> bf16 scalar multiply (q by bf16(1/128), k by bf16(128^-0.5)), plus the new conv
  state. One dispatch. It reads qkv with a row stride, so it can take the fused in_proj output directly.
- gated_norm35: Qwen3NextRMSNormGated: x = rms_norm(y, w, eps) = Multiply(w, bf16(y * inv)) in bf16, then
  _precise_swiglu: bf16(silu(fp32(z)) * fp32(x)). One dispatch; reads z with a row stride.
The recurrence stays mlx_lm's kernel (fp32 state, T at run time: per-token arithmetic does not depend on T).
Layout (the 35B): conv_dim 8192 = q 16 x 128 | k 16 x 128 | v 32 x 128, kernel 4; value heads 32 x 128."""
from functools import lru_cache
import mlx.core as mx

_HDR = r"""
struct TgSigmoid {                       // MLX 0.32.4's Sigmoid op struct, verbatim
  template <typename T>
  T operator()(T x) thread {
    auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));      // MLX 0.32.4 (0.32.2: metal::exp)
    return (x < 0) ? y : 1 - y;
  }
};
struct TgMultiply {
  template <typename T>
  T operator()(T x, T y) thread {
    return x * y;
  }
};
struct TgLogAddExp {                     // MLX 0.32.4's LogAddExp op struct, verbatim (nn.softplus = logaddexp(x, 0))
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
  }
};
"""

_PRE = r"""
    constexpr int C = 8192, QK = 2048, NG = C / 128;          // 64 groups of 128 channels: 16 q heads, 16 k, 32 v
    const uint lane = thread_index_in_simdgroup;
    const uint g = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    const uint s = threadgroup_position_in_grid.y;           // row
    if (g >= (uint)NG) return;
    const uint c0 = g * 128 + lane * 4;                       // this lane's 4 contiguous channels
    T sv[4];
    for (int i = 0; i < 4; ++i) {
        const uint c = c0 + i;
        // stream(t): t < 3 -> conv state row t, else qkv row t - 3; the window of row s = stream[s .. s+3]
        float acc = 0.0f;
        for (int t = 0; t < 4; ++t) {
            const int st = (int)s + t;
            const float xv = st < 3 ? (float)state[st * C + c] : (float)qkv[(size_t)(st - 3) * LD + c];
            acc = acc + (float)cw[c * 4 + t] * xv;
        }
        const T co = static_cast<T>(acc);                    // conv1d's bf16 output
        sv[i] = TgMultiply()(co, TgSigmoid()(co));            // nn.silu in bf16
    }
    if (g >= (uint)(2 * QK / 128)) {                          // v channels: silu only
        for (int i = 0; i < 4; ++i) v_out[s * (C - 2 * QK) + (c0 - 2 * QK) + i] = sv[i];
    } else {
        float xf[4], p = 0.0f;
        for (int i = 0; i < 4; ++i) xf[i] = (float)sv[i];
        for (int i = 0; i < 4; ++i) { const float sq = xf[i] * xf[i]; p += sq; }
        const float tot = simd_sum(p);
        const float r = metal::precise::rsqrt(tot / 128.0f + 1e-6f);
        for (int i = 0; i < 4; ++i) {
            const T nb = static_cast<T>(xf[i] * r);
            if (g < (uint)(QK / 128)) q_out[s * QK + c0 + i] = TgMultiply()(scale[0], nb);
            else k_out[s * QK + (c0 - QK) + i] = TgMultiply()(scale[1], nb);
        }
    }
    if (s == 0) {                                             // new conv state: rows S, S+1, S+2 of the stream
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 4; ++i) {
                const uint c = c0 + i;
                const int st = nrows + j;
                state_out[j * C + c] = st < 3 ? state[st * C + c] : qkv[(size_t)(st - 3) * LD + c];
            }
    }
"""

_GN = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    if (row >= (uint)nrows) return;                           // row = token * 32 + value head
    const device T* hr = h + (size_t)row * 128 + lane * 4;
    const device T* gr = gate + (size_t)(row / 32) * LDZ + ZOFF + (row % 32) * 128 + lane * 4;
    float acc = 0.0f;
    float hv[4];
    for (int i = 0; i < 4; ++i) { hv[i] = (float)hr[i]; acc += hv[i] * hv[i]; }
    acc = simd_sum(acc);
    const float inv = metal::precise::rsqrt(acc / 128.0f + epsv[0]);
    for (int i = 0; i < 4; ++i) {
        const T xn = static_cast<T>(hv[i] * inv);
        const T xw = TgMultiply()(w[lane * 4 + i], xn);
        const float zf = (float)gr[i];
        const float sl = TgMultiply()(zf, TgSigmoid()(zf));  // nn.silu on fp32
        out[(size_t)row * 128 + lane * 4 + i] = static_cast<T>(sl * (float)xw);
    }
"""


@lru_cache(None)
def _kpre():
    return mx.fast.metal_kernel(name="gdn35_prework_exact", input_names=["qkv", "state", "cw", "scale", "nrows"],
                                output_names=["q_out", "k_out", "v_out", "state_out"], source=_PRE, header=_HDR)


@lru_cache(None)
def _kgn():
    return mx.fast.metal_kernel(name="gdn35_gated_norm_exact", input_names=["h", "gate", "w", "epsv", "nrows"],
                                output_names=["out"], source=_GN, header=_HDR)


def scales35(dtype=mx.bfloat16):
    """the two Python scalars of qwen3_5's GatedDeltaNet as MLX turns them into operands (weak typing: bf16)"""
    inv = 128 ** -0.5
    return mx.concatenate([mx.array([inv ** 2], dtype), mx.array([inv], dtype)])


def prework35(qkv, conv_state, conv_w, scales, sg=4):
    """qkv [S, LD] (first 8192 columns used), conv_state [3, 8192], conv_w [8192, 4] -> q [S, 2048], k [S, 2048],
    v [S, 4096], new conv state [3, 8192]"""
    S, LD = qkv.shape          # S is a run-time input (one compiled kernel for every row count)
    return _kpre()(inputs=[qkv, conv_state.reshape(3, 8192), conv_w.reshape(8192, 4), scales, S],
                   template=[("T", qkv.dtype), ("LD", LD)], grid=(64 // sg * 32 * sg, S, 1),
                   threadgroup=(32 * sg, 1, 1), output_shapes=[(S, 2048), (S, 2048), (S, 4096), (3, 8192)],
                   output_dtypes=[qkv.dtype] * 4)


def gated_norm35(h, z, weight, eps, zoff=0):
    """h [S, 32, 128] (or [S * 32 * 128]) bf16, z [S, LDZ] with the 32 x 128 gate values in columns zoff .. zoff + 4095
    (the fused in_proj output: zoff 8192) -> Qwen3NextRMSNormGated(h, z) [S * 32, 128], bitwise"""
    n = h.size // 128
    ldz = z.shape[-1]
    (o,) = _kgn()(inputs=[h.reshape(-1), z, weight, mx.array([eps], mx.float32), n],
                  template=[("T", h.dtype), ("LDZ", ldz), ("ZOFF", zoff)],
                  grid=((n + 3) // 4 * 128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(n, 128)],
                  output_dtypes=[h.dtype])
    return o


# ---- the recurrence: mlx_lm's gated_delta kernel (models/gated_delta.py, mlx_lm 0.31.3), same per-token expressions in
# the same order, but each chunk of up to TM tokens' q, k (one key head), v (the TG's 4 value rows), g and beta is first
# loaded into threadgroup memory, so the serial loop does not wait on device loads. T stays a run-time input and TM is
# fixed, so decode (T = 1) and verify (T = 16) run the same binary.
_GDR = r"""
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;
    constexpr int TM = 16;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.y * 32 + dk_idx;   // 0..127
    const uint dv0 = threadgroup_position_in_grid.y * 4;
    const int TT = T;

    threadgroup InT sq[TM * Dk];
    threadgroup InT sk[TM * Dk];
    threadgroup InT sv[TM * 4];
    threadgroup float sg_[TM];
    threadgroup InT sb[TM];

    // state_in, state_out: [B, Hv, Dv, Dk]
    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = static_cast<float>(i_state[s_idx]);
    }

    for (int t0 = 0; t0 < TT; t0 += TM) {
      const int nt = min(TM, TT - t0);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int j = tid; j < nt * Dk; j += 128) {
        const int t = t0 + j / Dk, d = j % Dk;
        sq[j] = q[((b_idx * TT + t) * Hk + hk_idx) * Dk + d];
        sk[j] = k[((b_idx * TT + t) * Hk + hk_idx) * Dk + d];
      }
      for (int j = tid; j < nt * 4; j += 128) {
        const int t = t0 + j / 4;
        sv[j] = v[((b_idx * TT + t) * Hv + hv_idx) * Dv + dv0 + j % 4];
      }
      for (int j = tid; j < nt; j += 128) {
        sg_[j] = g[(b_idx * TT + t0 + j) * Hv + hv_idx];
        sb[j] = beta[(b_idx * TT + t0 + j) * Hv + hv_idx];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (int tt = 0; tt < nt; ++tt) {
        float kv_mem = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = state[i] * sg_[tt];
          kv_mem += state[i] * sk[tt * Dk + s_idx];
        }
        kv_mem = simd_sum(kv_mem);

        auto delta = (sv[tt * 4 + (dv_idx - dv0)] - kv_mem) * sb[tt];

        float out = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = state[i] + sk[tt * Dk + s_idx] * delta;
          out += state[i] * sq[tt * Dk + s_idx];
        }
        out = simd_sum(out);
        if (thread_index_in_simdgroup == 0) {
          y[((b_idx * TT + t0 + tt) * Hv + hv_idx) * Dv + dv_idx] = static_cast<InT>(out);
        }
      }
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      o_state[s_idx] = static_cast<StT>(state[i]);
    }
"""


@lru_cache(None)
def _kgdr():
    return mx.fast.metal_kernel(name="gdn35_recurrence_tg", input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
                                output_names=["y", "state_out"], source=_GDR)


def recurrence_tg(q, k, v, g, beta, state):
    """drop-in for mlx_lm's gated_delta_kernel (no mask): q, k [B, T, Hk, Dk], v [B, T, Hv, Dv], g [B, T, Hv] fp32,
    beta [B, T, Hv], state [B, Hv, Dv, Dk] fp32 -> y [B, T, Hv, Dv], new state"""
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    return _kgdr()(inputs=[q, k, v, g, beta, state, T],
                   template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
                   grid=(32, Dv, B * Hv), threadgroup=(32, 4, 1),
                   output_shapes=[(B, T, Hv, Dv), state.shape], output_dtypes=[q.dtype, state.dtype])


# ---- the recurrence with 4 value rows per thread: the same per-row expressions in the same order as mlx_lm's kernel,
# interleaved over 4 independent rows (latency hiding) and 4x fewer simdgroups (the T loop is serial per row, so the
# T = 16 cost was waves x steps x step latency). One binary for every T (decode and verify).
_GDR4 = r"""
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;
    constexpr int TM = 16, R = 4;

    auto dk_idx = thread_position_in_threadgroup.x;
    const uint sgi = thread_position_in_threadgroup.y;              // 0..3
    const uint tid = sgi * 32 + dk_idx;                             // 0..127
    const uint dv0 = threadgroup_position_in_grid.y * 16;           // this TG's 16 value rows
    const uint dvr = dv0 + sgi * R;                                 // this simdgroup's 4 rows
    const int TT = T;

    threadgroup InT sq[TM * Dk];
    threadgroup InT sk[TM * Dk];
    threadgroup InT sv[TM * 16];
    threadgroup float sg_[TM];
    threadgroup InT sb[TM];

    float state[R][n_per_t];
    for (int r = 0; r < R; ++r) {
      auto i_state = state_in + (n * Dv + dvr + r) * Dk;
      for (int i = 0; i < n_per_t; ++i) state[r][i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
    }

    for (int t0 = 0; t0 < TT; t0 += TM) {
      const int nt = min(TM, TT - t0);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int j = tid; j < nt * Dk; j += 128) {
        const int t = t0 + j / Dk, d = j % Dk;
        sq[j] = q[((b_idx * TT + t) * Hk + hk_idx) * Dk + d];
        sk[j] = k[((b_idx * TT + t) * Hk + hk_idx) * Dk + d];
      }
      for (int j = tid; j < nt * 16; j += 128) {
        const int t = t0 + j / 16;
        sv[j] = v[((b_idx * TT + t) * Hv + hv_idx) * Dv + dv0 + j % 16];
      }
      for (int j = tid; j < nt; j += 128) {
        sg_[j] = g[(b_idx * TT + t0 + j) * Hv + hv_idx];
        sb[j] = beta[(b_idx * TT + t0 + j) * Hv + hv_idx];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (int tt = 0; tt < nt; ++tt) {
        float kv_mem[R];
        for (int r = 0; r < R; ++r) {
          kv_mem[r] = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[r][i] = state[r][i] * sg_[tt];
            kv_mem[r] += state[r][i] * sk[tt * Dk + s_idx];
          }
        }
        for (int r = 0; r < R; ++r) kv_mem[r] = simd_sum(kv_mem[r]);
        float out[R];
        for (int r = 0; r < R; ++r) {
          auto delta = (sv[tt * 16 + sgi * R + r] - kv_mem[r]) * sb[tt];
          out[r] = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[r][i] = state[r][i] + sk[tt * Dk + s_idx] * delta;
            out[r] += state[r][i] * sq[tt * Dk + s_idx];
          }
        }
        for (int r = 0; r < R; ++r) out[r] = simd_sum(out[r]);
        if (thread_index_in_simdgroup == 0) {
          for (int r = 0; r < R; ++r)
            y[((b_idx * TT + t0 + tt) * Hv + hv_idx) * Dv + dvr + r] = static_cast<InT>(out[r]);
        }
      }
    }
    for (int r = 0; r < R; ++r) {
      auto o_state = state_out + (n * Dv + dvr + r) * Dk;
      for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * dk_idx + i] = static_cast<StT>(state[r][i]);
    }
"""


@lru_cache(None)
def _kgdr4():
    return mx.fast.metal_kernel(name="gdn35_recurrence_r4", input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
                                output_names=["y", "state_out"], source=_GDR4)


def recurrence_r4(q, k, v, g, beta, state):
    """as recurrence_tg, 4 value rows per thread (Dv % 16 == 0)"""
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    return _kgdr4()(inputs=[q, k, v, g, beta, state, T],
                    template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
                    grid=(32, Dv // 4, B * Hv), threadgroup=(32, 4, 1),
                    output_shapes=[(B, T, Hv, Dv), state.shape], output_dtypes=[q.dtype, state.dtype])


# ---- the recurrence with beta and g computed inside (mlx_lm: beta = sigmoid(b) in bf16, g = compute_g = exp(-exp(fp32
# A_log) * softplus(a + dt_bias)), fp32), reading b and a straight from the fused in_proj output, and R value rows per
# thread. The same per-row expressions as mlx_lm's kernel.
_GDRG = r"""
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;
    constexpr int TM = 16, RW = 4 * R;                              // value rows per threadgroup

    auto dk_idx = thread_position_in_threadgroup.x;
    const uint sgi = thread_position_in_threadgroup.y;              // 0..3
    const uint tid = sgi * 32 + dk_idx;                             // 0..127
    const uint dv0 = threadgroup_position_in_grid.y * RW;
    const uint dvr = dv0 + sgi * R;
    const int TT = T;

    threadgroup InT sq[TM * Dk];
    threadgroup InT sk[TM * Dk];
    threadgroup InT sv[TM * RW];
    threadgroup float sg_[TM];
    threadgroup InT sb[TM];

    float state[R][n_per_t];
    for (int r = 0; r < R; ++r) {
      auto i_state = state_in + (n * Dv + dvr + r) * Dk;
      for (int i = 0; i < n_per_t; ++i) state[r][i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
    }

    for (int t0 = 0; t0 < TT; t0 += TM) {
      const int nt = min(TM, TT - t0);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int j = tid; j < nt * Dk; j += 128) {
        const int t = t0 + j / Dk, d = j % Dk;
        sq[j] = q[((b_idx * TT + t) * Hk + hk_idx) * Dk + d];
        sk[j] = k[((b_idx * TT + t) * Hk + hk_idx) * Dk + d];
      }
      for (int j = tid; j < nt * RW; j += 128) {
        const int t = t0 + j / RW;
        sv[j] = v[((b_idx * TT + t) * Hv + hv_idx) * Dv + dv0 + j % RW];
      }
      for (int j = tid; j < nt; j += 128) {
        const size_t row = (size_t)(b_idx * TT + t0 + j) * LDP;
        sb[j] = TgSigmoid()(p[row + BOFF + hv_idx]);                                   // mx.sigmoid(b), bf16
        const InT apd = p[row + AOFF + hv_idx] + dt_bias[hv_idx];                       // a + dt_bias, bf16
        const InT spl = TgLogAddExp()(apd, InT(0));                                     // nn.softplus, bf16
        const float ea = metal::precise::exp(static_cast<float>(A_log[hv_idx]));        // exp(fp32 A_log)
        sg_[j] = metal::precise::exp(-ea * static_cast<float>(spl));                    // g, fp32
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (int tt = 0; tt < nt; ++tt) {
        float kv_mem[R];
        for (int r = 0; r < R; ++r) {
          kv_mem[r] = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[r][i] = state[r][i] * sg_[tt];
            kv_mem[r] += state[r][i] * sk[tt * Dk + s_idx];
          }
        }
        for (int r = 0; r < R; ++r) kv_mem[r] = simd_sum(kv_mem[r]);
        float out[R];
        for (int r = 0; r < R; ++r) {
          auto delta = (sv[tt * RW + sgi * R + r] - kv_mem[r]) * sb[tt];
          out[r] = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[r][i] = state[r][i] + sk[tt * Dk + s_idx] * delta;
            out[r] += state[r][i] * sq[tt * Dk + s_idx];
          }
        }
        for (int r = 0; r < R; ++r) out[r] = simd_sum(out[r]);
        if (thread_index_in_simdgroup == 0) {
          for (int r = 0; r < R; ++r)
            y[((b_idx * TT + t0 + tt) * Hv + hv_idx) * Dv + dvr + r] = static_cast<InT>(out[r]);
        }
      }
    }
    for (int r = 0; r < R; ++r) {
      auto o_state = state_out + (n * Dv + dvr + r) * Dk;
      for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * dk_idx + i] = static_cast<StT>(state[r][i]);
    }
"""


@lru_cache(None)
def _kgdrg():
    return mx.fast.metal_kernel(name="gdn35_recurrence_gates", header=_HDR, source=_GDRG,
                                input_names=["q", "k", "v", "p", "A_log", "dt_bias", "state_in", "T"],
                                output_names=["y", "state_out"])


def recurrence_gates(q, k, v, p, A_log, dt_bias, state, boff=12288, aoff=12320, r=4):
    """the recurrence with beta = sigmoid(b) and g = compute_g(A_log, a, dt_bias) inside; b and a read from p [B*T, LDP]
    (the fused in_proj output) at columns boff / aoff. r value rows per thread (Dv % (4 r) == 0)."""
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    return _kgdrg()(inputs=[q, k, v, p, A_log, dt_bias, state, T],
                    template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv),
                              ("R", r), ("LDP", p.shape[-1]), ("BOFF", boff), ("AOFF", aoff)],
                    grid=(32, Dv // r, B * Hv), threadgroup=(32, 4, 1),
                    output_shapes=[(B, T, Hv, Dv), state.shape], output_dtypes=[q.dtype, state.dtype])
