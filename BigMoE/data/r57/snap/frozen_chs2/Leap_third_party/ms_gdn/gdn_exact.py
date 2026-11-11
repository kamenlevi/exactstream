"""Round 32l: exact fused GDN verify prework. One dispatch for the chain between the in_proj and gated_delta_update in
MTPLX's GatedDeltaNet (eager branch): concat [conv_state; qkv] -> depthwise conv1d k=4 -> silu -> split -> per-head l2norm
of q, k -> q * inv_scale, plus the new conv state (last 3 rows of the stream). Unlike MTPLX's gdn_conv_norm (fp32
throughout, one rounding), every rounding and order of the eager chain is replicated (gdn_probe.py pinned them):
conv in fp32 rounded to bf16 (any order is exact: bf16 products are exact in fp32), silu = Multiply(x, Sigmoid(x)) in
bf16 (mlx.nn.silu, compiled), xf = fp32(x), squares rounded in fp32, sum = lane l adds its 4 contiguous squares in order
then simd_sum over the 32 lanes of the head (MLX's row reduce, 0/4096 mismatches), + 1e-6, metal::precise::rsqrt (MLX's
Rsqrt), xf * r in fp32 -> bf16, q: Multiply(bf16(inv_scale), q) in bf16. Target: bitwise == the eager chain at any S.
Layout (the 125B's GDN): conv_dim 10240 = q 16 x 128 | k 16 x 128 | v 48 x 128, kernel 4.
gdn_prework_exact(qkv [S, 10240] bf16, conv_state [3, 10240] bf16, conv_w [10240, 4] bf16, scale_bf16 [1] bf16)
  -> q [S, 2048], k [S, 2048], v [S, 6144], new_state [3, 10240]"""
from functools import lru_cache
import mlx.core as mx
from tierglu import _HDR                                    # TgSigmoid, TgMultiply (MLX's ops, verbatim)

_SRC = r"""
    constexpr int C = 10240, QK = 2048;
    const uint lane = thread_index_in_simdgroup;
    const uint g = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    const uint s = threadgroup_position_in_grid.y;           // row
    if (g >= (uint)(C / 128)) return;                         // 80 groups of 128 channels: 16 q heads, 16 k, 48 v
    const uint c0 = g * 128 + lane * 4;                       // this lane's 4 contiguous channels
    T sv[4];
    for (int i = 0; i < 4; ++i) {
        const uint c = c0 + i;
        // stream(t): t < 3 -> conv state row t, else qkv row t - 3; window of row s = stream[s .. s+3]
        float acc = 0.0f;
        for (int t = 0; t < 4; ++t) {
            const int st = (int)s + t;
            const float xv = st < 3 ? (float)state[st * C + c] : (float)qkv[(st - 3) * C + c];
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
        const float r = metal::precise::rsqrt(tot + 1e-6f);
        for (int i = 0; i < 4; ++i) {
            const T nb = static_cast<T>(xf[i] * r);
            if (g < (uint)(QK / 128)) q_out[s * QK + c0 + i] = TgMultiply()(scale[0], nb);
            else k_out[s * QK + (c0 - QK) + i] = nb;
        }
    }
    if (s == 0) {                                             // new conv state: rows S, S+1, S+2 of the stream
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 4; ++i) {
                const uint c = c0 + i;
                const int st = S_C + j;
                state_out[j * C + c] = st < 3 ? state[st * C + c] : qkv[(st - 3) * C + c];
            }
    }
"""


@lru_cache(None)
def _k():
    return mx.fast.metal_kernel(name="gdn_prework_exact", input_names=["qkv", "state", "cw", "scale"],
                                output_names=["q_out", "k_out", "v_out", "state_out"], source=_SRC, header=_HDR)


def gdn_prework_exact(qkv, conv_state, conv_w, scale_bf16, sg=4):
    S = qkv.shape[0]
    q, k, v, ns = _k()(inputs=[qkv.reshape(-1), conv_state.reshape(-1), conv_w.reshape(-1), scale_bf16.reshape(-1)],
                       template=[("T", qkv.dtype), ("S_C", S)], grid=(80 // sg * 32 * sg, S, 1), threadgroup=(32 * sg, 1, 1),
                       output_shapes=[(S, 2048), (S, 2048), (S, 6144), (3, 10240)], output_dtypes=[qkv.dtype] * 4)
    return q, k, v, ns


# ---- exact SigmoidRMSNormGated for decode/verify rows (MTPLX's eager expression below 32 rows):
#   x = mx.fast.rms_norm(h, weight, eps)   (bf16: lane l squares its 4 contiguous values in fp32, simd_sum, precise
#                                           rsqrt(sum / 128 + eps), out = Multiply(w, bf16(h * inv)) in bf16)
#   g = sigmoid(fp32(gate)); out = bf16(g * fp32(x))
_GN = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    if (row >= (uint)NROWS) return;
    const device T* hr = h + (size_t)row * 128 + lane * 4;
    const device T* gr = gate + (size_t)row * 128 + lane * 4;
    float acc = 0.0f;
    float hv[4];
    for (int i = 0; i < 4; ++i) { hv[i] = (float)hr[i]; acc += hv[i] * hv[i]; }
    acc = simd_sum(acc);
    const float inv = metal::precise::rsqrt(acc / 128.0f + epsv[0]);
    for (int i = 0; i < 4; ++i) {
        const T xn = static_cast<T>(hv[i] * inv);
        const T xw = TgMultiply()(w[lane * 4 + i], xn);
        const float g = TgSigmoid()((float)gr[i]);
        out[(size_t)row * 128 + lane * 4 + i] = static_cast<T>(g * (float)xw);
    }
"""


@lru_cache(None)
def _kgn():
    return mx.fast.metal_kernel(name="gdn_gated_norm_exact", input_names=["h", "gate", "w", "epsv"], output_names=["out"],
                                source=_GN, header=_HDR)


def gated_norm_exact(h, gate, weight, eps):
    """h, gate [..., 128] bf16 (per head), weight [128] bf16 -> SigmoidRMSNormGated(h, gate) bitwise."""
    n = h.size // 128
    (o,) = _kgn()(inputs=[h.reshape(-1), gate.reshape(-1), weight.astype(h.dtype), mx.array([eps], mx.float32)],
                  template=[("T", h.dtype), ("NROWS", n)],
                  grid=((n + 3) // 4 * 128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(h.size,)], output_dtypes=[h.dtype])
    return o.reshape(h.shape)
