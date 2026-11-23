"""R-70 FUSED ROUTER (Leap, Sun 2026-11-01; exact-safe candidate, NewIdeas 07:5x): the MoE block's routing chain in one
kernel, bitwise equal to the MLX ops it replaces (EvalSuite's block and BigMoE's tier.install_block run the same chain):
    gates = mx.softmax(logits, axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    order = mx.argsort(-scores, axis=-1)
    inds, scores = mx.take_along_axis(inds, order, axis=-1), mx.take_along_axis(scores, order, axis=-1)
(~9 kernels per layer, ~430 per 16-row check: R-65b). How each step is replicated (MLX 0.32's own kernel sources):
- softmax: softmax_single_row with SOFTMAX_N_READS 4 and 128 threads for 512 experts, line by line: fp32 loads, max by
  simd_max over the 4 simdgroups, fast::exp(x - max), per-thread sums of 4 then simd_sum, 1 / sum, bf16 out;
- argpartition: MLX sends it to its merge sort ("ArgPartition -> sort"), which is stable (odd-even swaps on strict <,
  merges take the left run on ties). So the last k are the k largest keys (value, index), in ascending key order;
  a bf16 >= 0 orders like its bits, so key = bits << 16 | index (+ 1, so 0 can mean none);
- sum: row_reduce_small's thread loop, accumulating in the output type (bf16), left to right in argpartition order;
- divide: bf16 / bf16 as MLX's Divide;
- argsort(-scores): stable, so rank = #(larger score) + #(equal score earlier).
router_fused(logits [..., E] bf16, k) -> (inds [..., k] uint32, scores [..., k] bf16); router_chain = the MLX chain.
ACC (template) is only for the probe: 0 = bf16 accumulation (MLX's), 1 = fp32."""
from functools import lru_cache
import mlx.core as mx

_SRC = r"""
    constexpr int E = E_C, K = K_C, NR = 4, NPL = E / 32;
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.x;
    threadgroup float local_max[32];
    threadgroup float local_norm[32];
    threadgroup uint keys[E];
    // softmax_single_row<T, float, 4> (MLX 0.32), line by line
    float ld[NR];
    const device T* in = logits + (size_t)row * E + lid * NR;
    for (int i = 0; i < NR; ++i) ld[i] = float(in[i]);
    if (sg == 0) { local_max[lane] = -INFINITY; local_norm[lane] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maxval = -FLT_MAX;
    for (int i = 0; i < NR; ++i) maxval = (maxval < ld[i]) ? ld[i] : maxval;
    maxval = simd_max(maxval);
    if (lane == 0) local_max[sg] = maxval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) { maxval = simd_max(local_max[lane]); if (lane == 0) local_max[0] = maxval; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maxval = local_max[0];
    float normalizer = 0.0f;
    for (int i = 0; i < NR; ++i) { const float ex = fast::exp(ld[i] - maxval); ld[i] = ex; normalizer += ex; }
    normalizer = simd_sum(normalizer);
    if (lane == 0) local_norm[sg] = normalizer;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) { normalizer = simd_sum(local_norm[lane]); if (lane == 0) local_norm[0] = normalizer; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normalizer = 1 / local_norm[0];
    for (int i = 0; i < NR; ++i) {
        const T p = T(ld[i] * normalizer);
        keys[lid * NR + i] = ((uint(as_type<ushort>(p)) << 16) | (lid * NR + i)) + 1u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    // top k keys, largest first (keys are distinct: the index is in the low bits)
    uint mk[NPL];
    for (int j = 0; j < NPL; ++j) mk[j] = keys[lane * NPL + j];
    uint q[K];
    uint prev = 0xFFFFFFFFu;
    for (int r = 0; r < K; ++r) {
        uint best = 0u;
        for (int j = 0; j < NPL; ++j) { const uint v = mk[j]; best = (v < prev && v > best) ? v : best; }
        best = simd_max(best);
        q[r] = best; prev = best;
    }
    // argpartition order: position j holds q[K - 1 - j] (ascending keys)
    T s[K];
    for (int j = 0; j < K; ++j) s[j] = as_type<T>(ushort((q[K - 1 - j] - 1u) >> 16));
    // sum: row_reduce_small's thread loop, op(vals[j], total) = vals[j] + total, left to right
#if ACC_F32
    float tot = 0.0f;
    for (int j = 0; j < K; ++j) tot = float(s[j]) + tot;
    const T sum = T(tot);
#else
    T tot = T(0);
    for (int j = 0; j < K; ++j) tot = s[j] + tot;
    const T sum = tot;
#endif
    T nv[K];
    for (int j = 0; j < K; ++j) nv[j] = s[j] / sum;
    if (lane < K) {
        const int j = lane;
        int rank = 0;
        for (int i = 0; i < K; ++i) rank += (nv[i] > nv[j]) || (i < j && nv[i] == nv[j]);
        out_inds[(size_t)row * K + rank] = (q[K - 1 - j] - 1u) & 0xFFFFu;
        out_scores[(size_t)row * K + rank] = nv[j];
    }
"""


@lru_cache(None)
def _k(acc_f32=False):
    return mx.fast.metal_kernel(name=f"router_fused{'_f32' if acc_f32 else ''}", input_names=["logits"],
                                output_names=["out_inds", "out_scores"],
                                source=("#define ACC_F32 1\n" if acc_f32 else "#define ACC_F32 0\n") + _SRC)


def router_fused(logits, k, acc_f32=False):
    """logits [..., E] (bf16, E % 128 == 0) -> (inds [..., k] uint32, scores [..., k] bf16), bitwise == router_chain"""
    E = logits.shape[-1]
    assert E % 128 == 0 and logits.dtype == mx.bfloat16 and k <= 32, (logits.shape, logits.dtype, k)
    R = logits.size // E
    lead = logits.shape[:-1]
    inds, scores = _k(acc_f32)(inputs=[logits.reshape(R, E)], template=[("T", logits.dtype), ("E_C", E), ("K_C", k)],
                               grid=((E // 4) * R, 1, 1), threadgroup=(E // 4, 1, 1), output_shapes=[(R, k), (R, k)],
                               output_dtypes=[mx.uint32, logits.dtype])
    return inds.reshape(lead + (k,)), scores.reshape(lead + (k,))


def router_chain(logits, k, norm=True):
    gates = mx.softmax(logits, axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if norm:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    order = mx.argsort(-scores, axis=-1)
    return mx.take_along_axis(inds, order, axis=-1), mx.take_along_axis(scores, order, axis=-1)
