"""R-77 CHAIN (Leap, Sun 2026-11-01): DFlash2's candidate-selector walk (r57_screen's chain: 15 sequential steps of
codebook gathers, two bf16 products, a 256-wide bf16 sum, add, argmax, take and an fp32 softmax, ~168 kernels) in ~6.
- edges for every (step, predecessor candidate, successor candidate) at once with the same MLX ops, in the same order:
  (pred_cb[pred] * hp[p]) * succ_cb[cand[p]], summed over the rank by MLX's own reduction (bitwise per row as long as
  the reduction kernel treats a row the same: checked against the shipped chain in r77_chain_test.py). The step-0
  predecessor is the block's anchor; step p >= 1's is one of cand[p - 1].
- the walk (bf16 add of the unary logit, argmax with the first index on ties, the fp32 softmax's value at the argmax)
  in ONE threadgroup: the softmax replays MLX's softmax_single_row for 16 values (4 lanes x 4 reads, simd max / sum,
  fast::exp, 1 / sum), as in R-70.
chain_fused(blk, unary, cand, hp, sel) -> (path [B, P] uint32, conf [B, P] float32), == chain(...) bitwise."""
from functools import lru_cache
import mlx.core as mx

_WALK = r"""
    constexpr int P = P_C, KC = K_C;
    const uint lane = thread_index_in_simdgroup;
    int s = 0;                                   // step 0: the anchor's row of E (all 16 rows of it are the anchor)
    for (int p = 0; p < P; ++p) {
        // sc[k] = unary[p, k] + E[p, s, k] (bf16 add), lanes 0..15
        T sc = T(0);
        if (lane < KC) sc = unary[p * KC + lane] + E[(p * KC + s) * KC + lane];
        // argmax, first index on ties (MLX's ArgMax over 16)
        float v = lane < KC ? float(sc) : -INFINITY;
        int idx = lane < KC ? int(lane) : 0x7fffffff;
        for (int off = 16; off >= 1; off >>= 1) {
            const float ov = simd_shuffle_xor(v, off);
            const int oi = simd_shuffle_xor(idx, off);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        const int sel = simd_broadcast(idx, 0);
        // softmax_single_row<float, float, 4> over the 16 fp32 values: lanes 0..3 hold 4 each, lanes 4..31 Limits::min
        const float x = float(sc);                                    // fp32 copy of each lane's value (lane < 16)
        float ld[4];
        for (int i = 0; i < 4; ++i) {
            const int src = 4 * (int)lane + i;
            const float xi = simd_shuffle(x, (ushort)(src < KC ? src : 0));
            ld[i] = (lane < 4) ? xi : -INFINITY;
        }
        float maxval = -FLT_MAX;
        for (int i = 0; i < 4; ++i) maxval = (maxval < ld[i]) ? ld[i] : maxval;
        maxval = simd_max(maxval);                                    // exact in any order (the second stage too)
        float normalizer = 0.0f;
        for (int i = 0; i < 4; ++i) { const float ex = fast::exp(ld[i] - maxval); ld[i] = ex; normalizer += ex; }
        normalizer = simd_sum(normalizer);
        normalizer = simd_sum(lane == 0 ? normalizer : 0.0f);            // local_normalizer[lane]: lane 0 real, rest 0
        normalizer = 1 / normalizer;
        // the softmax output at sel: thread sel / 4 holds it at slot sel % 4
        float pv = 0.0f;
        for (int i = 0; i < 4; ++i) if (4 * (int)lane + i == sel) pv = ld[i] * normalizer;
        pv = simd_sum(pv);                                            // exactly one lane is nonzero
        const uint pred = cand[p * KC + sel];
        if (lane == 0) { path[p] = pred; conf[p] = pv; }
        s = sel;
    }
"""


@lru_cache(None)
def _walk_k():
    return mx.fast.metal_kernel(name="dflash_chain_walk", input_names=["unary", "E", "cand"],
                                output_names=["path", "conf"], source=_WALK)


def chain_fused(blk, unary, cand, hp, sel):
    """blk [1, R] ids (anchor = blk[:, 0]); unary [1, P, K] bf16; cand [1, P, K] uint32; hp [1, P, rank] bf16"""
    B, P, K = cand.shape
    assert B == 1 and K <= 16 and unary.dtype == mx.bfloat16, (cand.shape, unary.dtype)
    prev = mx.concatenate([mx.broadcast_to(blk[:, :1].astype(cand.dtype), (1, 1, K)), cand[:, :-1]], axis=1)  # [1, P, K]
    ah = sel.predecessor_codebook(prev) * hp[:, :, None, :]           # [1, P, K, rank]: pred_cb[pred] * hp[p]
    E = mx.sum(ah[:, :, :, None, :] * sel.successor_codebook(cand)[:, :, None, :, :], axis=-1)   # [1, P, K, K]
    path, conf = _walk_k()(inputs=[unary.reshape(P, K), E.reshape(P, K, K), cand.reshape(P, K)],
                           template=[("T", unary.dtype), ("P_C", P), ("K_C", K)], grid=(32, 1, 1), threadgroup=(32, 1, 1),
                           output_shapes=[(P,), (P,)], output_dtypes=[cand.dtype, mx.float32])
    return path.reshape(1, P), conf.reshape(1, P)
