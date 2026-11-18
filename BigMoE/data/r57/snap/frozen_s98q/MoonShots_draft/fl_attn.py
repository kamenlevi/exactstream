"""WEEK98 BET 7: attention (step S5). Part 1: which stock SDPA form runs at decode (q_len 1, 24 q heads, 2 kv heads,
D = V = 256, K / V as strided views of the cache buffer), by bit-matching the shipped sdpa_vector.h kernels:
  v1            sdpa_vector (one pass, 32 simdgroups per head, keys interleaved over simdgroups)
  v2 b<B>       sdpa_vector_2pass_1 (B interleaved key blocks per head) + sdpa_vector_2pass_2
  gqa b<B> h<H> sdpa_vector_2pass_1_gqa<G = 12, HPT = H> (B contiguous chunks per kv head) + sdpa_vector_2pass_2
The kernel bodies are the shipped text; the function constants become literals (no mask, causal, sinks)."""
import os
import re

import mlx.core as mx

KDIR = os.path.join(os.path.dirname(mx.__file__), "include", "mlx", "backend", "metal", "kernels")  # [bundle] the running mlx's headers (was ~/BigMoE/.venv/.../kernels)
_S = open(os.path.join(KDIR, "sdpa_vector.h")).read()


def _body(name):
    """the body text of a [[kernel]] in sdpa_vector.h (between its signature's ') {' and the matching '}')"""
    a = _S.index(f"void {name}(")
    b = _S.index(") {", a) + 3
    depth, i = 1, b
    while depth:
        c = _S[i]
        depth += (c == "{") - (c == "}")
        i += 1
    return _S[b:i - 1]


_CONSTS = """
  constexpr bool has_mask = false, query_transposed = false, do_causal = false, bool_mask = false,
                 float_mask = false, has_sinks = false;
  const device bool* bmask = nullptr;
  const device T* fmask = nullptr;
  const device T* sinks = nullptr;
  const int mask_kv_seq_stride = 0, mask_q_seq_stride = 0, mask_head_stride = 0;
  constexpr int D = 256, V = 256;
  const int N = NKEYS;
  const size_t k_head_stride = KHS, k_seq_stride = 256, v_head_stride = KHS, v_seq_stride = 256;
  const float scale = 0.0625f;
  uint3 tid = threadgroup_position_in_grid;
  uint3 tpg = threadgroups_per_grid;
  uint3 tptg = threads_per_threadgroup;
  uint3 tidtg = thread_position_in_threadgroup;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;
"""

_SRC_V1 = _CONSTS + "  const int gqa_factor = 12, num_q_heads = 0;\n" + _body("sdpa_vector").replace(
    "(o[i] / sum_exp_score)", "metal::precise::divide(o[i], sum_exp_score)")
_SRC_P1 = _CONSTS + "  const int blocks = NBLOCKS;\n" + _body("sdpa_vector_2pass_1")
_SRC_GQA = (_CONSTS + "  const int blocks = NBLOCKS;\n" +
            _body("sdpa_vector_2pass_1_gqa").replace("const int blocks = tpg.z;", "")
            .replace("1 / sum_exp_score[j]", "metal::precise::divide(1.0f, sum_exp_score[j])"))
_SRC_P2 = ("""
  uint3 tid = threadgroup_position_in_grid;
  uint3 tpg = threadgroups_per_grid;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;
  const int blocks = NBLOCKS;
  constexpr int D = 256;
""" + _body("sdpa_vector_2pass_2").replace("(o[i] / sum_exp_score)", "metal::precise::divide(o[i], sum_exp_score)"))

_K = {}


def _kern(name, src, inputs, outputs, header=""):
    if name not in _K:
        _K[name] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs, source=src,
                                        header=header)
    return _K[name]


def sdpa_candidate(kind, q, kbuf, vbuf, T, blocks=32, hpt=1):
    """q [24, 256] bf16; kbuf / vbuf [2, cap, 256] (cache backing; keys 0..T-1 valid). Returns out [24, 256]"""
    cap = kbuf.shape[1]
    tmpl = [("T", mx.bfloat16), ("NKEYS", T), ("KHS", cap * 256)]
    if kind == "v1":
        k = _kern("fls_v1", _SRC_V1, ["queries", "keys", "values"], ["out"])
        return k(inputs=[q, kbuf, vbuf], template=tmpl, grid=(24 * 1024, 1, 1), threadgroup=(1024, 1, 1),
                 output_shapes=[(24, 256)], output_dtypes=[mx.bfloat16])[0]
    tmpl = tmpl + [("NBLOCKS", blocks)]
    if kind == "v2":
        k1 = _kern(f"fls_p1", _SRC_P1, ["queries", "keys", "values"], ["out", "sums", "maxs"])
        part, sums, maxs = k1(inputs=[q, kbuf, vbuf], template=tmpl,
                              grid=(2 * 32, 12, blocks), threadgroup=(32, 12, 1),
                              output_shapes=[(24 * blocks * 256,), (24 * blocks,), (24 * blocks,)],
                              output_dtypes=[mx.bfloat16, mx.float32, mx.float32])
    else:
        hdr = f"constexpr constant int G = 12; constexpr constant int HPT = {hpt};\n"
        k1 = _kern(f"fls_gqa_{hpt}", _SRC_GQA, ["queries", "keys", "values"], ["out", "sums", "maxs"], header=hdr)
        part, sums, maxs = k1(inputs=[q, kbuf, vbuf], template=tmpl,
                              grid=(2 * 32, 12, blocks), threadgroup=(32, 12, 1),
                              output_shapes=[(24 * blocks * 256,), (24 * blocks,), (24 * blocks,)],
                              output_dtypes=[mx.bfloat16, mx.float32, mx.float32])
    k2 = _kern("fls_p2", _SRC_P2, ["partials", "sums", "maxs"], ["out"])
    return k2(inputs=[part, sums, maxs], template=[("T", mx.bfloat16), ("NBLOCKS", blocks)],
              grid=(24 * 1024, 1, 1), threadgroup=(1024, 1, 1), output_shapes=[(24, 256)],
              output_dtypes=[mx.bfloat16])[0]


def stock_sdpa(q, kbuf, vbuf, T):
    """the stock call: q [1, 24, 1, 256], K / V = the cache backing's [:, :T] views (strided like the KV cache)"""
    K = kbuf[None, :, :T, :]
    Vv = vbuf[None, :, :T, :]
    return mx.fast.scaled_dot_product_attention(q.reshape(1, 24, 1, 256), K, Vv, scale=0.0625, mask=None).reshape(24, 256)


if __name__ == "__main__":
    import argparse
    import json
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default=os.path.expanduser("~/MoonShots/week98/bet7/fl_attn_match.json"))
    a = ap.parse_args()
    res = {}
    for T in (400, 1000, 1024, 2000, 4000):
        cap = ((T + 255) // 256) * 256 + 256
        hits = []
        # pass 2 reduces blocks / 32 per lane: blocks is a multiple of 32; the gqa form's threadgroup memory
        # (G * HPT * V floats) allows HPT <= 2
        cands = [("v1", {})] + [("v2", dict(blocks=b)) for b in (32, 64, 128, 256)] + \
                [("gqa", dict(blocks=b, hpt=h)) for b in (32, 64, 128, 256) for h in (1, 2)]
        ok = {f"{k} {kw}": True for k, kw in cands}
        for i in range(a.n):
            kk = mx.random.key(T * 10 + i)
            q = (mx.random.normal((24, 256), key=kk) * 2.0).astype(mx.bfloat16)
            kb = (mx.random.normal((2, cap, 256), key=mx.random.key(T * 10 + i + 1)) * 1.5).astype(mx.bfloat16)
            vb = mx.random.normal((2, cap, 256), key=mx.random.key(T * 10 + i + 2)).astype(mx.bfloat16)
            ref = stock_sdpa(q, kb, vb, T)
            mx.eval(ref)
            for k, kw in cands:
                key = f"{k} {kw}"
                if not ok[key]:
                    continue
                try:
                    o = sdpa_candidate(k, q, kb, vb, T, **kw)
                    mx.eval(o)
                    ok[key] = bool(mx.array_equal(o.view(mx.uint16), ref.view(mx.uint16)).item())
                except Exception as e:                       # noqa: BLE001
                    ok[key] = False
                    if i == 0:
                        print(f"  T {T} {key}: error {str(e)[:400]}")
        hits = [k for k, v in ok.items() if v]
        res[T] = hits
        print(f"T {T}: {len(hits)} matching: {hits}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


# ---------------------------------------------------------------------------------------------------------------------
# Part 2: the fused S5 (one pass form): per q head h (kv head g = h / 12), 32 simdgroups (sdpa_vector's layout):
#   q_norm and RoPE of q[h], k_norm and RoPE of the new key k[g] (the stock's separate fp32 Multiply / Multiply / Add
#   of _apply_partial_rope, so no FMA contraction), sdpa_vector's loop over keys 0..T (key / value T from the new row,
#   the rest from the cache backing), the output gate out * sigmoid(gate) (standalone sigmoid, bf16 Multiply).
#   Outputs: gated [6144] and the rotated new keys [2, 256] (the cache write).
from fl_exact import RMS_HEADER, EW_HEADER          # noqa: E402
from fl_hc import _const                             # noqa: E402

S5_HEADER = RMS_HEADER + EW_HEADER + """
// no FMA contraction: stock RoPE is three separate fp32 kernels (Multiply, Multiply, Add)
METAL_FUNC float fl_nofma_mul(float a, float b) { return as_type<float>(as_type<uint>(a * b)); }
"""

_S5 = """
  #pragma clang fp contract(off)
  constexpr int D = 256, BN = 32, BD = 32, qk_per_thread = 8, v_per_thread = 8;
  uint h = threadgroup_position_in_grid.x;
  uint g = h / 12;
  uint lid = thread_position_in_threadgroup.x;
  uint simd_gid = simdgroup_index_in_threadgroup, simd_lid = thread_index_in_simdgroup;
  uint nsg = threads_per_threadgroup.x / 32;
  threadgroup float part[33];
  threadgroup bfloat16_t qr[256];
  threadgroup bfloat16_t kr[256];
  threadgroup float cs[64];
  threadgroup float sn[64];
  typedef float U;
  // RoPE tables at position T (rope_start = cache offset): angles = float(T) * inv_freq (fp32 Multiply)
  if (lid < 64) {
    float a = float(TPOS) * inv_freq[lid % 32];
    cs[lid] = metal::precise::cos(a);
    sn[lid] = metal::precise::sin(a);
  }
  // q_norm / k_norm (nn.RMSNorm(256): w * T(x * inv))
  const device bfloat16_t* xq = qf + h * 512;
  const device bfloat16_t* xk = kf + g * 256;
  float invq = fl_rms_inv_p<256>(xq, eps[0], part, simd_lid, simd_gid, nsg);
  float invk = fl_rms_inv_p<256>(xk, eps[0], part, simd_lid, simd_gid, nsg);
  threadgroup bfloat16_t qn[256];
  threadgroup bfloat16_t kn[256];
  if (lid < 256) {
    qn[lid] = qnw[lid] * static_cast<bfloat16_t>(float(xq[lid]) * invq);
    kn[lid] = knw[lid] * static_cast<bfloat16_t>(float(xk[lid]) * invk);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < 256) {
    uint i = lid;
    if (i < 64) {
      bfloat16_t rq = (i < 32) ? static_cast<bfloat16_t>(-qn[i + 32]) : qn[i - 32];
      bfloat16_t rk = (i < 32) ? static_cast<bfloat16_t>(-kn[i + 32]) : kn[i - 32];
      float q1 = fl_nofma_mul(float(qn[i]), cs[i]);
      float q2 = fl_nofma_mul(float(rq), sn[i]);
      qr[i] = static_cast<bfloat16_t>(q1 + q2);
      float k1 = fl_nofma_mul(float(kn[i]), cs[i]);
      float k2 = fl_nofma_mul(float(rk), sn[i]);
      kr[i] = static_cast<bfloat16_t>(k1 + k2);
    } else {
      qr[i] = qn[i];
      kr[i] = kn[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (h % 12 == 0 && lid < 256) { krot[g * 256 + lid] = kr[lid]; }
  // ---- sdpa_vector (one pass), the shipped loop with key / value T taken from the new row
  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];
  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];
  const float scale = 0.0625f;
  for (int i = 0; i < qk_per_thread; i++) { q[i] = static_cast<U>(scale) * qr[simd_lid * qk_per_thread + i]; }
  for (int i = 0; i < v_per_thread; i++) { o[i] = 0; }
  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  const int N = TPOS + 1;
  for (int i = simd_gid; i < N; i += BN) {
    const bool is_new = (i == TPOS);
    const device bfloat16_t* kp = kbuf + size_t(g) * KHS + size_t(i) * 256 + simd_lid * qk_per_thread;
    const device bfloat16_t* vp = vbuf + size_t(g) * KHS + size_t(i) * 256 + simd_lid * v_per_thread;
    for (int j = 0; j < qk_per_thread; j++) { k[j] = is_new ? float(kr[simd_lid * qk_per_thread + j]) : float(kp[j]); }
    U score = 0;
    for (int j = 0; j < qk_per_thread; j++) { score += q[j] * k[j]; }
    score = simd_sum(score);
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    for (int j = 0; j < v_per_thread; j++) {
      float vv = is_new ? float(vf[g * 256 + simd_lid * v_per_thread + j]) : float(vp[j]);
      o[j] = o[j] * factor + exp_score * vv;
    }
  }
  if (simd_lid == 0) { max_scores[simd_gid] = max_score; sum_exp_scores[simd_gid] = sum_exp_score; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);
  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : metal::precise::divide(o[i], sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      uint d = simd_gid * v_per_thread + i;
      bfloat16_t ov = static_cast<bfloat16_t>(o[i]);
      bfloat16_t gt = fl_sigmoid_pre(qf[h * 512 + 256 + d]);
      gated[h * 256 + d] = ov * gt;                                          // attention_gate (bf16 Multiply)
    }
  }
"""


def fused_attn1(attn, qf, kf, vf, kbuf, vbuf, T, eps=None):
    """S5 (one-pass form). qf [12288], kf / vf [512]: S4's outputs; kbuf / vbuf [2, cap, 256]: the cache backing with
    keys 0..T-1 valid. Returns (gated [6144], the rotated new keys [512])"""
    cap = int(kbuf.shape[-2])
    eps = attn.q_norm.eps if eps is None else eps
    k = _kern("fl_s5_v1", _S5, ["qf", "kf", "vf", "kbuf", "vbuf", "qnw", "knw", "inv_freq", "eps"],
              ["gated", "krot"], header=S5_HEADER)
    return k(inputs=[qf.reshape(-1), kf.reshape(-1), vf.reshape(-1), kbuf.reshape(-1), vbuf.reshape(-1),
                     attn.q_norm.weight, attn.k_norm.weight, attn._inv_freq, _const(("eps", eps), lambda: mx.array([eps], mx.float32))],
             template=[("TPOS", int(T)), ("KHS", cap * 256)], grid=(24 * 1024, 1, 1), threadgroup=(1024, 1, 1),
             output_shapes=[(6144,), (512,)], output_dtypes=[mx.bfloat16, mx.bfloat16])


# ---------------------------------------------------------------------------------------------------------------------
# Part 3: the fused S5 in the stock 2-pass form (N >= 1024 keys: 64 blocks at N = 1024, 128 above; found above).
#   pass 1: threadgroup = (kv head g, block b), 12 simdgroups = the group's 12 q heads; each simdgroup normalizes and
#           rotates its own q head (the stock 256-wide RMS statistic emulated in one simdgroup: 2 virtual simdgroups),
#           simdgroup 0 the new key; then sdpa_vector_2pass_1's loop over keys b, b + blocks, ...
#   pass 2: sdpa_vector_2pass_2 as shipped, then the output gate.
S5P_HEADER = S5_HEADER + """
// the stock RMS statistic for 256 values computed inside ONE simdgroup (virtual lanes l and l + 32)
template <typename P>
METAL_FUNC float fl_rms256_simd(P x, float eps, uint lane) {
  float a0 = 0.0f, a1 = 0.0f;
  for (int i = 0; i < 4; i++) { float xi = x[lane * 4 + i]; a0 += xi * xi; }
  for (int i = 0; i < 4; i++) { float xi = x[128 + lane * 4 + i]; a1 += xi * xi; }
  float p0 = simd_sum(a0), p1 = simd_sum(a1);
  float v = (lane == 0) ? p0 : ((lane == 1) ? p1 : 0.0f);
  float tot = simd_sum(v);
  return metal::precise::rsqrt(tot / 256.0f + eps);
}
"""

_S5P1 = """
  #pragma clang fp contract(off)
  constexpr int BD = 32, qk_per_thread = 8, v_per_thread = 8, V = 256, D = 256;
  uint g = threadgroup_position_in_grid.x;
  uint block_idx = threadgroup_position_in_grid.z;
  uint hs = thread_position_in_threadgroup.y;                   // the q head within the group (one simdgroup)
  uint simd_lid = thread_index_in_simdgroup;
  uint h = g * 12 + hs;
  threadgroup bfloat16_t qn[12][256];
  threadgroup bfloat16_t qr[12][256];
  threadgroup bfloat16_t kn[256];
  threadgroup bfloat16_t kr[256];
  threadgroup float cs[64];
  threadgroup float sn[64];
  uint tlin = simd_lid + 32 * hs;
  if (tlin < 64) {
    float a = float(TPOS) * inv_freq[tlin % 32];
    cs[tlin] = metal::precise::cos(a);
    sn[tlin] = metal::precise::sin(a);
  }
  const device bfloat16_t* xq = qf + h * 512;
  float invq = fl_rms256_simd(xq, eps[0], simd_lid);
  for (int j = 0; j < 8; j++) {
    uint d = simd_lid * 8 + j;
    qn[hs][d] = qnw[d] * static_cast<bfloat16_t>(float(xq[d]) * invq);
  }
  if (hs == 0) {
    const device bfloat16_t* xk = kf + g * 256;
    float invk = fl_rms256_simd(xk, eps[0], simd_lid);
    for (int j = 0; j < 8; j++) {
      uint d = simd_lid * 8 + j;
      kn[d] = knw[d] * static_cast<bfloat16_t>(float(xk[d]) * invk);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int j = 0; j < 8; j++) {
    uint d = simd_lid * 8 + j;
    if (d < 64) {
      bfloat16_t rq = (d < 32) ? static_cast<bfloat16_t>(-qn[hs][d + 32]) : qn[hs][d - 32];
      qr[hs][d] = static_cast<bfloat16_t>(fl_nofma_mul(float(qn[hs][d]), cs[d]) + fl_nofma_mul(float(rq), sn[d]));
      if (hs == 0) {
        bfloat16_t rk = (d < 32) ? static_cast<bfloat16_t>(-kn[d + 32]) : kn[d - 32];
        kr[d] = static_cast<bfloat16_t>(fl_nofma_mul(float(kn[d]), cs[d]) + fl_nofma_mul(float(rk), sn[d]));
      }
    } else {
      qr[hs][d] = qn[hs][d];
      if (hs == 0) { kr[d] = kn[d]; }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (block_idx == 0 && hs == 0) { for (int j = 0; j < 8; j++) { krot[g * 256 + simd_lid * 8 + j] = kr[simd_lid * 8 + j]; } }
  // ---- sdpa_vector_2pass_1's loop
  typedef float U;
  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};
  const float scale = 0.0625f;
  for (int i = 0; i < qk_per_thread; i++) { q[i] = static_cast<U>(scale) * qr[hs][simd_lid * qk_per_thread + i]; }
  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  const int N = TPOS + 1;
  for (int i = block_idx; i < N; i += NBLOCKS) {
    const bool is_new = (i == TPOS);
    const device bfloat16_t* kp = kbuf + size_t(g) * KHS + size_t(i) * 256 + simd_lid * qk_per_thread;
    const device bfloat16_t* vp = vbuf + size_t(g) * KHS + size_t(i) * 256 + simd_lid * v_per_thread;
    U score = 0;
    for (int j = 0; j < qk_per_thread; j++) {
      score += q[j] * (is_new ? float(kr[simd_lid * qk_per_thread + j]) : float(kp[j]));
    }
    score = simd_sum(score);
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    for (int j = 0; j < v_per_thread; j++) {
      float vv = is_new ? float(vf[g * 256 + simd_lid * v_per_thread + j]) : float(vp[j]);
      o[j] = o[j] * factor + exp_score * vv;
    }
  }
  if (simd_lid == 0) {
    sums[h * NBLOCKS + block_idx] = sum_exp_score;
    maxs[h * NBLOCKS + block_idx] = max_score;
  }
  for (int i = 0; i < v_per_thread; i++) {
    part[size_t(h) * NBLOCKS * V + block_idx * V + simd_lid * v_per_thread + i] = static_cast<bfloat16_t>(o[i]);
  }
"""

_S5P2 = """
  constexpr int BN = 32, BD = 32, D = 256, elem_per_thread = 8;
  const int blocks = NBLOCKS;
  uint head_idx = threadgroup_position_in_grid.x;
  uint simd_gid = simdgroup_index_in_threadgroup, simd_lid = thread_index_in_simdgroup;
  typedef float U;
  thread U o[elem_per_thread] = {0};
  threadgroup U outputs[BN * BD];
  const device bfloat16_t* partials = part + size_t(head_idx) * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  const device float* sp = sums + head_idx * blocks;
  const device float* mp = maxs + head_idx * blocks;
  U sum_exp_score = 0.0;
  U max_score = Limits<U>::finite_min;
  for (int b = 0; b < blocks / BN; ++b) { max_score = max(max_score, mp[simd_lid + BN * b]); }
  max_score = simd_max(max_score);
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(mp[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * sp[simd_lid + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(mp[simd_gid] - max_score);
    for (int i = 0; i < elem_per_thread; i++) { o[i] += factor * static_cast<U>(partials[i]); }
    mp += BN;
    sp += BN;
    partials += BN * D;
  }
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
    o[i] = sum_exp_score == 0 ? o[i] : metal::precise::divide(o[i], sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      uint d = simd_gid * elem_per_thread + i;
      bfloat16_t ov = static_cast<bfloat16_t>(o[i]);
      gated[head_idx * 256 + d] = ov * fl_sigmoid_pre(qf[head_idx * 512 + 256 + d]);
    }
  }
"""


def fused_attn2(attn, qf, kf, vf, kbuf, vbuf, T, eps=None):
    """S5 in the stock 2-pass form (key count N = T + 1 >= 1024): two kernels. Returns (gated [6144], krot [512])"""
    N = int(T) + 1
    blocks = 64 if N == 1024 else 128
    cap = int(kbuf.shape[-2])
    eps = attn.q_norm.eps if eps is None else eps
    k1 = _kern("fl_s5_p1", _S5P1, ["qf", "kf", "vf", "kbuf", "vbuf", "qnw", "knw", "inv_freq", "eps"],
               ["part", "sums", "maxs", "krot"], header=S5P_HEADER)
    part, sums, maxs, krot = k1(
        inputs=[qf.reshape(-1), kf.reshape(-1), vf.reshape(-1), kbuf.reshape(-1), vbuf.reshape(-1),
                attn.q_norm.weight, attn.k_norm.weight, attn._inv_freq, _const(("eps", eps), lambda: mx.array([eps], mx.float32))],
        template=[("TPOS", int(T)), ("KHS", cap * 256), ("NBLOCKS", blocks)],
        grid=(2 * 32, 12, blocks), threadgroup=(32, 12, 1),
        output_shapes=[(24 * blocks * 256,), (24 * blocks,), (24 * blocks,), (512,)],
        output_dtypes=[mx.bfloat16, mx.float32, mx.float32, mx.bfloat16])
    k2 = _kern("fl_s5_p2", _S5P2, ["part", "sums", "maxs", "qf"], ["gated"], header=S5P_HEADER)
    gated = k2(inputs=[part, sums, maxs, qf.reshape(-1)], template=[("NBLOCKS", blocks)],
               grid=(24 * 1024, 1, 1), threadgroup=(1024, 1, 1), output_shapes=[(6144,)],
               output_dtypes=[mx.bfloat16])[0]
    return gated, krot


def fused_attn(attn, qf, kf, vf, kbuf, vbuf, T):
    """the stock form for this key count: one pass below 1,024 keys, two passes from 1,024"""
    return (fused_attn1 if int(T) + 1 < 1024 else fused_attn2)(attn, qf, kf, vf, kbuf, vbuf, T)
