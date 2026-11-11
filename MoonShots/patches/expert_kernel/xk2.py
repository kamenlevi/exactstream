"""xk2: routed-expert matvecs for verify passes that read each distinct expert ONCE and are bit-identical to MLX.
mlx-lm's SwitchGLU (verify rows x top-k < 64 ids: unsorted) runs gather_qmm as one qmv per (row, expert) pair, re-reading an
expert's weights for every row that picked it. Here the first pair of each expert (the owner) serves all its pairs. Each pair's
arithmetic is a clone of MLX 0.32.2's affine qmv_fast (gate/up, K % 512 == 0) and qmv (down, K=640 incl. its 16-lane tail):
same lane split of K, load_vector/qdot expressions, per-block accumulation and simd_sum. So per pair it is exactly MLX's
result, whatever simdgroup computes it.
Kernels: A = gate+up fused (one x load per pair feeds both), B = down. The activation and combine stay MLX ops."""
import mlx.core as mx

HDR = """
template <int VPT>
inline float lv4(const device bfloat16_t* x, thread float* x_thread) {      // MLX load_vector, bits = 4
  float sum = 0;
  for (int i = 0; i < VPT; i += 4) {
    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 16.0f;
    x_thread[i + 2] = x[i + 2] / 256.0f;
    x_thread[i + 3] = x[i + 3] / 4096.0f;
  }
  return sum;
}
template <int VPT>
inline float qd4(const device uint8_t* w, const thread float* x_thread, float scale, float bias, float sum) {   // MLX qdot
  float accum = 0;
  const device uint16_t* ws = (const device uint16_t*)w;
  for (int i = 0; i < (VPT / 4); i++) {
    accum +=
        (x_thread[4 * i] * (ws[i] & 0x000f) +
         x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
         x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
         x_thread[4 * i + 3] * (ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}
"""

_OWN = """
    const uint lane = thread_index_in_simdgroup;
    const int gsg = (int)(threadgroup_position_in_grid.x * SGTG + simdgroup_index_in_threadgroup);
    constexpr int NB = N / RSG;
    const int p = gsg / NB, rb = gsg % NB;
    if (p >= P) return;
    const int e = ids[p];
    for (int q = 0; q < p; q++) if (ids[q] == e) return;          // not the first pair of this expert
    int srv[PMAX]; int ns = 0;
    for (int q = p; q < P; q++) if (ids[q] == e && ns < PMAX) srv[ns++] = q;
    constexpr int IW = K / 2, IG = K / 64;                          // bytes and groups per weight row
    const int r0 = rb * RSG;
    const size_t wrow = (size_t)e * N + r0;
"""

SRC_A = _OWN + """
    // gate+up: MLX affine_gather_qmv_fast per pair (values_per_thread 16, block 512, scale step 4)
    const device uint8_t* wg = (const device uint8_t*)gw + wrow * IW + lane * 8;
    const device uint8_t* wu = (const device uint8_t*)uw + wrow * IW + lane * 8;
    const device bfloat16_t* sgp = gs + wrow * IG + lane / 4;
    const device bfloat16_t* bgp = gb + wrow * IG + lane / 4;
    const device bfloat16_t* sup = us + wrow * IG + lane / 4;
    const device bfloat16_t* bup = ub + wrow * IG + lane / 4;
    float rg[PMAX][RSG], ru[PMAX][RSG];
    for (int i = 0; i < PMAX; i++) for (int r = 0; r < RSG; r++) { rg[i][r] = 0; ru[i][r] = 0; }
    for (int k = 0; k < K; k += 512) {
      for (int i = 0; i < PMAX; i++) {
        if (i >= ns) break;
        float xt[16];
        float sum = lv4<16>(x + (size_t)xrow[srv[i]] * K + lane * 16 + k, xt);
        for (int r = 0; r < RSG; r++) {
          float s = sgp[r * IG + k / 64], b = bgp[r * IG + k / 64];
          rg[i][r] += qd4<16>(wg + r * IW + k / 2, xt, s, b, sum);
          float s2 = sup[r * IG + k / 64], b2 = bup[r * IG + k / 64];
          ru[i][r] += qd4<16>(wu + r * IW + k / 2, xt, s2, b2, sum);
        }
      }
    }
    for (int i = 0; i < PMAX; i++) {
      if (i >= ns) break;
      for (int r = 0; r < RSG; r++) {
        float a = simd_sum(rg[i][r]), c = simd_sum(ru[i][r]);
        if (lane == 0) { yg[srv[i] * N + r0 + r] = static_cast<bfloat16_t>(a); yu[srv[i] * N + r0 + r] = static_cast<bfloat16_t>(c); }
      }
    }
"""

SRC_B = _OWN + """
    // down: MLX affine_gather_qmv (qmv_impl) per pair: values_per_thread 8, block 256, full blocks then a tail
    const device uint8_t* wd = (const device uint8_t*)dw + wrow * IW + lane * 4;
    const device bfloat16_t* sdp = ds + wrow * IG + lane / 8;
    const device bfloat16_t* bdp = db + wrow * IG + lane / 8;
    float rd[PMAX][RSG];
    for (int i = 0; i < PMAX; i++) for (int r = 0; r < RSG; r++) rd[i][r] = 0;
    int k = 0;
    for (; k < K - 256; k += 256) {
      for (int i = 0; i < PMAX; i++) {
        if (i >= ns) break;
        float xt[8];
        float sum = lv4<8>(h + (size_t)srv[i] * K + lane * 8 + k, xt);
        for (int r = 0; r < RSG; r++) {
          float s = sdp[r * IG + k / 64], b = bdp[r * IG + k / 64];
          rd[i][r] += qd4<8>(wd + r * IW + k / 2, xt, s, b, sum);
        }
      }
    }
    const int remaining = clamp((int)(K - k - (int)lane * 8), 0, 8);
    if (remaining > 0) {                                             // = load_vector_safe / qdot_safe at N = 8
      for (int i = 0; i < PMAX; i++) {
        if (i >= ns) break;
        float xt[8];
        float sum = lv4<8>(h + (size_t)srv[i] * K + lane * 8 + k, xt);
        for (int r = 0; r < RSG; r++) {
          float s = sdp[r * IG + k / 64], b = bdp[r * IG + k / 64];
          rd[i][r] += qd4<8>(wd + r * IW + k / 2, xt, s, b, sum);
        }
      }
    }
    for (int i = 0; i < PMAX; i++) {
      if (i >= ns) break;
      for (int r = 0; r < RSG; r++) {
        float a = simd_sum(rd[i][r]);
        if (lane == 0) y[srv[i] * N + r0 + r] = static_cast<bfloat16_t>(a);
      }
    }
"""

_KA = mx.fast.metal_kernel(name="xk2_gateup", input_names=["x", "xrow", "ids", "gw", "gs", "gb", "uw", "us", "ub"],
                           output_names=["yg", "yu"], header=HDR, source=SRC_A)
_KB = mx.fast.metal_kernel(name="xk2_down", input_names=["h", "ids", "dw", "ds", "db"], output_names=["y"],
                           header=HDR, source=SRC_B)
RSG_A, RSG_B, SGTG = 4, 4, 2
_xrow = {}


def gate_up(x, idx, gate, up, pmax=None):
    """x [B, D] bf16, idx [B, KT] int -> (gate, up) outputs [B*KT, F] bf16, per pair bit-identical to gather_qmm."""
    B, D = x.shape
    KT = idx.shape[-1]; P = B * KT; F = gate[0].shape[-2]
    if (B, KT) not in _xrow:
        _xrow[(B, KT)] = mx.arange(P, dtype=mx.int32) // KT
    n_sg = P * (F // RSG_A)
    return _KA(inputs=[x, _xrow[(B, KT)], idx.reshape(-1).astype(mx.int32), gate[0], gate[1], gate[2], up[0], up[1], up[2]],
               template=[("K", D), ("N", F), ("P", P), ("RSG", RSG_A), ("PMAX", pmax or B), ("SGTG", SGTG)],
               grid=(-(-n_sg // SGTG) * 32 * SGTG, 1, 1), threadgroup=(32 * SGTG, 1, 1),
               output_shapes=[(P, F), (P, F)], output_dtypes=[mx.bfloat16, mx.bfloat16])


def down(h, idx, dn, pmax=None):
    """h [P, F] bf16 (one row per pair), idx [B, KT] -> [P, D] bf16, per pair bit-identical to gather_qmm."""
    P, F = h.shape
    D = dn[0].shape[-2]
    n_sg = P * (D // RSG_B)
    return _KB(inputs=[h, idx.reshape(-1).astype(mx.int32), dn[0], dn[1], dn[2]],
               template=[("K", F), ("N", D), ("P", P), ("RSG", RSG_B), ("PMAX", pmax or idx.shape[0]), ("SGTG", SGTG)],
               grid=(-(-n_sg // SGTG) * 32 * SGTG, 1, 1), threadgroup=(32 * SGTG, 1, 1),
               output_shapes=[(P, D)], output_dtypes=[mx.bfloat16])[0]


def switchglu(sw, x, idx):
    """Drop-in for mlx-lm SwitchGLU.__call__(x, idx) on the unsorted path (idx.size < 64): same activation op and shapes."""
    assert idx.size < 64, "SwitchGLU sorts at >= 64 ids (another MLX kernel)"
    lead = idx.shape
    B, KT = idx.size // idx.shape[-1], idx.shape[-1]
    xr = x.reshape(B, x.shape[-1])
    q = lambda m: (m.weight, m.scales, m.biases)
    g, u = gate_up(xr, idx.reshape(B, KT), q(sw.gate_proj), q(sw.up_proj))
    F = g.shape[-1]
    h = sw.activation(u.reshape(*lead, 1, F), g.reshape(*lead, 1, F))
    d = down(h.reshape(B * KT, F), idx.reshape(B, KT), q(sw.down_proj))
    return d.reshape(*lead, d.shape[-1])


# ---------------------------------------------------------------------------------------------------------------- xk3
# Same per-pair arithmetic as above, but each weight is dequantized ONCE per simdgroup and reused for every pair it serves:
# q = float(w & mask) exactly as qdot's inline int -> float, then x_thread * q in the same expression order. An expert's
# pairs are split into owner chunks of at most PMAX (registers), so PMAX = 2 keeps occupancy and still shares the dequant.
_OWN3 = """
    const uint lane = thread_index_in_simdgroup;
    const int gsg = (int)(threadgroup_position_in_grid.x * SGTG + simdgroup_index_in_threadgroup);
    constexpr int NB = N / RSG;
    const int p = gsg / NB, rb = gsg % NB;
    if (p >= P) return;
    const int e = ids[p];
    int j = 0;
    for (int q = 0; q < p; q++) j += ids[q] == e;
    if (j % PMAX) return;                                           // owner of a chunk of up to PMAX pairs of e
    int srv[PMAX]; int ns = 0;
    for (int q = p; q < P && ns < PMAX; q++) if (ids[q] == e) srv[ns++] = q;
    constexpr int IW = K / 2, IG = K / 64;
    const int r0 = rb * RSG;
    const size_t wrow = (size_t)e * N + r0;
"""

_DQ = """
inline void dq16(const device uint8_t* w, thread float* q) {          // the 4 x 4 masked values qdot converts inline
  const device uint16_t* ws = (const device uint16_t*)w;
  for (int i = 0; i < 4; i++) {
    const uint v = ws[i];
    q[4 * i] = (v & 0x000f); q[4 * i + 1] = (v & 0x00f0); q[4 * i + 2] = (v & 0x0f00); q[4 * i + 3] = (v & 0xf000);
  }
}
inline void dq8(const device uint8_t* w, thread float* q) {
  const device uint16_t* ws = (const device uint16_t*)w;
  for (int i = 0; i < 2; i++) {
    const uint v = ws[i];
    q[4 * i] = (v & 0x000f); q[4 * i + 1] = (v & 0x00f0); q[4 * i + 2] = (v & 0x0f00); q[4 * i + 3] = (v & 0xf000);
  }
}
template <int VPT>
inline float qdq(const thread float* q, const thread float* x_thread, float scale, float bias, float sum) {   // qdot on q
  float accum = 0;
  for (int i = 0; i < (VPT / 4); i++) {
    accum +=
        (x_thread[4 * i] * q[4 * i] +
         x_thread[4 * i + 1] * q[4 * i + 1] +
         x_thread[4 * i + 2] * q[4 * i + 2] +
         x_thread[4 * i + 3] * q[4 * i + 3]);
  }
  return scale * accum + sum * bias;
}
"""

SRC_A3 = _OWN3 + """
    const device uint8_t* wg = (const device uint8_t*)gw + wrow * IW + lane * 8;
    const device uint8_t* wu = (const device uint8_t*)uw + wrow * IW + lane * 8;
    const device bfloat16_t* sgp = gs + wrow * IG + lane / 4;
    const device bfloat16_t* bgp = gb + wrow * IG + lane / 4;
    const device bfloat16_t* sup = us + wrow * IG + lane / 4;
    const device bfloat16_t* bup = ub + wrow * IG + lane / 4;
    float rg[PMAX][RSG], ru[PMAX][RSG];
    for (int i = 0; i < PMAX; i++) for (int r = 0; r < RSG; r++) { rg[i][r] = 0; ru[i][r] = 0; }
    for (int k = 0; k < K; k += 512) {
      float xt[PMAX][16], sm[PMAX];
      for (int i = 0; i < PMAX; i++) if (i < ns) sm[i] = lv4<16>(x + (size_t)xrow[srv[i]] * K + lane * 16 + k, xt[i]);
      for (int r = 0; r < RSG; r++) {
        float q[16];
        dq16(wg + r * IW + k / 2, q);
        float s = sgp[r * IG + k / 64], b = bgp[r * IG + k / 64];
        for (int i = 0; i < PMAX; i++) if (i < ns) rg[i][r] += qdq<16>(q, xt[i], s, b, sm[i]);
        dq16(wu + r * IW + k / 2, q);
        float s2 = sup[r * IG + k / 64], b2 = bup[r * IG + k / 64];
        for (int i = 0; i < PMAX; i++) if (i < ns) ru[i][r] += qdq<16>(q, xt[i], s2, b2, sm[i]);
      }
    }
    for (int i = 0; i < PMAX; i++) {
      if (i >= ns) break;
      for (int r = 0; r < RSG; r++) {
        float a = simd_sum(rg[i][r]), c = simd_sum(ru[i][r]);
        if (lane == 0) { yg[srv[i] * N + r0 + r] = static_cast<bfloat16_t>(a); yu[srv[i] * N + r0 + r] = static_cast<bfloat16_t>(c); }
      }
    }
"""

SRC_B3 = _OWN3 + """
    const device uint8_t* wd = (const device uint8_t*)dw + wrow * IW + lane * 4;
    const device bfloat16_t* sdp = ds + wrow * IG + lane / 8;
    const device bfloat16_t* bdp = db + wrow * IG + lane / 8;
    float rd[PMAX][RSG];
    for (int i = 0; i < PMAX; i++) for (int r = 0; r < RSG; r++) rd[i][r] = 0;
    for (int k = 0; k < K; k += 256) {                               // = qmv_impl's full blocks + its 8-value tail at K=640
      if (k + (int)lane * 8 >= K) continue;
      float xt[PMAX][8], sm[PMAX];
      for (int i = 0; i < PMAX; i++) if (i < ns) sm[i] = lv4<8>(h + (size_t)srv[i] * K + lane * 8 + k, xt[i]);
      for (int r = 0; r < RSG; r++) {
        float q[8];
        dq8(wd + r * IW + k / 2, q);
        float s = sdp[r * IG + k / 64], b = bdp[r * IG + k / 64];
        for (int i = 0; i < PMAX; i++) if (i < ns) rd[i][r] += qdq<8>(q, xt[i], s, b, sm[i]);
      }
    }
    for (int i = 0; i < PMAX; i++) {
      if (i >= ns) break;
      for (int r = 0; r < RSG; r++) {
        float a = simd_sum(rd[i][r]);
        if (lane == 0) y[srv[i] * N + r0 + r] = static_cast<bfloat16_t>(a);
      }
    }
"""

_KA3 = mx.fast.metal_kernel(name="xk3_gateup", input_names=["x", "xrow", "ids", "gw", "gs", "gb", "uw", "us", "ub"],
                            output_names=["yg", "yu"], header=HDR + _DQ, source=SRC_A3)
_KB3 = mx.fast.metal_kernel(name="xk3_down", input_names=["h", "ids", "dw", "ds", "db"], output_names=["y"],
                            header=HDR + _DQ, source=SRC_B3)
import os as _os
PMAX3, RSG3A, RSG3B, SGTG3 = (int(_os.environ.get(k, d)) for k, d in (("XK3_PMAX", 2), ("XK3_RSGA", 4), ("XK3_RSGB", 4), ("XK3_SGTG", 2)))


def switchglu3(sw, x, idx):
    assert idx.size < 64, "SwitchGLU sorts at >= 64 ids (another MLX kernel)"
    lead = idx.shape
    B, KT = idx.size // idx.shape[-1], idx.shape[-1]
    D = x.shape[-1]; P = B * KT
    q = lambda m: (m.weight, m.scales, m.biases)
    g_, u_, d_ = q(sw.gate_proj), q(sw.up_proj), q(sw.down_proj)
    F = g_[0].shape[-2]
    ids = idx.reshape(-1).astype(mx.int32)
    if (B, KT) not in _xrow:
        _xrow[(B, KT)] = mx.arange(P, dtype=mx.int32) // KT
    n_sg = P * (F // RSG3A)
    g, u = _KA3(inputs=[x.reshape(B, D), _xrow[(B, KT)], ids, *g_, *u_],
                template=[("K", D), ("N", F), ("P", P), ("RSG", RSG3A), ("PMAX", PMAX3), ("SGTG", SGTG3)],
                grid=(-(-n_sg // SGTG3) * 32 * SGTG3, 1, 1), threadgroup=(32 * SGTG3, 1, 1),
                output_shapes=[(P, F), (P, F)], output_dtypes=[mx.bfloat16, mx.bfloat16])
    hh = sw.activation(u.reshape(*lead, 1, F), g.reshape(*lead, 1, F))
    n_sg = P * (D // RSG3B)
    d = _KB3(inputs=[hh.reshape(P, F), ids, *d_],
             template=[("K", F), ("N", D), ("P", P), ("RSG", RSG3B), ("PMAX", PMAX3), ("SGTG", SGTG3)],
             grid=(-(-n_sg // SGTG3) * 32 * SGTG3, 1, 1), threadgroup=(32 * SGTG3, 1, 1),
             output_shapes=[(P, D)], output_dtypes=[mx.bfloat16])[0]
    return d.reshape(*lead, D)
