"""v2 down projection for the 125B experts ([2560 out x 640 in], 4-bit g64): deterministic and batch-invariant, NOT
bit-identical to MLX (a v2-batch item). MLX's qmv splits each 640-long row over 32 lanes (8 values per lane per block,
2 blocks + a 16-lane tail) and pays a simd_sum per 320-byte row. Here ONE THREAD computes a whole output row:
  acc_g = sum_v x[64g+v] * q[v]   (v = 0..63, fixed order)   out = sum_g (s_g * acc_g + b_g * xsum_g)   (g = 0..9, fixed)
Rows of one expert are served for all pairs routed to it (dedup; per-pair order fixed -> batch-invariant). h is staged in
threadgroup memory as float, with per-group sums.
Layouts:  "mlx"  = the pools as they are (row stride 320 B; lanes read different rows, L1 absorbs the reuse)
          "pack" = offline repack: [expert][row/32][chunk c=0..19][lane] uint4 + [expert][row/32][group][lane] (s, b) pairs
                   -> every simdgroup load is 512 contiguous bytes."""
import mlx.core as mx
import numpy as np

HDR = """
inline float b2f(ushort u) { return as_type<float>((uint)u << 16); }
inline float nib(uint w, uint j) { return (float)((w >> (4 * j)) & 15u); }
"""

_PRE = """
    const uint t = thread_position_in_threadgroup.x, tg = threadgroup_position_in_grid.x;
    constexpr int RB = TPG;                          // rows per threadgroup (one per thread)
    constexpr int NRB = N / RB;
    const int p = (int)tg / NRB, rb = (int)tg % NRB;
    if (p >= P) return;
    const int e = ids[p];
    int jj = 0;
    for (int q = 0; q < p; q++) jj += ids[q] == e;
    if (jj % PMAX) return;                           // owner of a chunk of up to PMAX pairs of expert e
    int srv[PMAX]; int ns = 0;
    for (int q = p; q < P && ns < PMAX; q++) if (ids[q] == e) srv[ns++] = q;
    threadgroup float xs[PMAX][K];
    threadgroup float xg[PMAX][K / 64];
    for (int i = 0; i < ns; i++)
      for (int k = (int)t; k < K; k += RB) xs[i][k] = (float)h[(size_t)srv[i] * K + k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)t < ns * (K / 64)) {                    // per-group sums of each pair's input, fixed order
      const int i = (int)t / (K / 64), g = (int)t % (K / 64);
      float s = 0; for (int v = 0; v < 64; v++) s += xs[i][64 * g + v];
      xg[i][g] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int row = rb * RB + (int)t;
    float out[PMAX];
    for (int i = 0; i < PMAX; i++) out[i] = 0;
"""

_POST = """
    for (int i = 0; i < PMAX; i++) if (i < ns) y[(size_t)srv[i] * N + row] = static_cast<bfloat16_t>(out[i]);
"""

SRC_MLX = _PRE + """
    const device uint4* wr = (const device uint4*)(dw + ((size_t)e * N + row) * (K / 8));
    const device ushort* sr = (const device ushort*)ds + ((size_t)e * N + row) * (K / 64);
    const device ushort* br = (const device ushort*)db + ((size_t)e * N + row) * (K / 64);
    for (int g = 0; g < K / 64; g++) {
      const uint4 w0 = wr[2 * g], w1 = wr[2 * g + 1];
      const uint ws[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
      float acc[PMAX];
      for (int i = 0; i < PMAX; i++) acc[i] = 0;
      for (int u = 0; u < 8; u++)
        for (uint j = 0; j < 8; j++) {
          const float qv = nib(ws[u], j);
          for (int i = 0; i < PMAX; i++) if (i < ns) acc[i] += xs[i][64 * g + 8 * u + j] * qv;
        }
      const float s = b2f(sr[g]), b = b2f(br[g]);
      for (int i = 0; i < PMAX; i++) if (i < ns) out[i] += s * acc[i] + b * xg[i][g];
    }
""" + _POST

SRC_PACK = _PRE + """
    // packed: chunk c of row (rb*RB + t) at wp[((e*NRB + rb) * (K/32) + c) * RB + t]; (s, b) at sp[((e*NRB + rb) * (K/64) + g) * RB + t]
    const device uint4* wp = (const device uint4*)pw + ((size_t)(e * NRB + rb) * (K / 32)) * RB + t;
    const device uint* sp = (const device uint*)psb + ((size_t)(e * NRB + rb) * (K / 64)) * RB + t;
    for (int g = 0; g < K / 64; g++) {
      const uint4 w0 = wp[(2 * g) * RB], w1 = wp[(2 * g + 1) * RB];
      const uint ws[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
      float acc[PMAX];
      for (int i = 0; i < PMAX; i++) acc[i] = 0;
      for (int u = 0; u < 8; u++)
        for (uint j = 0; j < 8; j++) {
          const float qv = nib(ws[u], j);
          for (int i = 0; i < PMAX; i++) if (i < ns) acc[i] += xs[i][64 * g + 8 * u + j] * qv;
        }
      const uint sb = sp[g * RB];
      const float s = as_type<float>(sb << 16), b = as_type<float>(sb & 0xFFFF0000u);
      for (int i = 0; i < PMAX; i++) if (i < ns) out[i] += s * acc[i] + b * xg[i][g];
    }
""" + _POST

_KM = mx.fast.metal_kernel(name="xkd_mlx", input_names=["h", "ids", "dw", "ds", "db"], output_names=["y"], header=HDR, source=SRC_MLX)
_KP = mx.fast.metal_kernel(name="xkd_pack", input_names=["h", "ids", "pw", "psb"], output_names=["y"], header=HDR, source=SRC_PACK)


def pack(w, s, b, rb=128):
    """MLX layout [E, N, K/8] uint32 + [E, N, K/64] bf16 x2 -> (pw [E*N/rb*K/32*rb] uint4 words, psb [E*N/rb*K/64*rb] uint32)."""
    E, N, KW = w.shape; K = KW * 8
    wn = np.array(w).reshape(E, N // rb, rb, K // 32, 4)               # row-in-block, chunk, 4 words
    pw = np.ascontiguousarray(wn.transpose(0, 1, 3, 2, 4)).reshape(-1)  # [E][rb][chunk][row-in-block][4]
    sn = np.array(s.view(mx.uint16)).astype(np.uint32).reshape(E, N // rb, rb, K // 64)
    bn = np.array(b.view(mx.uint16)).astype(np.uint32).reshape(E, N // rb, rb, K // 64)
    sb = (sn | (bn << 16)).transpose(0, 1, 3, 2)                        # [E][rb][group][row-in-block]: s low, b high
    return mx.array(pw), mx.array(np.ascontiguousarray(sb).reshape(-1))


def down(h, idx, dn, packed=None, pmax=4, tpg=128):
    """h [P, 640] bf16, idx [B, KT] -> [P, 2560] bf16."""
    P, K = h.shape
    N = dn[0].shape[-2] if dn is not None else packed[2]
    ids = idx.reshape(-1).astype(mx.int32)
    tm = [("K", K), ("N", N), ("P", P), ("PMAX", pmax), ("TPG", tpg)]
    grid = (P * (N // tpg) * tpg, 1, 1)
    if packed is None:
        return _KM(inputs=[h, ids, *dn], template=tm, grid=grid, threadgroup=(tpg, 1, 1),
                   output_shapes=[(P, N)], output_dtypes=[mx.bfloat16])[0]
    return _KP(inputs=[h, ids, packed[0], packed[1]], template=tm, grid=grid, threadgroup=(tpg, 1, 1),
               output_shapes=[(P, N)], output_dtypes=[mx.bfloat16])[0]


# ---------------------------------------------------------------------------------------------------------- v2b: pack4
# Each thread owns R4 = 4 consecutive output rows; K is walked in 16-value chunks so a chunk of h (16 floats per pair) loaded
# once feeds 4 rows. Layout per (expert, block of 128 rows): chunk c (0..39) -> [lane][row 0..3] x 8 bytes (1 KB contiguous);
# (s, b) per group g -> [lane][row] x 4 bytes. Same fixed per-row order as above (v = 0..63 within a group, groups 0..9).
SRC_P4 = """
    const uint t = thread_position_in_threadgroup.x, tg = threadgroup_position_in_grid.x;
    constexpr int R4 = 4, RB = TPG * R4, NRB = N / RB, NC = K / 16, NG = K / 64;
    const int p = (int)tg / NRB, rb = (int)tg % NRB;
    if (p >= P) return;
    const int e = ids[p];
    int jj = 0;
    for (int q = 0; q < p; q++) jj += ids[q] == e;
    if (jj % PMAX) return;
    int srv[PMAX]; int ns = 0;
    for (int q = p; q < P && ns < PMAX; q++) if (ids[q] == e) srv[ns++] = q;
    threadgroup float4 xs[PMAX][K / 4];
    threadgroup float xg[PMAX][NG];
    for (int i = 0; i < ns; i++)
      for (int k = (int)t; k < K / 4; k += TPG) {
        const device bfloat16_t* hp = h + (size_t)srv[i] * K + 4 * k;
        xs[i][k] = float4((float)hp[0], (float)hp[1], (float)hp[2], (float)hp[3]);
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)t < ns * NG) {
      const int i = (int)t / NG, g = (int)t % NG;
      float s = 0;
      for (int v = 0; v < 16; v++) { const float4 a = xs[i][16 * g + v]; s += a.x; s += a.y; s += a.z; s += a.w; }
      xg[i][g] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device uint4* wp = (const device uint4*)pw + ((size_t)(e * NRB + rb) * NC) * (TPG * 2) + t * 2;
    const device uint4* sp = (const device uint4*)psb + ((size_t)(e * NRB + rb) * NG) * TPG + t;
    float out[PMAX][R4];
    for (int i = 0; i < PMAX; i++) for (int r = 0; r < R4; r++) out[i][r] = 0;
    for (int g = 0; g < NG; g++) {
      float acc[PMAX][R4];
      for (int i = 0; i < PMAX; i++) for (int r = 0; r < R4; r++) acc[i][r] = 0;
      for (int cc = 0; cc < 4; cc++) {
        const int c = 4 * g + cc;
        const uint4 wa = wp[(size_t)c * (TPG * 2)], wb = wp[(size_t)c * (TPG * 2) + 1];   // rows 0,1 | rows 2,3: 2 words each
        const uint wr[R4][2] = {{wa.x, wa.y}, {wa.z, wa.w}, {wb.x, wb.y}, {wb.z, wb.w}};
#if SHAREDQ
        float qv[R4][16];                              // decode once, reuse for every pair
        for (int r = 0; r < R4; r++) for (int v = 0; v < 16; v++) qv[r][v] = nib(wr[r][v / 8], v % 8);
#endif
        for (int i = 0; i < PMAX; i++) {
          if (i >= ns) break;
          const float4 x0 = xs[i][4 * c], x1 = xs[i][4 * c + 1], x2 = xs[i][4 * c + 2], x3 = xs[i][4 * c + 3];
          const float xv[16] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w, x2.x, x2.y, x2.z, x2.w, x3.x, x3.y, x3.z, x3.w};
          for (int r = 0; r < R4; r++)
#if SHAREDQ
            for (int v = 0; v < 16; v++) acc[i][r] += xv[v] * qv[r][v];
#else
            for (int v = 0; v < 16; v++) acc[i][r] += xv[v] * nib(wr[r][v / 8], v % 8);
#endif
        }
      }
      const uint4 sb = sp[(size_t)g * TPG];
      const uint sbr[R4] = {sb.x, sb.y, sb.z, sb.w};
      for (int i = 0; i < PMAX; i++) {
        if (i >= ns) break;
        for (int r = 0; r < R4; r++)
          out[i][r] += as_type<float>(sbr[r] << 16) * acc[i][r] + as_type<float>(sbr[r] & 0xFFFF0000u) * xg[i][g];
      }
    }
    const int row0 = rb * RB + (int)t * R4;
    for (int i = 0; i < PMAX; i++) {
      if (i >= ns) break;
      for (int r = 0; r < R4; r++) y[(size_t)srv[i] * N + row0 + r] = static_cast<bfloat16_t>(out[i][r]);
    }
"""
import os as _os
_KP4 = mx.fast.metal_kernel(name="xkd_pack4", input_names=["h", "ids", "pw", "psb"], output_names=["y"],
                            header=f"#define SHAREDQ {int(_os.environ.get('XKD_SHAREDQ', '0'))}\n" + HDR, source=SRC_P4)


def pack4(w, s, b, tpg=32):
    """[E, N, K/8] + scales/biases -> pw: [E][N/(4 tpg)][K/16 chunks][tpg lanes][4 rows][2 words], psb: [E][..][K/64][tpg][4]."""
    E, N, KW = w.shape; K = KW * 8; RB = 4 * tpg
    wn = np.array(w).reshape(E, N // RB, tpg, 4, K // 16, 2)              # row = rb*RB + lane*4 + r; chunk c = 2 words
    pw = np.ascontiguousarray(wn.transpose(0, 1, 4, 2, 3, 5)).reshape(-1)  # [E][rb][c][lane][r][2]
    sn = np.array(s.view(mx.uint16)).astype(np.uint32).reshape(E, N // RB, tpg, 4, K // 64)
    bn = np.array(b.view(mx.uint16)).astype(np.uint32).reshape(E, N // RB, tpg, 4, K // 64)
    sb = (sn | (bn << 16)).transpose(0, 1, 4, 2, 3)                        # [E][rb][g][lane][r]
    return mx.array(pw), mx.array(np.ascontiguousarray(sb).reshape(-1))


def down4(h, idx, packed, N=2560, pmax=2, tpg=32):
    P, K = h.shape
    ids = idx.reshape(-1).astype(mx.int32)
    return _KP4(inputs=[h, ids, packed[0], packed[1]], template=[("K", K), ("N", N), ("P", P), ("PMAX", pmax), ("TPG", tpg)],
                grid=(P * (N // (4 * tpg)) * tpg, 1, 1), threadgroup=(tpg, 1, 1),
                output_shapes=[(P, N)], output_dtypes=[mx.bfloat16])[0]


# MLX-layout variant of pack4 (no repack): rows 4t..4t+3 read row-major from the pools
SRC_P4M = """
    const uint t = thread_position_in_threadgroup.x, tg = threadgroup_position_in_grid.x;
    constexpr int R4 = 4, RB = TPG * R4, NRB = N / RB, NC = K / 16, NG = K / 64;
    const int p = (int)tg / NRB, rb = (int)tg % NRB;
    if (p >= P) return;
    const int e = ids[p];
    int jj = 0;
    for (int q = 0; q < p; q++) jj += ids[q] == e;
    if (jj % PMAX) return;
    int srv[PMAX]; int ns = 0;
    for (int q = p; q < P && ns < PMAX; q++) if (ids[q] == e) srv[ns++] = q;
    threadgroup float4 xs[PMAX][K / 4];
    threadgroup float xg[PMAX][NG];
    for (int i = 0; i < ns; i++)
      for (int k = (int)t; k < K / 4; k += TPG) {
        const device bfloat16_t* hp = h + (size_t)srv[i] * K + 4 * k;
        xs[i][k] = float4((float)hp[0], (float)hp[1], (float)hp[2], (float)hp[3]);
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)t < ns * NG) {
      const int i = (int)t / NG, g = (int)t % NG;
      float s = 0;
      for (int v = 0; v < 16; v++) { const float4 a = xs[i][16 * g + v]; s += a.x; s += a.y; s += a.z; s += a.w; }
      xg[i][g] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const size_t row0m = (size_t)e * N + rb * RB + t * R4;               // MLX layout: row-major pools
    const device uint2* wm = (const device uint2*)(dw + row0m * (K / 8));
    const device ushort* sm_ = (const device ushort*)ds + row0m * NG;
    const device ushort* bm_ = (const device ushort*)db + row0m * NG;
    float out[PMAX][R4];
    for (int i = 0; i < PMAX; i++) for (int r = 0; r < R4; r++) out[i][r] = 0;
    for (int g = 0; g < NG; g++) {
      float acc[PMAX][R4];
      for (int i = 0; i < PMAX; i++) for (int r = 0; r < R4; r++) acc[i][r] = 0;
      for (int cc = 0; cc < 4; cc++) {
        const int c = 4 * g + cc;
        const uint2 a0 = wm[c], a1 = wm[NC + c], a2 = wm[2 * NC + c], a3 = wm[3 * NC + c];   // rows 0..3, chunk c
        const uint wr[R4][2] = {{a0.x, a0.y}, {a1.x, a1.y}, {a2.x, a2.y}, {a3.x, a3.y}};
#if SHAREDQ
        float qv[R4][16];                              // decode once, reuse for every pair
        for (int r = 0; r < R4; r++) for (int v = 0; v < 16; v++) qv[r][v] = nib(wr[r][v / 8], v % 8);
#endif
        for (int i = 0; i < PMAX; i++) {
          if (i >= ns) break;
          const float4 x0 = xs[i][4 * c], x1 = xs[i][4 * c + 1], x2 = xs[i][4 * c + 2], x3 = xs[i][4 * c + 3];
          const float xv[16] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w, x2.x, x2.y, x2.z, x2.w, x3.x, x3.y, x3.z, x3.w};
          for (int r = 0; r < R4; r++)
#if SHAREDQ
            for (int v = 0; v < 16; v++) acc[i][r] += xv[v] * qv[r][v];
#else
            for (int v = 0; v < 16; v++) acc[i][r] += xv[v] * nib(wr[r][v / 8], v % 8);
#endif
        }
      }
      const uint sbr[R4] = {(uint)sm_[g] | ((uint)bm_[g] << 16), (uint)sm_[NG + g] | ((uint)bm_[NG + g] << 16),
                            (uint)sm_[2 * NG + g] | ((uint)bm_[2 * NG + g] << 16), (uint)sm_[3 * NG + g] | ((uint)bm_[3 * NG + g] << 16)};
      for (int i = 0; i < PMAX; i++) {
        if (i >= ns) break;
        for (int r = 0; r < R4; r++)
          out[i][r] += as_type<float>(sbr[r] << 16) * acc[i][r] + as_type<float>(sbr[r] & 0xFFFF0000u) * xg[i][g];
      }
    }
    const int row0 = rb * RB + (int)t * R4;
    for (int i = 0; i < PMAX; i++) {
      if (i >= ns) break;
      for (int r = 0; r < R4; r++) y[(size_t)srv[i] * N + row0 + r] = static_cast<bfloat16_t>(out[i][r]);
    }
"""
import os as _os
_KP4M = mx.fast.metal_kernel(name="xkd_p4m", input_names=["h", "ids", "dw", "ds", "db"], output_names=["y"],
                             header=f"#define SHAREDQ 0\n" + HDR, source=SRC_P4M)


def down4m(h, idx, dn, pmax=2, tpg=64):
    P, K = h.shape
    N = dn[0].shape[-2]
    ids = idx.reshape(-1).astype(mx.int32)
    return _KP4M(inputs=[h, ids, *dn], template=[("K", K), ("N", N), ("P", P), ("PMAX", pmax), ("TPG", tpg)],
                 grid=(P * (N // (4 * tpg)) * tpg, 1, 1), threadgroup=(tpg, 1, 1),
                 output_shapes=[(P, N)], output_dtypes=[mx.bfloat16])[0]


# ------------------------------------------------------------------------------------------------ rev (outside review)
# 4 lanes per output row x 5 chunks of 32 values (16 B) each, 8 rows per simdgroup: step s, lane l reads chunk s*32 + l of the
# row block (512 contiguous bytes per step, no tail). Weight-stationary: each chunk is dequantized once and applied to all of
# the expert's token rows (inputs and half-group sums in threadgroup memory). Per chunk: acc += s * dot32(q, x) + b * xsum32.
# The 4 lanes of a row combine with a fixed simd_shuffle_xor tree (1, then 2). Deterministic, batch-invariant.
SRC_REV = """
    const uint t = thread_position_in_threadgroup.x, tg = threadgroup_position_in_grid.x;
    const uint lane = t % 32, sg = t / 32;
    constexpr int RPT = SGPT * 8, NRB = N / RPT, NH = K / 32;     // rows per threadgroup; half-groups per row
    const int p = (int)tg / NRB, rb = (int)tg % NRB;
    if (p >= P) return;
    const int e = ids[p];
    int jj = 0;
    for (int q = 0; q < p; q++) jj += ids[q] == e;
    if (jj % PMAX) return;
    int srv[PMAX]; int ns = 0;
    for (int q = p; q < P && ns < PMAX; q++) if (ids[q] == e) srv[ns++] = q;
    threadgroup float4 xs[PMAX][K / 4];
    threadgroup float xh[PMAX][NH];
    for (int i = 0; i < ns; i++)
      for (int k = (int)t; k < K / 4; k += SGPT * 32) {
        const device bfloat16_t* hp = h + (size_t)srv[i] * K + 4 * k;
        xs[i][k] = float4((float)hp[0], (float)hp[1], (float)hp[2], (float)hp[3]);
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int w = (int)t; w < ns * NH; w += SGPT * 32) {           // half-group (32-value) sums per token, fixed order
      const int i = w / NH, hg = w % NH;
      float s = 0;
      for (int v = 0; v < 8; v++) { const float4 a = xs[i][8 * hg + v]; s += a.x; s += a.y; s += a.z; s += a.w; }
      xh[i][hg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int sblk = e * (N / 8) + rb * SGPT + (int)sg;           // this simdgroup's 8-row block of expert e
    const device uint4* wp = (const device uint4*)pw + (size_t)sblk * 160 + lane;
    const device uint* sp = (const device uint*)psb + (size_t)sblk * 160 + lane;
    const int kp = lane % 4;                                       // chunks kp*5 .. kp*5+4 of row (lane / 4)
    float acc[PMAX];
    for (int i = 0; i < PMAX; i++) acc[i] = 0;
    for (int s = 0; s < 5; s++) {
      const uint4 w = wp[s * 32];
      const uint sb = sp[s * 32];
      const float sc = as_type<float>(sb << 16), bi = as_type<float>(sb & 0xFFFF0000u);
      const uint ws[4] = {w.x, w.y, w.z, w.w};
      float q[32];
      for (int u = 0; u < 4; u++) for (uint j = 0; j < 8; j++) q[8 * u + j] = nib(ws[u], j);
      const int c = kp * 5 + s;                                    // half-group index within the row
      for (int i = 0; i < PMAX; i++) {
        if (i >= ns) break;
        float d = 0;
        for (int v = 0; v < 8; v++) {
          const float4 a = xs[i][8 * c + v];
          d += a.x * q[4 * v]; d += a.y * q[4 * v + 1]; d += a.z * q[4 * v + 2]; d += a.w * q[4 * v + 3];
        }
        acc[i] += sc * d + bi * xh[i][c];
      }
    }
    for (int i = 0; i < PMAX; i++) {
      if (i >= ns) break;
      float v = acc[i];
      v += simd_shuffle_xor(v, 1);
      v += simd_shuffle_xor(v, 2);
      if (kp == 0) y[(size_t)srv[i] * N + rb * RPT + sg * 8 + lane / 4] = static_cast<bfloat16_t>(v);
    }
"""
_KREV = mx.fast.metal_kernel(name="xkd_rev", input_names=["h", "ids", "pw", "psb"], output_names=["y"], header=HDR, source=SRC_REV)


def pack_rev(w, s, b):
    """[E, N, K/8] uint32 + scales/biases -> pw [E][N/8][step 5][lane 32] uint4, psb [E][N/8][5][32] (s | b << 16) of the
    chunk's group. Lane l: row 8*blk + l//4, chunk (l%4)*5 + step (32 values = 4 words)."""
    E, N, KW = w.shape; K = KW * 8
    assert K == 640 and N % 8 == 0
    wn = np.array(w).reshape(E, N // 8, 8, 4, 5, 4)                 # [E][blk][row][kp][s][4 words]: chunk = kp*5 + s
    pw = np.ascontiguousarray(wn.transpose(0, 1, 4, 2, 3, 5)).reshape(-1)   # [E][blk][s][row][kp][4] -> lane = row*4 + kp
    sn = np.array(s.view(mx.uint16)).astype(np.uint32).reshape(E, N // 8, 8, K // 64)
    bn = np.array(b.view(mx.uint16)).astype(np.uint32).reshape(E, N // 8, 8, K // 64)
    sb = sn | (bn << 16)                                             # [E][blk][row][group]
    kp = np.arange(4)[:, None]; st = np.arange(5)[None, :]
    grp = (kp * 5 + st) // 2                                         # group of chunk kp*5 + s  -> [4][5]
    sbc = sb[:, :, :, grp]                                           # [E][blk][row][kp][s]
    psb = np.ascontiguousarray(sbc.transpose(0, 1, 4, 2, 3)).reshape(-1)    # [E][blk][s][row][kp]
    return mx.array(pw), mx.array(psb)


def down_rev(h, idx, packed, N=2560, pmax=4, sgpt=8):
    P, K = h.shape
    ids = idx.reshape(-1).astype(mx.int32)
    return _KREV(inputs=[h, ids, packed[0], packed[1]], template=[("K", K), ("N", N), ("P", P), ("PMAX", pmax), ("SGPT", sgpt)],
                 grid=(P * (N // (8 * sgpt)) * sgpt * 32, 1, 1), threadgroup=(sgpt * 32, 1, 1),
                 output_shapes=[(P, N)], output_dtypes=[mx.bfloat16])[0]
