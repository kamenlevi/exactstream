"""FUSED TREE LEVEL, the tail (MoonShots, Tue 17 Nov 2026): topk_gpu + GPUFrontier.step as ONE Metal kernel, bitwise.

After node t's logits lg (fp32, [NS]): one threadgroup of 1024 threads
  1. lse = logsumexp(lg)                       the stock logsumexp_looped order (fused2's S15: fast exp, precise log)
  2. the top-K of lg by (value desc, index asc) = argpartition(-lg)[:K] then the stable descending sort (topk_gpu)
     (per-thread lists, then a two-level all-lane merge under that total order)
  3. tp_k = exp(tv_k - lse) (the subtract rounded alone, precise exp); token ids through the screen's id map
  4. the frontier: push node t's K children (float64 path products exact in integer arithmetic, as sf_tree's
     f64_mul_f32), then pop node t + 1 = the live candidate with the largest product, the earliest pushed on ties
     (heapq's order); its parent, token, depth, path product and ancestor row. For the root call (tt = 0) the
     children are the roots (parent -1, path 1.0 x tp).
Replaces 21 dispatches (topk 20 + step 1; 62 + 24 us GPU per pop at the real format, sf_gprof) with one.
usage: imported by sf_tree.py (--tail)."""
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sf_tree import HDR as F64_HDR                         # noqa: E402  (f64_mul_f32)

HDR = """
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;
// Limits<float> comes from MLX's utils.h (included for every custom kernel)
// the total order of argpartition(-lg)'s stable sort: a before b iff a.v > b.v or (a.v == b.v and a.i < b.i)
METAL_FUNC bool tl_before(float av, uint ai, float bv, uint bi) { return av > bv || (av == bv && ai < bi); }
""" + F64_HDR

SRC = """
  const uint lid = thread_position_in_threadgroup.x;
  const uint lsize = threads_per_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sgi = simdgroup_index_in_threadgroup;
  const int tt = tti[0];
  threadgroup float local_max[32];
  threadgroup float local_norm[32];
  threadgroup float lse_s[1];
  // ---- 1. logsumexp, the stock looped layout (4 values per lane per round)
  float prevmax;
  float maxval = Limits<float>::finite_min;
  float normalizer = 0;
  for (int r = 0; r < int((NS + 4 * lsize - 1) / (4 * lsize)); r++) {
    int offset = r * lsize * 4 + lid * 4;
    float vals[4];
    if (offset + 4 <= NS) {
      for (int i = 0; i < 4; i++) { vals[i] = lg[offset + i]; }
    } else {
      for (int i = 0; i < 4; i++) { vals[i] = (offset + i < NS) ? lg[offset + i] : Limits<float>::min; }
    }
    prevmax = maxval;
    for (int i = 0; i < 4; i++) { maxval = (maxval < vals[i]) ? vals[i] : maxval; }
    normalizer *= fast::exp(prevmax - maxval);
    for (int i = 0; i < 4; i++) { normalizer += fast::exp(vals[i] - maxval); }
  }
  prevmax = maxval;
  maxval = simd_max(maxval);
  normalizer *= fast::exp(prevmax - maxval);
  normalizer = simd_sum(normalizer);
  prevmax = maxval;
  if (lane == 0) { local_max[sgi] = maxval; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = simd_max(local_max[lane]);
  normalizer *= fast::exp(prevmax - maxval);
  if (lane == 0) { local_norm[sgi] = normalizer; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer = simd_sum(local_norm[lane]);
  if (lid == 0) { lse_s[0] = isinf(maxval) ? maxval : (metal::precise::log(normalizer) + maxval); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float lse = lse_s[0];
  // ---- 2. the top-K of lg by (value desc, index asc)
  float bv[KK]; uint bi[KK];
  for (int k = 0; k < KK; k++) { bv[k] = -INFINITY; bi[k] = 0xffffffffu; }
  int nfill = 0;
  for (uint j = lid; j < uint(NS); j += lsize) {
    float v = lg[j];
    if (nfill < KK || tl_before(v, j, bv[KK - 1], bi[KK - 1])) {
      int p = min(nfill, KK - 1);
      while (p > 0 && tl_before(v, j, bv[p - 1], bi[p - 1])) { bv[p] = bv[p - 1]; bi[p] = bi[p - 1]; p--; }
      bv[p] = v; bi[p] = j;
      nfill = min(nfill + 1, KK);
    }
  }
  threadgroup float sgv[32 * KK]; threadgroup uint sgx[32 * KK];
  threadgroup float topv[KK]; threadgroup uint topi[KK];
  int head = 0;
  for (int k = 0; k < KK; k++) {
    bool live = head < nfill;
    float mv = live ? bv[head] : -INFINITY;
    uint mi = live ? bi[head] : 0xffffffffu;
    for (ushort off = 16; off >= 1; off >>= 1) {
      float ov = simd_shuffle_xor(mv, off); uint oi = simd_shuffle_xor(mi, off);
      if (tl_before(ov, oi, mv, mi)) { mv = ov; mi = oi; }
    }
    if (live && bi[head] == mi && bv[head] == mv) { head++; }
    if (lane == 0) { sgv[sgi * KK + k] = mv; sgx[sgi * KK + k] = mi; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgi == 0) {
    int hd = 0;
    for (int k = 0; k < KK; k++) {
      bool live = hd < KK;
      float mv = live ? sgv[lane * KK + hd] : -INFINITY;
      uint mi = live ? sgx[lane * KK + hd] : 0xffffffffu;
      float lv = mv; uint li = mi;
      for (ushort off = 16; off >= 1; off >>= 1) {
        float ov = simd_shuffle_xor(mv, off); uint oi = simd_shuffle_xor(mi, off);
        if (tl_before(ov, oi, mv, mi)) { mv = ov; mi = oi; }
      }
      if (live && li == mi && lv == mv) { hd++; }
      if (lane == 0) { topv[k] = mv; topi[k] = mi; }
    }
  }
  // ---- 4a. copy the frontier state (all threads), in parallel with nothing else touching it
  for (int i = lid; i < CAP; i += lsize) { ok[i] = fk[i]; opar[i] = fpar[i]; otok[i] = ftok[i]; odep[i] = fdep[i]; }
  for (int i = lid; i < NMAX; i += lsize) { onpp[i] = npp[i]; ondep[i] = ndep[i]; }
  for (int i = lid; i < NMAX * NMAX; i += lsize) { onanc[i] = nanc[i]; }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  // ---- 3 + 4b. probabilities, token ids, the pushes (one thread per child)
  threadgroup int badv[1];
  if (lid == 0) { badv[0] = 0; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int n0 = nf[0];
  if (lid < (uint)KK) {
    float d = topv[lid] - lse;
    float p = metal::precise::exp(d);
    int tok = idmap[topi[lid]];
    otis[lid] = tok; otp[lid] = p;
    ulong pp = (tt > 0) ? npp[tt - 1] : 0x3FF0000000000000ul;
    int pd = (tt > 0) ? ndep[tt - 1] + 1 : 0;
    int bad = 0;
    int slot = n0 + int(lid);
    if (tt < NMAX) {                                   // the last call (tt = NMAX) only reports node tt - 1's top-K
      if (slot < CAP) {
        ok[slot] = f64_mul_f32(pp, p, bad); opar[slot] = tt - 1; otok[slot] = tok; odep[slot] = pd;
      } else { bad = 2; }
    }
    if (bad) { badv[0] = bad; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  const int n = (tt < NMAX) ? min(n0 + KK, CAP) : n0;
  // ---- 4c. pop node tt: the max key, the earliest index on ties (one simdgroup per 32 slots, then simdgroup 0)
  threadgroup ulong pk[32]; threadgroup int px[32];
  {
    ulong kv = 0ul; int ix = 0x7fffffff;
    for (int i = lid; i < n; i += lsize) {
      ulong k2 = ok[i];
      if (k2 > kv || (k2 == kv && k2 != 0ul && i < ix)) { kv = k2; ix = i; }
    }
    for (ushort off = 16; off >= 1; off >>= 1) {
      ulong ok2 = simd_shuffle_xor(kv, off); int ox = simd_shuffle_xor(ix, off);
      if (ok2 > kv || (ok2 == kv && ox < ix)) { kv = ok2; ix = ox; }
    }
    if (lane == 0) { pk[sgi] = kv; px[sgi] = ix; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    ulong bk = 0ul; int best = -1;
    for (int s = 0; s < int((lsize + 31) / 32); s++) {
      if (pk[s] > bk || (pk[s] == bk && pk[s] != 0ul && px[s] < best)) { bk = pk[s]; best = px[s]; }
    }
    int bad = badv[0];
    onf[0] = n;
    if (tt < NMAX) {
      if (best < 0) { bad = 3; best = 0; }
      ok[best] = 0ul;
      int par = opar[best], d = odep[best];
      onpp[tt] = bk; ondep[tt] = d;
      for (int j = 0; j < NMAX; ++j) { onanc[tt * NMAX + j] = -1; }
      if (par >= 0) {
        for (int j = 0; j < d - 1; ++j) { onanc[tt * NMAX + j] = nanc[par * NMAX + j]; }
        onanc[tt * NMAX + d - 1] = par;
      }
      opop[0] = par; opop[1] = otok[best]; opop[2] = d; opop[3] = bad;
    } else {
      opop[0] = -1; opop[1] = 0; opop[2] = 0; opop[3] = bad;
    }
  }
"""
_K = {}


def tail(G, lg, idmap, tt):
    """G: sf_tree.GPUFrontier (its arrays are replaced); lg [1, NS] f32; idmap [NS] int32 (None: identity);
    tt: the node index to pop (node tt - 1's children are pushed; tt = 0 pushes the roots).
    Returns (pop [4] = par, tok, depth, bad of node tt; tis [K] token ids; tp [K] f32) for node tt - 1's children."""
    NS = int(lg.shape[-1])
    if idmap is None:
        idmap = mx.arange(NS, dtype=mx.int32)
    k = _K.setdefault("tail", mx.fast.metal_kernel(
        name="sf_tail", input_names=["lg", "idmap", "fk", "fpar", "ftok", "fdep", "nf", "npp", "ndep", "nanc", "tti"],
        output_names=["ok", "opar", "otok", "odep", "onf", "onpp", "ondep", "onanc", "opop", "otis", "otp"],
        header=HDR, source=SRC))
    C, N, K = G.CAP, G.N, G.K
    o = k(inputs=[lg.reshape(-1), idmap.astype(mx.int32), G.fk, G.fpar, G.ftok, G.fdep, G.nf, G.npp, G.ndep, G.nanc,
                  mx.array([tt], mx.int32)],
          template=[("NS", NS), ("CAP", C), ("NMAX", N), ("KK", K)], grid=(1024, 1, 1), threadgroup=(1024, 1, 1),
          output_shapes=[(C,), (C,), (C,), (C,), (1,), (N,), (N,), (N * N,), (4,), (K,), (K,)],
          output_dtypes=[mx.uint64, mx.int32, mx.int32, mx.int32, mx.int32, mx.uint64, mx.int32, mx.int32, mx.int32,
                         mx.int32, mx.float32])
    (G.fk, G.fpar, G.ftok, G.fdep, G.nf, G.npp, G.ndep, G.nanc, pop, tis, tp) = o
    return pop, tis, tp
