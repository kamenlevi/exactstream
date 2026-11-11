// Bitshift-trellis Viterbi encoder on the CPU (Breakthrough D2, 2026-10-20). QTIP-style trellis: L-bit state, k new
// bits per value (state_t = ((state_{t-1} << k) | b_t) & (2^L - 1)), reconstruction lut[state]. Exact MSE Viterbi per
// tile of T values with a free initial state. Predecessors of state s are p = h * NJ + (s >> k), h < 2^k, NJ = 2^(L-k),
// so the min over predecessors is shared by the 2^k states with the same s >> k. Backpointers stay in cache per tile.
// Multi-threaded with GCD over blocks of tiles. Build: clang -O3 -mcpu=native -shared -o libtrellis.dylib trellis_enc.c
#include <dispatch/dispatch.h>
#include <stdint.h>
#include <stdlib.h>

void viterbi_tiles(const float* x, float* out, int32_t* states, int ntiles, int T, int L, int k, const float* lut) {
  const int S = 1 << L, K = 1 << k, NJ = S >> k, BLK = 64;
  int nblk = (ntiles + BLK - 1) / BLK;
  dispatch_apply((size_t)nblk, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t blk) {
    float* cost = (float*)malloc(sizeof(float) * S);
    float* best = (float*)malloc(sizeof(float) * NJ);
    uint8_t* bp = (uint8_t*)malloc((size_t)(T - 1) * NJ);
    int t0 = (int)blk * BLK, t1 = t0 + BLK < ntiles ? t0 + BLK : ntiles;
    for (int tile = t0; tile < t1; tile++) {
      const float* xt = x + (size_t)tile * T;
      float x0 = xt[0];
      for (int s = 0; s < S; s++) { float d = x0 - lut[s]; cost[s] = d * d; }
      for (int t = 1; t < T; t++) {
        uint8_t* b = bp + (size_t)(t - 1) * NJ;
        for (int j = 0; j < NJ; j++) { best[j] = cost[j]; b[j] = 0; }
        for (int h = 1; h < K; h++) {
          const float* c = cost + (size_t)h * NJ;
          for (int j = 0; j < NJ; j++) { int lt = c[j] < best[j]; best[j] = lt ? c[j] : best[j]; b[j] = lt ? (uint8_t)h : b[j]; }
        }
        float xv = xt[t];
        for (int j = 0; j < NJ; j++) {
          float bj = best[j];
          const float* l = lut + (size_t)j * K;
          float* c = cost + (size_t)j * K;
          for (int q = 0; q < K; q++) { float d = xv - l[q]; c[q] = bj + d * d; }
        }
      }
      int sb = 0; float bv = cost[0];
      for (int s = 1; s < S; s++) if (cost[s] < bv) { bv = cost[s]; sb = s; }
      float* o = out + (size_t)tile * T;
      int32_t* st = states ? states + (size_t)tile * T : 0;
      int s = sb;
      o[T - 1] = lut[s]; if (st) st[T - 1] = s;
      for (int t = T - 1; t > 0; t--) {
        int j = s >> k, h = bp[(size_t)(t - 1) * NJ + j];
        s = (h << (L - k)) | j;
        o[t - 1] = lut[s]; if (st) st[t - 1] = s;
      }
    }
    free(cost); free(best); free(bp);
  });
}
