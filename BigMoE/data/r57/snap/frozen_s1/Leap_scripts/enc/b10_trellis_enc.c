// Leap copy (Sun 2026-10-25) of ~/Fit12/scripts/fit_trellis_enc.c (itself a copy of Breakthrough's D2 trellis_enc.c); verbatim below.
// Fit12 B3 copy (2026-10-25) of Breakthrough's scripts/trellis_enc.c (D2, 2026-10-20; read, never edited there): the same
// exact-MSE bitshift-trellis Viterbi per tile of T values (L-bit state, k new bits per value, free initial state, recon
// lut[state]); added: a worker count nthr (tiles strided over nthr GCD workers) so a CPU-share lock's core budget holds.
// Build: clang -O3 -mcpu=native -shared -fblocks -o fit_trellis.dylib fit_trellis_enc.c
#include <dispatch/dispatch.h>
#include <stdint.h>
#include <stdlib.h>

void viterbi_tiles_n(const float* x, float* out, int ntiles, int T, int L, int k, const float* lut, int nthr) {
  const int S = 1 << L, K = 1 << k, NJ = S >> k;
  if (nthr < 1) nthr = 1;
  dispatch_apply((size_t)nthr, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t w) {
    float* cost = (float*)malloc(sizeof(float) * S);
    float* best = (float*)malloc(sizeof(float) * NJ);
    uint8_t* bp = (uint8_t*)malloc((size_t)(T - 1) * NJ);
    for (int tile = (int)w; tile < ntiles; tile += nthr) {
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
      int s = sb;
      o[T - 1] = lut[s];
      for (int t = T - 1; t > 0; t--) {
        int j = s >> k, h = bp[(size_t)(t - 1) * NJ + j];
        s = (h << (L - k)) | j;
        o[t - 1] = lut[s];
      }
    }
    free(cost); free(best); free(bp);
  });
}
