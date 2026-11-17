// scripts/slc_bench.m -- SLC microbench for ZERO's IDEAS_1117c section 1.4 (MoonShots, Tue 17 Nov 2026).
// (i)  GPU re-read bandwidth vs buffer size (2 -> 256 MB): the knee = the SLC's effective size for GPU reads.
// (ii) A 21 MB buffer read once by the GPU: cold / after a CPU touch / after a GPU touch, with a 100 MB GPU stream
//      (the expert phase) placed before the touch (A), concurrent with a CPU touch (B), or between the touch and the
//      read (C, the spec's literal "in between"). PASS if CPU-touched ~= GPU-touched (~x2 faster than cold).
// Exact: scheduling only, no numerics. MB = 10^6 bytes. GPU time = command buffer GPUEndTime - GPUStartTime.
// build: clang -O2 -fobjc-arc -framework Metal -framework Foundation -o scratch/slc_bench scripts/slc_bench.m
// run:   scratch/slc_bench data/slc_bench.json | tee data/slc_bench.txt     (SLC_QUICK=1: smoke run)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *SRC =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void rd(device const uint4 *src [[buffer(0)]], device uint *out [[buffer(1)]],\n"
    "               constant uint &n16 [[buffer(2)]], constant uint &passes [[buffer(3)]],\n"
    "               uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {\n"
    "  uint4 acc = uint4(0);\n"
    "  for (uint p = 0; p < passes; ++p) {\n"
    "    uint i = gid;\n"
    "    for (; i + 3 * gsz < n16; i += 4 * gsz) {\n"
    "      uint4 a = src[i], b = src[i + gsz], c = src[i + 2 * gsz], d = src[i + 3 * gsz];\n"
    "      acc += (a ^ b) + (c ^ d) + uint4(p);\n"
    "    }\n"
    "    for (; i < n16; i += gsz) acc += src[i];\n"
    "  }\n"
    "  uint r = acc.x ^ acc.y ^ acc.z ^ acc.w;\n"
    "  if (r == 0x9e3779b9u) out[gid & 1023u] = r;\n"
    "}\n"
    "kernel void spin(device uint *out [[buffer(1)]], constant uint &iters [[buffer(3)]],\n"
    "                 uint gid [[thread_position_in_grid]]) {\n"
    "  uint x = gid;\n"
    "  for (uint i = 0; i < iters; ++i) x = x * 1664525u + 1013904223u;\n"
    "  if (x == 0x9e3779b9u) out[gid & 1023u] = x;\n"
    "}\n";

static id<MTLDevice> dev;
static id<MTLCommandQueue> cq;
static id<MTLComputePipelineState> pso, spso;
static id<MTLBuffer> outb;
static NSUInteger TG = 256, NT = 65536;

static double now_s(void) {
  static mach_timebase_info_data_t tb;
  if (tb.denom == 0) mach_timebase_info(&tb);
  return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9;
}

static id<MTLBuffer> newbuf(size_t n, uint64_t seed) {
  id<MTLBuffer> b = [dev newBufferWithLength:n options:MTLResourceStorageModeShared];
  if (!b) { fprintf(stderr, "alloc %zu failed\n", n); exit(1); }
  uint64_t *w = (uint64_t *)b.contents, x = seed * 0x9E3779B97F4A7C15ull + 1;
  for (size_t i = 0; i < n / 8; i++) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; w[i] = x; }
  return b;
}

static void enc_read(id<MTLCommandBuffer> cb, id<MTLBuffer> b, size_t bytes, uint32_t reps) {
  // reps whole-buffer reads, one dispatch each (serial encoder): every pass sweeps the whole buffer, reuse distance = bytes
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:pso];
  [e setBuffer:b offset:0 atIndex:0];
  [e setBuffer:outb offset:0 atIndex:1];
  uint32_t n16 = (uint32_t)(bytes / 16), one = 1;
  [e setBytes:&n16 length:4 atIndex:2];
  [e setBytes:&one length:4 atIndex:3];
  NSUInteger nt = NT;
  if (nt > n16) nt = n16 / TG * TG;
  if (nt < TG) nt = TG;
  for (uint32_t r = 0; r < reps; r++)
    [e dispatchThreads:MTLSizeMake(nt, 1, 1) threadsPerThreadgroup:MTLSizeMake(TG, 1, 1)];
  [e endEncoding];
}

static double gpu_read(id<MTLBuffer> b, size_t bytes, uint32_t reps) {
  id<MTLCommandBuffer> cb = [cq commandBuffer];
  enc_read(cb, b, bytes, reps);
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.status != MTLCommandBufferStatusCompleted) { fprintf(stderr, "command buffer failed\n"); exit(1); }
  return cb.GPUEndTime - cb.GPUStartTime;
}

// ALU-only spin (no memory traffic): keeps the GPU busy, and its clocks up, while the CPU touches
static void enc_spin(id<MTLCommandBuffer> cb, uint32_t iters) {
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:spso];
  [e setBuffer:outb offset:0 atIndex:1];
  [e setBytes:&iters length:4 atIndex:3];
  [e dispatchThreads:MTLSizeMake(16384, 1, 1) threadsPerThreadgroup:MTLSizeMake(TG, 1, 1)];
  [e endEncoding];
}

// CPU touch: one 8-byte load per 64 bytes (every cache line), split over nth threads.
static volatile uint64_t sink;
static double cpu_touch(const void *p, size_t n, int nth) {
  double t0 = now_s();
  __block uint64_t tot = 0;
  size_t chunk = n / (size_t)nth / 128 * 128;
  dispatch_apply((size_t)nth, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t t) {
    const uint64_t *w = (const uint64_t *)((const uint8_t *)p + t * chunk);
    size_t m = (t + 1 == (size_t)nth ? n - t * chunk : chunk) / 8;
    uint64_t s = 0;
    for (size_t i = 0; i < m; i += 8) s += w[i];
    __atomic_fetch_add(&tot, s, __ATOMIC_RELAXED);
  });
  sink += tot;
  return now_s() - t0;
}

static double cpu_pf3(const void *p, size_t n, int nth) {
  double t0 = now_s();
  size_t chunk = n / (size_t)nth / 128 * 128;
  dispatch_apply((size_t)nth, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t t) {
    const char *w = (const char *)p + t * chunk;
    size_t m = t + 1 == (size_t)nth ? n - t * chunk : chunk;
    for (size_t i = 0; i < m; i += 64) __builtin_prefetch(w + i, 0, 1);  // prfm pldl3keep
  });
  return now_s() - t0;
}

static double cpu_write(void *dst, const void *src, size_t n, int nth) {
  double t0 = now_s();
  size_t chunk = n / (size_t)nth / 128 * 128;
  dispatch_apply((size_t)nth, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t t) {
    size_t m = t + 1 == (size_t)nth ? n - t * chunk : chunk;
    memcpy((char *)dst + t * chunk, (const char *)src + t * chunk, m);
  });
  return now_s() - t0;
}

static int cmpd(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}
static double pct(const double *v, int n, double p) {
  double c[256];
  memcpy(c, v, (size_t)n * sizeof(double));
  qsort(c, (size_t)n, sizeof(double), cmpd);
  double pos = p * (n - 1);
  int lo = (int)pos, hi = lo + 1 < n ? lo + 1 : lo;
  return c[lo] + (c[hi] - c[lo]) * (pos - lo);
}

enum { HOT, COLD, A_GPU, A_CPU, A_EV16, A_EV48, A_PF3, A_CPUW, B_CPU, C_GPU, C_CPU, NCOND };
static const char *CN[NCOND] = {
    "hot: GPU read, gap, read (no stream)", "cold: stream, gap, read", "A_gpu: stream, GPU touch, gap, read",
    "A_cpu: stream, gap with CPU touch, read", "A_ev16: as A_cpu + CPU reads 16 MB other",
    "A_ev48: as A_cpu + CPU reads 48 MB other", "A_pf3: as A_cpu, PRFM PLDL3KEEP not loads",
    "A_cpuw: as A_cpu, CPU writes it (memcpy)",
    "B_cpu: CPU touch during the stream, gap, read",
    "C_gpu: GPU touch, stream, gap, read", "C_cpu: CPU touch, stream, gap, read"};
static const char *CK[NCOND] = {"hot", "cold", "A_gpu", "A_cpu", "A_ev16", "A_ev48", "A_pf3", "A_cpuw", "B_cpu", "C_gpu", "C_cpu"};

int main(int argc, char **argv) {
  @autoreleasepool {
    const char *outp = argc > 1 ? argv[1] : "slc_bench.json";
    int quick = getenv("SLC_QUICK") != NULL;
    if (getenv("SLC_NT")) NT = strtoul(getenv("SLC_NT"), NULL, 10);
    dev = MTLCreateSystemDefaultDevice();
    cq = [dev newCommandQueue];
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(SRC) options:nil error:&err];
    if (!lib) { fprintf(stderr, "compile: %s\n", err.localizedDescription.UTF8String); return 1; }
    pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rd"] error:&err];
    if (!pso) { fprintf(stderr, "pso: %s\n", err.localizedDescription.UTF8String); return 1; }
    spso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"spin"] error:&err];
    if (!spso) { fprintf(stderr, "spin pso: %s\n", err.localizedDescription.UTF8String); return 1; }
    outb = [dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    const size_t MB = 1000000;
    size_t bF = 512 * MB, bCF = 256 * MB, bX = 100 * MB, bB = 21 * MB, bS = 256 * MB;
    id<MTLBuffer> F = newbuf(bF, 1), X = newbuf(bX, 2), B = newbuf(bB, 3), S = newbuf(bS, 4);
    void *CF = malloc(bCF), *E = malloc(48 * MB);
    memset(CF, 0x5a, bCF);
    memset(E, 0xa5, 48 * MB);
    FILE *js = fopen(outp, "w");
    if (!js) { fprintf(stderr, "cannot write %s\n", outp); return 1; }
    printf("device %s, threads %lu, threadgroup %lu, quick %d\n", dev.name.UTF8String, (unsigned long)NT,
           (unsigned long)TG, quick);
    fprintf(js, "{\"device\": \"%s\", \"threads\": %lu, \"threadgroup\": %lu, \"MB\": 1000000, \"quick\": %d,\n",
            dev.name.UTF8String, (unsigned long)NT, (unsigned long)TG, quick);
    for (int i = 0; i < 5; i++) gpu_read(F, bF, 1);  // clocks up

    // (i) re-read bandwidth vs size
    int sizes[] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 32, 36, 40, 48, 56, 64, 85, 96, 128, 192, 256};
    int ns = (int)(sizeof sizes / sizeof sizes[0]), R1 = quick ? 2 : 7;
    double hot_med[32], cold_med[32];
    uint32_t passes_of[32];
    fprintf(js, "\"sizes\": [\n");
    for (int k = 0; k < ns; k++) {
      size_t n = (size_t)sizes[k] * MB;
      uint32_t P = (uint32_t)fmax(10.0, ceil(256.0 * MB / (double)n));
      double tc[16], th[16];
      for (int r = 0; r < R1; r++) {
        gpu_read(F, bF, 1);  // flush
        tc[r] = gpu_read(S, n, 1);
        th[r] = gpu_read(S, n, P);
      }
      cold_med[k] = pct(tc, R1, 0.5);
      hot_med[k] = pct(th, R1, 0.5);
      passes_of[k] = P;
      fprintf(js, "  {\"mb\": %d, \"passes\": %u, \"cold_s\": [", sizes[k], P);
      for (int r = 0; r < R1; r++) fprintf(js, "%s%.9f", r ? ", " : "", tc[r]);
      fprintf(js, "], \"hot_s\": [");
      for (int r = 0; r < R1; r++) fprintf(js, "%s%.9f", r ? ", " : "", th[r]);
      fprintf(js, "]}%s\n", k + 1 < ns ? "," : "");
    }
    fprintf(js, "],\n");
    double dram = (double)sizes[ns - 1] * MB * passes_of[ns - 1] / hot_med[ns - 1] / 1e9;
    printf("\n(i) GPU re-read bandwidth vs buffer size (median of %d; cold = 1 read after a 512 MB flush, "
           "hot = passes x whole-buffer re-reads (one dispatch each) right after it; DRAM = hot at 256 MB)\n", R1);
    printf("%6s %7s %10s %10s %11s %10s %9s\n", "MB", "passes", "cold us", "cold GB/s", "hot us/pass", "hot GB/s", "hot/DRAM");
    int knee15 = 0;
    for (int k = 0; k < ns; k++) {
      double n = (double)sizes[k] * MB, hot = n * passes_of[k] / hot_med[k] / 1e9;
      if (hot / dram >= 1.5) knee15 = sizes[k];
      printf("%6d %7u %10.1f %10.1f %11.1f %10.1f %9.2f\n", sizes[k], passes_of[k], cold_med[k] * 1e6,
             n / cold_med[k] / 1e9, hot_med[k] / passes_of[k] * 1e6, hot, hot / dram);
    }
    printf("largest size with hot >= 1.5 x DRAM: %d MB (DRAM %.1f GB/s)\n", knee15, dram);

    // CPU touch speed for 21 MB from cold
    double t1[8], t4[8];
    for (int r = 0; r < 5; r++) { gpu_read(F, bF, 1); cpu_touch(CF, bCF, 4); t1[r] = cpu_touch(B.contents, bB, 1); }
    for (int r = 0; r < 5; r++) { gpu_read(F, bF, 1); cpu_touch(CF, bCF, 4); t4[r] = cpu_touch(B.contents, bB, 4); }
    // spin calibrated to ~1.2 ms of GPU time (longer than the longest CPU touch, 21 + 48 MB)
    uint32_t spin_iters = 20000;
    {
      double ts = 0;
      for (int r = 0; r < 3; r++) {
        id<MTLCommandBuffer> cb = [cq commandBuffer];
        enc_spin(cb, spin_iters);
        [cb commit];
        [cb waitUntilCompleted];
        ts = cb.GPUEndTime - cb.GPUStartTime;
      }
      spin_iters = (uint32_t)(spin_iters * 1.2e-3 / ts);
    }
    printf("\nCPU touch of 21 MB from cold: 1 thread %.0f us (%.0f GB/s), 4 threads %.0f us (%.0f GB/s); "
           "between the stream and the timed read the GPU runs an ALU-only spin (%u iters, ~1.2 ms) in every condition\n",
           pct(t1, 5, .5) * 1e6, bB / pct(t1, 5, .5) / 1e9, pct(t4, 5, .5) * 1e6, bB / pct(t4, 5, .5) / 1e9, spin_iters);
    fprintf(js, "\"touch21_1thread_s\": %.9f, \"touch21_4threads_s\": %.9f, \"spin_iters\": %u,\n", pct(t1, 5, .5),
            pct(t4, 5, .5), spin_iters);

    // (ii) 21 MB read: cold / CPU-touched / GPU-touched
    int R2 = quick ? 3 : 30, idle_gaps = 0;
    static double tr[NCOND][64], tt[NCOND][64], tspin[NCOND][64];
    for (int r = 0; r < R2; r++) {
      for (int j = 0; j < NCOND; j++) {
        int c = (j + r) % NCOND;
        gpu_read(F, bF, 1);     // flush the SLC
        cpu_touch(CF, bCF, 4);  // flush the CPU caches
        double touch = 0;
        if (c == C_CPU) touch = cpu_touch(B.contents, bB, 4);
        id<MTLCommandBuffer> cb1 = [cq commandBuffer], cb2 = [cq commandBuffer], cb3 = [cq commandBuffer];
        if (c == HOT || c == C_GPU) enc_read(cb1, B, bB, 1);
        if (c != HOT) enc_read(cb1, X, bX, 1);
        if (c == A_GPU) enc_read(cb1, B, bB, 1);
        enc_spin(cb2, spin_iters);
        [cb1 commit];
        [cb2 commit];
        if (c == B_CPU) touch = cpu_touch(B.contents, bB, 4);
        [cb1 waitUntilCompleted];
        if (c == A_CPU || c == A_EV16 || c == A_EV48) touch = cpu_touch(B.contents, bB, 4);
        if (c == A_EV16) touch += cpu_touch(E, 16 * MB, 4);
        if (c == A_EV48) touch += cpu_touch(E, 48 * MB, 4);
        if (c == A_PF3) touch = cpu_pf3(B.contents, bB, 4);
        if (c == A_CPUW) touch = cpu_write(B.contents, E, bB, 4);
        enc_read(cb3, B, bB, 1);
        [cb3 commit];
        [cb3 waitUntilCompleted];
        if (cb3.GPUStartTime - cb2.GPUEndTime > 50e-6) idle_gaps++;
        tr[c][r] = cb3.GPUEndTime - cb3.GPUStartTime;
        tt[c][r] = touch;
        tspin[c][r] = cb3.GPUStartTime - cb2.GPUEndTime;
      }
    }
    printf("trials whose timed read started > 50 us after the spin ended (GPU idled): %d of %d\n", idle_gaps, R2 * NCOND);
    double cold = pct(tr[COLD], R2, .5);
    printf("\n(ii) 21 MB read once by the GPU; %d interleaved trials per condition, each after a 512 MB GPU flush "
           "+ 256 MB CPU flush; stream = 100 MB GPU read\n", R2);
    printf("%-50s %8s %8s %8s %8s %7s %9s\n", "condition", "med us", "p25", "p75", "GB/s", "cold/x", "touch us");
    fprintf(js, "\"read21\": {\n");
    for (int c = 0; c < NCOND; c++) {
      double m = pct(tr[c], R2, .5);
      printf("%-50s %8.1f %8.1f %8.1f %8.1f %7.2f %9.0f\n", CN[c], m * 1e6, pct(tr[c], R2, .25) * 1e6,
             pct(tr[c], R2, .75) * 1e6, bB / m / 1e9, cold / m, pct(tt[c], R2, .5) * 1e6);
      fprintf(js, "  \"%s\": {\"read_s\": [", CK[c]);
      for (int r = 0; r < R2; r++) fprintf(js, "%s%.9f", r ? ", " : "", tr[c][r]);
      fprintf(js, "], \"touch_s\": [");
      for (int r = 0; r < R2; r++) fprintf(js, "%s%.9f", r ? ", " : "", tt[c][r]);
      fprintf(js, "]}%s\n", c + 1 < NCOND ? "," : "");
    }
    fprintf(js, "},\n\"knee_hot_ge_1p5x_dram_mb\": %d, \"dram_gbs\": %.2f}\n", knee15, dram);
    fclose(js);
    double a_gpu = cold / pct(tr[A_GPU], R2, .5), a_cpu = cold / pct(tr[A_CPU], R2, .5);
    printf("\nA: GPU-touched x%.2f vs cold, CPU-touched x%.2f vs cold -> %s\n", a_gpu, a_cpu,
           (a_cpu < 1.1) ? "KILL (CPU touch does not help the GPU)"
           : (a_gpu >= 1.5 && a_cpu >= 0.85 * a_gpu) ? "PASS (CPU-touched ~= GPU-touched)"
                                                       : "PARTIAL");
  }
  return 0;
}
