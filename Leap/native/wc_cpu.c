// wc_cpu.c - the CPU arms of the WHOLE-CHIP screen (Leap, Tue 2026-11-03 17:1x; TABLE_350 'WHOLE-CHIP').
// Build: clang -O3 -mcpu=native -o wc_cpu wc_cpu.c -framework Accelerate
// Every mode prints "ready" after its setup, then one line a second, "<unix time> <fields>", and stops on SIGTERM or
// after <seconds>:
//   bw    <threads> <seconds>             streaming reads of a private 1 GB buffer -> gbps
//   q4    <threads> <seconds> <routing>   a 4-bit expert kernel (gate + up, SiLU, down, scores) over the windows'
//                                         rows per expert: int8 dot products (i8mm USDOT), activations Q8 per 64-group,
//                                         fp16 scales / biases; synthetic experts of the real shapes
//                                         -> exp_per_ms, gbps (expert bytes), win_per_s
//   check <routing>                       q4 on window 0 vs a float64 reference of the same experts -> cos
//   sgemm <threads> <seconds>             Accelerate's cblas_sgemm (SME) on the dense 16-row expert shape, fp32
//                                         weights, one BLAS thread per caller -> exp_per_ms, gflops
// routing file: int32 nwin, then per window int32 ids[16 * 10] and float32 scores[16 * 10].
#include <arm_neon.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#define HD 2560
#define ID 640
#define NE 512
#define S 16
#define TOPK 10
#define QG ((size_t)ID * HD / 2)       // gate / up nibbles
#define QD ((size_t)HD * ID / 2)       // down nibbles
#define SG ((size_t)ID * (HD / 64))    // gate / up groups
#define SD ((size_t)HD * (ID / 64))    // down groups
#define EXB (2 * QG + QD + 2 * (2 * SG + SD) * sizeof(_Float16))

static volatile sig_atomic_t stop_flag = 0;
static void on_term(int s) { (void)s; stop_flag = 1; }
static double now(void) { struct timeval tv; gettimeofday(&tv, 0); return tv.tv_sec + tv.tv_usec * 1e-6; }
static uint64_t rs = 88172645463325252ull;
static inline uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static float gauss(void) { double u = (rnd() >> 11) * (1.0 / 9007199254740992.0) + 1e-12, v = (rnd() >> 11) * (1.0 / 9007199254740992.0); return (float)(sqrt(-2 * log(u)) * cos(6.283185307179586 * v)); }

// ---------------------------------------------------------------- experts
typedef struct { const uint8_t *q; const _Float16 *s, *b; } proj_t;
typedef struct { proj_t g, u, d; } expert_t;
static uint8_t *POOL;
static expert_t EX[NE];
static void make_pool(void) {
    POOL = malloc(EXB * NE);
    uint64_t *p = (uint64_t *)POOL;
    for (size_t i = 0; i < EXB * NE / 8; i++) p[i] = rnd();
    for (int e = 0; e < NE; e++) {
        uint8_t *B = POOL + (size_t)e * EXB;
        _Float16 *sb = (_Float16 *)(B + 2 * QG + QD);
        EX[e].g = (proj_t){B, sb, sb + SG};
        EX[e].u = (proj_t){B + QG, sb + 2 * SG, sb + 3 * SG};
        EX[e].d = (proj_t){B + 2 * QG, sb + 4 * SG, sb + 4 * SG + SD};
        _Float16 *sp[3] = {sb, sb + 2 * SG, sb + 4 * SG}, *bp[3] = {sb + SG, sb + 3 * SG, sb + 4 * SG + SD};
        size_t n[3] = {SG, SG, SD};
        for (int k = 0; k < 3; k++)
            for (size_t i = 0; i < n[k]; i++) {                  // weights = s * q + b, centred: b = -7.5 s
                float s = 0.012f * (1.0f + 0.1f * gauss());
                sp[k][i] = (_Float16)s;
                bp[k][i] = (_Float16)(-7.5f * s);
            }
    }
}

// activations of one row set, Q8 per 64-group, each group's 64 values stored as [32 even k | 32 odd k] to match the
// nibble order (byte j of a group: low nibble = k 2j, high nibble = k 2j + 1)
static void quant_rows(const float *x, int m, int K, int8_t *xq, float *xs, float *xsum) {
    int G = K / 64;
    for (int i = 0; i < m; i++)
        for (int g = 0; g < G; g++) {
            const float *v = x + (size_t)i * K + g * 64;
            float mx = 0;
            for (int k = 0; k < 64; k++) mx = fmaxf(mx, fabsf(v[k]));
            float sc = mx > 0 ? mx / 127.0f : 1.0f, inv = 1.0f / sc;
            int8_t *o = xq + (size_t)i * K + g * 64;
            int sm = 0;                                           // the bias term uses the QUANTIZED sum, so it
            for (int j = 0; j < 32; j++) {                        // cancels the codes' zero point exactly
                o[j] = (int8_t)lrintf(v[2 * j] * inv);
                o[32 + j] = (int8_t)lrintf(v[2 * j + 1] * inv);
                sm += o[j] + o[32 + j];
            }
            xs[i * G + g] = sc;
            xsum[i * G + g] = sc * (float)sm;
        }
}

// y[i][n] = sum_g s[n][g] xs[i][g] (q[n] . xq[i])_g + b[n][g] xsum[i][g], for MM rows at once (unrolled by MM)
#define PROJ_MM(MM)                                                                                                    \
    static void proj##MM(proj_t P, int N, int K, const int8_t *xq, const float *xs, const float *xsum, float *y,       \
                         int ystride) {                                                                                \
        int G = K / 64;                                                                                                \
        const uint8x16_t m4 = vdupq_n_u8(0xF);                                                                         \
        for (int n = 0; n < N; n++) {                                                                                  \
            const uint8_t *qr = P.q + (size_t)n * (K / 2);                                                             \
            const _Float16 *sr = P.s + (size_t)n * G, *br = P.b + (size_t)n * G;                                       \
            float32x4_t acc[MM];                                                                                       \
            float bs[MM];                                                                                              \
            for (int i = 0; i < MM; i++) { acc[i] = vdupq_n_f32(0); bs[i] = 0; }                                       \
            for (int g = 0; g < G; g++) {                                                                              \
                uint8x16_t B0 = vld1q_u8(qr + g * 32), B1 = vld1q_u8(qr + g * 32 + 16);                                \
                uint8x16_t l0 = vandq_u8(B0, m4), h0 = vshrq_n_u8(B0, 4), l1 = vandq_u8(B1, m4), h1 = vshrq_n_u8(B1, 4); \
                float sc = (float)sr[g], bi = (float)br[g];                                                            \
                for (int i = 0; i < MM; i++) {                                                                         \
                    const int8_t *xr = xq + (size_t)i * K + g * 64;                                                    \
                    int32x4_t d = vusdotq_s32(vdupq_n_s32(0), l0, vld1q_s8(xr));                                       \
                    d = vusdotq_s32(d, l1, vld1q_s8(xr + 16));                                                         \
                    d = vusdotq_s32(d, h0, vld1q_s8(xr + 32));                                                         \
                    d = vusdotq_s32(d, h1, vld1q_s8(xr + 48));                                                         \
                    acc[i] = vfmaq_n_f32(acc[i], vcvtq_f32_s32(d), sc * xs[i * G + g]);                               \
                    bs[i] += bi * xsum[i * G + g];                                                                     \
                }                                                                                                      \
            }                                                                                                          \
            for (int i = 0; i < MM; i++) y[(size_t)i * ystride + n] = vaddvq_f32(acc[i]) + bs[i];                      \
        }                                                                                                              \
    }
PROJ_MM(1)
PROJ_MM(2)
PROJ_MM(3)
PROJ_MM(4)
static void proj(proj_t P, int N, int K, const int8_t *xq, const float *xs, const float *xsum, int m, float *y) {
    int G = K / 64;
    for (int i0 = 0; i0 < m; i0 += 4) {
        int mm = m - i0 < 4 ? m - i0 : 4;
        const int8_t *a = xq + (size_t)i0 * K;
        const float *b = xs + i0 * G, *c = xsum + i0 * G;
        float *o = y + (size_t)i0 * N;
        if (mm == 4) proj4(P, N, K, a, b, c, o, N);
        else if (mm == 3) proj3(P, N, K, a, b, c, o, N);
        else if (mm == 2) proj2(P, N, K, a, b, c, o, N);
        else proj1(P, N, K, a, b, c, o, N);
    }
}

// ---------------------------------------------------------------- windows
typedef struct { int ne; int e[S * TOPK]; int m[S * TOPK]; int row[S * TOPK][S]; float w[S * TOPK][S]; } win_t;
static int NW;
static win_t *WIN;
static float X[S * HD];
static int8_t XQ[S * HD];
static float XS[S * HD / 64], XSUM[S * HD / 64];
static void load_routing(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f || fread(&NW, 4, 1, f) != 1) { fprintf(stderr, "bad routing file\n"); exit(1); }
    WIN = calloc(NW, sizeof(win_t));
    for (int w = 0; w < NW; w++) {
        int ids[S * TOPK];
        float sc[S * TOPK];
        if (fread(ids, 4, S * TOPK, f) != S * TOPK || fread(sc, 4, S * TOPK, f) != S * TOPK) { fprintf(stderr, "short routing file\n"); exit(1); }
        win_t *W = &WIN[w];
        for (int r = 0; r < S; r++)
            for (int k = 0; k < TOPK; k++) {
                int e = ids[r * TOPK + k], j;
                for (j = 0; j < W->ne && W->e[j] != e; j++);
                if (j == W->ne) { W->e[j] = e; W->m[j] = 0; W->ne++; }
                W->row[j][W->m[j]] = r;
                W->w[j][W->m[j]] = sc[r * TOPK + k];
                W->m[j]++;
            }
    }
    fclose(f);
    for (int i = 0; i < S * HD; i++) X[i] = gauss();
    quant_rows(X, S, HD, XQ, XS, XSUM);
}

typedef struct { int8_t xq[S * HD]; float xs[S * HD / 64], xsum[S * HD / 64]; float g[S * ID], u[S * ID], h[S * ID];
                 int8_t hq[S * ID]; float hs[S * ID / 64], hsum[S * ID / 64]; float y[S * HD]; float out[S * HD]; } tbuf_t;
static void run_expert(const win_t *W, int j, tbuf_t *T) {
    int m = W->m[j];
    const expert_t *E = &EX[W->e[j]];
    for (int i = 0; i < m; i++) {
        int r = W->row[j][i];
        memcpy(T->xq + (size_t)i * HD, XQ + (size_t)r * HD, HD);
        memcpy(T->xs + i * (HD / 64), XS + r * (HD / 64), sizeof(float) * (HD / 64));
        memcpy(T->xsum + i * (HD / 64), XSUM + r * (HD / 64), sizeof(float) * (HD / 64));
    }
    proj(E->g, ID, HD, T->xq, T->xs, T->xsum, m, T->g);
    proj(E->u, ID, HD, T->xq, T->xs, T->xsum, m, T->u);
    for (int i = 0; i < m * ID; i++) { float a = T->g[i]; T->h[i] = a / (1.0f + expf(-a)) * T->u[i]; }
    quant_rows(T->h, m, ID, T->hq, T->hs, T->hsum);
    proj(E->d, HD, ID, T->hq, T->hs, T->hsum, m, T->y);
    for (int i = 0; i < m; i++) {
        float *o = T->out + (size_t)W->row[j][i] * HD;
        const float *yy = T->y + (size_t)i * HD;
        float32x4_t wv = vdupq_n_f32(W->w[j][i]);
        for (int n = 0; n < HD; n += 4) vst1q_f32(o + n, vfmaq_f32(vld1q_f32(o + n), vld1q_f32(yy + n), wv));
    }
}

// ---------------------------------------------------------------- threads
static int NT;
static double T_END;
static atomic_int next_j, bar_count, bar_gen;
static atomic_long done_bytes, done_exp, done_win;
static void barrier(void) {
    int gen = atomic_load(&bar_gen);
    if (atomic_fetch_add(&bar_count, 1) == NT - 1) { atomic_store(&bar_count, 0); atomic_fetch_add(&bar_gen, 1); }
    else while (atomic_load(&bar_gen) == gen) __builtin_arm_yield();
}
static atomic_int cur_win;
static void *q4_worker(void *arg) {
    (void)arg;
    tbuf_t *T = malloc(sizeof(tbuf_t));
    for (;;) {
        barrier();                                                // window start
        int w = atomic_load(&cur_win);
        if (w < 0) break;
        const win_t *W = &WIN[w];
        int j;
        while ((j = atomic_fetch_add(&next_j, 1)) < W->ne) run_expert(W, j, T);
        barrier();                                                // window end
    }
    free(T);
    return 0;
}
static void report_bw(void) {
    double last = now();
    long b0 = 0;
    printf("ready\n");
    fflush(stdout);
    while (!stop_flag && now() < T_END) {
        struct timespec ts = {0, 20000000};
        nanosleep(&ts, 0);
        double t = now();
        if (t - last >= 1.0) {
            long b = atomic_load(&done_bytes);
            printf("%.3f gbps %.2f\n", t, (b - b0) / (t - last) / 1e9);
            fflush(stdout);
            b0 = b; last = t;
        }
    }
}

static void run_q4(int nt, double secs, const char *rpath) {
    load_routing(rpath);
    make_pool();
    NT = nt;                                                      // the driver is worker 0, so nt busy threads in all
    pthread_t th[16];
    tbuf_t *T0 = malloc(sizeof(tbuf_t));
    for (long i = 1; i < nt; i++) pthread_create(&th[i], 0, q4_worker, (void *)i);
    T_END = now() + secs;
    double last = now();
    long b0 = 0, e0 = 0, w0 = 0;
    printf("ready %d windows, %.1f unique experts per window, %.2f MB per expert\n", NW,
           ({ double s = 0; for (int w = 0; w < NW; w++) s += WIN[w].ne; s / NW; }), EXB / 1e6);
    fflush(stdout);
    int w = 0;
    while (!stop_flag && now() < T_END) {
        atomic_store(&cur_win, w);
        atomic_store(&next_j, 0);
        barrier();                                                // release the workers into window w
        const win_t *W = &WIN[w];
        int j;
        while ((j = atomic_fetch_add(&next_j, 1)) < W->ne) run_expert(W, j, T0);
        barrier();                                                // wait for window w to finish
        atomic_fetch_add(&done_exp, W->ne);
        atomic_fetch_add(&done_bytes, (long)W->ne * (long)EXB);
        atomic_fetch_add(&done_win, 1);
        w = (w + 1) % NW;
        double t = now();
        if (t - last >= 1.0) {
            long b = atomic_load(&done_bytes), e = atomic_load(&done_exp), ww = atomic_load(&done_win);
            double dt = t - last;
            printf("%.3f gbps %.2f exp_per_ms %.3f win_per_s %.1f\n", t, (b - b0) / dt / 1e9, (e - e0) / dt / 1e3, (ww - w0) / dt);
            fflush(stdout);
            b0 = b; e0 = e; w0 = ww; last = t;
        }
    }
    atomic_store(&cur_win, -1);
    barrier();
    for (int i = 1; i < nt; i++) pthread_join(th[i], 0);
}

// ---------------------------------------------------------------- check
static void run_check(const char *rpath) {
    load_routing(rpath);
    make_pool();
    tbuf_t *T = calloc(1, sizeof(tbuf_t));
    const win_t *W = &WIN[0];
    for (int j = 0; j < W->ne; j++) run_expert(W, j, T);
    double *ref = calloc(S * HD, sizeof(double)), *g = malloc(ID * 8), *u = malloc(ID * 8), *h = malloc(ID * 8);
    for (int j = 0; j < W->ne; j++) {
        const expert_t *E = &EX[W->e[j]];
        for (int i = 0; i < W->m[j]; i++) {
            const float *x = X + (size_t)W->row[j][i] * HD;
            for (int n = 0; n < ID; n++) {
                double ag = 0, au = 0;
                for (int k = 0; k < HD; k++) {
                    size_t gi = (size_t)n * (HD / 64) + k / 64;
                    int bidx = (k % 64) / 2, hi = k & 1;
                    uint8_t bg = E->g.q[(size_t)n * (HD / 2) + (k / 64) * 32 + bidx], bu = E->u.q[(size_t)n * (HD / 2) + (k / 64) * 32 + bidx];
                    int qg = hi ? bg >> 4 : bg & 15, qu = hi ? bu >> 4 : bu & 15;
                    ag += ((double)E->g.s[gi] * qg + (double)E->g.b[gi]) * x[k];
                    au += ((double)E->u.s[gi] * qu + (double)E->u.b[gi]) * x[k];
                }
                g[n] = ag; u[n] = au;
                h[n] = ag / (1 + exp(-ag)) * au;
            }
            double *o = ref + (size_t)W->row[j][i] * HD;
            for (int n = 0; n < HD; n++) {
                double a = 0;
                for (int k = 0; k < ID; k++) {
                    size_t gi = (size_t)n * (ID / 64) + k / 64;
                    uint8_t bd = E->d.q[(size_t)n * (ID / 2) + (k / 64) * 32 + (k % 64) / 2];
                    int qd = (k & 1) ? bd >> 4 : bd & 15;
                    a += ((double)E->d.s[gi] * qd + (double)E->d.b[gi]) * h[k];
                }
                o[n] += W->w[j][i] * a;
            }
        }
    }
    double dot = 0, na = 0, nb = 0, mn = 1;
    for (int r = 0; r < S; r++) {
        double d = 0, a = 0, b = 0;
        for (int n = 0; n < HD; n++) { double p = T->out[r * HD + n], q = ref[r * HD + n]; d += p * q; a += p * p; b += q * q; }
        mn = fmin(mn, d / sqrt(a * b + 1e-30));
        dot += d; na += a; nb += b;
    }
    printf("check window 0 (%d experts, 160 pairs): cos %.6f, min row cos %.6f\n", W->ne, dot / sqrt(na * nb), mn);
}

// ---------------------------------------------------------------- bw
static uint8_t *BUF;
static size_t BUFN = (size_t)1 << 30;
static void *bw_worker(void *arg) {
    long tid = (long)arg;
    size_t chunk = BUFN / NT, lo = chunk * tid, step = (size_t)1 << 20;
    uint8x16_t acc = vdupq_n_u8(0);
    size_t p = lo;
    while (!stop_flag && now() < T_END) {
        const uint8_t *q = BUF + p;
        for (size_t i = 0; i < step; i += 128) {
            uint8x16_t a0 = vld1q_u8(q + i), a1 = vld1q_u8(q + i + 16), a2 = vld1q_u8(q + i + 32), a3 = vld1q_u8(q + i + 48);
            uint8x16_t a4 = vld1q_u8(q + i + 64), a5 = vld1q_u8(q + i + 80), a6 = vld1q_u8(q + i + 96), a7 = vld1q_u8(q + i + 112);
            acc = veorq_u8(acc, veorq_u8(veorq_u8(veorq_u8(a0, a1), veorq_u8(a2, a3)), veorq_u8(veorq_u8(a4, a5), veorq_u8(a6, a7))));
        }
        atomic_fetch_add(&done_bytes, (long)step);
        p += step;
        if (p + step > lo + chunk) p = lo;
    }
    return (void *)(uintptr_t)vgetq_lane_u8(acc, 0);
}
static void run_bw(int nt, double secs) {
    BUF = malloc(BUFN);
    for (size_t i = 0; i < BUFN; i += 8) *(uint64_t *)(BUF + i) = rnd();
    NT = nt;
    T_END = now() + secs;
    pthread_t th[16];
    for (long i = 0; i < nt; i++) pthread_create(&th[i], 0, bw_worker, (void *)i);
    report_bw();
    stop_flag = 1;
    for (int i = 0; i < nt; i++) pthread_join(th[i], 0);
}

// ---------------------------------------------------------------- sgemm
#define SG_EXP 16
static void *sg_worker(void *arg) {
    (void)arg;
    float *W = malloc(sizeof(float) * (size_t)SG_EXP * 3 * ID * HD), *x = malloc(sizeof(float) * S * HD);
    float *g = malloc(sizeof(float) * S * ID), *u = malloc(sizeof(float) * S * ID), *y = malloc(sizeof(float) * S * HD);
    for (size_t i = 0; i < (size_t)SG_EXP * 3 * ID * HD; i++) W[i] = 0.02f * gauss();
    for (int i = 0; i < S * HD; i++) x[i] = gauss();
    barrier();
    int e = 0;
    while (!stop_flag && now() < T_END) {
        const float *Wg = W + (size_t)e * 3 * ID * HD, *Wu = Wg + (size_t)ID * HD, *Wd = Wu + (size_t)ID * HD;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, S, ID, HD, 1.0f, x, HD, Wg, HD, 0.0f, g, ID);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, S, ID, HD, 1.0f, x, HD, Wu, HD, 0.0f, u, ID);
        for (int i = 0; i < S * ID; i++) g[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, S, HD, ID, 1.0f, g, ID, Wd, ID, 0.0f, y, HD);
        atomic_fetch_add(&done_exp, 1);
        atomic_fetch_add(&done_bytes, (long)3 * ID * HD * 4);
        e = (e + 1) % SG_EXP;
    }
    return 0;
}
static void run_sgemm(int nt, double secs) {
    NT = nt + 1;
    T_END = now() + 1e9;
    pthread_t th[16];
    for (long i = 0; i < nt; i++) pthread_create(&th[i], 0, sg_worker, (void *)i);
    barrier();                                                    // all weights built
    T_END = now() + secs;
    double last = now();
    long e0 = 0;
    printf("ready\n");
    fflush(stdout);
    while (!stop_flag && now() < T_END) {
        struct timespec ts = {0, 20000000};
        nanosleep(&ts, 0);
        double t = now();
        if (t - last >= 1.0) {
            long e = atomic_load(&done_exp);
            double r = (e - e0) / (t - last);
            printf("%.3f exp_per_ms %.3f gflops %.0f gbps_fp32 %.1f\n", t, r / 1e3, r * 2.0 * S * 3 * ID * HD / 1e9, r * 3.0 * ID * HD * 4 / 1e9);
            fflush(stdout);
            e0 = e; last = t;
        }
    }
    stop_flag = 1;
    for (int i = 0; i < nt; i++) pthread_join(th[i], 0);
}

int main(int argc, char **argv) {
    setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
    signal(SIGTERM, on_term);
    signal(SIGINT, on_term);
    setvbuf(stdout, 0, _IOLBF, 0);
    if (argc >= 4 && !strcmp(argv[1], "bw")) run_bw(atoi(argv[2]), atof(argv[3]));
    else if (argc >= 5 && !strcmp(argv[1], "q4")) run_q4(atoi(argv[2]), atof(argv[3]), argv[4]);
    else if (argc >= 3 && !strcmp(argv[1], "check")) run_check(argv[2]);
    else if (argc >= 4 && !strcmp(argv[1], "sgemm")) run_sgemm(atoi(argv[2]), atof(argv[3]));
    else { fprintf(stderr, "usage: wc_cpu bw|q4|check|sgemm ...\n"); return 2; }
    return 0;
}
