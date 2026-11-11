// SSD batch policy for the dh_w4 loader (MoonShots 2026-10-22). v3f's loader reads a layer's late experts as a second
// batch after the waited batch lands (two ensure() calls), paying a second request latency; the speed model says that
// costs ~3-6 ms per verify pass. Per trial: nw "waited" + nl "late" MSB-3 reads (2,162,688 B each at record offsets of
// BigMoE's records3 files, F_NOCACHE, one pread per read into a 16 KB-aligned buffer, GCD global queue as the loader),
// records drawn without replacement (cold), from an idle SSD (2 ms pause). Policies:
//   seq        waited batch (dispatch_apply), then the late batch: the loader today
//   d<us>      waited reads issued, the late ones <us> microseconds later (d0 = all at once, waited first)
// Prints per (nw, nl, policy): median time to the last waited read (phase A) and to the last read (phase B), in ms.
// build: clang -O2 -o ssd_batch ssd_batch.c    usage: ssd_batch RECORDS3_DIR [trials=15]   (quiet lock: SSD timing)
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define REC 2932736LL
#define MSB 2162688LL
#define NE 512
#define NL 48
#define MAXR 16

static int fds[NL];
static uint8_t* bufs[MAXR];
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return t.tv_sec * 1e3 + t.tv_nsec * 1e-6; }
static int cmpd(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return (x > y) - (x < y); }
static int* order; static int next_rec = 0;
static void read_one(int rec, int slot) {
  int L = rec / NE, e = rec % NE;
  int64_t done = 0;
  while (done < MSB) {
    ssize_t r = pread(fds[L], bufs[slot] + done, MSB - done, (int64_t)e * REC + done);
    if (r <= 0) { perror("pread"); exit(1); }
    done += r;
  }
}

// one trial: returns phase A (all waited done) and phase B (all done) times, ms from the first issue
static int g_recs[MAXR];
static _Atomic int g_left;
static double g_t0, g_ta;
static void trial(int nw, int nl, int delay_us, int seq, double* tA, double* tB) {
  for (int i = 0; i < nw + nl; i++) g_recs[i] = order[next_rec++];
  dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
  g_t0 = now_ms();
  if (seq) {
    dispatch_apply(nw, q, ^(size_t i) { read_one(g_recs[i], (int)i); });
    *tA = now_ms() - g_t0;
    if (nl) dispatch_apply(nl, q, ^(size_t i) { read_one(g_recs[nw + i], nw + (int)i); });
    *tB = now_ms() - g_t0;
    return;
  }
  atomic_store(&g_left, nw);
  dispatch_group_t g = dispatch_group_create();
  for (int i = 0; i < nw; i++)
    dispatch_group_async(g, q, ^{
      read_one(g_recs[i], i);
      if (atomic_fetch_sub(&g_left, 1) == 1) g_ta = now_ms() - g_t0;
    });
  if (delay_us > 0) { double t1 = now_ms(); while (now_ms() - t1 < delay_us * 1e-3) {} }
  for (int i = 0; i < nl; i++) dispatch_group_async(g, q, ^{ read_one(g_recs[nw + i], nw + i); });
  dispatch_group_wait(g, DISPATCH_TIME_FOREVER);
  dispatch_release(g);
  *tB = now_ms() - g_t0;
  *tA = g_ta;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: ssd_batch RECORDS3_DIR [trials]\n"); return 1; }
  int T = argc > 2 ? atoi(argv[2]) : 15;
  for (int L = 0; L < NL; L++) {
    char p[1024]; snprintf(p, sizeof p, "%s/L%02d.bin", argv[1], L);
    fds[L] = open(p, O_RDONLY); if (fds[L] < 0) { perror(p); return 1; }
    fcntl(fds[L], F_NOCACHE, 1);
  }
  for (int i = 0; i < MAXR; i++) if (posix_memalign((void**)&bufs[i], 16384, MSB)) return 1;
  order = malloc(sizeof(int) * NL * NE);
  for (int i = 0; i < NL * NE; i++) order[i] = i;
  srand(12345);
  for (int i = NL * NE - 1; i > 0; i--) { int j = rand() % (i + 1); int t = order[i]; order[i] = order[j]; order[j] = t; }
  const int NW[] = {1, 2, 3}, NLT[] = {1, 2, 4, 6};
  const int POL[] = {-1, 0, 50, 100, 150, 250};          // -1 = seq
  const int NP = sizeof POL / sizeof POL[0];
  printf("nw nl policy   phaseA_ms phaseB_ms   (median of %d cold trials each)\n", T);
  for (int a = 0; a < 3; a++)
    for (int b = 0; b < 4; b++) {
      int nw = NW[a], nl = NLT[b];
      if ((nw + nl) * T * NP + next_rec > NL * NE) { printf("out of cold records\n"); return 0; }
      double A[16][64], B[16][64];
      for (int t = 0; t < T; t++)
        for (int k = 0; k < NP; k++) {
          int p = POL[(k + t) % NP];                       // rotate the policy order across trials (drift)
          int kk = (k + t) % NP;
          usleep(2000);                                    // idle SSD between layers
          trial(nw, nl, p < 0 ? 0 : p, p < 0, &A[kk][t], &B[kk][t]);
        }
      for (int k = 0; k < NP; k++) {
        qsort(A[k], T, sizeof(double), cmpd); qsort(B[k], T, sizeof(double), cmpd);
        char nm[16]; if (POL[k] < 0) snprintf(nm, sizeof nm, "seq"); else snprintf(nm, sizeof nm, "d%d", POL[k]);
        printf("%2d %2d %-7s  %8.3f  %8.3f\n", nw, nl, nm, A[k][T / 2], B[k][T / 2]);
      }
      fflush(stdout);
    }
  return 0;
}
