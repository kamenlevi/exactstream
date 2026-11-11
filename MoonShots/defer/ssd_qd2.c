// NewIdeas S10 (Sun ~10:00): CONTINUOUS-queue SSD throughput at the B10 read units, the upper bound for a cross-layer read
// pipeline (ssd_prefix measured per-layer bursts: ~0.3 ms fixed per burst). Per cell (size, QD): QD threads (user-
// interactive QoS) keep QD cold F_NOCACHE preads in flight, each pulling the next record from a shared counter (records
// drawn without replacement from BigMoE's records3 files, reshuffled per round), N reads per cell. GB/s over the middle
// 80 % of completions (no ramp-up / drain) and overall; median per-read latency. Rounds interleave the cells (order
// rotated); median over rounds. Quiet research mode: waits out ~/.local/share/quiet/PAUSE before a cell, redoes a cell
// that spans one. Sizes: base 1.29 MB, residuals 0.9 / 1.0 / 1.2 MB, MSB-3 2.16 MB (reference).
// build: clang -O2 -o ssd_qd2 ssd_qd2.c    usage: ssd_qd2 RECORDS3_DIR [reads=600] [rounds=3]   (quiet lock: SSD timing)
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define REC 2932736LL
#define NE 512
#define NL 48
#define MAXQ 32
#define NS 5
#define NQ 6
#define MAXN 2048
#define MAXRND 8

static const int64_t SZ[NS] = {917504, 1015808, 1212416, 1290240, 2162688};
static const char* SN[NS] = {"res0.9", "res1.0", "res1.2", "base", "MSB3"};
static const int QD[NQ] = {1, 2, 4, 8, 16, 32};
static int fds[NL];
static uint8_t* bufs[MAXQ];
static int order[NL * NE];
static int next_rec = 0;
static char pause_f[1024];
static int paused(void) { return access(pause_f, F_OK) == 0; }
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return t.tv_sec * 1e3 + t.tv_nsec * 1e-6; }
static int cmpd(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return (x > y) - (x < y); }

static struct { int N, base; int64_t size; atomic_int ctr; double done[MAXN], lat[MAXN]; } C;

static void read_one(int rec, uint8_t* buf, int64_t size) {
  int L = rec / NE, e = rec % NE;
  int64_t got = 0;
  while (got < size) {
    ssize_t r = pread(fds[L], buf + got, size - got, (int64_t)e * REC + got);
    if (r <= 0) { perror("pread"); exit(1); }
    got += r;
  }
}

static void* worker(void* arg) {
  uint8_t* buf = bufs[(intptr_t)arg];
  for (;;) {
    int k = atomic_fetch_add(&C.ctr, 1);
    if (k >= C.N) return NULL;
    double t0 = now_ms();
    read_one(order[C.base + k], buf, C.size);
    double t1 = now_ms();
    C.done[k] = t1; C.lat[k] = t1 - t0;
  }
}

static void shuffle(unsigned seed) {
  for (int i = 0; i < NL * NE; i++) order[i] = i;
  srand(seed);
  for (int i = NL * NE - 1; i > 0; i--) { int j = rand() % (i + 1); int t = order[i]; order[i] = order[j]; order[j] = t; }
  next_rec = 0;
}

// one cell: returns mid-80 % GB/s; *all = overall GB/s, *lat = median per-read latency (ms)
static double cell(int s, int q, int N, double* all, double* lat) {
  pthread_t th[MAXQ];
  pthread_attr_t at; pthread_attr_init(&at); pthread_attr_set_qos_class_np(&at, QOS_CLASS_USER_INTERACTIVE, 0);
  C.N = N; C.base = next_rec; C.size = SZ[s]; atomic_store(&C.ctr, 0); next_rec += N;
  double t0 = now_ms();
  for (int i = 0; i < q; i++) pthread_create(&th[i], &at, worker, (void*)(intptr_t)i);
  for (int i = 0; i < q; i++) pthread_join(th[i], NULL);
  qsort(C.done, N, sizeof(double), cmpd); qsort(C.lat, N, sizeof(double), cmpd);
  int a = N / 10, b = N - N / 10;
  *all = (double)N * SZ[s] / ((C.done[N - 1] - t0) * 1e6);
  *lat = C.lat[N / 2];
  return (double)(b - a) * SZ[s] / ((C.done[b] - C.done[a]) * 1e6);
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: ssd_qd2 RECORDS3_DIR [reads] [rounds]\n"); return 1; }
  int N = argc > 2 ? atoi(argv[2]) : 600, R = argc > 3 ? atoi(argv[3]) : 3, redo = 0;
  if (N > MAXN - 1 || R > MAXRND || NS * NQ * N > NL * NE) { fprintf(stderr, "reads / rounds too large\n"); return 1; }
  snprintf(pause_f, sizeof pause_f, "%s/.local/share/quiet/PAUSE", getenv("HOME"));
  for (int L = 0; L < NL; L++) {
    char p[1024]; snprintf(p, sizeof p, "%s/L%02d.bin", argv[1], L);
    fds[L] = open(p, O_RDONLY); if (fds[L] < 0) { perror(p); return 1; }
    fcntl(fds[L], F_NOCACHE, 1);
  }
  for (int i = 0; i < MAXQ; i++) if (posix_memalign((void**)&bufs[i], 16384, REC)) return 1;
  static double g[NS][NQ][MAXRND], ga[NS][NQ][MAXRND], gl[NS][NQ][MAXRND];
  for (int r = 0; r < R; r++) {
    shuffle(4242 + r);
    for (int c = 0; c < NS * NQ; c++) {
      int cc = (c + r * 7) % (NS * NQ), s = cc / NQ, qi = cc % NQ;
      for (;;) {                                              // wait out a PAUSE; redo a cell that spans one
        while (paused()) usleep(200000);
        usleep(2000);
        if (next_rec + N > NL * NE) shuffle(9000 + r * 100 + redo);
        g[s][qi][r] = cell(s, QD[qi], N, &ga[s][qi][r], &gl[s][qi][r]);
        if (!paused()) break;
        redo++;
      }
    }
  }
  printf("continuous queue: %d cold reads per cell, %d rounds, median (min-max) over rounds; GB/s over the middle 80 %% of "
         "completions | overall | median read latency\n", N, R);
  double med[NS][NQ];
  for (int s = 0; s < NS; s++)
    for (int qi = 0; qi < NQ; qi++) {
      double v[MAXRND], va[MAXRND], vl[MAXRND];
      memcpy(v, g[s][qi], R * sizeof(double)); memcpy(va, ga[s][qi], R * sizeof(double)); memcpy(vl, gl[s][qi], R * sizeof(double));
      qsort(v, R, sizeof(double), cmpd); qsort(va, R, sizeof(double), cmpd); qsort(vl, R, sizeof(double), cmpd);
      med[s][qi] = v[R / 2];
      printf("%-6s %8lld QD %2d: %6.2f GB/s (%5.2f-%5.2f) | %6.2f | %6.3f ms\n", SN[s], (long long)SZ[s], QD[qi], v[R / 2], v[0],
             v[R - 1], va[R / 2], vl[R / 2]);
    }
  printf("ratio to MSB-3 at the same QD (mid-80 GB/s):\n");
  for (int s = 0; s < NS - 1; s++) {
    printf("%-6s", SN[s]);
    for (int qi = 0; qi < NQ; qi++) printf("  QD%-2d %.3f", QD[qi], med[s][qi] / med[NS - 1][qi]);
    printf("\n");
  }
  printf("cells redone after a PAUSE: %d\n", redo);
  return 0;
}
