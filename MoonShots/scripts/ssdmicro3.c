/* M-30 QUEUEMIX (MoonShots, Tue 2026-11-03 22:4x; WALL 1 sprint lever 2 "never let the queue empty"; pre-registered in
   ~/NewIdeas/QUEUE.md M-30 before any timing). Lever 2 (SSD_WALL2 design 2) issues layer L+1's forecast reads with
   layer L's demand burst so the drive never idles: the next demand reads should then skip the first byte (~119 us) and
   the idle wake (~48 us). The same forecast reads share the drive's bandwidth with the demand reads they ride beside.
   This measures both sides on the loader's call shape, plus whether macOS's per-thread I/O policy (setiopolicy_np)
   lets demand reads go first.
   Demand = 2 base records (2,162,688 B parts) by one GCD dispatch_apply from the coordinator (BigMoE's load_jobs; the
   default I/O policy). Forecast = F records (F = 1, 2, 4) from F pre-spun threads, each one pread of a part, issued
   Delta us before the demand call (0 = with it), at the default policy, IOPOL_UTILITY or IOPOL_THROTTLE; or chained,
   issued the moment the demand call returns. Every point starts after a busy-waited 0.8 ms idle gap; F_NOCACHE; fresh
   records (bytes no other read of the run touches, none resident in the page cache); points interleaved per round.
   Per call: the demand call's time, its critical read, and every forecast read's start / end against the demand call.
   build: clang -O2 -o scratch/ssdmicro3 scripts/ssdmicro3.c   run: scripts/run_ssdmicro3.sh (quiet lock, ~10 s) */
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include "records3_guard.h"                   /* refuses a converted (non row-major) records3 */

#define PART 2162688
#define PG 16384
#define NF 4
#define CAP_S 720

static int64_t REC;
static int fds[48];
static uint8_t *map[48];
static int skipped, recycles;
static uint64_t rs = 20261105;
static dispatch_queue_t GQ;

static inline uint64_t now(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
static void busy_until(uint64_t t) { while (now() < t) {} }
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

static int32_t *pv;
static uint8_t *pused;
static int pn, pnext;
static int resident(int f, int64_t off, int64_t len) {
  static char vec[1024];
  int64_t ps = getpagesize(), a = off / ps * ps, n = (off + len - a + ps - 1) / ps;
  if (n > 1024 || mincore(map[f] + a, (size_t)(n * ps), vec)) return 0;
  for (int64_t i = 0; i < n; i++) if (vec[i] & MINCORE_INCORE) return 1;
  return 0;
}
static void pshuffle(void) {
  for (int i = pn - 1; i > 0; i--) { int j = (int)(rnd() % (uint64_t)(i + 1)); int32_t t = pv[i]; pv[i] = pv[j]; pv[j] = t; }
  memset(pused, 0, pn); pnext = 0;
}
static int take(void) {
  for (;;) {
    if (pnext == pn) { pshuffle(); recycles++; }
    int id = pv[pnext++];
    if (pused[id]) continue;
    pused[id] = 1;
    if (resident(id / 512, (int64_t)(id % 512) * REC, PART)) { skipped++; continue; }
    return id;
  }
}

typedef struct { int fd; int64_t off; uint8_t *buf; uint64_t t0, t1; } Job;
static void full_read(Job *j) {
  j->t0 = now();
  for (int64_t d = 0; d < PART;) {
    ssize_t r = pread(j->fd, j->buf + d, (size_t)(PART - d), j->off + d);
    if (r <= 0) { perror("ssdmicro3: pread"); exit(2); }
    d += r;
  }
  j->t1 = now();
}
static Job D[2], F[NF];

/* forecast threads: spin only while fc_on (a point's gap and calls); thread i reads F[i] when fc_gen moves and i < nf,
   under the point's I/O policy */
static atomic_int fc_on, fc_spinning, fc_done, nf, fc_pol;
static atomic_uint fc_gen;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
static void *fc_main(void *a) {
  const int me = (int)(intptr_t)a;
  int pol = -1;
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  for (;;) {
    pthread_mutex_lock(&mu);
    while (!atomic_load(&fc_on)) pthread_cond_wait(&cv, &mu);
    pthread_mutex_unlock(&mu);
    unsigned seen = atomic_load(&fc_gen);
    int want = atomic_load(&fc_pol);
    if (want != pol) {
      if (setiopolicy_np(IOPOL_TYPE_DISK, IOPOL_SCOPE_THREAD, want)) perror("ssdmicro3: setiopolicy_np");
      pol = want;
    }
    atomic_fetch_add(&fc_spinning, 1);
    while (atomic_load_explicit(&fc_on, memory_order_relaxed)) {
      if (atomic_load_explicit(&fc_gen, memory_order_acquire) != seen) {
        seen = atomic_load(&fc_gen);
        if (me < atomic_load(&nf)) { full_read(&F[me]); atomic_fetch_add(&fc_done, 1); }
      } else
        __asm__ volatile("yield");
    }
    atomic_fetch_sub(&fc_spinning, 1);
  }
  return NULL;
}
static void fc_fire(void) { atomic_fetch_add_explicit(&fc_gen, 1, memory_order_release); }

/* a point: F forecast reads Delta us before the demand call (0 = with it; -1 = chained after it), I/O policy pol */
typedef struct { const char *set; int f, delta, pol; } Pt;
static Pt P[32];
static int NP;
static const char *pname(int pol) {
  return pol == IOPOL_THROTTLE ? "throttle" : pol == IOPOL_UTILITY ? "utility" : pol == IOPOL_IMPORTANT ? "important" : "default";
}

static void run_point(int round, int pi, int gap_us) {
  const Pt *p = &P[pi];
  for (int i = 0; i < 2; i++) { int id = take(); D[i].fd = fds[id / 512]; D[i].off = (int64_t)(id % 512) * REC; D[i].t0 = D[i].t1 = 0; }
  for (int i = 0; i < p->f; i++) { int id = take(); F[i].fd = fds[id / 512]; F[i].off = (int64_t)(id % 512) * REC; F[i].t0 = F[i].t1 = 0; }
  if (p->f) {
    atomic_store(&nf, p->f); atomic_store(&fc_done, 0); atomic_store(&fc_pol, p->pol);
    pthread_mutex_lock(&mu); atomic_store(&fc_on, 1); pthread_cond_broadcast(&cv); pthread_mutex_unlock(&mu);
    while (atomic_load(&fc_spinning) < NF) {}
  }
  uint64_t tc = now() + (uint64_t)gap_us * 1000 + (p->delta > 0 ? (uint64_t)p->delta * 1000 : 0);
  if (p->f && p->delta > 0) { busy_until(tc - (uint64_t)p->delta * 1000); fc_fire(); }
  busy_until(tc);
  tc = now();
  if (p->f && p->delta == 0) fc_fire();
  dispatch_apply(2, GQ, ^(size_t j) { full_read(&D[j]); });
  uint64_t tr = now();
  if (p->f && p->delta < 0) fc_fire();
  if (p->f) {
    while (atomic_load(&fc_done) < p->f) {}
    atomic_store(&fc_on, 0);
    while (atomic_load(&fc_spinning) > 0) {}
  }
  int c = D[1].t1 > D[0].t1;
  double fs = 0, fe = 0, fr = 0;
  for (int i = 0; i < p->f; i++) {                                          /* first forecast start, last end */
    double s = ((double)F[i].t0 - (double)tc) / 1e3, e = ((double)F[i].t1 - (double)tc) / 1e3;
    if (!i || s < fs) fs = s;
    if (!i || e > fe) fe = e;
    fr += (F[i].t1 - F[i].t0) / 1e3;
  }
  printf("%d\t%d\t%s\t%d\t%d\t%s\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\n", round, pi, p->set, p->f, p->delta, pname(p->pol),
         (tr - tc) / 1e3, (D[c].t1 - D[c].t0) / 1e3, (D[c].t0 - (double)tc) / 1e3, fs, fe, p->f ? fr / p->f : 0.0);
}

int main(int argc, char **argv) {
  int R = argc > 1 ? atoi(argv[1]) : 150;
  records3_guard();
  setvbuf(stdout, NULL, _IOFBF, 16 << 20);
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  GQ = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
  char path[512];
  int64_t fsz = 0;
  for (int f = 0; f < 48; f++) {
    snprintf(path, sizeof path, "%s/BigMoE/models/bare/records3/L%02d.bin", getenv("HOME"), f);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return 1; }
    fcntl(fd, F_NOCACHE, 1);
    fds[f] = fd;
    if (f == 0) { struct stat st; fstat(fd, &st); fsz = st.st_size; REC = fsz / 512 / 4096 * 4096; }
    map[f] = mmap(NULL, (size_t)fsz, PROT_READ, MAP_SHARED, fd, 0);   /* residency checks only; never touched */
    if (map[f] == MAP_FAILED) { perror("ssdmicro3: mmap"); return 1; }
  }
  if (REC < PART) { fprintf(stderr, "ssdmicro3: record stride %lld too small\n", (long long)REC); return 1; }
  pn = 48 * 512;
  pv = malloc(sizeof(int32_t) * pn); pused = calloc(pn, 1);
  for (int i = 0; i < pn; i++) pv[i] = i;
  pshuffle();
  for (int i = 0; i < 2; i++) { if (posix_memalign((void **)&D[i].buf, PG, PART)) return 1; memset(D[i].buf, 1, PART); }
  for (int i = 0; i < NF; i++) { if (posix_memalign((void **)&F[i].buf, PG, PART)) return 1; memset(F[i].buf, 1, PART); }
  pthread_t th;
  for (int i = 0; i < NF; i++) pthread_create(&th, NULL, fc_main, (void *)(intptr_t)i);

  P[NP++] = (Pt){"A", 0, 0, IOPOL_DEFAULT};                                /* alone (the gap varies below) */
  for (int f = 1; f <= 4; f *= 2) P[NP++] = (Pt){"M", f, 0, IOPOL_DEFAULT};  /* mixed: with the demand call */
  P[NP++] = (Pt){"M", 2, 0, IOPOL_UTILITY};
  for (int f = 2; f <= 4; f *= 2) P[NP++] = (Pt){"M", f, 0, IOPOL_THROTTLE};
  static const int DL[] = {100, 200, 400, 800};                             /* busy: forecast Delta us earlier */
  for (int f = 2; f <= 4; f *= 2)
    for (int i = 0; i < 4; i++) P[NP++] = (Pt){"B", f, DL[i], IOPOL_DEFAULT};
  for (int i = 1; i <= 2; i++) P[NP++] = (Pt){"B", 2, DL[i], IOPOL_THROTTLE};
  for (int f = 2; f <= 4; f *= 2) P[NP++] = (Pt){"C", f, -1, IOPOL_DEFAULT}; /* chained: right after the demand call */

  for (int i = 0; i < 3; i++) { int id = take(); D[0].fd = fds[id / 512]; D[0].off = (int64_t)(id % 512) * REC; full_read(&D[0]); }
  printf("# M-30 QUEUEMIX raw: record stride %lld, %d points (+ alone back to back), %d rounds asked\n", (long long)REC, NP, R);
  printf("round\tpoint\tset\tf\tdelta\tpolicy\tT_us\tcrit_read_us\tcrit_start_us\tfc_start_us\tfc_end_us\tfc_read_us\n");
  uint64_t T0 = now();
  int r = 0;
  for (; r < R && now() - T0 < (uint64_t)CAP_S * 1000000000ull; r++) {
    int idx[33], m = 0;
    for (int i = 0; i < NP; i++) idx[m++] = i;
    idx[m++] = -1;                                                          /* alone, back to back */
    for (int i = m - 1; i > 0; i--) { int j = (int)(rnd() % (uint64_t)(i + 1)), t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
    for (int i = 0; i < m; i++) {
      if (idx[i] < 0) { P[NP] = (Pt){"A0", 0, 0, IOPOL_DEFAULT}; run_point(r, NP, 0); }
      else run_point(r, idx[i], 800);
    }
  }
  printf("# done: %d rounds in %.1f s; record pool recycles %d; %d resident records skipped\n", r, (now() - T0) / 1e9,
         recycles, skipped);
  fflush(stdout);
  return 0;
}
