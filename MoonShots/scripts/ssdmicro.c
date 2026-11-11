/* M-28 SSDMICRO (MoonShots, Tue 2026-11-03 19:xx; WALL 1 sprint lever 1 "kill the per-call overhead", the orchestrator
   Tue 18:3x, charter ~/FableGod/push/WALL1_SPRINT.md; pre-registered in ~/NewIdeas/QUEUE.md M-28 before any timing):
   what makes the ~0.28 ms fixed cost per loader call with misses (Fable A's SSD_WALL2 fit at C 256: SSD busy =
   0.283 ms per call + MB / 12.7 GB/s)? Drive first byte, idle-gap wake, queue depth, piece size, keepalive spacing,
   F_NOCACHE vs the page cache, GCD wake, one burst vs two.
   Reads mimic BigMoE's record loader (ext/bigmoe_ext.cpp load_jobs): one page-aligned F_NOCACHE pread per missed record
   part (base 2,162,688 B, sidecar 770,048 B), a call's reads issued by one dispatch_apply on the
   QOS_CLASS_USER_INTERACTIVE global queue. Every read lands on bytes of BigMoE's records3 store (read-only; the bytes
   themselves are never parsed) that no other read of the run touches, so nothing is warm in the page cache or the drive:
   files 0-31 whole records, 32-35 1 MB chunks, 36-43 256 KB chunks, 44-47 records read through the page cache only;
   bytes already resident in the page cache (mincore on a read-only mapping; e.g. earlier plain reads) are skipped.
   Each call follows a busy-waited idle gap (the engine's ~0.3-1 ms between calls; 0.8 ms unless the set varies it).
   Rounds visit every point once, in blocks by keepalive setting (block order and point order shuffled per round; each
   block starts with 2 ms of its keepalive and one untimed record read).
   S1 size / first byte: one in-thread read of 4 KB .. 4 MB (QD 1) at gaps 0 and 0.8 ms.
   S2 queue depth: k parallel 256 KB reads, k = 1 .. 64 (gap 0.8; k = 1, 8, 64 also back to back).
   S3 pieces: 1 base record as k = 1 .. 32 parallel pieces (4 KB units, even split); 2 base records at k = 1, 4, 8.
   S4 idle gap: R0 (2 base records as 2 whole reads: the median C 256 call) and R0 in 8 pieces, gaps 0 .. 10 ms.
   S5 keepalive spacing: a thread doing 4 KB F_NOCACHE preads + usleep(n), BIGMOE_SSD_KEEPALIVE_US's loop, n = 25 .. 500.
   S6 page cache: R0 / 8 pieces / 1 record through a plain fd (F_NOCACHE off, cold records).
   S7 dispatch: GCD vs 8 pre-spun threads (no wake-up) vs in-thread serial.
   S8 JOINT: base call then sidecar call (two bursts) vs one call with both, whole and as 8 + 3 pieces.
   Output: one TSV line per call on stdout (held in a 16 MB buffer, written at exit); ssdmicro_sum.py prices it.
   build: clang -O2 -o scratch/ssdmicro scripts/ssdmicro.c   run: scripts/run_ssdmicro.sh (quiet lock, ~1 min) */
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include "records3_guard.h"                   /* refuses a converted (non row-major) records3 */

#define BASE 2162688                      /* records3 MSB part: a base row's read */
#define SIDE 770048                       /* records3 LSB part: a sidecar row's read */
#define PG 16384
#define ARENA (32 << 20)
#define MAXJ 80
#define NSPIN 8
#define CAP_S 720                         /* no new round after 12 min (the set's cap is 15) */

enum { GCD, SPIN, SERIAL };
static const char *DN[] = {"gcd", "spin", "serial"};
static int64_t REC;                       /* record stride: file size / 512 (records3 today 2,932,736) */
static int fnc[48], fpc[48];              /* F_NOCACHE fds (files 0-43); plain fds (files 44-47) */
static uint8_t *arena, *map[48];
static int skipped;
static int64_t arena_off;
static uint64_t rs = 20261103;
static dispatch_queue_t GQ;

static inline uint64_t now(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
static void busy(uint64_t ns) { uint64_t t = now() + ns; while (now() < t) {} }
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

/* fresh-byte pools: entry id = (file - f0) * per + index; offset = index * unit; each entry is read once */
typedef struct { int32_t *v; uint8_t *used; int n, i, f0, per, recycles; int64_t unit; } Pool;
static Pool pa, pb1, pb2, pc;

static void reshuffle(Pool *p) {
  for (int i = p->n - 1; i > 0; i--) { int j = (int)(rnd() % (uint64_t)(i + 1)); int32_t t = p->v[i]; p->v[i] = p->v[j]; p->v[j] = t; }
  memset(p->used, 0, p->n); p->i = 0;
}
static void pool_init(Pool *p, int f0, int nf, int64_t unit, int64_t fsz) {
  p->f0 = f0; p->unit = unit; p->per = (int)(fsz / unit); p->n = nf * p->per; p->recycles = 0;
  p->v = malloc(sizeof(int32_t) * p->n); p->used = calloc(p->n, 1);
  for (int i = 0; i < p->n; i++) p->v[i] = i;
  reshuffle(p);
}
static int resident(int f, int64_t off, int64_t len) {   /* any page of [off, off + len) in the page cache */
  static char vec[1024];
  int64_t ps = getpagesize(), a = off / ps * ps, n = (off + len - a + ps - 1) / ps;
  if (n > 1024 || mincore(map[f] + a, (size_t)(n * ps), vec)) return 0;
  for (int64_t i = 0; i < n; i++) if (vec[i] & MINCORE_INCORE) return 1;
  return 0;
}
static int take(Pool *p, int pair) {     /* pair: the entry and the next one in its file (a 4 MB read) */
  for (;;) {
    if (p->i == p->n) { reshuffle(p); p->recycles++; }
    int id = p->v[p->i++];
    if (p->used[id] || (pair && (id % p->per == p->per - 1 || p->used[id + 1]))) continue;
    if (resident(p->f0 + id / p->per, (int64_t)(id % p->per) * p->unit, p->unit * (pair ? 2 : 1))) { p->used[id] = 1; skipped++; continue; }
    p->used[id] = 1;
    if (pair) p->used[id + 1] = 1;
    return id;
  }
}
static int pfd(Pool *p, int id) { int f = p->f0 + id / p->per; return p == &pc ? fpc[f] : fnc[f]; }
static int64_t poff(Pool *p, int id) { return (int64_t)(id % p->per) * p->unit; }

typedef struct { int fd; int64_t off, len; uint8_t *dst; uint64_t t0, t1; } Job;
static Job J[MAXJ];
static int NJ, JHI;

static void add_job(int fd, int64_t off, int64_t len) {
  if (NJ == MAXJ || arena_off + len > ARENA) { fprintf(stderr, "ssdmicro: call too large\n"); exit(2); }
  J[NJ++] = (Job){fd, off, len, arena + arena_off, 0, 0};
  arena_off += (len + PG - 1) / PG * PG;  /* page-aligned destinations (unaligned F_NOCACHE ones are slow) */
}
static void add_part(int fd, int64_t off, int64_t len, int k) {   /* one record part as k parallel pieces */
  int64_t u = len / 4096, q = u / k, r = u % k, o = 0;
  for (int i = 0; i < k; i++) { int64_t n = (q + (i < r)) * 4096; add_job(fd, off + o, n); o += n; }
}
static void run_job(int j) {
  Job *p = &J[j];
  p->t0 = now();
  for (int64_t d = 0; d < p->len;) {
    ssize_t r = pread(p->fd, p->dst + d, (size_t)(p->len - d), p->off + d);
    if (r <= 0) { perror("ssdmicro: pread"); exit(2); }
    d += r;
  }
  p->t1 = now();
}

/* pre-spun workers: spin only while spin_on (a SPIN point's gap and call), park on the condvar otherwise */
static atomic_int spin_on, spinning, nxt, ndone;
static atomic_uint gen;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv = PTHREAD_COND_INITIALIZER;

static void *spin_main(void *a) {
  (void)a;
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  for (;;) {
    pthread_mutex_lock(&mu);
    while (!atomic_load(&spin_on)) pthread_cond_wait(&cv, &mu);
    pthread_mutex_unlock(&mu);
    unsigned seen = atomic_load(&gen);
    atomic_fetch_add(&spinning, 1);
    while (atomic_load_explicit(&spin_on, memory_order_relaxed)) {
      if (atomic_load_explicit(&gen, memory_order_acquire) != seen) {
        seen = atomic_load(&gen);
        for (int j; (j = atomic_fetch_add(&nxt, 1)) < JHI;) { run_job(j); atomic_fetch_add(&ndone, 1); }
      } else
        __asm__ volatile("yield");
    }
    atomic_fetch_sub(&spinning, 1);
  }
  return NULL;
}
static void spin_start(void) {
  pthread_mutex_lock(&mu); atomic_store(&spin_on, 1); pthread_cond_broadcast(&cv); pthread_mutex_unlock(&mu);
  while (atomic_load(&spinning) < NSPIN) {}
}
static void spin_stop(void) { atomic_store(&spin_on, 0); while (atomic_load(&spinning) > 0) {} }

/* keepalive: BigMoE's keepalive_main loop (one 4 KB F_NOCACHE pread of a random block, then usleep(n)); off: no I/O */
static atomic_int ka_us;
static atomic_long ka_reads;
static void *ka_main(void *a) {
  (void)a;
  uint8_t *b;
  if (posix_memalign((void **)&b, PG, 4096)) return NULL;
  memset(b, 0, 4096);
  uint64_t x = 0x9E3779B97F4A7C15ull;
  for (;;) {
    int us = atomic_load(&ka_us);
    if (us <= 0) { usleep(200); continue; }
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    if (pread(fnc[32 + (int)(x % 12)], b, 4096, (int64_t)((x >> 16) % (uint64_t)(512 * REC / 4096)) * 4096) == 4096)
      atomic_fetch_add(&ka_reads, 1);
    usleep((useconds_t)us);
  }
  return NULL;
}

static uint64_t do_call(int disp, int lo, int hi) {   /* returns the call's start; jobs J[lo, hi) */
  uint64_t t;
  if (disp == GCD) { t = now(); dispatch_apply((size_t)(hi - lo), GQ, ^(size_t i) { run_job(lo + (int)i); }); }
  else if (disp == SERIAL) { t = now(); for (int j = lo; j < hi; j++) run_job(j); }
  else {
    JHI = hi; atomic_store(&nxt, lo); atomic_store(&ndone, 0);
    t = now();
    atomic_fetch_add_explicit(&gen, 1, memory_order_release);
    while (atomic_load(&ndone) < hi - lo) {}
  }
  return t;
}

/* a point: dispatch, gap (us), keepalive (us, 0 off), page cache, nb base records in kb pieces, ns sidecar records in
   ks pieces, nm single reads of msz bytes, seq (base reads and sidecar reads as two calls, back to back) */
typedef struct { const char *set; int disp, gap, ka, pc, nb, kb, ns, ks, nm, msz, seq; } Pt;
static Pt P[96];
static int NP;
static void pt(const char *set, int disp, int gap, int ka, int pcache, int nb, int kb, int ns, int ks, int nm, int msz, int seq) {
  P[NP++] = (Pt){set, disp, gap, ka, pcache, nb, kb, ns, ks, nm, msz, seq};
}

static int build(const Pt *p) {           /* fills J; returns where the second call starts (seq), else NJ */
  NJ = 0; arena_off = 0;
  Pool *q = p->pc ? &pc : &pa;
  for (int i = 0; i < p->nb; i++) { int id = take(q, 0); add_part(pfd(q, id), poff(q, id), BASE, p->kb); }
  int split = NJ;
  for (int i = 0; i < p->ns; i++) { int id = take(q, 0); add_part(pfd(q, id), poff(q, id) + REC - SIDE, SIDE, p->ks); }
  for (int i = 0; i < p->nm; i++) {
    Pool *s = p->msz > (1 << 20) ? &pa : p->msz > (256 << 10) ? &pb1 : &pb2;
    int id = take(s, s == &pa);
    add_job(pfd(s, id), poff(s, id), p->msz);
  }
  return p->seq ? split : NJ;
}

static void run_point(int round, int pi) {
  const Pt *p = &P[pi];
  int split = build(p);
  if (p->disp == SPIN) spin_start();
  busy((uint64_t)p->gap * 1000);
  uint64_t ta = do_call(p->disp, 0, split), tb = now(), tc = ta, td = tb;
  if (split < NJ) { tc = do_call(p->disp, split, NJ); td = now(); }
  if (p->disp == SPIN) spin_stop();
  int lo = split < NJ ? split : 0, crit = lo, fast = 0;      /* the critical read of the (last) call */
  double mins = 1e18, maxs = 0, rd = 0;
  int64_t bytes = 0;
  for (int j = 0; j < NJ; j++) {
    bytes += J[j].len;
    rd += (J[j].t1 - J[j].t0) / 1e3;
    if (J[j].len >= (1 << 20) && (double)J[j].len / (J[j].t1 - J[j].t0) > 16.0) fast++;   /* > 16 GB/s: not the drive */
    if (j < lo) continue;
    if (J[j].t1 > J[crit].t1) crit = j;
    double s = (J[j].t0 - tc) / 1e3;
    if (s < mins) mins = s;
    if (s > maxs) maxs = s;
  }
  printf("%d\t%d\t%s\t%s\tg%d\tka%d\t%s\tb%dx%d\ts%dx%d\tm%dx%d\t%s\t%d\t%lld\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%d\n",
         round, pi, p->set, DN[p->disp], p->gap, p->ka, p->pc ? "pc" : "nc", p->nb, p->kb, p->ns, p->ks, p->nm, p->msz,
         p->seq ? "seq" : "one", NJ, (long long)bytes, (td - ta) / 1e3, (tb - ta) / 1e3, (J[crit].t0 - tc) / 1e3,
         (J[crit].t1 - J[crit].t0) / 1e3, (td - J[crit].t1) / 1e3, mins, maxs, rd / NJ, fast);
}

int main(int argc, char **argv) {
  int R = argc > 1 ? atoi(argv[1]) : 100;
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
    if (f < 44) { fcntl(fd, F_NOCACHE, 1); fnc[f] = fd; } else fpc[f] = fd;
    if (f == 0) { struct stat st; fstat(fd, &st); fsz = st.st_size; REC = fsz / 512 / 4096 * 4096; }
    map[f] = mmap(NULL, (size_t)fsz, PROT_READ, MAP_SHARED, fd, 0);   /* residency checks only; never touched */
    if (map[f] == MAP_FAILED) { perror("ssdmicro: mmap"); return 1; }
  }
  if (REC < BASE || 2 * REC < (4 << 20)) { fprintf(stderr, "ssdmicro: record stride %lld too small\n", (long long)REC); return 1; }
  pool_init(&pa, 0, 32, REC, 512 * REC);
  pool_init(&pb1, 32, 4, 1 << 20, fsz);
  pool_init(&pb2, 36, 8, 256 << 10, fsz);
  pool_init(&pc, 44, 4, REC, 512 * REC);
  if (posix_memalign((void **)&arena, PG, ARENA)) return 1;
  memset(arena, 1, ARENA);                /* fault every destination page in before timing */
  pthread_t th;
  for (int i = 0; i < NSPIN; i++) pthread_create(&th, NULL, spin_main, NULL);
  pthread_create(&th, NULL, ka_main, NULL);

  static const int SZ[] = {4096, 65536, 262144, 770048, 1 << 20, 4 << 20};
  for (int g = 0; g <= 800; g += 800) {                                   /* S1 */
    for (int i = 0; i < 6; i++) pt("S1", SERIAL, g, 0, 0, 0, 1, 0, 1, 1, SZ[i], 0);
    pt("S1", SERIAL, g, 0, 0, 1, 1, 0, 1, 0, 0, 0);
  }
  for (int k = 1; k <= 64; k *= 2) pt("S2", GCD, 800, 0, 0, 0, 1, 0, 1, k, 256 << 10, 0);
  for (int k = 1; k <= 64; k *= 8) pt("S2", GCD, 0, 0, 0, 0, 1, 0, 1, k, 256 << 10, 0);
  for (int k = 1; k <= 32; k *= 2) pt("S3", GCD, 800, 0, 0, 1, k, 0, 1, 0, 0, 0);
  static const int K2[] = {1, 4, 8};
  for (int i = 0; i < 3; i++) pt("S3", GCD, 800, 0, 0, 2, K2[i], 0, 1, 0, 0, 0);               /* 2 x 1 = R0 */
  static const int GAP[] = {0, 100, 250, 500, 2000, 5000, 10000};
  for (int i = 0; i < 7; i++) { pt("S4", GCD, GAP[i], 0, 0, 2, 1, 0, 1, 0, 0, 0); pt("S4", GCD, GAP[i], 0, 0, 2, 8, 0, 1, 0, 0, 0); }
  static const int KAS[] = {25, 50, 100, 200, 500};
  for (int i = 0; i < 5; i++) { pt("S5", GCD, 0, KAS[i], 0, 2, 1, 0, 1, 0, 0, 0); pt("S5", GCD, 800, KAS[i], 0, 2, 1, 0, 1, 0, 0, 0); }
  pt("S5", GCD, 800, 50, 0, 2, 8, 0, 1, 0, 0, 0);
  pt("S5", GCD, 800, 50, 0, 1, 1, 0, 1, 0, 0, 0);
  pt("S5", GCD, 800, 50, 0, 1, 8, 0, 1, 0, 0, 0);
  pt("S6", GCD, 800, 0, 1, 2, 1, 0, 1, 0, 0, 0);
  pt("S6", GCD, 800, 0, 1, 2, 8, 0, 1, 0, 0, 0);
  pt("S6", GCD, 800, 0, 1, 1, 1, 0, 1, 0, 0, 0);
  pt("S7", SPIN, 800, 0, 0, 1, 1, 0, 1, 0, 0, 0);
  pt("S7", SPIN, 800, 0, 0, 2, 1, 0, 1, 0, 0, 0);
  pt("S7", SPIN, 800, 0, 0, 1, 8, 0, 1, 0, 0, 0);
  pt("S7", SPIN, 800, 0, 0, 2, 4, 0, 1, 0, 0, 0);
  pt("S7", SERIAL, 800, 0, 0, 2, 1, 0, 1, 0, 0, 0);
  pt("S7", SPIN, 800, 50, 0, 2, 1, 0, 1, 0, 0, 0);
  pt("S7", SPIN, 800, 50, 0, 2, 4, 0, 1, 0, 0, 0);
  for (int ka = 0; ka <= 50; ka += 50) {                                  /* S8 */
    pt("S8", GCD, 800, ka, 0, 1, 1, 1, 1, 0, 0, 1);
    pt("S8", GCD, 800, ka, 0, 1, 1, 1, 1, 0, 0, 0);
    pt("S8", GCD, 800, ka, 0, 1, 8, 1, 3, 0, 0, 0);
    pt("S8", GCD, 800, ka, 0, 1, 8, 1, 3, 0, 0, 1);
  }
  pt("S8", GCD, 800, 0, 0, 2, 1, 2, 1, 0, 0, 1);
  pt("S8", GCD, 800, 0, 0, 2, 1, 2, 1, 0, 0, 0);
  pt("S8", GCD, 800, 0, 0, 2, 8, 2, 3, 0, 0, 0);

  Pt warm = {"warm", GCD, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0};
  for (int i = 0; i < 3; i++) { build(&warm); do_call(GCD, 0, NJ); }
  printf("# M-28 SSDMICRO raw: record stride %lld, %d points, %d rounds asked\n", (long long)REC, NP, R);
  printf("round\tpoint\tset\tdisp\tgap\tka\tcache\tbase\tside\tsmall\tcalls\tnjobs\tbytes\tT_us\tT1_us\tcrit_start_us\t"
         "crit_read_us\ttail_us\tmin_start_us\tmax_start_us\tmean_read_us\tfast\n");
  static const int KA[] = {0, 25, 50, 100, 200, 500};
  uint64_t T0 = now();
  int r = 0;
  for (; r < R && now() - T0 < (uint64_t)CAP_S * 1000000000ull; r++) {
    int ord[6] = {0, 1, 2, 3, 4, 5};
    for (int i = 5; i > 0; i--) { int j = (int)(rnd() % (uint64_t)(i + 1)), t = ord[i]; ord[i] = ord[j]; ord[j] = t; }
    for (int b = 0; b < 6; b++) {
      int ka = KA[ord[b]], idx[96], m = 0;
      for (int i = 0; i < NP; i++) if (P[i].ka == ka) idx[m++] = i;
      for (int i = m - 1; i > 0; i--) { int j = (int)(rnd() % (uint64_t)(i + 1)), t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
      atomic_store(&ka_us, ka);
      busy(2000000);
      build(&warm); do_call(GCD, 0, NJ);
      long k0 = atomic_load(&ka_reads);
      uint64_t tb = now();
      for (int i = 0; i < m; i++) run_point(r, idx[i]);
      if (ka) printf("#ka\t%d\t%ld\t%.6f\n", ka, atomic_load(&ka_reads) - k0, (now() - tb) / 1e9);
    }
    atomic_store(&ka_us, 0);
  }
  printf("# done: %d rounds in %.1f s; pool recycles: records %d, 1 MB %d, 256 KB %d, page-cache records %d; %d resident "
         "entries skipped\n", r, (now() - T0) / 1e9, pa.recycles, pb1.recycles, pb2.recycles, pc.recycles, skipped);
  fflush(stdout);
  return 0;
}
