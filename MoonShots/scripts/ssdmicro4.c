/* M-31 HEADPING (MoonShots, Tue 2026-11-03 22:4x; WALL 1, the floor under lever 1; pre-registered in
   ~/NewIdeas/QUEUE.md M-31 before any timing). After JOINT, the keepalive / wake-ahead ping and zero-copy, a loader
   burst still pays the drive's first byte (~119 us, M-28's QD 1 intercept), and a busy drive does not hide it (M-30).
   Is part of it a cold lookup inside the drive (the FTL's logical-to-physical map, or a NAND page not yet sensed) that a
   tiny read can warm? If a 4 KB "head ping" of a record shortens that record's later full read, the loader could ping the
   heads of every forecast expert (a 4 KB read each: no bandwidth wasted on wrong forecasts) and the demand reads of the
   right ones would start faster.
   Probe: one full base part (2,162,688 B) read in-thread (QD 1) from a fresh record, after a busy-waited 0.8 ms gap, with
   BigMoE's keepalive (usleep 50, 4 KB) running so the drive's idle wake stays out of it; a few points with it off.
   H0 no ping; H1 4 KB of the record's head pinged Delta = 100 / 300 / 1,000 / 5,000 us before the full read (from the
   ping's end); H2 the part's last 4 KB; H3 the next record's head (~2.9 MB away); H4 a far random record's head; H5 64 KB
   of the head; G0 / G1 the loader's 2-record GCD call without / with both heads pinged (300 us).
   build: clang -O2 -o scratch/ssdmicro4 scripts/ssdmicro4.c   run: scripts/run_ssdmicro4.sh (quiet lock, ~10 s) */
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

#define PART 2162688
#define PG 16384
#define CAP_S 720

static int64_t REC;
static int fds[48];
static uint8_t *map[48];
static int skipped, recycles;
static uint64_t rs = 20261106;
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
static int take(int pair) {                       /* pair: the record and the next one in its file, both fresh */
  for (;;) {
    if (pnext == pn) { pshuffle(); recycles++; }
    int id = pv[pnext++];
    if (pused[id] || (pair && (id % 512 == 511 || pused[id + 1]))) continue;
    pused[id] = 1;
    if (pair) pused[id + 1] = 1;
    if (resident(id / 512, (int64_t)(id % 512) * REC, REC * (pair ? 2 : 1))) { skipped++; continue; }
    return id;
  }
}

static uint8_t *buf[2], *pbuf;
static void rd(int fd, uint8_t *b, int64_t len, int64_t off) {
  for (int64_t d = 0; d < len;) {
    ssize_t r = pread(fd, b + d, (size_t)(len - d), off + d);
    if (r <= 0) { perror("ssdmicro4: pread"); exit(2); }
    d += r;
  }
}

/* keepalive: BigMoE's keepalive_main loop (one 4 KB F_NOCACHE pread of a random block, then usleep(n)); off: no I/O */
static atomic_int ka_us;
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
    if (pread(fds[x % 48], b, 4096, (int64_t)((x >> 16) % (uint64_t)(512 * REC / 4096)) * 4096) != 4096) perror("ka");
    usleep((useconds_t)us);
  }
  return NULL;
}

/* a point: what is pinged (0 none, 1 head, 2 part's tail, 3 next record's head, 4 a far record's head, 5 64 KB head),
   Delta us from the ping's end to the full read, the keepalive (us, 0 off), records (1 in-thread, 2 = the GCD call) */
typedef struct { const char *name; int what, delta, ka, nrec; } Pt;
static Pt P[16];
static int NP;

static void run_point(int round, int pi) {
  const Pt *p = &P[pi];
  int fd[2]; int64_t off[2];
  for (int i = 0; i < p->nrec; i++) { int id = take(p->what == 3); fd[i] = fds[id / 512]; off[i] = (int64_t)(id % 512) * REC; }
  int far = p->what == 4 ? take(0) : 0;
  busy_until(now() + 800000);
  double ping = 0;
  if (p->what) {
    uint64_t t0 = now();
    for (int i = 0; i < p->nrec; i++) {
      int f = p->what == 4 ? fds[far / 512] : fd[i];
      int64_t o = p->what == 2 ? off[i] + PART - 4096 : p->what == 3 ? off[i] + REC : p->what == 4 ? (int64_t)(far % 512) * REC : off[i];
      rd(f, pbuf, p->what == 5 ? 65536 : 4096, o);
    }
    uint64_t t1 = now();
    ping = (t1 - t0) / 1e3;
    busy_until(t1 + (uint64_t)p->delta * 1000);
  }
  uint64_t tc = now();
  if (p->nrec == 1) rd(fd[0], buf[0], PART, off[0]);
  else {
    const int *fdp = fd;
    const int64_t *offp = off;
    dispatch_apply(2, GQ, ^(size_t j) { rd(fdp[j], buf[j], PART, offp[j]); });
  }
  uint64_t tr = now();
  printf("%d\t%d\t%s\t%d\t%d\t%d\t%d\t%.1f\t%.1f\n", round, pi, p->name, p->what, p->delta, p->ka, p->nrec, (tr - tc) / 1e3, ping);
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
    if (map[f] == MAP_FAILED) { perror("ssdmicro4: mmap"); return 1; }
  }
  if (REC < PART) { fprintf(stderr, "ssdmicro4: record stride %lld too small\n", (long long)REC); return 1; }
  pn = 48 * 512;
  pv = malloc(sizeof(int32_t) * pn); pused = calloc(pn, 1);
  for (int i = 0; i < pn; i++) pv[i] = i;
  pshuffle();
  for (int i = 0; i < 2; i++) { if (posix_memalign((void **)&buf[i], PG, PART)) return 1; memset(buf[i], 1, PART); }
  if (posix_memalign((void **)&pbuf, PG, 65536)) return 1;
  memset(pbuf, 1, 65536);
  pthread_t th;
  pthread_create(&th, NULL, ka_main, NULL);

  P[NP++] = (Pt){"H0", 0, 0, 50, 1};
  static const int DL[] = {100, 300, 1000, 5000};
  for (int i = 0; i < 4; i++) P[NP++] = (Pt){"H1", 1, DL[i], 50, 1};
  P[NP++] = (Pt){"H2", 2, 300, 50, 1};
  P[NP++] = (Pt){"H3", 3, 300, 50, 1};
  P[NP++] = (Pt){"H4", 4, 300, 50, 1};
  P[NP++] = (Pt){"H5", 5, 300, 50, 1};
  P[NP++] = (Pt){"G0", 0, 0, 50, 2};
  P[NP++] = (Pt){"G1", 1, 300, 50, 2};
  P[NP++] = (Pt){"H0", 0, 0, 0, 1};
  P[NP++] = (Pt){"H1", 1, 100, 0, 1};
  P[NP++] = (Pt){"H1", 1, 300, 0, 1};

  for (int i = 0; i < 3; i++) { int id = take(0); rd(fds[id / 512], buf[0], PART, (int64_t)(id % 512) * REC); }
  printf("# M-31 HEADPING raw: record stride %lld, %d points, %d rounds asked\n", (long long)REC, NP, R);
  printf("round\tpoint\tname\twhat\tdelta\tka\tnrec\tT_us\tping_us\n");
  uint64_t T0 = now();
  int r = 0;
  for (; r < R && now() - T0 < (uint64_t)CAP_S * 1000000000ull; r++) {
    const int first = (int)(rnd() & 1);
    for (int b = 0; b < 2; b++) {                                           /* blocks: keepalive on / off, random order */
      int ka = (b ^ first) ? 50 : 0, idx[16], m = 0;
      for (int i = 0; i < NP; i++) if (P[i].ka == ka) idx[m++] = i;
      for (int i = m - 1; i > 0; i--) { int j = (int)(rnd() % (uint64_t)(i + 1)), t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
      atomic_store(&ka_us, ka);
      busy_until(now() + 2000000);
      for (int i = 0; i < m; i++) run_point(r, idx[i]);
    }
    atomic_store(&ka_us, 0);
  }
  printf("# done: %d rounds in %.1f s; record pool recycles %d; %d resident records skipped\n", r, (now() - T0) / 1e9,
         recycles, skipped);
  fflush(stdout);
  return 0;
}
