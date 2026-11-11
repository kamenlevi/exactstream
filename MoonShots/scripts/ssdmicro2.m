/* M-29 SSDMICRO2 (MoonShots, Tue 2026-11-03 22:xx; WALL 1 sprint lever 1, the orchestrator's brief 22:3x: "reading
   straight into aligned slot memory (the zero-copy candidate), the copy-tail cost, keepalive spacing variants";
   pre-registered in ~/NewIdeas/QUEUE.md M-29 before any timing). Follows M-28 (data/ssdmicro.txt): of the median loader
   call's ~0.28 ms fixed cost, ~0.19 ms is the I/O path and ~0.10 ms sits outside it. The tier loader
   (~/BigMoE/ext/bigmoe_ext.cpp read_expert, rec_fmt 1 on a record part) reads each missed part into a pooled aligned
   bounce, then memcpys its 9 (base) or 6 (sidecar) pieces into the pool slots, on the job's own thread.
   BIGMOE_RECORDS_DIRECT (REC1: preadv straight into the slots) was killed for unaligned F_NOCACHE destinations.
   Calls go out by one GCD dispatch_apply (user-interactive QoS), F_NOCACHE, on fresh records (bytes no other read of
   the run touches, none resident in the page cache), after a busy-waited 0.8 ms idle gap unless stated.
   Z  zero-copy, 1 and 2 base records per call:
      Z0   today: an aligned read into a bounce, then memcpy of the 9 pieces into engine-stride pool slots;
      Z0n  the same read without the copy; Z0p the copy split over the 3 projections (dispatch_apply(3));
      Z1   one read straight into a contiguous 16 KB-aligned slot (the GPU would index the pieces inside it);
      Z2   preadv of today's 9 pieces into engine-stride slots (REC1);
      Z3   preadv of 4 KB-padded pieces (s / b 51,200 -> 53,248 B; the part stays 2,162,688 B) into 16 KB-aligned slots;
      Z3b  the same into slots that are 4 KB- but never 16 KB-aligned (pool strides padded to 4 KB only);
      Z4   16 KB-padded pieces (part 2,260,992 B) into 16 KB-aligned slots;
      "m"  Z0 / Z1 / Z3 / Z3b with the slots in Metal shared buffers (MLX's pool memory).
   S  the sidecar part (770,048 B, 6 pieces), 1 record: today (bounce + copy), no copy, contiguous, 4 KB-padded preadv
      into 16 KB- and 4 KB-aligned slots.
   K  keepalive spacing: BigMoE's loop (pread, then usleep(n)) at n = 50 .. 200 with 4 KB reads, and n = 200 with 16 /
      64 KB reads; R0 = 2 base records into bounces, back to back and after 0.8 ms.
   P  wake-ahead ping (no trickle): one 4 KB read issued T = 25 .. 400 us before the R0 call by a spinning thread.
   C  keepalive 50 with today's copy, and with the 4 KB-aligned zero-copy (Z3b).
   Output: one TSV line per call on stdout (16 MB buffer, written at exit); ssdmicro2_sum.py prices it.
   M-29b (argv[2] = "big", pre-registered in QUEUE M-29b): only Z0m / Z0pm / Z0n / Z3bm at 4, 8 and 15 base records per
   call, the 24 GB line's burst sizes (does the copy tail grow when many copies run side by side, and does zero-copy
   keep its gain at queue depth 15?).
   build: clang -O2 -fobjc-arc -framework Metal -framework Foundation -o scratch/ssdmicro2 scripts/ssdmicro2.m
   run: scripts/run_ssdmicro2.sh (quiet lock, ~20 s; ~1.1 GB of pre-faulted slot memory) */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
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
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>
#include "records3_guard.h"                   /* refuses a converted (non row-major) records3 */

#define PG 16384
#define NS 48                             /* slots per pool: ~100 MB per base pool set, cold destinations */
#define MAXJ 16                           /* M-29b: up to 15 records per call (the 24 GB line's bursts) */
#define CAP_S 720

static int64_t REC;
static int fds[48];
static uint8_t *map[48];
static int skipped, recycles;
static uint64_t rs = 20261104;
static dispatch_queue_t GQ;
static id<MTLDevice> dev;
static NSMutableArray *keep;

static inline uint64_t now(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
static void busy_until(uint64_t t) { while (now() < t) {} }
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

/* fresh records of all 48 files, each read once; any page resident in the page cache disqualifies a record */
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
    if (resident(id / 512, (int64_t)(id % 512) * REC, REC)) { skipped++; continue; }
    return id;
  }
}

/* piece layouts of a record part: per projection w3 614,400 + s3 + b3 (base) or l1 204,800 + b (sidecar), s / b
   51,200 B today; pad rounds every piece length up (4 KB or 16 KB) */
typedef struct { int n; int64_t len[9], foff[9], part, sum; } Layout;
static Layout LB, LB4, LB16, LS, LS4;
static void mklayout(Layout *L, int side, int64_t pad) {
  int64_t o = 0;
  L->n = side ? 6 : 9;
  for (int p = 0; p < 3; p++)
    for (int q = 0; q < (side ? 2 : 3); q++) {
      int64_t l = q ? 51200 : side ? 204800 : 614400;
      int i = p * (side ? 2 : 3) + q;
      L->len[i] = (l + pad - 1) / pad * pad; L->foff[i] = o; o += L->len[i];
    }
  L->sum = o;
  L->part = (o + PG - 1) / PG * PG;                  /* today: 2,162,688 / 770,048, as records3 */
}

/* slot pools: mode 0 = engine strides (= piece length), 16 = strides rounded to 16 KB, 4 = strides rounded to 4 KB
   with bases at +4 KB (only slots k % 4 == 0 are used: every destination is 4 KB- and never 16 KB-aligned),
   -1 = one contiguous part-sized slot per expert */
typedef struct { uint8_t *base[9]; int64_t stride[9]; int n, mode; } Pools;
static uint8_t *alloc_mem(int64_t len, int metal) {
  uint8_t *p;
  if (metal) {
    id<MTLBuffer> b = [dev newBufferWithLength:(NSUInteger)len options:MTLResourceStorageModeShared];
    if (!b) { fprintf(stderr, "ssdmicro2: Metal buffer failed\n"); exit(1); }
    [keep addObject:b];
    p = (uint8_t *)b.contents;
  } else if (posix_memalign((void **)&p, PG, (size_t)len)) exit(1);
  memset(p, 1, (size_t)len);                         /* fault every page in before timing */
  return p;
}
static void mkpools(Pools *P, const Layout *L, int mode, int metal) {
  P->mode = mode;
  P->n = mode < 0 ? 1 : L->n;
  if (mode < 0) { P->stride[0] = L->part; P->base[0] = alloc_mem(NS * L->part, metal); return; }
  for (int i = 0; i < L->n; i++) {
    int64_t r = mode == 0 ? 1 : mode == 16 ? PG : 4096, off = mode == 4 ? 4096 : 0;
    P->stride[i] = (L->len[i] + r - 1) / r * r;
    P->base[i] = alloc_mem(NS * P->stride[i] + PG, metal) + off;
  }
}
static uint8_t *slot(const Pools *P, int i, int k) { return P->base[i] + (int64_t)k * P->stride[i]; }

enum { BOUNCE, COPY, COPYPAR, DIRECT };
typedef struct { int fd; int64_t off; int kind; const Layout *L; const Pools *P; int k; uint8_t *bounce; uint64_t t0, t1, t2; } Job;
static Job J[MAXJ];
static int NJ;
static uint8_t *bounces[MAXJ];

static void read_record(int fd, int64_t off, struct iovec *iov, int n) {   /* BigMoE's loop: one preadv, resumed */
  int i = 0;
  while (i < n) {
    ssize_t r = preadv(fd, iov + i, n - i, off);
    if (r <= 0) { perror("ssdmicro2: preadv"); exit(2); }
    off += r;
    while (i < n && r >= (ssize_t)iov[i].iov_len) { r -= iov[i].iov_len; i++; }
    if (i < n && r > 0) { iov[i].iov_base = (uint8_t *)iov[i].iov_base + r; iov[i].iov_len -= r; }
  }
}
static void run_job(int j) {
  Job *p = &J[j];
  const Layout *L = p->L;
  struct iovec iov[9];
  p->t0 = now();
  if (p->kind == DIRECT) {
    int n = p->P->n;
    if (n == 1) { iov[0].iov_base = slot(p->P, 0, p->k); iov[0].iov_len = (size_t)L->part; }
    else for (int i = 0; i < n; i++) { iov[i].iov_base = slot(p->P, i, p->k); iov[i].iov_len = (size_t)L->len[i]; }
    read_record(p->fd, p->off, iov, n);
    p->t1 = p->t2 = now();
    return;
  }
  iov[0].iov_base = p->bounce; iov[0].iov_len = (size_t)L->part;
  read_record(p->fd, p->off, iov, 1);
  p->t1 = now();
  if (p->kind == COPY)
    for (int i = 0; i < L->n; i++) memcpy(slot(p->P, i, p->k), p->bounce + L->foff[i], (size_t)L->len[i]);
  else if (p->kind == COPYPAR) {
    const int per = L->n / 3;
    dispatch_apply(3, GQ, ^(size_t q) {
      for (int i = (int)q * per; i < ((int)q + 1) * per; i++) memcpy(slot(p->P, i, p->k), p->bounce + L->foff[i], (size_t)L->len[i]);
    });
  }
  p->t2 = now();
}

/* keepalive: BigMoE's keepalive_main loop (one F_NOCACHE pread of a random 4 KB-aligned block, then usleep(n)) */
static atomic_int ka_us, ka_sz;
static atomic_long ka_reads;
static void *ka_main(void *a) {
  (void)a;
  uint8_t *b;
  if (posix_memalign((void **)&b, PG, 65536)) return NULL;
  memset(b, 0, 65536);
  uint64_t x = 0x9E3779B97F4A7C15ull;
  for (;;) {
    int us = atomic_load(&ka_us), sz = atomic_load(&ka_sz);
    if (us <= 0) { usleep(200); continue; }
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    if (pread(fds[x % 48], b, (size_t)sz, (int64_t)((x >> 16) % (uint64_t)((512 * REC - 65536) / 4096)) * 4096) == sz)
      atomic_fetch_add(&ka_reads, 1);
    usleep((useconds_t)us);
  }
  return NULL;
}

/* wake-ahead ping: spins only while ping_on (a P point's gap and call); one 4 KB read at ping_at */
static atomic_int ping_on, ping_spinning;
static _Atomic uint64_t ping_at;
static uint64_t ping_t0, ping_t1;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
static void *ping_main(void *a) {
  (void)a;
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  uint8_t *b;
  if (posix_memalign((void **)&b, PG, 4096)) return NULL;
  memset(b, 0, 4096);
  uint64_t x = 0x2545F4914F6CDD1Dull;
  for (;;) {
    pthread_mutex_lock(&mu);
    while (!atomic_load(&ping_on)) pthread_cond_wait(&cv, &mu);
    pthread_mutex_unlock(&mu);
    atomic_store(&ping_spinning, 1);
    while (atomic_load(&ping_on)) {
      uint64_t t = atomic_load(&ping_at);
      if (t && now() >= t) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        ping_t0 = now();
        if (pread(fds[x % 48], b, 4096, (int64_t)((x >> 16) % (uint64_t)(512 * REC / 4096)) * 4096) != 4096) perror("ping");
        ping_t1 = now();
        atomic_store(&ping_at, 0);
      } else
        __asm__ volatile("yield");
    }
    atomic_store(&ping_spinning, 0);
  }
  return NULL;
}

typedef struct { const char *set, *arm; int nrec, side, kind, gap, ka, kasz, ping; const Layout *L; const Pools *P; } Pt;
static Pt P[64];
static int NP;
static void pt(const char *set, const char *arm, int nrec, int side, int kind, const Layout *L, const Pools *Q, int gap,
               int ka, int kasz, int ping) {
  P[NP++] = (Pt){set, arm, nrec, side, kind, gap, ka, kasz, ping, L, Q};
}

static void run_point(int round, int pi) {
  const Pt *p = &P[pi];
  NJ = 0;
  for (int r = 0; r < p->nrec; r++) {
    int id = take();
    int64_t off = (int64_t)(id % 512) * REC + (p->side ? REC - 786432 : 0);   /* sidecar reads end inside the record */
    int k = p->P ? (int)(rnd() % (p->P->mode == 4 ? NS / 4 : NS)) * (p->P->mode == 4 ? 4 : 1) : 0;
    J[NJ] = (Job){fds[id / 512], off, p->kind, p->L, p->P, k, bounces[NJ], 0, 0, 0};
    NJ++;
  }
  if (p->ping) {
    ping_t0 = ping_t1 = 0;
    pthread_mutex_lock(&mu); atomic_store(&ping_on, 1); pthread_cond_broadcast(&cv); pthread_mutex_unlock(&mu);
    while (!atomic_load(&ping_spinning)) {}
  }
  uint64_t tc = now() + (uint64_t)p->gap * 1000;
  if (p->ping) atomic_store(&ping_at, tc - (uint64_t)p->ping * 1000);
  busy_until(tc);
  tc = now();
  dispatch_apply((size_t)NJ, GQ, ^(size_t j) { run_job((int)j); });
  uint64_t tr = now();
  if (p->ping) {
    while (atomic_load(&ping_at)) {}
    atomic_store(&ping_on, 0);
    while (atomic_load(&ping_spinning)) {}
  }
  int crit = 0, fast = 0;
  int64_t bytes = 0;
  uint64_t t1max = 0;
  for (int j = 0; j < NJ; j++) {
    if (J[j].t2 > J[crit].t2) crit = j;
    if (J[j].t1 > t1max) t1max = J[j].t1;
    int64_t b = J[j].kind == DIRECT && J[j].P->n > 1 ? J[j].L->sum : J[j].L->part;
    bytes += b;
    if ((double)b / (J[j].t1 - J[j].t0) > 16.0) fast++;   /* > 16 GB/s: not the drive */
  }
  printf("%d\t%d\t%s\t%s\t%d\tg%d\tka%dx%d\tp%d\t%lld\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%d\n", round, pi,
         p->set, p->arm, p->nrec, p->gap, p->ka, p->kasz, p->ping, (long long)bytes, (tr - tc) / 1e3,
         (J[crit].t0 - (int64_t)tc) / 1e3, (J[crit].t1 - J[crit].t0) / 1e3, (J[crit].t2 - J[crit].t1) / 1e3,
         (tr - J[crit].t2) / 1e3, (t1max - tc) / 1e3, p->ping ? (ping_t1 - ping_t0) / 1e3 : 0.0,
         p->ping ? ((double)tc - (double)ping_t1) / 1e3 : 0.0, fast);
}

int main(int argc, char **argv) {
  @autoreleasepool {
    int R = argc > 1 ? atoi(argv[1]) : 150;
    records3_guard();
    setvbuf(stdout, NULL, _IOFBF, 16 << 20);
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    GQ = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
    dev = MTLCreateSystemDefaultDevice();
    keep = [NSMutableArray array];
    if (!dev) { fprintf(stderr, "ssdmicro2: no Metal device\n"); return 1; }
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
      if (map[f] == MAP_FAILED) { perror("ssdmicro2: mmap"); return 1; }
    }
    if (REC < 2260992) {                               /* the 16 KB-padded base part must fit in one record */ fprintf(stderr, "ssdmicro2: record stride %lld too small\n", (long long)REC); return 1; }
    pn = 48 * 512;
    pv = malloc(sizeof(int32_t) * pn); pused = calloc(pn, 1);
    for (int i = 0; i < pn; i++) pv[i] = i;
    pshuffle();
    mklayout(&LB, 0, 1); mklayout(&LB4, 0, 4096); mklayout(&LB16, 0, PG); mklayout(&LS, 1, 1); mklayout(&LS4, 1, 4096);
    static Pools PB0, PB0m, PC, PCm, P416, P416m, P44, P44m, P1616, PS0, PSC, PS416, PS44;
    mkpools(&PB0, &LB, 0, 0); mkpools(&PB0m, &LB, 0, 1); mkpools(&PC, &LB, -1, 0); mkpools(&PCm, &LB, -1, 1);
    mkpools(&P416, &LB4, 16, 0); mkpools(&P416m, &LB4, 16, 1); mkpools(&P44, &LB4, 4, 0); mkpools(&P44m, &LB4, 4, 1);
    mkpools(&P1616, &LB16, 16, 0);
    mkpools(&PS0, &LS, 0, 0); mkpools(&PSC, &LS, -1, 0); mkpools(&PS416, &LS4, 16, 0); mkpools(&PS44, &LS4, 4, 0);
    for (int j = 0; j < MAXJ; j++) bounces[j] = alloc_mem(2260992, 0);
    pthread_t th;
    pthread_create(&th, NULL, ka_main, NULL);
    pthread_create(&th, NULL, ping_main, NULL);

    const int wed = argc > 2 && !strcmp(argv[2], "wed");           /* Wed 09:0x ticket: M-29b repeat + pings at the engine's gaps */
    const int big = (argc > 2 && !strcmp(argv[2], "big")) || wed;
    static const int NB[] = {4, 8, 15};
    for (int i = 0; big && i < 3; i++) {                                   /* M-29b */
      pt("Zb", "Z0m", NB[i], 0, COPY, &LB, &PB0m, 800, 0, 0, 0);
      if (!wed) pt("Zb", "Z0pm", NB[i], 0, COPYPAR, &LB, &PB0m, 800, 0, 0, 0);
      pt("Zb", "Z0n", NB[i], 0, BOUNCE, &LB, NULL, 800, 0, 0, 0);
      pt("Zb", "Z3bm", NB[i], 0, DIRECT, &LB4, &P44m, 800, 0, 0, 0);
      if (argc > 3 && !wed) pt("Zb", "Z1m", NB[i], 0, DIRECT, &LB, &PCm, 800, 0, 0, 0);   /* exploratory: one contiguous read per record */
    }
    static const int WG[] = {1200, 2000}, WT[] = {150, 250};
    for (int i = 0; wed && i < 2; i++) {                                   /* pings at the engine's 1.2-2 ms gaps */
      pt("Pw", "R0", 2, 0, BOUNCE, &LB, NULL, WG[i], 0, 0, 0);
      for (int j = 0; j < 2; j++) pt("Pw", "R0", 2, 0, BOUNCE, &LB, NULL, WG[i], 0, 0, WT[j]);
    }
    if (wed) pt("Pw", "R0", 2, 0, BOUNCE, &LB, NULL, 0, 0, 0, 0);
    for (int n = 1; !big && n <= 2; n++) {                                 /* Z */
      pt("Z", "Z0", n, 0, COPY, &LB, &PB0, 800, 0, 0, 0);
      pt("Z", "Z0n", n, 0, BOUNCE, &LB, NULL, 800, 0, 0, 0);
      pt("Z", "Z1", n, 0, DIRECT, &LB, &PC, 800, 0, 0, 0);
      pt("Z", "Z3", n, 0, DIRECT, &LB4, &P416, 800, 0, 0, 0);
      pt("Z", "Z3b", n, 0, DIRECT, &LB4, &P44, 800, 0, 0, 0);
      pt("Z", "Z0m", n, 0, COPY, &LB, &PB0m, 800, 0, 0, 0);
      pt("Z", "Z1m", n, 0, DIRECT, &LB, &PCm, 800, 0, 0, 0);
      pt("Z", "Z3m", n, 0, DIRECT, &LB4, &P416m, 800, 0, 0, 0);
      pt("Z", "Z3bm", n, 0, DIRECT, &LB4, &P44m, 800, 0, 0, 0);
    }
    if (!big) {
    pt("Z", "Z0p", 1, 0, COPYPAR, &LB, &PB0, 800, 0, 0, 0);
    pt("Z", "Z2", 1, 0, DIRECT, &LB, &PB0, 800, 0, 0, 0);
    pt("Z", "Z4", 1, 0, DIRECT, &LB16, &P1616, 800, 0, 0, 0);
    pt("S", "S0", 1, 1, COPY, &LS, &PS0, 800, 0, 0, 0);                     /* S */
    pt("S", "S0n", 1, 1, BOUNCE, &LS, NULL, 800, 0, 0, 0);
    pt("S", "S1", 1, 1, DIRECT, &LS, &PSC, 800, 0, 0, 0);
    pt("S", "S3", 1, 1, DIRECT, &LS4, &PS416, 800, 0, 0, 0);
    pt("S", "S3b", 1, 1, DIRECT, &LS4, &PS44, 800, 0, 0, 0);
    pt("K", "R0", 2, 0, BOUNCE, &LB, NULL, 0, 0, 0, 0);                     /* K (gap 800, ka off = Z0n x 2) */
    static const int KA[] = {50, 100, 125, 150, 175, 200};
    for (int i = 0; i < 6; i++)
      for (int g = 0; g <= 800; g += 800) pt("K", "R0", 2, 0, BOUNCE, &LB, NULL, g, KA[i], 4096, 0);
    for (int sz = 16384; sz <= 65536; sz *= 4)
      for (int g = 0; g <= 800; g += 800) pt("K", "R0", 2, 0, BOUNCE, &LB, NULL, g, 200, sz, 0);
    static const int PING[] = {25, 50, 100, 200, 400};                     /* P */
    for (int i = 0; i < 5; i++) pt("P", "R0", 2, 0, BOUNCE, &LB, NULL, 800, 0, 0, PING[i]);
    pt("C", "Z0", 2, 0, COPY, &LB, &PB0, 800, 50, 4096, 0);                 /* C */
    pt("C", "Z3b", 2, 0, DIRECT, &LB4, &P44, 800, 50, 4096, 0);
    }

    static const int BK[][2] = {{0, 0}, {50, 4096}, {100, 4096}, {125, 4096}, {150, 4096}, {175, 4096}, {200, 4096}, {200, 16384}, {200, 65536}};
    Layout *wl = &LB;
    for (int i = 0; i < 3; i++) {                                          /* warm-up: threads, GCD, page tables */
      NJ = 1; J[0] = (Job){fds[take() / 512], 0, BOUNCE, wl, NULL, 0, bounces[0], 0, 0, 0};
      run_job(0);
    }
    printf("# M-29 SSDMICRO2 raw: record stride %lld, %d points, %d rounds asked, %d slots per pool\n", (long long)REC, NP, R, NS);
    printf("round\tpoint\tset\tarm\tnrec\tgap\tka\tping\tbytes\tT_us\tcrit_start_us\tread_us\tcopy_us\ttail_us\t"
           "reads_done_us\tping_read_us\tping_lead_us\tfast\n");
    uint64_t T0 = now();
    int r = 0;
    for (; r < R && now() - T0 < (uint64_t)CAP_S * 1000000000ull; r++) {
      int ord[9];
      for (int i = 0; i < 9; i++) ord[i] = i;
      for (int i = 8; i > 0; i--) { int j = (int)(rnd() % (uint64_t)(i + 1)), t = ord[i]; ord[i] = ord[j]; ord[j] = t; }
      for (int b = 0; b < 9; b++) {
        int ka = BK[ord[b]][0], sz = BK[ord[b]][1], idx[64], m = 0;
        for (int i = 0; i < NP; i++) if (P[i].ka == ka && P[i].kasz == sz) idx[m++] = i;
        for (int i = m - 1; i > 0; i--) { int j = (int)(rnd() % (uint64_t)(i + 1)), t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
        if (!m) continue;                                                  /* M-29b runs the off block only */
        atomic_store(&ka_sz, sz ? sz : 4096);
        atomic_store(&ka_us, ka);
        busy_until(now() + 2000000);
        int id = take();                                                   /* one untimed record read per block */
        NJ = 1; J[0] = (Job){fds[id / 512], (int64_t)(id % 512) * REC, BOUNCE, wl, NULL, 0, bounces[0], 0, 0, 0};
        run_job(0);
        long k0 = atomic_load(&ka_reads);
        uint64_t tb = now();
        for (int i = 0; i < m; i++) run_point(r, idx[i]);
        if (ka) printf("#ka\t%d\t%d\t%ld\t%.6f\n", ka, sz, atomic_load(&ka_reads) - k0, (now() - tb) / 1e9);
      }
      atomic_store(&ka_us, 0);
    }
    printf("# done: %d rounds in %.1f s; record pool recycles %d; %d resident records skipped\n", r, (now() - T0) / 1e9,
           recycles, skipped);
    fflush(stdout);
  }
  return 0;
}
