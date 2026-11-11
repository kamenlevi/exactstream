// QUEUE 3B prerequisite (1), NewIdeas Sun ~07:00: does a 2-bit PREFIX read (records4's P2 part, the first 1,540,096 B of a
// record) stream at the GB/s of a full MSB-3 read (2,162,688 B)? Per trial: n cold records (drawn without replacement from
// BigMoE's records3 files, F_NOCACHE, one pread per record from the record start into a 16 KB-aligned buffer, all n
// issued at once on the GCD interactive queue = the loader's load_jobs pattern), 2 ms idle between trials, the size order
// rotated across trials. Sizes: LSB 770,048 | P2 1,540,096 | MSB-3 2,162,688 | records4 P2+B3 2,310,144 | full 2,932,736.
// Prints per (n, size): median ms and GB/s; per n a least-squares t = a + b * bytes over the sizes (a = fixed cost per call).
// build: clang -O2 -o ssd_prefix ssd_prefix.c    usage: ssd_prefix RECORDS3_DIR [trials=20]   (quiet lock: SSD timing)
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define REC 2932736LL
#define NE 512
#define NL 48
#define MAXR 16
#define NS 7

static const int64_t SZ[NS] = {770048, 917504, 1015808, 1212416, 1290240, 2162688, 2932736};   // B10 v2 units (16 KB aligned)
static const char* SN[NS] = {"LSB", "res0.9", "res1.0", "res1.2", "base", "MSB3", "full"};
static int fds[NL];
static uint8_t* bufs[MAXR];
static int* order; static int next_rec = 0;
static char pause_f[1024];                                    // quiet research mode: jobs wait out ~/.local/share/quiet/PAUSE
static int paused(void) { return access(pause_f, F_OK) == 0; }
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return t.tv_sec * 1e3 + t.tv_nsec * 1e-6; }
static int cmpd(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return (x > y) - (x < y); }

static void read_one(int rec, int slot, int64_t size) {
  int L = rec / NE, e = rec % NE;
  int64_t done = 0;
  while (done < size) {
    ssize_t r = pread(fds[L], bufs[slot] + done, size - done, (int64_t)e * REC + done);
    if (r <= 0) { perror("pread"); exit(1); }
    done += r;
  }
}

static double trial(int n, int64_t size) {
  const int* recs = order + next_rec;                         // blocks can't capture arrays: a pointer
  next_rec += n;
  dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
  double t0 = now_ms();
  dispatch_apply(n, q, ^(size_t i) { read_one(recs[i], (int)i, size); });
  return now_ms() - t0;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: ssd_prefix RECORDS3_DIR [trials]\n"); return 1; }
  int T = argc > 2 ? atoi(argv[2]) : 20, redo = 0;
  snprintf(pause_f, sizeof pause_f, "%s/.local/share/quiet/PAUSE", getenv("HOME"));
  for (int L = 0; L < NL; L++) {
    char p[1024]; snprintf(p, sizeof p, "%s/L%02d.bin", argv[1], L);
    fds[L] = open(p, O_RDONLY); if (fds[L] < 0) { perror(p); return 1; }
    fcntl(fds[L], F_NOCACHE, 1);
  }
  for (int i = 0; i < MAXR; i++) if (posix_memalign((void**)&bufs[i], 16384, REC)) return 1;
  order = malloc(sizeof(int) * NL * NE);
  for (int i = 0; i < NL * NE; i++) order[i] = i;
  srand(4242);
  for (int i = NL * NE - 1; i > 0; i--) { int j = rand() % (i + 1); int t = order[i]; order[i] = order[j]; order[j] = t; }
  const int NN[] = {1, 2, 4, 8, 13, 16};
  printf("n  size     bytes     median_ms   GB/s   (%d cold trials each; sizes rotated)\n", T);
  for (int a = 0; a < 6; a++) {
    int n = NN[a];
    if (n * (T * NS + 64) + next_rec > NL * NE) { printf("out of cold records\n"); return 0; }
    double t[NS][64], med[NS];
    for (int k = 0; k < T; k++)
      for (int s = 0; s < NS; s++) {
        int ss = (s + k) % NS;
        for (;;) {                                            // wait out a PAUSE; redo a trial that spans one
          while (paused()) usleep(200000);
          usleep(2000);
          if (next_rec + n > NL * NE) { printf("out of cold records\n"); return 0; }
          t[ss][k] = trial(n, SZ[ss]);
          if (!paused()) break;
          redo++;
        }
      }
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int s = 0; s < NS; s++) {
      qsort(t[s], T, sizeof(double), cmpd); med[s] = t[s][T / 2];
      printf("%2d %-6s %9lld  %8.3f  %6.2f\n", n, SN[s], (long long)SZ[s], med[s], n * SZ[s] / (med[s] * 1e6));
      double x = (double)SZ[s]; sx += x; sy += med[s]; sxx += x * x; sxy += x * med[s];
    }
    double b = (NS * sxy - sx * sy) / (NS * sxx - sx * sx), a0 = (sy - b * sx) / NS;
    printf("%2d fit: t = %.3f ms + bytes / (%.2f GB/s)   base / MSB3 GB/s ratio %.3f\n", n, a0, n / (b * 1e6),
           (SZ[4] / med[4]) / (SZ[5] / med[5]));
    fflush(stdout);
  }
  printf("trials redone after a PAUSE: %d\n", redo);
  return 0;
}
