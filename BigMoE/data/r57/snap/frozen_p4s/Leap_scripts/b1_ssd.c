// B1 step 2 (Leap, Sun 2026-10-25): random-read throughput of this Mac's SSD by chunk size, page cache bypassed
// (F_NOCACHE), T threads doing pread at 16 KB-aligned random offsets over the given read-only files (the 125B's
// safetensors shards). Prints GB/s and reads/s per (chunk, threads), then the median single-read latency (QD1).
// Build: clang -O2 -o b1_ssd b1_ssd.c -lpthread. Usage: b1_ssd <seconds per point> <file>...
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static int nf;
static int fds[64];
static off_t sizes[64];
static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
typedef struct {
    size_t chunk;
    double until;
    uint64_t seed, reads;
    char *buf;
} arg_t;
static uint64_t rnd(uint64_t *s) {
    *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17;
    return *s;
}
static void *worker(void *p) {
    arg_t *a = p;
    while (now() < a->until) {
        for (int k = 0; k < 16; k++) {
            int f = (int)(rnd(&a->seed) % nf);
            off_t span = (sizes[f] - (off_t)a->chunk) / 16384;
            off_t off = (off_t)(rnd(&a->seed) % (uint64_t)span) * 16384;
            if (pread(fds[f], a->buf, a->chunk, off) != (ssize_t)a->chunk) { perror("pread"); exit(1); }
            a->reads++;
        }
    }
    return 0;
}
static int cmpd(const void *x, const void *y) {
    double a = *(const double *)x, b = *(const double *)y;
    return (a > b) - (a < b);
}
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: b1_ssd <seconds> <file>...\n"); return 1; }
    double secs = atof(argv[1]);
    nf = argc - 2;
    for (int i = 0; i < nf; i++) {
        fds[i] = open(argv[i + 2], O_RDONLY);
        if (fds[i] < 0) { perror(argv[i + 2]); return 1; }
        fcntl(fds[i], F_NOCACHE, 1);
        struct stat st; fstat(fds[i], &st); sizes[i] = st.st_size;
    }
    size_t chunks[] = {4096, 8192, 16384, 32768, 65536, 131072, 262144, 1048576, 2899968};
    int threads[] = {8, 32, 64};
    for (size_t c = 0; c < sizeof chunks / sizeof *chunks; c++) {
        for (size_t t = 0; t < sizeof threads / sizeof *threads; t++) {
            int T = threads[t];
            pthread_t th[64]; arg_t a[64];
            double t0 = now();
            for (int i = 0; i < T; i++) {
                a[i].chunk = chunks[c]; a[i].until = t0 + secs; a[i].seed = 0x9E3779B97F4A7C15ull * (i + 1) + c;
                a[i].reads = 0;
                posix_memalign((void **)&a[i].buf, 16384, chunks[c]);
                pthread_create(&th[i], 0, worker, &a[i]);
            }
            uint64_t n = 0;
            for (int i = 0; i < T; i++) { pthread_join(th[i], 0); n += a[i].reads; free(a[i].buf); }
            double dt = now() - t0;
            printf("chunk %7zu B, %2d threads: %6.2f GB/s, %8.0f reads/s\n", chunks[c], T, n * (double)chunks[c] / dt / 1e9,
                   n / dt);
            fflush(stdout);
        }
    }
    size_t lat[] = {16384, 1048576, 2899968};
    for (size_t c = 0; c < 3; c++) {
        double v[200]; char *buf; posix_memalign((void **)&buf, 16384, lat[c]);
        uint64_t s = 12345 + c;
        for (int i = 0; i < 200; i++) {
            int f = (int)(rnd(&s) % nf);
            off_t off = (off_t)(rnd(&s) % (uint64_t)((sizes[f] - (off_t)lat[c]) / 16384)) * 16384;
            double t0 = now();
            if (pread(fds[f], buf, lat[c], off) != (ssize_t)lat[c]) { perror("pread"); return 1; }
            v[i] = now() - t0;
        }
        qsort(v, 200, sizeof *v, cmpd);
        printf("QD1 latency, %7zu B: median %.0f us, p90 %.0f us\n", lat[c], v[100] * 1e6, v[180] * 1e6);
        free(buf);
    }
    return 0;
}
