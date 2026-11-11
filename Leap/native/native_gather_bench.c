// Leap, Mon 2026-11-02 23:5x: the CPU cost of BigMoE's planned native-layout load (BIGMOE_NAX_NATIVE): one 4-bit expert's
// record pieces (MSB w3 / s3 / b3 and LSB l1 / b for gate, up, down) gathered from a row-major bounce buffer into the
// fragment-native slot layout (tierglu_nax6n.repack: native[((t*KG+g)*32+r)*per+w] = rowmajor[((t*32+r)*KG+g)*per+w])
// vs a plain memcpy of the same bytes. One thread, 48 experts in rotation (~140 MB each side, past the caches).
// Build: clang -O2 native_gather_bench.c -o native_gather_bench
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const int PO[3] = {640, 640, 2560}, PKG[3] = {40, 40, 10};
static void gw(uint32_t *d, const uint32_t *s, int O, int KG, int per) {      // 6- or 2-word runs
    for (int t = 0; t < O / 32; ++t)
        for (int g = 0; g < KG; ++g)
            for (int r = 0; r < 32; ++r)
                memcpy(d + ((size_t)(t * KG + g) * 32 + r) * per, s + ((size_t)(t * 32 + r) * KG + g) * per, 4 * per);
}
static void gh(uint16_t *d, const uint16_t *s, int O, int KG) {               // bf16 scales / biases
    for (int t = 0; t < O / 32; ++t)
        for (int g = 0; g < KG; ++g)
            for (int r = 0; r < 32; ++r)
                d[(t * KG + g) * 32 + r] = s[(t * 32 + r) * KG + g];
}
static size_t expert_bytes(void) {
    size_t n = 0;
    for (int p = 0; p < 3; ++p) n += (size_t)PO[p] * PKG[p] * (24 + 2 + 2 + 8 + 2);
    return n;
}
static void gather_expert(uint8_t *d, const uint8_t *s) {
    size_t off = 0;
    for (int p = 0; p < 3; ++p) {
        const size_t n = (size_t)PO[p] * PKG[p];
        gw((uint32_t *)(d + off), (const uint32_t *)(s + off), PO[p], PKG[p], 6); off += 24 * n;
        gh((uint16_t *)(d + off), (const uint16_t *)(s + off), PO[p], PKG[p]); off += 2 * n;
        gh((uint16_t *)(d + off), (const uint16_t *)(s + off), PO[p], PKG[p]); off += 2 * n;
        gw((uint32_t *)(d + off), (const uint32_t *)(s + off), PO[p], PKG[p], 2); off += 8 * n;
        gh((uint16_t *)(d + off), (const uint16_t *)(s + off), PO[p], PKG[p]); off += 2 * n;
    }
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
int main(void) {
    const int NB = 48, REP = 8;
    const size_t B = expert_bytes();
    uint8_t *src = malloc(NB * B), *dst = malloc(NB * B);
    for (size_t i = 0; i < NB * B; ++i) src[i] = (uint8_t)(i * 2654435761u >> 13), dst[i] = 0;
    double best_m = 1e9, best_g = 1e9;
    for (int rep = 0; rep < REP; ++rep) {
        double t0 = now();
        for (int e = 0; e < NB; ++e) memcpy(dst + e * B, src + e * B, B);
        double t1 = now();
        for (int e = 0; e < NB; ++e) gather_expert(dst + e * B, src + e * B);
        double t2 = now();
        if (t1 - t0 < best_m) best_m = t1 - t0;
        if (t2 - t1 < best_g) best_g = t2 - t1;
    }
    unsigned x = 0;
    for (size_t i = 0; i < NB * B; i += 4093) x += dst[i];
    printf("expert record %.2f MB; per expert (best of %d passes over %d): memcpy %.0f us (%.1f GB/s), native gather %.0f us (%.1f GB/s) [%u]\n",
           B / 1e6, REP, NB, 1e6 * best_m / NB, NB * B / best_m / 1e9, 1e6 * best_g / NB, NB * B / best_g / 1e9, x & 1);
    return 0;
}
