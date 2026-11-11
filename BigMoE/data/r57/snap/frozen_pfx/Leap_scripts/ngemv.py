"""NGEMV (Leap, Sun 2026-11-08 01:1x; CHAT-150 G1, FAST): one GEMV kernel on ref2's native 4-bit layout (DENSE-N / SMALL-N
copies: wn [N/32, K/64, 32, 8] u32, sn / bn [N/32, K/64, 32] bf16) for 1-4-row dense calls. Today every <= 16-row dense call
runs qmm16n's 16-row MMA tiles: at chat's 1-2 rows that pays 16 rows of tile work, and narrow calls (a|b N 96, k / v
N 512, shared down K 640) are latency-bound (~16 us each, 60+ per check).
Layout (ref2._dn_repack): lane l (qid = l >> 2, fm = (qid & 4) | ((l >> 1) & 3), fq = (qid & 2) | (l & 1)) holds 8 words
per group; word 2 st + p: low half = 4 codes of row 16 p + fm, high half = row 16 p + fm + 8, at k = 16 fq + 4 st + j
(j = nibble). So each lane owns 4 rows x 16 k of a group and the simdgroup reads the group's 1 KB contiguously.
Kernel: one threadgroup per 32-row tile, KS simdgroups splitting the K groups. A lane accumulates s * (q . x) + b * sum(x)
for its 4 rows x M inputs over its groups (the affine trick per group), then xor-shuffles over its 4 fq lanes, then the
simdgroups' partials sum in threadgroup memory. Same operands as qmm16n (bf16 x, the codes, bf16 scales / biases), other
summation order: FAST (not bitwise). ngemv(x [M, K], wn, sn, bn, N) -> y [M, N]."""
import os
from functools import lru_cache

import mlx.core as mx

_SRC = r"""
    constexpr int KG = K_C / 64;
    constexpr int GPS = (KG + KS - 1) / KS;
    const uint lane = thread_index_in_simdgroup;
    const uint sgi = simdgroup_index_in_threadgroup;
    const uint tile = threadgroup_position_in_grid.x;
    const uint qid = lane >> 2;
    const uint fm = (qid & 4) | ((lane >> 1) & 3);
    const uint fq = (qid & 2) | (lane & 1);
    float acc[4][M_C];
    for (int r = 0; r < 4; r++)
        for (int m = 0; m < M_C; m++) acc[r][m] = 0.0f;
    const int g0 = int(sgi) * GPS;
    const int g1 = min(KG, g0 + GPS);
    for (int g = g0; g < g1; g++) {
        const size_t tg = (size_t)tile * KG + g;
        const device uint4* wp = (const device uint4*)(wn + (tg * 32 + lane) * 8);
        const uint4 w0 = wp[0], w1 = wp[1];
        const uint wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        const size_t sb = tg * 32;
        const float s0 = float(sn[sb + fm]), s1 = float(sn[sb + fm + 8]), s2 = float(sn[sb + 16 + fm]), s3 = float(sn[sb + 24 + fm]);
        const float b0 = float(bn[sb + fm]), b1 = float(bn[sb + fm + 8]), b2 = float(bn[sb + 16 + fm]), b3 = float(bn[sb + 24 + fm]);
        float xv[M_C][16], xs[M_C];
        for (int m = 0; m < M_C; m++) {
            const device uint4* xp = (const device uint4*)(x + (size_t)m * K_C + g * 64 + 16 * fq);
            const uint4 xa = xp[0], xb = xp[1];
            thread const T* xa_ = (thread const T*)&xa;
            thread const T* xb_ = (thread const T*)&xb;
            xs[m] = 0.0f;
            for (int j = 0; j < 8; j++) { xv[m][j] = float(xa_[j]); xv[m][8 + j] = float(xb_[j]); }
            for (int j = 0; j < 16; j++) xs[m] += xv[m][j];
        }
        float d[4][M_C];
        for (int r = 0; r < 4; r++)
            for (int m = 0; m < M_C; m++) d[r][m] = 0.0f;
        for (int st = 0; st < 4; st++) {                         // each code extracted once, used by every row
            const uint wa = wv[2 * st], wb = wv[2 * st + 1];
            for (int j = 0; j < 4; j++) {
                const float q0 = float((wa >> (4 * j)) & 0xFu), q1 = float((wa >> (16 + 4 * j)) & 0xFu);
                const float q2 = float((wb >> (4 * j)) & 0xFu), q3 = float((wb >> (16 + 4 * j)) & 0xFu);
                for (int m = 0; m < M_C; m++) {
                    const float xx = xv[m][4 * st + j];
                    d[0][m] += q0 * xx; d[1][m] += q1 * xx; d[2][m] += q2 * xx; d[3][m] += q3 * xx;
                }
            }
        }
        for (int m = 0; m < M_C; m++) {
            acc[0][m] += s0 * d[0][m] + b0 * xs[m];
            acc[1][m] += s1 * d[1][m] + b1 * xs[m];
            acc[2][m] += s2 * d[2][m] + b2 * xs[m];
            acc[3][m] += s3 * d[3][m] + b3 * xs[m];
        }
    }
    for (int r = 0; r < 4; r++)
        for (int m = 0; m < M_C; m++) {
            float v = acc[r][m];
            v += simd_shuffle_xor(v, 1);
            v += simd_shuffle_xor(v, 8);
            acc[r][m] = v;
        }
    threadgroup float red[KS * 32 * M_C];
    if (fq == 0) {
        const uint rows[4] = {fm, fm + 8, 16 + fm, 24 + fm};
        for (int r = 0; r < 4; r++)
            for (int m = 0; m < M_C; m++) red[(sgi * 32 + rows[r]) * M_C + m] = acc[r][m];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = thread_index_in_threadgroup; o < (uint)(32 * M_C); o += 32 * KS) {   // 32 x M outputs, 32 KS threads
        const uint row = o / M_C, m = o % M_C;
        float v = 0.0f;
        for (int s = 0; s < KS; s++) v += red[(s * 32 + row) * M_C + m];
        y[(size_t)m * N_C + tile * 32 + row] = static_cast<T>(v);
    }
"""


@lru_cache(None)
def _k():
    return mx.fast.metal_kernel(name="leap_ngemv", input_names=["x", "wn", "sn", "bn"], output_names=["y"], source=_SRC)


def ks_for(T, KG, target=640):
    """simdgroups per tile: >= target simdgroups in all, <= 32 per threadgroup, <= the groups"""
    return int(max(1, min(32, KG, -(-target // T))))


def ngemv(x, wn, sn, bn, N=None, ks=None):
    """x [M, K] bf16 (M 1-4), native wn [T, KG, 32, 8] / sn / bn [T, KG, 32] -> y [M, N] (N <= 32 T; padding columns cut)"""
    M, K = x.shape
    T, KG = wn.shape[0], wn.shape[1]
    assert 1 <= M <= 4 and K == KG * 64 and x.dtype == mx.bfloat16, (x.shape, wn.shape)
    ks = ks or ks_for(T, KG)
    (y,) = _k()(inputs=[x, wn, sn, bn], template=[("T", mx.bfloat16), ("M_C", M), ("K_C", K), ("N_C", 32 * T), ("KS", ks)],
                grid=(T * 32 * ks, 1, 1), threadgroup=(32 * ks, 1, 1), output_shapes=[(M, 32 * T)],
                output_dtypes=[mx.bfloat16])
    return y if N is None or N == 32 * T else y[:, :N]


KS_L = {(2560, 6144): 6, (96, 2560): 10, (512, 2560): 10, (1312, 2560): 8, (2560, 640): 5}   # ref2._DNKS (L's DN-KS 1)
SHAPES = [("gdn_inproj", 16384, 2560, 36), ("out_proj", 2560, 6144, 48), ("att_q", 12288, 2560, 12),
          ("shared_gu", 1312, 2560, 48), ("shared_down", 2560, 640, 48), ("gdn_ab", 96, 2560, 36), ("att_kv", 512, 2560, 24)]


def _native(rng, N, K):
    import numpy as np
    import ref2
    w = mx.array((rng.standard_normal((N, K)) * 0.02).astype(np.float32)).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    wn, sn, bn = ref2._dn_repack(wq, s, b)
    mx.eval(wq, s, b, wn, sn, bn)
    return (wq, s, b), (wn, sn, bn)


def check(n=2, seed=3, out=None):
    """ngemv vs MLX's quantized_matmul (same packs, MLX layout) and vs qmm16n (ref2's native 16-row kernel, padded rows),
    per shape and M 1 / 2 / 4: max relative error and the share of bf16 outputs that differ"""
    import numpy as np
    import qmm26m
    rng = np.random.default_rng(seed)
    lines = []
    for name, N, K, _ in SHAPES:
        worst_m = worst_n = 0.0
        diff = tot = 0
        for _ in range(n):
            (wq, s, b), (wn, sn, bn) = _native(rng, N, K)
            for M in (1, 2, 4):
                x = mx.array(rng.standard_normal((M, K)).astype(np.float32)).astype(mx.bfloat16)
                y0 = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)
                y1 = qmm26m.qmm16n_m(x, wn, sn, bn, N, 1, 1)[:M, :N]
                y2 = ngemv(x, wn, sn, bn, N)
                mx.eval(y0, y1, y2)
                f = lambda a: a.astype(mx.float32)
                den = max(1e-30, float(mx.max(mx.abs(f(y1))).item()))
                worst_m = max(worst_m, float(mx.max(mx.abs(f(y2) - f(y0))).item()) / den)
                worst_n = max(worst_n, float(mx.max(mx.abs(f(y2) - f(y1))).item()) / den)
                diff += int(mx.sum(y2 != y1).item())
                tot += y2.size
        lines.append(f"NGEMV check {name:12s} N {N:5d} K {K:5d}: max rel error vs MLX {worst_m:.1e}, vs qmm16n {worst_n:.1e}; "
                     f"outputs != qmm16n {diff} / {tot} ({100 * diff / max(1, tot):.1f} %)")
    for l in lines:
        print(l, flush=True)
        if out:
            open(out, "a").write(l + "\n")


def bench(Ms=(1, 2, 4), reps=7, out=None):
    """per call us in dependent chains of each shape's per-check call count over distinct cold weights: qmm16n_m (ref2's
    native kernel with pad_in at L's DN-KS split, today's FAST form) vs ngemv; x 'calls' -> ms per check"""
    import time
    import numpy as np
    import qmm26m
    rng = np.random.default_rng(4)
    tot = {M: 0.0 for M in Ms}
    for name, N, K, ncall in SHAPES:
        W = [_native(rng, N, K)[1] for _ in range(min(ncall, 12))]
        mb = sum(a.nbytes for a in W[0]) / 1e6
        line = []
        for M in Ms:
            X = mx.array(rng.standard_normal((M, K)).astype(np.float32)).astype(mx.bfloat16)
            mx.eval(X)
            ksl = KS_L.get((N, K), 1)                  # L's DN-KS (REF2_DN_KS=1) split per shape
            arms = {"qmm16n": lambda x, w: qmm26m.qmm16n_m(x, w[0], w[1], w[2], N, ksl, 1)[:x.shape[0]],
                    "ngemv": lambda x, w: ngemv(x, w[0], w[1], w[2], N)}

            def chain(fn):
                tt = []
                for calls in (False, True):
                    prev = mx.zeros((1,), mx.float32)
                    for i in range(ncall):
                        xx = X + (prev * 0).astype(mx.bfloat16)
                        prev = (fn(xx, W[i % len(W)]) if calls else xx).reshape(-1)[:1].astype(mx.float32)
                    mx.synchronize()
                    t0 = time.perf_counter()
                    mx.eval(prev)
                    tt.append(time.perf_counter() - t0)
                return 1e6 * (tt[1] - tt[0]) / ncall
            for a in arms:
                chain(arms[a])
            acc = {a: [] for a in arms}
            nm = list(arms)
            for r in range(reps):
                for a in (nm if r % 2 == 0 else nm[::-1]):
                    acc[a].append(chain(arms[a]))
            t = {a: float(np.median(v)) for a, v in acc.items()}
            d = ncall * (t["ngemv"] - t["qmm16n"]) / 1e3
            tot[M] += d
            line.append(f"M {M}: qmm16n {t['qmm16n']:.1f} us, ngemv {t['ngemv']:.1f} us ({mb / t['ngemv'] * 1e3:.0f} GB/s; "
                        f"x {ncall}: {d:+.2f} ms)")
        s = f"NGEMV bench {name:12s} N {N:5d} K {K:5d} {mb:5.2f} MB | " + " | ".join(line)
        print(s, flush=True)
        if out:
            open(out, "a").write(s + "\n")
    s = "NGEMV bench total per check: " + ", ".join(f"M {M} {tot[M]:+.2f} ms" for M in Ms)
    print(s, flush=True)
    if out:
        open(out, "a").write(s + "\n")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.expanduser("~/BigMoE"))
    sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_third_party/ms_gdn"))
    o = os.path.expanduser("~/Leap/data/ngemv.txt")
    open(o, "a").write(f"\nNGEMV {__import__('time').strftime('%a %H:%M:%S')} ({os.environ.get('NG_COND', 'share lock')})\n")
    check(out=o)
    if "bench" in sys.argv[1:]:
        bench(out=o)
