"""HC8-G (Leap, Sun 2026-11-08 01:0x; CHAT-150 G1, FAST): HC8's hyper-connection read in GEMV form for 1-4 real rows.
hc8r v4 (k1c + k2b + k3c) runs 16-row MMA tiles over few threadgroups (k1c: NP = 20, k3c: 80): ~57 us per read at any
row count, ~2.4x its ~7 MB of 8-bit mixers at the machine's rate. At chat's 1-4 rows a GEMV streams the same bytes with
far more simdgroups:
  K1G  per (stream, block of RPS output rows): one simdgroup walks the stream's 2560 inputs, 16 per lane (one uint4 of
       codes per row), rebuilding hn = h + bf16(bo * inj) and an = bf16(hn * wn) in registers (k1c's expressions);
       acc = s * (q . an) + b * sum(an) per 64-group (the affine trick), simd_sum -> part [HC, M, NC] (the stream's
       whole sum, so no partials); block 0 also writes hw (= hn) and the stream's sum of squares.
  K3G  per output channel d: the k2b math in a prologue (rinv per (row, stream), mix1 = silu(c / HC), inject, xsum)
       from part [HC, M, NC] (4 terms per value), then one simdgroup per d: lane = stream * 8 + group (5 of 8 active),
       64 codes of the up mixer per lane, the affine trick per group, shuffles over the groups, k3c's tail (sigmoid
       gate x hw x rinv x wn), shuffles over the streams -> mixed [M, D].
Two dispatches, no pads, no 16-row outputs. The operands are k1c / k2b / k3c's (bf16 an, mix1; fp32 sums), only the
summation order differs: FAST (not bitwise), gated with the chat FAST arm. check(): max relative error vs hc8r.read v4.
read_g(P, h [M, HCD], bo [M, D] | None, inj [M, HC] | None) -> (mixed [M, D], written [M, HCD], inject [M, HC])"""
import os
from functools import lru_cache

import mlx.core as mx

import hc8r as H

_K1G = r"""
    constexpr int G = 64, KG = HCD / G, NB = D / 512;
    const uint lane = thread_index_in_simdgroup;
    const int st = int(threadgroup_position_in_grid.y);
    const int blk = int(threadgroup_position_in_grid.x) * NSG + int(simdgroup_index_in_threadgroup);
    const int n0 = blk * RPS;
    if (n0 >= NC) return;                                       // no barriers in this kernel
    const bool writer = blk == 0;
    float acc[RPS][M_C];
    for (int r = 0; r < RPS; r++)
        for (int m = 0; m < M_C; m++) acc[r][m] = 0.0f;
    float ss[M_C], ia[M_C];
    for (int m = 0; m < M_C; m++) { ss[m] = 0.0f; ia[m] = PEND ? float(injp[m * HC + st]) : 0.0f; }
    for (int b = 0; b < NB; b++) {
        const int kl = b * 512 + int(lane) * 16;
        const int k = st * D + kl;
        const int g = k / G;
        const uint4 wv0 = *(const device uint4*)(wn + k), wv1 = *(const device uint4*)(wn + k + 8);
        thread const T* wp0 = (thread const T*)&wv0;
        thread const T* wp1 = (thread const T*)&wv1;
        float a[M_C][16], xs[M_C];
        for (int m = 0; m < M_C; m++) {
            const uint4 hv0 = *(const device uint4*)(h + (size_t)m * HCD + k), hv1 = *(const device uint4*)(h + (size_t)m * HCD + k + 8);
            thread const T* hp0 = (thread const T*)&hv0;
            thread const T* hp1 = (thread const T*)&hv1;
            T hn[16];
            if (PEND) {
                const uint4 bv0 = *(const device uint4*)(bo + (size_t)m * D + kl), bv1 = *(const device uint4*)(bo + (size_t)m * D + kl + 8);
                thread const T* bp0 = (thread const T*)&bv0;
                thread const T* bp1 = (thread const T*)&bv1;
                for (short j = 0; j < 8; j++) {
                    hn[j] = static_cast<T>(float(hp0[j]) + float(static_cast<T>(float(bp0[j]) * ia[m])));
                    hn[8 + j] = static_cast<T>(float(hp1[j]) + float(static_cast<T>(float(bp1[j]) * ia[m])));
                }
            } else {
                for (short j = 0; j < 8; j++) { hn[j] = hp0[j]; hn[8 + j] = hp1[j]; }
            }
            xs[m] = 0.0f;
            for (short j = 0; j < 16; j++) {
                ss[m] += float(hn[j]) * float(hn[j]);
                const T an = static_cast<T>(float(hn[j]) * float(j < 8 ? wp0[j] : wp1[j - 8]));
                a[m][j] = float(an);
                xs[m] += float(an);
            }
            if (writer) {
                *(device uint4*)(hw + (size_t)m * HCD + k) = *(thread uint4*)&hn[0];
                *(device uint4*)(hw + (size_t)m * HCD + k + 8) = *(thread uint4*)&hn[8];
            }
        }
        for (int r = 0; r < RPS; r++) {
            const int n = n0 + r;
            if (n >= NC) break;
            if (n < LR) {
                const uint4 cw = *(const device uint4*)((const device uchar*)wq + (size_t)n * HCD + k);
                const float s_ = float(ws[(size_t)n * KG + g]), b_ = float(wb[(size_t)n * KG + g]);
                float d[M_C];
                for (int m = 0; m < M_C; m++) d[m] = 0.0f;
                for (short w = 0; w < 4; w++) {
                    const uint word = cw[w];
                    for (short j = 0; j < 4; j++) {
                        const float q = float((word >> (8 * j)) & 0xFFu);
                        for (int m = 0; m < M_C; m++) d[m] += q * a[m][4 * w + j];
                    }
                }
                for (int m = 0; m < M_C; m++) acc[r][m] += s_ * d[m] + b_ * xs[m];
            } else {
                const device T* wr = wi + (size_t)(n - LR) * HCD + k;
                for (short j = 0; j < 16; j++) {
                    const float w_ = float(wr[j]);
                    for (int m = 0; m < M_C; m++) acc[r][m] += w_ * a[m][j];
                }
            }
        }
    }
    for (int r = 0; r < RPS; r++) {
        const int n = n0 + r;
        if (n >= NC) break;
        for (int m = 0; m < M_C; m++) {
            const float v = simd_sum(acc[r][m]);
            if (lane == 0) part[((size_t)st * M_C + m) * NC + n] = v;
        }
    }
    if (writer) {
        for (int m = 0; m < M_C; m++) {
            const float v = simd_sum(ss[m]);
            if (lane == 0) ssq[st * M_C + m] = v;
        }
    }
"""

_K3G = r"""
    constexpr int G = 64, KG = LR / G;
    static_assert(HC == 4 && KG <= 8, "K3G: lane = stream * 8 + group");
    const uint lane = thread_index_in_simdgroup;
    const uint t = thread_index_in_threadgroup;
    const int d = int(threadgroup_position_in_grid.x) * NSG + int(simdgroup_index_in_threadgroup);
    threadgroup T Ms[M_C * LR];
    threadgroup float xs_s[M_C * KG];
    threadgroup float rv_s[M_C * HC];
    for (uint o = t; o < (uint)(M_C * HC); o += 32 * NSG) {
        const uint m = o / HC, s = o % HC;
        rv_s[o] = metal::precise::rsqrt(ssq[s * M_C + m] / float(D) + EPS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = t; o < (uint)(M_C * NC); o += 32 * NSG) {
        const uint m = o / NC, n = o % NC;
        float c = 0.0f;
        for (int s = 0; s < HC; s++) c += part[((size_t)s * M_C + m) * NC + n] * rv_s[m * HC + s];
        if (n < (uint)LR) {
            const float z = c / float(HC);
            Ms[m * LR + n] = static_cast<T>(z / (1.0f + metal::precise::exp(-z)));
        } else if (threadgroup_position_in_grid.x == 0) {
            inject[m * HC + (n - LR)] = static_cast<T>(2.0f / (1.0f + metal::precise::exp(-c / float(HC))));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = t; o < (uint)(M_C * KG); o += 32 * NSG) {
        const uint m = o / KG, g = o % KG;
        float xs = 0.0f;
        for (int k = 0; k < G; k++) xs += float(Ms[m * LR + g * G + k]);
        xs_s[o] = xs;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int s = int(lane) >> 3, g = int(lane) & 7;
    const int col = s * D + d;
    float acc[M_C];
    for (int m = 0; m < M_C; m++) acc[m] = 0.0f;
    if (g < KG) {
        const device uchar* wr = (const device uchar*)uq + (size_t)col * LR + g * G;
        uint4 cs[4];
        for (short v = 0; v < 4; v++) cs[v] = *(const device uint4*)(wr + 16 * v);
        const float s_ = float(us[(size_t)col * KG + g]), b_ = float(ub[(size_t)col * KG + g]);
        float dd[M_C];
        for (int m = 0; m < M_C; m++) dd[m] = 0.0f;
        for (short v = 0; v < 4; v++)
            for (short w = 0; w < 4; w++) {
                const uint word = cs[v][w];
                for (short j = 0; j < 4; j++) {
                    const float q = float((word >> (8 * j)) & 0xFFu);
                    const int kk = g * G + v * 16 + w * 4 + j;
                    for (int m = 0; m < M_C; m++) dd[m] += q * float(Ms[m * LR + kk]);
                }
            }
        for (int m = 0; m < M_C; m++) acc[m] = s_ * dd[m] + b_ * xs_s[m * KG + g];
    }
    float tp[M_C];
    for (int m = 0; m < M_C; m++) {
        float a_ = acc[m];
        a_ += simd_shuffle_xor(a_, 1); a_ += simd_shuffle_xor(a_, 2); a_ += simd_shuffle_xor(a_, 4);
        const float gate = 1.0f / (1.0f + metal::precise::exp(-a_));
        const float nv = float(hw[(size_t)m * HCD + col]) * rv_s[m * HC + s] * float(wn[col]);
        float v = g == 0 ? gate * nv : 0.0f;
        v += simd_shuffle_xor(v, 8); v += simd_shuffle_xor(v, 16);
        tp[m] = v;
    }
    if (lane == 0)
        for (int m = 0; m < M_C; m++) mixed[(size_t)m * D + d] = static_cast<T>(tp[m] / float(HC));
"""


@lru_cache(None)
def _k1g():
    return mx.fast.metal_kernel(name="hc8g_k1g", input_names=["h", "bo", "injp", "wn", "wq", "ws", "wb", "wi"],
                                output_names=["part", "ssq", "hw"], source=_K1G)


@lru_cache(None)
def _k3g(eps):
    return mx.fast.metal_kernel(name=f"hc8g_k3g_{eps!r}".replace(".", "_").replace("-", "m"),
                                input_names=["part", "ssq", "uq", "us", "ub", "hw", "wn"],
                                output_names=["mixed", "inject"], source=f"constexpr float EPS = {eps!r}f;\n" + _K3G)


def read_g(P, h, bo=None, inj=None, rps=8, nsg1=4, nsg3=8):
    """HC8's read for M = h.shape[0] (1-4) real rows in GEMV form -> (mixed [M, D], written [M, HCD], inject [M, HC])"""
    HC, D, LR = P["hc"], P["d"], P["lr"]
    HCD, NC = HC * D, LR + HC
    M = int(h.shape[0])
    assert 1 <= M <= 4 and h.shape[1] == HCD and D % 512 == 0 and LR % 64 == 0 and D % nsg3 == 0
    pend = bo is not None
    BF = mx.bfloat16
    nblk = (NC + rps - 1) // rps
    tgs = (nblk + nsg1 - 1) // nsg1
    part, ssq, hw = _k1g()(
        inputs=[h, bo if pend else h, inj if pend else h, P["wn"], P["wq"], P["ws"], P["wb"], P["wi"]],
        template=[("T", BF), ("M_C", M), ("HCD", HCD), ("D", D), ("LR", LR), ("HC", HC), ("NC", NC), ("RPS", rps),
                  ("NSG", nsg1), ("PEND", int(pend))],
        grid=(tgs * 32 * nsg1, HC, 1), threadgroup=(32 * nsg1, 1, 1),
        output_shapes=[(HC, M, NC), (HC, M), (M, HCD)], output_dtypes=[mx.float32, mx.float32, BF])
    mixed, inject = _k3g(P["eps"])(
        inputs=[part, ssq, P["uq"], P["us"], P["ub"], hw, P["wn"]],
        template=[("T", BF), ("M_C", M), ("HCD", HCD), ("D", D), ("LR", LR), ("HC", HC), ("NC", NC), ("NSG", nsg3)],
        grid=(D // nsg3 * 32 * nsg3, 1, 1), threadgroup=(32 * nsg3, 1, 1),
        output_shapes=[(M, D), (M, HC)], output_dtypes=[BF, BF])
    return mixed, hw, inject


def _params(rng, D=2560, HC=4, LR=320):
    import numpy as np
    HCD = HC * D
    bf = lambda a: mx.array(a.astype(np.float32)).astype(mx.bfloat16)
    wq, ws, wb = mx.quantize(bf(rng.standard_normal((LR, HCD)) * 0.02), group_size=64, bits=8)
    uq, us, ub = mx.quantize(bf(rng.standard_normal((HCD, LR)) * 0.05), group_size=64, bits=8)
    P = dict(wq=wq, ws=ws, wb=wb, uq=uq, us=us, ub=ub, wi=bf(rng.standard_normal((HC, HCD)) * 0.02),
             wn=bf(rng.uniform(0.5, 1.5, (HCD,))), hc=HC, d=D, lr=LR, eps=1e-6)
    mx.eval([a for a in P.values() if isinstance(a, mx.array)])
    return P


def check(n=3, seed=2, out=None):
    """read_g vs hc8r.read v4 (padded) on the real rows: max relative error of mixed / inject, written bitwise"""
    import numpy as np
    rng = np.random.default_rng(seed)
    D, HC = 2560, 4
    HCD = HC * D
    bf = lambda a: mx.array(a.astype(np.float32)).astype(mx.bfloat16)
    pad = lambda a: mx.concatenate([a, mx.zeros((16 - a.shape[0], a.shape[1]), a.dtype)], axis=0)
    worst = {"mixed": 0.0, "inject": 0.0}
    wbad = flips = tot_el = 0
    for trial in range(n):
        P = _params(rng)
        for M in (1, 2, 3, 4):
            h, bo, inj = bf(rng.standard_normal((M, HCD))), bf(rng.standard_normal((M, D))), bf(rng.uniform(0.0, 2.0, (M, HC)))
            for pend in (False, True):
                r0 = H.read(P, pad(h), pad(bo) if pend else None, pad(inj) if pend else None, NP=20, v=4)
                r1 = read_g(P, h, bo if pend else None, inj if pend else None)
                mx.eval(r0, r1)
                for name, a, b in (("mixed", r0[0][:M], r1[0]), ("inject", r0[2][:M], r1[2])):
                    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
                    e = float(mx.max(mx.abs(a32 - b32)).item() / max(1e-30, mx.max(mx.abs(a32)).item()))
                    worst[name] = max(worst[name], e)
                    if name == "mixed":
                        flips += int(mx.sum(a32 != b32).item())
                        tot_el += a.size
                wbad += not bool(mx.array_equal(r0[1][:M], r1[1]))
    line = (f"HC8-G check: max rel error vs hc8r.read v4: mixed {worst['mixed']:.2e}, inject {worst['inject']:.2e}; "
            f"mixed values differing {flips} / {tot_el} ({100 * flips / max(1, tot_el):.2f} %, bf16 ulps); written bitwise "
            f"{'yes' if wbad == 0 else f'NO ({wbad} cases)'}")
    print(line, flush=True)
    if out:
        open(out, "a").write(line + "\n")
    return worst, wbad


def bench(Ms=(1, 2, 3, 4), nw=8, n=48, reps=7, out=None):
    """per HC read us, dependent chains over nw cold weight sets, pending on: hc8r.read v4 (pads + 3 kernels), hc8f's
    fold on M rows (K1M + K3F), read_g (K1G + K3G); calls minus the glue-only chain, median, arms interleaved"""
    import time
    import numpy as np
    import hc8f
    rng = np.random.default_rng(6)
    D, HC = 2560, 4
    HCD = HC * D
    BF = mx.bfloat16
    bf = lambda a: mx.array(a.astype(np.float32)).astype(BF)
    pad = lambda a: mx.concatenate([a, mx.zeros((16 - a.shape[0], a.shape[1]), a.dtype)], axis=0)
    Ps = [_params(rng) for _ in range(nw)]
    arms = {"read_v4_padded": lambda P, h, bo, inj, M: H.read(P, pad(h), pad(bo), pad(inj), NP=20, v=4),
            "fold_k1m": lambda P, h, bo, inj, M: hc8f.read_fold(P, h, bo, inj, NP=20, v=4, M=M),
            "gemv": lambda P, h, bo, inj, M: read_g(P, h, bo, inj)}
    for M in Ms:
        h, bo, inj = bf(rng.standard_normal((M, HCD))), bf(rng.standard_normal((M, D))), bf(rng.uniform(0.0, 2.0, (M, HC)))
        mx.eval(h, bo, inj)

        def chain(fn):
            tt = []
            for calls in (False, True):
                prev = mx.zeros((1,), mx.float32)
                for i in range(n):
                    hx = h + (prev * 0).astype(BF)
                    prev = (fn(Ps[i % nw], hx, bo, inj, M)[0] if calls else hx).reshape(-1)[:1].astype(mx.float32)
                mx.synchronize()
                t0 = time.perf_counter()
                mx.eval(prev)
                tt.append(time.perf_counter() - t0)
            return 1e6 * (tt[1] - tt[0]) / n
        for a in arms:
            chain(arms[a])
        acc = {a: [] for a in arms}
        names = list(arms)
        for r in range(reps):
            for a in (names if r % 2 == 0 else names[::-1]):
                acc[a].append(chain(arms[a]))
        t = {a: float(np.median(v)) for a, v in acc.items()}
        line = (f"HC8-G bench M {M}: " + " | ".join(f"{a} {t[a]:.1f} us" for a in arms)
                + f" | gemv x 96 reads: {96 * (t['gemv'] - t['read_v4_padded']) / 1e3:+.2f} ms vs v4, "
                f"{96 * (t['gemv'] - t['fold_k1m']) / 1e3:+.2f} ms vs fold_k1m per check (7 MB per read: "
                f"{7.0 / t['gemv'] * 1e3:.0f} GB/s)")
        print(line, flush=True)
        if out:
            open(out, "a").write(line + "\n")


def sweep(Ms=(1, 2, 4), nw=8, n=48, reps=5, out=None):
    """read_g at its tunables (rps, nsg1, nsg3), and K1G / K3G alone (the other kernel's inputs precomputed), per read us"""
    import time
    import numpy as np
    rng = np.random.default_rng(7)
    D, HC, LR = 2560, 4, 320
    HCD, NC = HC * D, LR + HC
    BF = mx.bfloat16
    bf = lambda a: mx.array(a.astype(np.float32)).astype(BF)
    Ps = [_params(rng) for _ in range(nw)]

    def k1only(P, h, bo, inj, rps=8, nsg1=4):
        M = h.shape[0]
        nblk = (NC + rps - 1) // rps
        tgs = (nblk + nsg1 - 1) // nsg1
        return _k1g()(inputs=[h, bo, inj, P["wn"], P["wq"], P["ws"], P["wb"], P["wi"]],
                      template=[("T", BF), ("M_C", M), ("HCD", HCD), ("D", D), ("LR", LR), ("HC", HC), ("NC", NC),
                                ("RPS", rps), ("NSG", nsg1), ("PEND", 1)],
                      grid=(tgs * 32 * nsg1, HC, 1), threadgroup=(32 * nsg1, 1, 1),
                      output_shapes=[(HC, M, NC), (HC, M), (M, HCD)], output_dtypes=[mx.float32, mx.float32, BF])

    for M in Ms:
        h, bo, inj = bf(rng.standard_normal((M, HCD))), bf(rng.standard_normal((M, D))), bf(rng.uniform(0.0, 2.0, (M, HC)))
        pre = [k1only(P, h, bo, inj) for P in Ps]
        mx.eval(h, bo, inj, pre)

        def k3only(P, h_, bo_, inj_, i, nsg3=8):
            part, ssq, hw = pre[i % nw]
            ssq = ssq + (h_.reshape(-1)[:1].astype(mx.float32) * 0)   # the chain's dependency (one tiny op, ~2 us)
            return _k3g(P["eps"])(inputs=[part, ssq, P["uq"], P["us"], P["ub"], hw, P["wn"]],
                                 template=[("T", BF), ("M_C", M), ("HCD", HCD), ("D", D), ("LR", LR), ("HC", HC), ("NC", NC),
                                           ("NSG", nsg3)],
                                 grid=(D // nsg3 * 32 * nsg3, 1, 1), threadgroup=(32 * nsg3, 1, 1),
                                 output_shapes=[(M, D), (M, HC)], output_dtypes=[BF, BF])

        arms = {}
        for rps in (2, 4, 8):
            for nsg1 in (2, 4, 8):
                arms[f"rg r{rps} s{nsg1}"] = (lambda rps_, nsg1_: lambda P, hx, i: read_g(P, hx, bo, inj, rps=rps_, nsg1=nsg1_))(rps, nsg1)
        arms["rg nsg3 4"] = lambda P, hx, i: read_g(P, hx, bo, inj, nsg3=4)
        for rps in (2, 4, 8):
            arms[f"K1G only r{rps}"] = (lambda rps_: lambda P, hx, i: k1only(P, hx, bo, inj, rps=rps_))(rps)
        arms["K3G only s8"] = lambda P, hx, i: k3only(P, hx, bo, inj, i)
        arms["K3G only s4"] = lambda P, hx, i: k3only(P, hx, bo, inj, i, nsg3=4)

        def chain(fn):
            tt = []
            for calls in (False, True):
                prev = mx.zeros((1,), mx.float32)
                for i in range(n):
                    hx = h + (prev * 0).astype(BF)
                    prev = (fn(Ps[i % nw], hx, i)[0] if calls else hx).reshape(-1)[:1].astype(mx.float32)
                mx.synchronize()
                t0 = time.perf_counter()
                mx.eval(prev)
                tt.append(time.perf_counter() - t0)
            return 1e6 * (tt[1] - tt[0]) / n
        for a in arms:
            chain(arms[a])
        acc = {a: [] for a in arms}
        names = list(arms)
        for r in range(reps):
            for a in (names if r % 2 == 0 else names[::-1]):
                acc[a].append(chain(arms[a]))
        t = {a: float(np.median(v)) for a, v in acc.items()}
        line = f"HC8-G sweep M {M}: " + " | ".join(f"{a} {t[a]:.1f}" for a in arms) + " (us per read)"
        print(line, flush=True)
        if out:
            open(out, "a").write(line + "\n")


if __name__ == "__main__":
    import sys
    o = os.path.expanduser("~/Leap/data/hc8g.txt")
    if "sweep" in sys.argv[1:]:
        sweep(out=o)
        sys.exit(0)
    check(out=o)
    if "bench" in sys.argv[1:]:
        bench(out=o)
