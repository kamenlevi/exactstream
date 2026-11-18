"""hc8r32.py (Leap, Mon 16 Nov 2026 13:0x; CHECK25): the FAST HC8 read (hc8r.read v4: K1C -> K2B -> K3C, ref2's hc8_fast
form at <= 16 rows) for 17-32 rows in ONE pass: the three dispatches each cover 32 rows, and every 8-bit mixer fragment
is decoded once for both 16-row MMA tiles (rows 0-15 and 16-31). Each row's arithmetic is hc8r v4's (the A builder, the
MMA K order, the epilogue, the gate, the tail's stream order), so read32 on 32 rows == hc8r.read on rows 0-15 and on rows
16-31 (each zero-padded to 16), bit for bit (check() proves it). K3C's tail runs one row tile at a time through one 8 KB
buffer (32 rows' tail at once would pass 32 KB of threadgroup memory).
  read32(P, h [32, HC * D], bo=None, inj=None, NP=20) -> (mixed [32, D], written [32, HC * D], inject [32, HC])
  check() vs two hc8r.read calls on random mixers / inputs (with and without the pending write)
usage: [BigMoE venv] hc8r32.py check"""
from functools import lru_cache
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98q/Leap_scripts"))
import hc8r as H8

BF = mx.bfloat16
RR = 32

_MMA2 = H8._MMA + r"""
    auto ct_a2 = op.template get_left_input_cooperative_tensor<T, T, float>();
    auto ct_c2 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ct_a)>,
                                                                metal::remove_addrspace_t<decltype(ct_b)>, float>();
    float acc2[16];
    for (short i = 0; i < 16; i++) acc2[i] = 0.0f;
"""

_K1C32 = _MMA2 + r"""
    constexpr int G = 64, NC = NT * 32, KG = HCD / G, NG = KP / G;
    const int part = int(threadgroup_position_in_grid.x);
    const int tile = int(simdgroup_index_in_threadgroup);
    const uint t = thread_index_in_threadgroup;
    const int n0 = tile * 32;
    const int kp0 = part * KP;
    const int st = kp0 / D;
    const bool itile = n0 >= LR;
    threadgroup T As[2][32 * G];
    threadgroup float Xs[2][32];
    // the A builders: t < 256 -> row t / 8, values (t % 8) * 8 .. + 7 of the group
    const int ar = int(t) / 8, ak = (int(t) % 8) * 8;
    float ss = 0.0f;
    const float ia = (PEND && t < 256) ? float(injp[ar * HC + st]) : 0.0f;
    uint4 hv_n = uint4(0), wv_n = uint4(0), bv_n = uint4(0);
    if (t < 256) {
        hv_n = *(const device uint4*)(h + (size_t)ar * HCD + kp0 + ak);
        wv_n = *(const device uint4*)(wn + kp0 + ak);
        if (PEND) bv_n = *(const device uint4*)(bo + (size_t)ar * D + (kp0 + ak - st * D));
    }
    uint4 q[4], qn[4];
    if (!itile) {
        for (short r = 0; r < 4; r++) {
            const int n = n0 + (r >> 1) * 16 + fm + (r & 1) * 8;
            q[r] = *(const device uint4*)((const device uchar*)wq + (size_t)n * HCD + kp0 + 16 * fq);
        }
    }
    for (int gi = 0; gi < NG; gi++) {
        const int k0 = kp0 + gi * G, g = k0 / G, buf = gi & 1;
        if (t < 256) {
            const int k = k0 + ak;
            const uint4 hv = hv_n, wv = wv_n;
            const uint4 bv_c = bv_n;
            if (gi + 1 < NG) {
                hv_n = *(const device uint4*)(h + (size_t)ar * HCD + k + G);
                wv_n = *(const device uint4*)(wn + k + G);
                if (PEND) bv_n = *(const device uint4*)(bo + (size_t)ar * D + (k + G - st * D));
            }
            thread const T* hp = (thread const T*)&hv;
            thread const T* wp = (thread const T*)&wv;
            T hn[8], an[8];
            float xs = 0.0f;
            if (PEND) {
                const uint4 bv = bv_c;
                thread const T* bp = (thread const T*)&bv;
                for (short j = 0; j < 8; j++)
                    hn[j] = static_cast<T>(float(hp[j]) + float(static_cast<T>(float(bp[j]) * ia)));
            } else {
                for (short j = 0; j < 8; j++) hn[j] = hp[j];
            }
            for (short j = 0; j < 8; j++) {
                ss += float(hn[j]) * float(hn[j]);
                an[j] = static_cast<T>(float(hn[j]) * float(wp[j]));
                xs += float(an[j]);
            }
            *(device uint4*)(hw + (size_t)ar * HCD + k) = *(thread uint4*)hn;
            *(threadgroup uint4*)(&As[buf][ar * G + ak]) = *(thread uint4*)an;
            xs += simd_shuffle_xor(xs, 1); xs += simd_shuffle_xor(xs, 2); xs += simd_shuffle_xor(xs, 4);
            if ((t & 7) == 0) Xs[buf][ar] = xs;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!itile && gi + 1 < NG) {
            for (short r = 0; r < 4; r++) {
                const int n = n0 + (r >> 1) * 16 + fm + (r & 1) * 8;
                qn[r] = *(const device uint4*)((const device uchar*)wq + (size_t)n * HCD + k0 + G + 16 * fq);
            }
        }
        const float xs0 = Xs[buf][fm], xs1 = Xs[buf][fm + 8], xs2 = Xs[buf][16 + fm], xs3 = Xs[buf][24 + fm];
        threadgroup const T* a0 = &As[buf][fm * G + 16 * fq];
        threadgroup const T* a1 = &As[buf][(fm + 8) * G + 16 * fq];
        threadgroup const T* a2 = &As[buf][(16 + fm) * G + 16 * fq];
        threadgroup const T* a3 = &As[buf][(24 + fm) * G + 16 * fq];
        for (short i = 0; i < 16; i++) { ct_c[i] = 0.0f; ct_c2[i] = 0.0f; }
        if (!itile) {
            for (short s = 0; s < 4; s++) {
                for (short j = 0; j < 4; j++) {
                    ct_a[j] = a0[4 * s + j]; ct_a[4 + j] = a1[4 * s + j];
                    ct_a2[j] = a2[4 * s + j]; ct_a2[4 + j] = a3[4 * s + j];
                }
                for (short r = 0; r < 4; r++) {
                    const uint word = q[r][s];
                    const short base = (r >> 1) * 8 + (r & 1) * 4;
                    for (short j = 0; j < 4; j++) ct_b[base + j] = static_cast<T>(float((word >> (8 * j)) & 0xFFu));
                }
                op.run(ct_a, ct_b, ct_c);
                op.run(ct_a2, ct_b, ct_c2);
            }
            for (short bk = 0; bk < 2; bk++)
                for (short j = 0; j < 4; j++) {
                    const int n = n0 + 16 * bk + 4 * fq + j;
                    const float s_ = float(ws[(size_t)n * KG + g]), b_ = float(wb[(size_t)n * KG + g]);
                    const short i0 = bk * 8 + j, i1 = bk * 8 + 4 + j;
                    acc[i0] += s_ * ct_c[i0] + b_ * xs0;
                    acc[i1] += s_ * ct_c[i1] + b_ * xs1;
                    acc2[i0] += s_ * ct_c2[i0] + b_ * xs2;
                    acc2[i1] += s_ * ct_c2[i1] + b_ * xs3;
                }
            for (short r = 0; r < 4; r++) q[r] = qn[r];
        } else {
            for (short s = 0; s < 4; s++) {
                for (short j = 0; j < 4; j++) {
                    ct_a[j] = a0[4 * s + j]; ct_a[4 + j] = a1[4 * s + j];
                    ct_a2[j] = a2[4 * s + j]; ct_a2[4 + j] = a3[4 * s + j];
                }
                for (short r = 0; r < 4; r++) {
                    const int nl = (r >> 1) * 16 + fm + (r & 1) * 8;
                    const short base = (r >> 1) * 8 + (r & 1) * 4;
                    for (short j = 0; j < 4; j++)
                        ct_b[base + j] = nl < HC ? wi[(size_t)nl * HCD + k0 + 16 * fq + 4 * s + j] : T(0);
                }
                op.run(ct_a, ct_b, ct_c);
                op.run(ct_a2, ct_b, ct_c2);
            }
            for (short i = 0; i < 16; i++) { acc[i] += ct_c[i]; acc2[i] += ct_c2[i]; }
        }
    }
    for (short i = 0; i < 16; i++) {
        const int m = fm + 8 * ((i & 7) >> 2);
        const int n = n0 + (i >> 3) * 16 + 4 * fq + (i & 3);
        part_out[((size_t)part * 32 + m) * NC + n] = acc[i];
        part_out[((size_t)part * 32 + 16 + m) * NC + n] = acc2[i];
    }
    if (t < 256) {
        ss += simd_shuffle_xor(ss, 1); ss += simd_shuffle_xor(ss, 2); ss += simd_shuffle_xor(ss, 4);
        if ((t & 7) == 0) ssq[part * 32 + ar] = ss;
    }
"""

_K3C32 = _MMA2 + r"""
    constexpr int G = 64, KG = LR / G;
    const int sg = int(simdgroup_index_in_threadgroup);
    const uint t = thread_index_in_threadgroup;
    const int ch0 = int(threadgroup_position_in_grid.x) * 32;
    const int c0 = sg * D + ch0;
    threadgroup T Ms[32 * LR];
    threadgroup float tp[HC * 16 * 32];
    uint4 q[4], qn[4];
    for (short r = 0; r < 4; r++) {
        const int n = c0 + (r >> 1) * 16 + fm + (r & 1) * 8;
        q[r] = *(const device uint4*)((const device uchar*)uq + (size_t)n * LR + 16 * fq);
    }
    T hv[16], hv2[16], wv[16];
    for (short i = 0; i < 16; i++) {
        const int m = fm + 8 * ((i & 7) >> 2);
        const int col = c0 + (i >> 3) * 16 + 4 * fq + (i & 3);
        hv[i] = hw[(size_t)m * HCD + col];
        hv2[i] = hw[(size_t)(16 + m) * HCD + col];
        wv[i] = wn[col];
    }
    const float r0 = rinv[fm * HC + sg], r1 = rinv[(fm + 8) * HC + sg];
    const float r2 = rinv[(16 + fm) * HC + sg], r3 = rinv[(24 + fm) * HC + sg];
    for (uint o = t; o < 32 * LR / 8; o += 32 * HC)
        *(threadgroup uint4*)(&Ms[o * 8]) = *(const device uint4*)(mix1 + o * 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = 0; g < KG; g++) {
        const int k0 = g * G;
        if (g + 1 < KG) {
            for (short r = 0; r < 4; r++) {
                const int n = c0 + (r >> 1) * 16 + fm + (r & 1) * 8;
                qn[r] = *(const device uint4*)((const device uchar*)uq + (size_t)n * LR + k0 + G + 16 * fq);
            }
        }
        threadgroup const T* a0 = &Ms[fm * LR + k0 + 16 * fq];
        threadgroup const T* a1 = &Ms[(fm + 8) * LR + k0 + 16 * fq];
        threadgroup const T* a2 = &Ms[(16 + fm) * LR + k0 + 16 * fq];
        threadgroup const T* a3 = &Ms[(24 + fm) * LR + k0 + 16 * fq];
        const float xs0 = xsum[fm * KG + g], xs1 = xsum[(fm + 8) * KG + g];
        const float xs2 = xsum[(16 + fm) * KG + g], xs3 = xsum[(24 + fm) * KG + g];
        for (short i = 0; i < 16; i++) { ct_c[i] = 0.0f; ct_c2[i] = 0.0f; }
        for (short s = 0; s < 4; s++) {
            for (short j = 0; j < 4; j++) {
                ct_a[j] = a0[4 * s + j]; ct_a[4 + j] = a1[4 * s + j];
                ct_a2[j] = a2[4 * s + j]; ct_a2[4 + j] = a3[4 * s + j];
            }
            for (short r = 0; r < 4; r++) {
                const uint word = q[r][s];
                const short base = (r >> 1) * 8 + (r & 1) * 4;
                for (short j = 0; j < 4; j++) ct_b[base + j] = static_cast<T>(float((word >> (8 * j)) & 0xFFu));
            }
            op.run(ct_a, ct_b, ct_c);
            op.run(ct_a2, ct_b, ct_c2);
        }
        for (short bk = 0; bk < 2; bk++)
            for (short j = 0; j < 4; j++) {
                const int n = c0 + 16 * bk + 4 * fq + j;
                const float s_ = float(us[(size_t)n * KG + g]), b_ = float(ub[(size_t)n * KG + g]);
                const short i0 = bk * 8 + j, i1 = bk * 8 + 4 + j;
                acc[i0] += s_ * ct_c[i0] + b_ * xs0;
                acc[i1] += s_ * ct_c[i1] + b_ * xs1;
                acc2[i0] += s_ * ct_c2[i0] + b_ * xs2;
                acc2[i1] += s_ * ct_c2[i1] + b_ * xs3;
            }
        for (short r = 0; r < 4; r++) q[r] = qn[r];
    }
    for (short tt = 0; tt < 2; tt++) {                          // the tail, one 16-row tile at a time (8 KB buffer)
        for (short i = 0; i < 16; i++) {
            const int m = fm + 8 * ((i & 7) >> 2);
            const int nl = (i >> 3) * 16 + 4 * fq + (i & 3);
            const float a_ = tt ? acc2[i] : acc[i];
            const float gate = 1.0f / (1.0f + metal::precise::exp(-a_));
            const float rr = tt ? (((i & 7) >> 2) ? r3 : r2) : (((i & 7) >> 2) ? r1 : r0);
            const float nv = float(tt ? hv2[i] : hv[i]) * rr * float(wv[i]);
            tp[(sg * 16 + m) * 32 + nl] = gate * nv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint o = t; o < 16 * 32; o += 32 * HC) {
            const uint m = o / 32, nl = o % 32;
            float sum = 0.0f;
            for (int s = 0; s < HC; s++) sum += tp[(s * 16 + m) * 32 + nl];
            mixed[(size_t)(16 * tt + m) * D + ch0 + nl] = static_cast<T>(sum / float(HC));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

_K2B32 = H8._K2B.replace("ssq[p * 16 + r]", "ssq[p * 32 + r]").replace(
    "part_in[((size_t)p * 16 + r) * NC + n]", "part_in[((size_t)p * 32 + r) * NC + n]")
assert _K2B32.count("p * 32") == 2


@lru_cache(None)
def _kern(which, eps=None):
    if which == 1:
        return mx.fast.metal_kernel(name="hc8r32_k1c", input_names=["h", "bo", "injp", "wn", "wq", "ws", "wb", "wi"],
                                    output_names=["part_out", "ssq", "hw"], source=_K1C32, header=H8._HDR)
    if which == 2:
        return mx.fast.metal_kernel(name=f"hc8r32_k2b_{eps!r}".replace(".", "_").replace("-", "m"),
                                    input_names=["part_in", "ssq"], output_names=["mix1", "inject", "rinv", "xsum"],
                                    source=f"constexpr float EPS = {eps!r}f;\n" + _K2B32)
    return mx.fast.metal_kernel(name="hc8r32_k3c", input_names=["mix1", "uq", "us", "ub", "hw", "rinv", "wn", "xsum"],
                                output_names=["mixed"], source=_K3C32, header=H8._HDR)


def read32(P, h, bo=None, inj=None, NP=20):
    """hc8r.read v4 on 32 rows in one pass: h [32, HC * D] bf16 (rows past the real ones zero), the pending (bo [32, D],
    inj [32, HC]) or None -> (mixed [32, D], written [32, HC * D], inject [32, HC])"""
    HC, D, LR = P["hc"], P["d"], P["lr"]
    HCD = HC * D
    KP, NT = HCD // NP, (LR + HC + 31) // 32
    assert D % KP == 0 and KP % 64 == 0 and NP % HC == 0 and h.shape == (RR, HCD) and NT * 32 >= 256
    assert 32 * LR * 2 + HC * 16 * 32 * 4 <= 32768, "K3C32's threadgroup memory"
    pend = bo is not None
    part, ssq, hw = _kern(1)(
        inputs=[h, bo if pend else h, inj if pend else h, P["wn"], P["wq"], P["ws"], P["wb"], P["wi"]],
        template=[("T", BF), ("HCD", HCD), ("D", D), ("LR", LR), ("HC", HC), ("NP", NP), ("KP", KP), ("NT", NT),
                  ("PEND", int(pend))],
        grid=(32 * NT * NP, 1, 1), threadgroup=(32 * NT, 1, 1),
        output_shapes=[(NP, RR, NT * 32), (NP, RR), (RR, HCD)], output_dtypes=[mx.float32, mx.float32, BF])
    mix1, inject, rinv, xsum = _kern(2, P["eps"])(
        inputs=[part, ssq], template=[("T", BF), ("LR", LR), ("HC", HC), ("NP", NP), ("NC", NT * 32), ("D", D)],
        grid=(RR * NT * 32, 1, 1), threadgroup=(NT * 32, 1, 1),
        output_shapes=[(RR, LR), (RR, HC), (RR, HC), (RR, LR // 64)], output_dtypes=[BF, BF, mx.float32, mx.float32])
    (mixed,) = _kern(3)(
        inputs=[mix1, P["uq"], P["us"], P["ub"], hw, rinv, P["wn"], xsum],
        template=[("T", BF), ("HCD", HCD), ("D", D), ("LR", LR), ("HC", HC)],
        grid=(32 * HC * (D // 32), 1, 1), threadgroup=(32 * HC, 1, 1),
        output_shapes=[(RR, D)], output_dtypes=[BF])
    return mixed, hw, inject


def check(seeds=3, NP=20):
    """read32 vs hc8r.read v4 on rows 0-15 and 16-31 (each zero-padded to 16), with / without the pending write, M 17 /
    24 / 32 real rows; random 8-bit mixers in the 125B's HC shapes (HC 4, D 2560, LR 320)"""
    import numpy as np
    rng = np.random.default_rng(1133)
    HC, D, LR = 4, 2560, 320
    lines, bad, n = [], 0, 0
    for sd in range(seeds):
        f32 = lambda *s_: mx.array(rng.standard_normal(s_).astype(np.float32))
        wq, ws, wb = mx.quantize((f32(LR, HC * D) * 0.02).astype(BF), group_size=64, bits=8)
        uq, us, ub = mx.quantize((f32(HC * D, LR) * 0.05).astype(BF), group_size=64, bits=8)
        P = dict(wq=wq, ws=ws, wb=wb, uq=uq, us=us, ub=ub, wi=(f32(HC, HC * D) * 0.02).astype(BF),
                 wn=(1.0 + 0.1 * f32(HC * D)).astype(BF), hc=HC, d=D, lr=LR, eps=1e-6)
        mx.eval([v for v in P.values() if isinstance(v, mx.array)])
        for pend in (False, True):
            for M in (32, 24, 17):
                h = (f32(M, HC * D) * 0.8).astype(BF)
                bo = (f32(M, D) * 0.5).astype(BF) if pend else None
                inj = (mx.sigmoid(f32(M, HC)) * 2).astype(BF) if pend else None
                z = lambda a, rows: mx.concatenate([a, mx.zeros((rows - a.shape[0],) + a.shape[1:], a.dtype)])
                o32 = read32(P, z(h, 32), z(bo, 32) if pend else None, z(inj, 32) if pend else None, NP=NP)
                oa = H8.read(P, h[:16], bo[:16] if pend else None, inj[:16] if pend else None, NP=NP, v=4)
                ob = H8.read(P, z(h[16:], 16), z(bo[16:], 16) if pend else None, z(inj[16:], 16) if pend else None,
                             NP=NP, v=4)
                mx.eval(*o32, *oa, *ob)
                ok = all(bool(mx.array_equal(o32[i][:M], mx.concatenate([oa[i], ob[i][:M - 16]])))
                         for i in range(3))
                n += 1
                bad += not ok
                lines.append(f"  seed {sd} pend {int(pend)} M {M}: (mixed, written, inject) == two 16-row reads: {ok}")
    hdr = f"HC8R32 check: {n - bad} of {n} cases bit for bit == hc8r.read v4 on rows 0-15 + 16-31 (NP {NP})"
    print(hdr, flush=True)
    for l in lines:
        print(l, flush=True)
    return bad, [hdr] + lines


def bench(rounds=7, NP=20, nread=96):
    """ms per check for its 96 HC reads (48 layers x 2), each on its own cold 8-bit mixers: one 16-row read (A16), two
    16-row reads (today's 17-32-row route), read32; 'ind' (independent) and 'chain' (each read after the previous one)"""
    import time
    import numpy as np
    HC, D, LR = 4, 2560, 320
    Ps = []
    for i in range(nread):
        k = mx.random.split(mx.random.key(1000 + i), 6)
        wq = mx.random.randint(0, 2 ** 31 - 1, (LR, HC * D // 4), key=k[0]).astype(mx.uint32)
        uq = mx.random.randint(0, 2 ** 31 - 1, (HC * D, LR // 4), key=k[1]).astype(mx.uint32)
        ws = mx.random.uniform(1e-4, 2e-4, (LR, HC * D // 64), key=k[2]).astype(BF)
        us = mx.random.uniform(1e-4, 2e-4, (HC * D, LR // 64), key=k[3]).astype(BF)
        Ps.append(dict(wq=wq, ws=ws, wb=-128 * ws, uq=uq, us=us, ub=-128 * us,
                       wi=(mx.random.normal((HC, HC * D), key=k[4]) * 0.02).astype(BF),
                       wn=mx.ones((HC * D,), BF), hc=HC, d=D, lr=LR, eps=1e-6))
    mx.eval([v for P in Ps for v in P.values() if isinstance(v, mx.array)])
    h32 = (mx.random.normal((RR, HC * D), key=mx.random.key(7)) * 0.8).astype(BF)
    mx.eval(h32)
    forms = {"16-row": lambda P, h: H8.read(P, h[:16], NP=NP, v=4)[0],
             "two 16-row": lambda P, h: mx.concatenate([H8.read(P, h[:16], NP=NP, v=4)[0], H8.read(P, h[16:], NP=NP, v=4)[0]]),
             "read32": lambda P, h: read32(P, h, NP=NP)[0]}
    out = {}
    for md in ("ind", "chain"):
        T = {f: [] for f in forms}
        for r in range(rounds):
            order = list(forms)[r % 3:] + list(forms)[:r % 3]
            for f in order:
                mx.synchronize()
                t0 = time.perf_counter()
                if md == "chain":
                    h = h32
                    for P in Ps:
                        y = forms[f](P, h)
                        h = h32 + (y[:1, :1] * 0).astype(BF)
                    mx.eval(h)
                else:
                    mx.eval([forms[f](P, h32) for P in Ps])
                mx.synchronize()
                T[f].append(time.perf_counter() - t0)
        out[md] = {f: 1e3 * float(np.median(v[1:])) for f, v in T.items()}
    lines = [f"HC8R32 bench ({nread} reads per check, cold 8-bit mixers, {rounds} rounds, median of {rounds - 1}): " +
             "; ".join(f"[{md}] " + ", ".join(f"{f} {v:.2f} ms" for f, v in o.items()) for md, o in out.items())]
    print(lines[0], flush=True)
    return lines


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        lines = bench()
        open(os.path.expanduser("~/Leap/data/week98/check25/hc8r32_bench.txt"), "a").write("\n".join(lines) + "\n")
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        b, lines = check()
        p = os.path.expanduser("~/Leap/data/week98/check25/hc8r32_check.txt")
        open(p, "w").write("\n".join(lines) + "\n")
        sys.exit(1 if b else 0)
