"""HC8-F (Leap, Sun 2026-11-08; CHAT-150 G1 / B1, exact): hc8r's HC read with K2 (k2b) folded into K3 (k3c) for <= 4 real
rows. At chat's 1-4 rows k2b's work is tiny (M x 324 outputs over NP partials), so every K3 threadgroup recomputes it in
a prologue from the partials (~M x 324 x NP floats from L2) instead of a separate launch: one dispatch less per HC read
(96 per check) and its dependency gap. The prologue repeats k2b's expressions in k2b's order (rinv per (row, stream);
c over the partials in p order; mix1 / inject; xsum over the bf16 mix1), so mix1 / rinv / xsum / inject are the same
values and K3's code MMA and tail (k3c verbatim) see the same operands: bitwise == k2b + k3c on the real rows (check()).
Padding rows (>= M) are zeros and their outputs are discarded by the caller, as today.
read_fold(P, part, ssq, hw, M, eps) -> (mixed [16, D], inject [16, HC]); K1 (k1c) runs as in hc8r.read."""
import os
from functools import lru_cache

import mlx.core as mx

import hc8r as H

_PRO_OLD = """    const float r0 = rinv[fm * HC + sg], r1 = rinv[(fm + 8) * HC + sg];
    for (uint o = t; o < 16 * LR / 8; o += 32 * HC)
        *(threadgroup uint4*)(&Ms[o * 8]) = *(const device uint4*)(mix1 + o * 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);"""
_PRO_NEW = """    // HC8-F prologue: k2b's math for the M_C real rows, in k2b's order (bitwise the same mix1 / rinv / xsum / inject)
    constexpr int N2 = LR + HC, PPS = NP / HC;
    threadgroup float rv_s[16 * HC];
    threadgroup float xs_s[16 * KG];
    for (uint o = t; o < (uint)(16 * HC); o += 32 * HC) {
        const uint r = o / HC, s = o % HC;
        float v = 0.0f;
        if (r < (uint)M_C) {
            float ss = 0.0f;
            for (int p = int(s) * PPS; p < (int(s) + 1) * PPS; p++) ss += ssq[p * 16 + r];
            v = metal::precise::rsqrt(ss / float(D) + EPS);
        }
        rv_s[o] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = t; o < (uint)(16 * N2); o += 32 * HC) {
        const uint r = o / N2, n = o % N2;
        if (r >= (uint)M_C) {
            if (n < (uint)LR) Ms[r * LR + n] = static_cast<T>(0.0f);
            continue;
        }
        float c = 0.0f;
        for (int p = 0; p < NP; p++) c += part_in[((size_t)p * 16 + r) * NC + n] * rv_s[r * HC + p / PPS];
        if (n < (uint)LR) {
            const float z = c / float(HC);
            Ms[r * LR + n] = static_cast<T>(z / (1.0f + metal::precise::exp(-z)));
        } else if (threadgroup_position_in_grid.x == 0) {
            inject[r * HC + (n - LR)] = static_cast<T>(2.0f / (1.0f + metal::precise::exp(-c / float(HC))));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = t; o < (uint)(16 * KG); o += 32 * HC) {
        const uint r = o / KG, g_ = o % KG;
        float xs = 0.0f;
        if (r < (uint)M_C)
            for (int k = 0; k < 64; k++) xs += float(Ms[r * LR + g_ * 64 + k]);
        xs_s[o] = xs;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float r0 = rv_s[fm * HC + sg], r1 = rv_s[(fm + 8) * HC + sg];"""
_XS_OLD = "        const float xs0 = xsum[fm * KG + g], xs1 = xsum[(fm + 8) * KG + g];"
_XS_NEW = "        const float xs0 = xs_s[fm * KG + g], xs1 = xs_s[(fm + 8) * KG + g];"


@lru_cache(None)
def _kf(eps):
    src = H._K3C
    for a, b in ((_PRO_OLD, _PRO_NEW), (_XS_OLD, _XS_NEW)):
        assert src.count(a) == 1, a[:60]
        src = src.replace(a, b)
    return mx.fast.metal_kernel(name=f"hc8f_k3f_{eps!r}".replace(".", "_").replace("-", "m"),
                                input_names=["part_in", "ssq", "uq", "us", "ub", "hw", "wn"],
                                output_names=["mixed", "inject"], source=f"constexpr float EPS = {eps!r}f;\n" + H._MMA + src[len(H._MMA):],
                                header=H._HDR)


def k3f(P, part, ssq, hw, M):
    """mixed [16, D], inject [16, HC] from K1's partials for M (1-16) real rows; == k2b + k3c on those rows"""
    D, HC, LR = P["d"], P["hc"], P["lr"]
    NP, NC = part.shape[0], part.shape[2]
    mixed, inject = _kf(P["eps"])(
        inputs=[part, ssq, P["uq"], P["us"], P["ub"], hw, P["wn"]],
        template=[("T", mx.bfloat16), ("HCD", HC * D), ("D", D), ("LR", LR), ("HC", HC), ("NP", NP), ("NC", NC), ("M_C", M)],
        grid=(32 * HC * (D // 32), 1, 1), threadgroup=(32 * HC, 1, 1),
        output_shapes=[(16, D), (16, HC)], output_dtypes=[mx.bfloat16, mx.bfloat16])
    return mixed, inject


_K1M_SUBS = [
    ("const float ia = (PEND && t < 128) ? float(injp[ar * HC + st]) : 0.0f;",
     "const bool arv = ar < M_C;                                // K1M: rows >= M read as the pad's zeros\n"
     "    const float ia = (PEND && t < 128 && arv) ? float(injp[ar * HC + st]) : 0.0f;"),
    ("        hv_n = *(const device uint4*)(h + (size_t)ar * HCD + kp0 + ak);",
     "        if (arv) hv_n = *(const device uint4*)(h + (size_t)ar * HCD + kp0 + ak);"),
    ("        if (PEND) bv_n = *(const device uint4*)(bo + (size_t)ar * D + (kp0 + ak - st * D));",
     "        if (PEND && arv) bv_n = *(const device uint4*)(bo + (size_t)ar * D + (kp0 + ak - st * D));"),
    ("                hv_n = *(const device uint4*)(h + (size_t)ar * HCD + k + G);",
     "                if (arv) hv_n = *(const device uint4*)(h + (size_t)ar * HCD + k + G);"),
    ("                if (PEND) bv_n = *(const device uint4*)(bo + (size_t)ar * D + (k + G - st * D));",
     "                if (PEND && arv) bv_n = *(const device uint4*)(bo + (size_t)ar * D + (k + G - st * D));")]


@lru_cache(None)
def _k1m():
    """hc8r's K1 (k1c) reading h / bo / inj with M rows (no 16-row pads): rows >= M load nothing, so their pending write,
    sum of squares and A rows are the pad's zeros, as with the padded inputs (bitwise, check())"""
    src = H._K1C
    for a, b in _K1M_SUBS:
        assert src.count(a) == 1, a[:60]
        src = src.replace(a, b)
    return mx.fast.metal_kernel(name="hc8f_k1m", input_names=["h", "bo", "injp", "wn", "wq", "ws", "wb", "wi"],
                                output_names=["part_out", "ssq", "hw"], source=src, header=H._HDR)


def k1(P, h, bo=None, inj=None, NP=16, v=4):
    """hc8r.read's K1 (v 4: k1c, v 5: k1d) -> (part [NP, 16, NC], ssq [NP, 16], hw [16, HCD]). h with M < 16 rows (bo / inj
    with M rows) runs K1M: the same outputs as K1 on the 16-row zero-padded inputs, pads included (bitwise, check_k1m())"""
    HC, D, LR = P["hc"], P["d"], P["lr"]
    HCD = HC * D
    KP, NT = HCD // NP, (LR + HC + 31) // 32
    M = int(h.shape[0])
    assert D % KP == 0 and KP % 64 == 0 and NP % HC == 0 and h.shape[1] == HCD and 1 <= M <= 16
    pend = bo is not None
    assert not pend or (bo.shape == (M, D) and inj.shape == (M, HC))
    BF = mx.bfloat16
    ins = [h, bo if pend else h, inj if pend else h, P["wn"], P["wq"], P["ws"], P["wb"], P["wi"]]
    tpl = [("T", BF), ("HCD", HCD), ("D", D), ("LR", LR), ("HC", HC), ("NP", NP), ("KP", KP), ("NT", NT), ("PEND", int(pend))]
    outs = dict(output_shapes=[(NP, 16, NT * 32), (NP, 16), (16, HCD)], output_dtypes=[mx.float32, mx.float32, BF])
    if M < 16:                                           # K1M: unpadded inputs, the pads' zeros in-kernel
        return _k1m()(inputs=ins, template=tpl + [("M_C", M)], grid=(32 * NT * NP, 1, 1), threadgroup=(32 * NT, 1, 1), **outs)
    if v == 5:
        TPT = 4
        NTG = (NT + TPT - 1) // TPT
        return H._kern(9, None)(inputs=ins, template=tpl + [("TPT", TPT)], grid=(32 * TPT * NTG * NP, 1, 1),
                                threadgroup=(32 * TPT, 1, 1), **outs)
    return H._kern(6, None)(inputs=ins, template=tpl, grid=(32 * NT * NP, 1, 1), threadgroup=(32 * NT, 1, 1), **outs)


def read_k1m(P, h, bo=None, inj=None, NP=16, v=4):
    """hc8r.read v4 / v5 on M-row inputs: K1M (no pad launches), then k2b + k3c as read() runs them. K1M's outputs equal K1's
    on the padded inputs on all 16 rows, so this == hc8r.read(padded) bitwise on all 16 rows (check_k1m)"""
    HC, D, LR = P["hc"], P["d"], P["lr"]
    NT = (LR + HC + 31) // 32
    BF = mx.bfloat16
    part, ssq, hw = k1(P, h, bo, inj, NP, v)
    mix1, inject, rinv, xsum = H._kern(7, P["eps"])(
        inputs=[part, ssq], template=[("T", BF), ("LR", LR), ("HC", HC), ("NP", NP), ("NC", NT * 32), ("D", D)],
        grid=(16 * NT * 32, 1, 1), threadgroup=(NT * 32, 1, 1),
        output_shapes=[(16, LR), (16, HC), (16, HC), (16, LR // 64)], output_dtypes=[BF, BF, mx.float32, mx.float32])
    (mixed,) = H._kern(8, None)(
        inputs=[mix1, P["uq"], P["us"], P["ub"], hw, rinv, P["wn"], xsum],
        template=[("T", BF), ("HCD", HC * D), ("D", D), ("LR", LR), ("HC", HC)],
        grid=(32 * HC * (D // 32), 1, 1), threadgroup=(32 * HC, 1, 1), output_shapes=[(16, D)], output_dtypes=[BF])
    return mixed, hw, inject


def read_fold(P, h, bo=None, inj=None, NP=16, v=4, M=16):
    """hc8r.read (v 4 / 5) with K2 folded into K3 for M real rows; h / bo / inj either padded to 16 rows (K1 as read()
    runs it) or with M rows (K1M, no pad launches). Returns (mixed [16, D], written [16, HC * D], inject [16, HC]) as
    read() does (rows >= M of mixed / inject unspecified)"""
    assert v in (4, 5) and h.shape[0] in (M, 16)
    part, ssq, hw = k1(P, h, bo, inj, NP, v)
    mixed, inject = k3f(P, part, ssq, hw, M)
    return mixed, hw, inject


def check(n=6, seed=0):
    """K3F vs k2b + k3c on random partials / sums of squares / 8-bit up mixers, M 1-4 (and 16): the real rows' mixed and
    inject bitwise"""
    import numpy as np
    rng = np.random.default_rng(seed)
    D, HC, LR, NP = 2560, 4, 320, 20
    NT = (LR + HC + 31) // 32
    NC = NT * 32
    bad = tot = 0
    for trial in range(n):
        w = mx.array(rng.standard_normal((HC * D, LR)).astype(np.float32) * 0.05).astype(mx.bfloat16)
        uq, us, ub = mx.quantize(w, group_size=64, bits=8)
        P = dict(uq=uq, us=us, ub=ub, wn=mx.array(rng.uniform(0.5, 1.5, (HC * D,)).astype(np.float32)).astype(mx.bfloat16),
                 d=D, hc=HC, lr=LR, eps=1e-6)
        part = mx.array(rng.standard_normal((NP, 16, NC)).astype(np.float32))
        ssq = mx.array(rng.uniform(100.0, 4000.0, (NP, 16)).astype(np.float32))
        hw = mx.array(rng.standard_normal((16, HC * D)).astype(np.float32)).astype(mx.bfloat16)
        mx.eval(uq, us, ub, P["wn"], part, ssq, hw)
        mix1, inject0, rinv, xsum = H._kern(7, P["eps"])(
            inputs=[part, ssq], template=[("T", mx.bfloat16), ("LR", LR), ("HC", HC), ("NP", NP), ("NC", NC), ("D", D)],
            grid=(16 * NT * 32, 1, 1), threadgroup=(NT * 32, 1, 1),
            output_shapes=[(16, LR), (16, HC), (16, HC), (16, LR // 64)], output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float32, mx.float32])
        (mixed0,) = H._kern(8, None)(
            inputs=[mix1, uq, us, ub, hw, rinv, P["wn"], xsum],
            template=[("T", mx.bfloat16), ("HCD", HC * D), ("D", D), ("LR", LR), ("HC", HC)],
            grid=(32 * HC * (D // 32), 1, 1), threadgroup=(32 * HC, 1, 1), output_shapes=[(16, D)], output_dtypes=[mx.bfloat16])
        for M in (1, 2, 3, 4, 16):
            mixed1, inject1 = k3f(P, part, ssq, hw, M)
            mx.eval(mixed0, inject0, mixed1, inject1)
            ok = bool(mx.array_equal(mixed0[:M], mixed1[:M])) and bool(mx.array_equal(inject0[:M], inject1[:M]))
            tot += 1
            bad += not ok
    print(f"HC8-F check: {tot - bad} of {tot} cases bitwise == k2b + k3c on the real rows (mixed, inject)", flush=True)
    return bad


def check_k1m(n=3, seed=1):
    """K1M (M-row inputs) vs K1 v4 / v5 on the zero-padded 16-row inputs: part / ssq / hw bitwise on all 16 rows, with and
    without the pending write, M 1-4; and read_fold(M rows) vs hc8r.read(padded) on the real rows (mixed, written, inject)"""
    import numpy as np
    rng = np.random.default_rng(seed)
    D, HC, LR, NP = 2560, 4, 320, 20
    HCD = HC * D
    BF = mx.bfloat16
    bf = lambda a: mx.array(a.astype(np.float32)).astype(BF)
    pad = lambda a: mx.concatenate([a, mx.zeros((16 - a.shape[0], a.shape[1]), a.dtype)], axis=0)
    bad = tot = 0
    for trial in range(n):
        wq, ws, wb = mx.quantize(bf(rng.standard_normal((LR, HCD)) * 0.02), group_size=64, bits=8)
        uq, us, ub = mx.quantize(bf(rng.standard_normal((HCD, LR)) * 0.05), group_size=64, bits=8)
        P = dict(wq=wq, ws=ws, wb=wb, uq=uq, us=us, ub=ub, wi=bf(rng.standard_normal((HC, HCD)) * 0.02),
                 wn=bf(rng.uniform(0.5, 1.5, (HCD,))), hc=HC, d=D, lr=LR, eps=1e-6)
        mx.eval([a for a in P.values() if isinstance(a, mx.array)])
        for M in (1, 2, 3, 4):
            h, bo, inj = bf(rng.standard_normal((M, HCD))), bf(rng.standard_normal((M, D))), bf(rng.uniform(0.0, 2.0, (M, HC)))
            for pend in (False, True):
                b_, i_ = (bo, inj) if pend else (None, None)
                bp, ip = (pad(bo), pad(inj)) if pend else (None, None)
                new = k1(P, h, b_, i_, NP)
                for v in (4, 5):
                    ref = k1(P, pad(h), bp, ip, NP, v)
                    mx.eval(new, ref)
                    ok = all(bool(mx.array_equal(a, b)) for a, b in zip(new, ref))
                    tot += 1
                    bad += not ok
                    if not ok:
                        print(f"  K1M differs: trial {trial} M {M} pend {pend} v {v}", flush=True)
                r0 = H.read(P, pad(h), bp, ip, NP=NP, v=4)
                r1 = read_fold(P, h, b_, i_, NP=NP, v=4, M=M)
                r2 = read_k1m(P, h, b_, i_, NP=NP, v=4)
                mx.eval(r0, r1, r2)
                ok = all(bool(mx.array_equal(a[:M], b[:M])) for a, b in zip(r0, r1))
                tot += 1
                bad += not ok
                if not ok:
                    print(f"  read_fold(M rows) differs from read(padded): trial {trial} M {M} pend {pend}", flush=True)
                ok = all(bool(mx.array_equal(a, b)) for a, b in zip(r0, r2))
                tot += 1
                bad += not ok
                if not ok:
                    print(f"  read_k1m(M rows) differs from read(padded) on the 16 rows: trial {trial} M {M} pend {pend}", flush=True)
    print(f"HC8-F K1M check: {tot - bad} of {tot} cases bitwise (K1 outputs on all 16 rows vs v4 / v5 on padded inputs; "
          f"read_fold(M rows) vs hc8r.read(padded) on the real rows)", flush=True)
    return bad


def bench(Ms=(1, 2, 3, 4), nw=8, n=48, reps=7, out=None):
    """per HC read us in dependent chains of n reads over nw distinct cold weight sets (8-bit mixers at the 125B's shapes,
    pending write on), M real rows; arms: hc8r.read v4 on the padded inputs (pads + k1c + k2b + k3c), read_fold on the
    padded inputs (pads + k1c + K3F), read_fold on M rows (K1M + K3F). Calls minus the glue-only chain, median of reps,
    arms interleaved"""
    import time
    import numpy as np
    rng = np.random.default_rng(5)
    D, HC, LR, NP = 2560, 4, 320, 20
    HCD = HC * D
    BF = mx.bfloat16
    bf = lambda a: mx.array(a.astype(np.float32)).astype(BF)
    pad = lambda a: mx.concatenate([a, mx.zeros((16 - a.shape[0], a.shape[1]), a.dtype)], axis=0)
    Ps = []
    for _ in range(nw):
        wq, ws, wb = mx.quantize(bf(rng.standard_normal((LR, HCD)) * 0.02), group_size=64, bits=8)
        uq, us, ub = mx.quantize(bf(rng.standard_normal((HCD, LR)) * 0.05), group_size=64, bits=8)
        P = dict(wq=wq, ws=ws, wb=wb, uq=uq, us=us, ub=ub, wi=bf(rng.standard_normal((HC, HCD)) * 0.02),
                 wn=bf(rng.uniform(0.5, 1.5, (HCD,))), hc=HC, d=D, lr=LR, eps=1e-6)
        mx.eval([a for a in P.values() if isinstance(a, mx.array)])
        Ps.append(P)
    arms = {"read_v4_padded": lambda P, h, bo, inj, M: H.read(P, pad(h), pad(bo), pad(inj), NP=NP, v=4),
            "fold_padded": lambda P, h, bo, inj, M: read_fold(P, pad(h), pad(bo), pad(inj), NP=NP, v=4, M=M),
            "fold_k1m": lambda P, h, bo, inj, M: read_fold(P, h, bo, inj, NP=NP, v=4, M=M),
            "k1m_unfolded": lambda P, h, bo, inj, M: read_k1m(P, h, bo, inj, NP=NP, v=4)}
    res = {}
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
        res[M] = t
        line = (f"HC8-F bench M {M}: " + " | ".join(f"{a} {t[a]:.1f} us" for a in arms)
                + " | x 96 reads vs v4: " + ", ".join(f"{a} {96 * (t[a] - t['read_v4_padded']) / 1e3:+.2f} ms"
                                                      for a in names[1:]) + " per check (GPU, dependent chain)")
        print(line, flush=True)
        if out:
            open(out, "a").write(line + "\n")
    return res


if __name__ == "__main__":
    import sys
    check()
    check_k1m()
    if "bench" in sys.argv[1:]:
        bench(out=os.path.expanduser("~/Leap/data/hc8f_bench.txt"))
