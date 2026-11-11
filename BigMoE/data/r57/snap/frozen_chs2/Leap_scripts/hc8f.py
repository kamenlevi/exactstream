"""HC8-F (Leap, Sun 2026-11-08; CHAT-150 G1 / B1, exact): hc8r's HC read with K2 (k2b) folded into K3 (k3c) for <= 4 real
rows. At chat's 1-4 rows k2b's work is tiny (M x 324 outputs over NP partials), so every K3 threadgroup recomputes it in
a prologue from the partials (~M x 324 x NP floats from L2) instead of a separate launch: one dispatch less per HC read
(96 per check) and its dependency gap. The prologue repeats k2b's expressions in k2b's order (rinv per (row, stream);
c over the partials in p order; mix1 / inject; xsum over the bf16 mix1), so mix1 / rinv / xsum / inject are the same
values and K3's code MMA and tail (k3c verbatim) see the same operands: bitwise == k2b + k3c on the real rows (check()).
Padding rows (>= M) are zeros and their outputs are discarded by the caller, as today.
read_fold(P, part, ssq, hw, M, eps) -> (mixed [16, D], inject [16, HC]); K1 (k1c) runs as in hc8r.read."""
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


def read_fold(P, h, bo=None, inj=None, NP=16, v=4, M=16):
    """hc8r.read (v 4 / 5) with K2 folded into K3 for M real rows: K1 is read()'s own call, unchanged; returns
    (mixed [16, D], written [16, HC * D], inject [16, HC]) as read() does (rows >= M of mixed / inject unspecified)"""
    assert v in (4, 5)
    HC, D, LR = P["hc"], P["d"], P["lr"]
    HCD = HC * D
    KP, NT = HCD // NP, (LR + HC + 31) // 32
    assert D % KP == 0 and KP % 64 == 0 and NP % HC == 0 and h.shape == (16, HCD)
    pend = bo is not None
    BF = mx.bfloat16
    ins = [h, bo if pend else h, inj if pend else h, P["wn"], P["wq"], P["ws"], P["wb"], P["wi"]]
    tpl = [("T", BF), ("HCD", HCD), ("D", D), ("LR", LR), ("HC", HC), ("NP", NP), ("KP", KP), ("NT", NT), ("PEND", int(pend))]
    outs = dict(output_shapes=[(NP, 16, NT * 32), (NP, 16), (16, HCD)], output_dtypes=[mx.float32, mx.float32, BF])
    if v == 5:
        TPT = 4
        NTG = (NT + TPT - 1) // TPT
        part, ssq, hw = H._kern(9, None)(inputs=ins, template=tpl + [("TPT", TPT)], grid=(32 * TPT * NTG * NP, 1, 1),
                                         threadgroup=(32 * TPT, 1, 1), **outs)
    else:
        part, ssq, hw = H._kern(6, None)(inputs=ins, template=tpl, grid=(32 * NT * NP, 1, 1), threadgroup=(32 * NT, 1, 1), **outs)
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


if __name__ == "__main__":
    check()
