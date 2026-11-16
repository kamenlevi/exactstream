"""hc8p32.py (Leap, Mon 16 Nov 2026 21:3x; CHECK25 module 4): the FAST HC8 read of hc8r32.read32 (K1C -> K2B -> K3C) in
ONE launch. One persistent kernel of 40 threadgroups x 352 threads runs the three phases with two grid barriers between
them (gbar_test.py: exact with device-scope fences; co-resident at 28 KB of threadgroup memory; ~1.5 us per barrier).
  phase 1 = K1C32's text on groups 0..NP-1 (the pending write, the partial projections, the sums of squares);
  phase 2 = K2B32's text on groups 0..31 (one row each: mix1, inject, rinv, xsum);
  phase 3 = K3C32 on all 40 groups, two 32-column items per group (simdgroups 0-3 and 4-7; 8-10 only meet the barriers),
            the shared mix1 staged once per group, the tail in 8-row passes (same per-element arithmetic and stream order).
So read32p == read32 bit for bit by construction (check() proves it). The barrier counter: COUNTER[0] = "init" (a zeroed
output per call, MLX's init_value) or "persist" (one persistent input the kernel resets itself at the end; the calls
must be ordered, as the forward's reads are).
  read32p(P, h, bo=None, inj=None, NP=20) -> (mixed, written, inject)     check()     bench()   (ROUGH under a share lock)
usage: [BigMoE venv] hc8p32.py check | bench"""
from functools import lru_cache
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts/week98"))
import hc8r as H8
import hc8r32 as R32
import hc4r as H4

BF = mx.bfloat16
GROUPS = 40
SMEM = 28672
SPIN = 1 << 20
COUNTER = ["persist"]
_CTR = []

_GBAR = r"""
#define GBAR(k) { \
    threadgroup_barrier(mem_flags::mem_device); \
    if (thread_index_in_threadgroup == 0) { \
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device); \
        atomic_fetch_add_explicit(ctr_a, 1u, memory_order_relaxed); \
        uint it_ = 0; \
        while (atomic_load_explicit(ctr_a, memory_order_relaxed) < (k) * uint(P_C)) { \
            if (++it_ > uint(SPIN)) { atomic_fetch_add_explicit(ctr_a + 1, 1u, memory_order_relaxed); break; } } \
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device); } \
    threadgroup_barrier(mem_flags::mem_device); \
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device); }
"""


def _phase1():
    b = R32._K1C32
    for a, z in (("    const int part = int(threadgroup_position_in_grid.x);\n", "    const int part = gid;\n"),
                 ("    constexpr int G = 64, NC = NT * 32, KG = HCD / G, NG = KP / G;\n",
                  "    constexpr int G = 64, KG = HCD / G, NG = KP / G;\n"),
                 ("    threadgroup T As[2][32 * G];\n    threadgroup float Xs[2][32];\n",
                  "    threadgroup T (*As)[32 * G] = (threadgroup T (*)[32 * G])smem;\n"
                  "    threadgroup float (*Xs)[32] = (threadgroup float (*)[32])(smem + 2 * 32 * G * sizeof(T));\n")):
        assert b.count(a) == 1, a
        b = b.replace(a, z)
    assert "threadgroup_position_in_grid" not in b
    return b


def _phase2():
    b = R32._K2B32
    for a, z in (("    const uint r = threadgroup_position_in_grid.x;\n", "    const uint r = uint(gid);\n"),
                 ("    threadgroup T ms[LR];\n", "    threadgroup T* ms = (threadgroup T*)smem;\n")):
        assert b.count(a) == 1, a
        b = b.replace(a, z)
    b = b.replace("part_in[", "part_out[")
    assert "threadgroup_position_in_grid" not in b and "part_in" not in b
    return b


def _phase3():
    k3 = R32._K3C32
    loop = k3[k3.index("    for (int g = 0; g < KG; g++) {"):k3.index("    for (short tt = 0; tt < 2; tt++)")]
    return H8._MMA + R32._MMA2[len(H8._MMA):] + r"""
    constexpr int G = 64, KG = LR / G;
    const int sgg = int(simdgroup_index_in_threadgroup);
    const uint t0 = thread_index_in_threadgroup;
    const bool act = sgg < 2 * HC;
    const int it3 = act ? sgg / HC : 0;
    const int sg = sgg % HC;
    const uint t = t0 - uint(it3) * (32 * HC);
    const int ch0 = (2 * gid + it3) * 32;
    const int c0 = sg * D + ch0;
    threadgroup T* Ms = (threadgroup T*)smem;
    threadgroup float* tp = (threadgroup float*)(smem + 32 * LR * sizeof(T)) + it3 * (HC * 8 * 32);
    uint4 q[4], qn[4];
    T hv[16], hv2[16], wv[16];
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
    if (act) {
        for (short r = 0; r < 4; r++) {
            const int n = c0 + (r >> 1) * 16 + fm + (r & 1) * 8;
            q[r] = *(const device uint4*)((const device uchar*)uq + (size_t)n * LR + 16 * fq);
        }
        for (short i = 0; i < 16; i++) {
            const int m = fm + 8 * ((i & 7) >> 2);
            const int col = c0 + (i >> 3) * 16 + 4 * fq + (i & 3);
            hv[i] = hw[(size_t)m * HCD + col];
            hv2[i] = hw[(size_t)(16 + m) * HCD + col];
            wv[i] = wn[col];
        }
        r0 = rinv[fm * HC + sg]; r1 = rinv[(fm + 8) * HC + sg];
        r2 = rinv[(16 + fm) * HC + sg]; r3 = rinv[(24 + fm) * HC + sg];
    }
    for (uint o = t0; o < 32 * LR / 8; o += NT * 32)
        *(threadgroup uint4*)(&Ms[o * 8]) = *(const device uint4*)(mix1 + o * 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (act) {
""" + loop + r"""    }
    for (short pass = 0; pass < 4; pass++) {                    // the tail: 8 rows per pass (K3C32: 16), same order
        const short tt = pass >> 1, hf = pass & 1;
        if (act) {
            for (short i = 0; i < 16; i++) {
                const int m = fm + 8 * ((i & 7) >> 2);
                if ((m >> 3) != hf) continue;
                const int nl = (i >> 3) * 16 + 4 * fq + (i & 3);
                const float a_ = tt ? acc2[i] : acc[i];
                const float gate = 1.0f / (1.0f + metal::precise::exp(-a_));
                const float rr = tt ? (((i & 7) >> 2) ? r3 : r2) : (((i & 7) >> 2) ? r1 : r0);
                const float nv = float(tt ? hv2[i] : hv[i]) * rr * float(wv[i]);
                tp[(sg * 8 + (m & 7)) * 32 + nl] = gate * nv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (act) {
            for (uint o = t; o < 8 * 32; o += 32 * HC) {
                const uint m = o / 32, nl = o % 32;
                float sum = 0.0f;
                for (int s = 0; s < HC; s++) sum += tp[(s * 8 + m) * 32 + nl];
                mixed[(size_t)(16 * tt + 8 * hf + m) * D + ch0 + nl] = static_cast<T>(sum / float(HC));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""


@lru_cache(None)
def _kern(eps, persist, bits=8):
    head = (f"    constexpr float EPS = {eps!r}f;\n    constexpr int NC = NT * 32;\n"
            "    threadgroup uint4 smem4[SMEM / 16];\n    threadgroup uchar* smem = (threadgroup uchar*)smem4;\n"
            + ("    device atomic_uint* ctr_a = (device atomic_uint*)(const device uint*)ctr_in;\n" if persist else
               "    device atomic_uint* ctr_a = (device atomic_uint*)ctr;\n")
            + "    const int gid = int(threadgroup_position_in_grid.x);\n")
    tail = ""
    if persist:   # the last group out resets the counter for the next call (all groups have passed both barriers)
        tail = r"""
    threadgroup_barrier(mem_flags::mem_device);
    if (thread_index_in_threadgroup == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        if (atomic_fetch_add_explicit(ctr_a + 2, 1u, memory_order_relaxed) == uint(P_C) - 1u) {
            atomic_store_explicit(ctr_a, 0u, memory_order_relaxed);
            atomic_store_explicit(ctr_a + 2, 0u, memory_order_relaxed);
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        }
    }
"""
    p1, p3 = _phase1(), _phase3()
    if bits == 4:   # V3's 4-bit mixers: hc4r's own load / decode transforms on the same phase texts
        p1, p3 = H4._to4(p1, "wq", "HCD"), H4._k3_to4(p3)
    src = (_GBAR + head + "    if (gid < NP) {\n" + p1 + "    }\n    GBAR(1u)\n"
           + "    if (gid < 32) {\n" + _phase2() + "    }\n    GBAR(2u)\n    {\n" + p3 + "    }\n" + tail)
    ins = ["h", "bo", "injp", "wn", "wq", "ws", "wb", "wi", "uq", "us", "ub"] + (["ctr_in"] if persist else [])
    outs = ["part_out", "ssq", "hw", "mix1", "inject", "rinv", "xsum", "mixed"] + ([] if persist else ["ctr"])
    return mx.fast.metal_kernel(name=f"hc{bits}p32_{'p' if persist else 'i'}_{eps!r}".replace(".", "_").replace("-", "m"),
                                input_names=ins, output_names=outs, source=src, header=H8._HDR)


def _ctr():
    if not _CTR:
        c = mx.zeros((1024,), dtype=mx.uint32)      # 4 KB: MLX passes tiny inputs in constant space (read-only)
        mx.eval(c)
        _CTR.append(c)
    return _CTR[0]


def read32p(P, h, bo=None, inj=None, NP=20, ret_ctr=False, bits=8):
    HC, D, LR = P["hc"], P["d"], P["lr"]
    HCD = HC * D
    KP, NT = HCD // NP, (LR + HC + 31) // 32
    assert D % KP == 0 and KP % 64 == 0 and NP % HC == 0 and h.shape == (32, HCD) and NT * 32 >= 256
    assert D // 32 == 2 * GROUPS and NP <= GROUPS and 32 <= GROUPS and NT * 32 >= 2 * 32 * HC
    assert 32 * LR * 2 + 2 * HC * 8 * 32 * 4 <= SMEM and 2 * 32 * 64 * 2 + 2 * 32 * 4 <= SMEM and LR * 2 <= SMEM
    pend = bo is not None
    persist = COUNTER[0] == "persist"
    ins = [h, bo if pend else h, inj if pend else h, P["wn"], P["wq"], P["ws"], P["wb"], P["wi"], P["uq"], P["us"],
           P["ub"]] + ([_ctr()] if persist else [])
    shapes = [(NP, 32, NT * 32), (NP, 32), (32, HCD), (32, LR), (32, HC), (32, HC), (32, LR // 64), (32, D)]
    dts = [mx.float32, mx.float32, BF, BF, BF, mx.float32, mx.float32, BF]
    if not persist:
        shapes.append((4,))
        dts.append(mx.uint32)
    o = _kern(P["eps"], persist, bits)(
        inputs=ins, template=[("T", BF), ("HCD", HCD), ("D", D), ("LR", LR), ("HC", HC), ("NP", NP), ("KP", KP),
                              ("NT", NT), ("PEND", int(pend)), ("P_C", GROUPS), ("SPIN", SPIN), ("SMEM", SMEM)],
        grid=(GROUPS * NT * 32, 1, 1), threadgroup=(NT * 32, 1, 1), output_shapes=shapes, output_dtypes=dts,
        **({} if persist else dict(init_value=0)))
    r = (o[7], o[2], o[4])
    return r + ((o[8] if not persist else _ctr()),) if ret_ctr else r


def check(seeds=3, NP=20):
    """read32p vs read32 on random 8-bit mixers in the 125B's HC shapes, with / without the pending write, M 17 / 24 / 32,
    both counter modes; and the barrier counter's fail flag (a spin that hit its cap) stays 0"""
    import numpy as np
    rng = np.random.default_rng(1634)
    HC, D, LR = 4, 2560, 320
    lines, bad, n = [], 0, 0
    for sd in range(seeds):
        f32 = lambda *s_: mx.array(rng.standard_normal(s_).astype(np.float32))
        wd, wu = (f32(LR, HC * D) * 0.02).astype(BF), (f32(HC * D, LR) * 0.05).astype(BF)
        wi, wn = (f32(HC, HC * D) * 0.02).astype(BF), (1.0 + 0.1 * f32(HC * D)).astype(BF)
        for bits in (8, 4):
            wq, ws, wb = mx.quantize(wd, group_size=64, bits=bits)
            uq, us, ub = mx.quantize(wu, group_size=64, bits=bits)
            P = dict(wq=wq, ws=ws, wb=wb, uq=uq, us=us, ub=ub, wi=wi, wn=wn, hc=HC, d=D, lr=LR, eps=1e-6)
            mx.eval([v for v in P.values() if isinstance(v, mx.array)])
            for pend in (False, True):
                for M in (32, 24, 17):
                    z = lambda a: mx.concatenate([a, mx.zeros((32 - a.shape[0],) + a.shape[1:], a.dtype)])
                    h = z((f32(M, HC * D) * 0.8).astype(BF))
                    bo = z((f32(M, D) * 0.5).astype(BF)) if pend else None
                    inj = z((mx.sigmoid(f32(M, HC)) * 2).astype(BF)) if pend else None
                    ref = (R32.read32 if bits == 8 else H4.read32)(P, h, bo, inj, NP=NP)
                    for mode in ("init", "persist"):
                        COUNTER[0] = mode
                        o = read32p(P, h, bo, inj, NP=NP, ret_ctr=True, bits=bits)
                        mx.eval(*ref, *o)
                        same = [bool(mx.array_equal(o[i], ref[i])) for i in range(3)]
                        fails = int(o[3][1].item())
                        ok = all(same) and fails == 0
                        n += 1
                        bad += not ok
                        lines.append(f"  seed {sd} {bits}-bit mixers pend {int(pend)} M {M} counter {mode:7s}: (mixed, written, inject) == read32 "
                                     f"{same}; barrier spin-cap fails {fails} -> {'OK' if ok else 'BAD'}")
                        if not ok:
                            print(lines[-1], flush=True)
    COUNTER[0] = "persist"
    hdr = (f"HC8P32 check: {n - bad} of {n} cases bit for bit == read32 (8-bit mixers: hc8r32; 4-bit: hc4r; one launch, "
           f"NP {NP}, {GROUPS} groups)")
    print(hdr, flush=True)
    return bad, [hdr] + lines


def bench(rounds=9, NP=20, nread=96):
    """ROUGH (share lock): ms per check for 96 chained reads (each read's h = the previous read's written streams), each
    on its own cold mixers: read32 (3 launches) vs read32p (1 launch, persistent counter)"""
    import time
    import numpy as np
    rng = np.random.default_rng(1635)
    HC, D, LR = 4, 2560, 320
    f32 = lambda *s_: mx.array(rng.standard_normal(s_).astype(np.float32))
    Ps, Pd = [], {}
    for _ in range(nread):
        wd, wu = (f32(LR, HC * D) * 0.02).astype(BF), (f32(HC * D, LR) * 0.05).astype(BF)
        wq, ws, wb = mx.quantize(wd, group_size=64, bits=8)
        uq, us, ub = mx.quantize(wu, group_size=64, bits=8)
        Ps.append(dict(wq=wq, ws=ws, wb=wb, uq=uq, us=us, ub=ub, wi=(f32(HC, HC * D) * 0.02).astype(BF),
                       wn=(1.0 + 0.1 * f32(HC * D)).astype(BF), hc=HC, d=D, lr=LR, eps=1e-6))
        Pd[id(Ps[-1])] = (wd, wu)
    mx.eval([v for P in Ps for v in P.values() if isinstance(v, mx.array)])
    h0 = (f32(32, HC * D) * 0.8).astype(BF)
    bo = (f32(32, D) * 0.5).astype(BF)
    inj = (mx.sigmoid(f32(32, HC)) * 2).astype(BF)
    mx.eval(h0, bo, inj)

    def chain(fn, Pl):
        h = h0
        outs = []
        for P in Pl:
            m, w, j = fn(P, h, bo, inj, NP=NP)
            h = w
            outs.append(m)
        return outs + [h]
    COUNTER[0] = "persist"
    Ps4 = []
    for P in Ps:
        q4 = {}
        for k_, w_ in (("w", Pd[id(P)][0]), ("u", Pd[id(P)][1])):
            q4[k_] = mx.quantize(w_, group_size=64, bits=4)
        Ps4.append(dict(P, wq=q4["w"][0], ws=q4["w"][1], wb=q4["w"][2], uq=q4["u"][0], us=q4["u"][1], ub=q4["u"][2]))
    mx.eval([v for P in Ps4 for v in P.values() if isinstance(v, mx.array)])
    FN = {"read32": (R32.read32, Ps), "read32p": (read32p, Ps), "hc4r.read32": (H4.read32, Ps4),
          "read32p 4-bit": (lambda *a, **k: read32p(*a, bits=4, **k), Ps4)}
    ts = {k: [] for k in FN}
    for r in range(rounds):
        for nm in (list(FN) if r % 2 == 0 else list(FN)[::-1]):
            fn, Pl = FN[nm]
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(chain(fn, Pl))
            mx.synchronize()
            ts[nm].append((time.perf_counter() - t0) * 1e3)
    md = {k: float(np.median(v[1:])) for k, v in ts.items()}
    line = (f"  HC read, {nread} chained reads at 32 rows (ROUGH, share lock), ms per check: 8-bit mixers read32 "
            f"{md['read32']:.2f} -> read32p {md['read32p']:.2f} ({md['read32p'] - md['read32']:+.2f}); 4-bit (V3) hc4r.read32 "
            f"{md['hc4r.read32']:.2f} -> read32p {md['read32p 4-bit']:.2f} ({md['read32p 4-bit'] - md['hc4r.read32']:+.2f}); "
            f"launches {3 * nread} -> {nread}")
    print(line, flush=True)
    return [line]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    p = os.path.expanduser(f"~/Leap/data/week98/check25/hc8p32_{cmd}.txt")
    if cmd == "check":
        b, lines = check()
        open(p, "w").write("\n".join(lines) + f"\nverdict: {'PASS' if not b else 'FAIL'}\n")
        print("PASS" if not b else "FAIL")
        sys.exit(1 if b else 0)
    if cmd == "bench":
        open(p, "a").write("\n".join(bench()) + "\n")
