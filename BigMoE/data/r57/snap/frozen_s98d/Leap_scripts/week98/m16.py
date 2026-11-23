"""m16.py (Leap, Tue 17 Nov 2026 09:4x; BREAK98, the orchestrator 09:4x answer (b)): the check modules at 16 ROWS, for
Saturday's gated path (t30b + TREE-16 + CC(1) + G1's pick: a 16-row lossy check, int8-free, 8-bit HC mixers). Today's
16-row lossy check (c25 F16) is 38.6 ms, 97 % GPU-busy, 3,242 dispatches (data/week98/c25/gpubB_run.txt). Two switches,
both BITWISE by construction (no token changes; bit tests below and as a c25 pair before any timing counts):
  HC16P  ref2's hc8_fast read at 5-16 rows (hc8r.read v4: K1C -> K2B -> K3C, three dependent launches) in ONE launch:
         hc8p32's persistent kernel (40 groups, two grid barriers) with the INPUT rows bounded by MR = 16: the h / bo / injp
         loads of rows >= MR are skipped (those A-builder rows stay 0, as zero padding would make them), and K3C's tail runs
         2 of its 4 8-row passes (rows 0-15). Every row < MR keeps read32p's arithmetic, and read32p == hc8r32.read32 ==
         hc8r.read per 16-row half bit for bit (hc8p32.check, hc8r32.check), so rows 0-15 == today's read. No pad copies:
         h [16, HC*D], bo [16, D], inj [16, HC] go in as they are; the outputs are 32-row buffers returned as [:16] views.
         The barrier counter's spin-cap fail count (must stay 0, else the read is not exact) is fails().
  GDN16  the GDN core at 2-16 rows: gdn_fast.fast (C 2, U 4) in place of mlx_lm's gated_delta_kernel (bit for bit; this
         check covers C 2 / U 4 at T 2..16 through the hook). Engaged at any phase for T 2..16 (bitwise wherever it runs).
RESULTS 11-17 09:5x (shared GPU, share locks): check 32 / 32 bit for bit (HC16P incl. two concurrent calls, 0 spin-cap
fails; GDN16 T 2..16); bench (ROUGH, 96 chained 16-row reads with the pending write): hc8r.read v4 18.27 -> 20.60 ms
(1.127x) with tile 2 still computed, 15.34 -> 15.97 (1.041x) with it compiled out. HC16P = KILL at 16 rows: phase 1 is
K1C32's A builder (8 threads per row over 32 rows), 2x the 16-row K1C's per-thread work, more than the two launches it
saves; only a 16-row phase-1 rewrite could change that (upside <= ~1 ms). GDN16 goes to the c25 / slot #3 engine arms.
  HC32P  the 32-row one-launch HC read for c25's 32-row arms (slot #3's integration A/B, the 550 line): hc8r32.read32 (HC8,
         f32one) and hc4r.read32 (V3's 4-bit mixers, hc4i) -> this kernel at MR = 32 (every guard open: hc8p32.read32p's
         text and bits, 8- or 4-bit) with per-call counter slots (hc8p32's one shared counter is unsafe when MLX runs two
         reads concurrently). Bit for bit == read32 (check).
install() once, after f32one / hc4i / int8d if those are installed (it wraps whatever hc8r.read / gated_delta_kernel /
hc8r32.read32 / hc4r.read32 are); HC16P[0] / GDN16[0] / HC32P[0] switch per call; COUNTS counts the engaged calls.
usage: [BigMoE venv] m16.py check | bench   (GPU, < 1 GB; share lock; bench is ROUGH)"""
from functools import lru_cache
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98d/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98d/Leap_scripts/week98"))
import hc8r as H8
import hc8p32 as HP

BF = mx.bfloat16
MR = 16
HC16P = [False]
GDN16 = [False]
HC32P = [False]
COUNTS = dict(hc16p=0, gdn16=0, hc32p=0)
_ORIG = {}
_CTR16 = []

# rows >= MR: no h / bo / injp loads (the A builder's row stays 0; hv_n / bv_n start at uint4(0))
_P1_REPS = (
    ("    const float ia = (PEND && t < 256) ? float(injp[ar * HC + st]) : 0.0f;\n",
     "    const float ia = (PEND && t < 256 && ar < MR) ? float(injp[ar * HC + st]) : 0.0f;\n"),
    ("        hv_n = *(const device uint4*)(h + (size_t)ar * HCD + kp0 + ak);\n",
     "        if (ar < MR) hv_n = *(const device uint4*)(h + (size_t)ar * HCD + kp0 + ak);\n"),
    ("        if (PEND) bv_n = *(const device uint4*)(bo + (size_t)ar * D + (kp0 + ak - st * D));\n",
     "        if (PEND && ar < MR) bv_n = *(const device uint4*)(bo + (size_t)ar * D + (kp0 + ak - st * D));\n"),
    ("                hv_n = *(const device uint4*)(h + (size_t)ar * HCD + k + G);\n",
     "                if (ar < MR) hv_n = *(const device uint4*)(h + (size_t)ar * HCD + k + G);\n"),
    ("                if (PEND) bv_n = *(const device uint4*)(bo + (size_t)ar * D + (k + G - st * D));\n",
     "                if (PEND && ar < MR) bv_n = *(const device uint4*)(bo + (size_t)ar * D + (k + G - st * D));\n"))
_P3_REP = ("    for (short pass = 0; pass < 4; pass++) {",
           "    for (short pass = 0; pass < (MR > 16 ? 4 : 2); pass++) {")
# rows 16-31's tile (11-17 09:5x: with it, the one launch was 1.127x the three at 16 rows, ROUGH): its MMA runs, its A-tile
# loads, its partial-sum writes, its K3C inputs and the second half of the mix1 staging are compiled out at MR <= 16 (MR is a
# template constant); tile 1's arithmetic is untouched, so rows 0-15 keep read32p's bits
_T2_RUN = ("op.run(ct_a2, ct_b, ct_c2);", "if (MR > 16) op.run(ct_a2, ct_b, ct_c2);")
_T2_LD = ("ct_a2[j] = a2[4 * s + j]; ct_a2[4 + j] = a3[4 * s + j];",
          "if (MR > 16) { ct_a2[j] = a2[4 * s + j]; ct_a2[4 + j] = a3[4 * s + j]; }")
_T2_P1 = (("        part_out[((size_t)part * 32 + 16 + m) * NC + n] = acc2[i];\n",
           "        if (MR > 16) part_out[((size_t)part * 32 + 16 + m) * NC + n] = acc2[i];\n"),)
_T2_P3 = (("            hv2[i] = hw[(size_t)(16 + m) * HCD + col];\n",
           "            if (MR > 16) hv2[i] = hw[(size_t)(16 + m) * HCD + col];\n"),
          ("    for (uint o = t0; o < 32 * LR / 8; o += NT * 32)\n",
           "    for (uint o = t0; o < (MR > 16 ? 32 : 16) * LR / 8; o += NT * 32)\n"))


def _sub(b, a, z, n):
    assert b.count(a) == n, (a, b.count(a))
    return b.replace(a, z)


def _phase1():
    b = HP._phase1()
    for a, z in _P1_REPS + _T2_P1:
        b = _sub(b, a, z, 1)
    b = _sub(b, *_T2_RUN, 2)
    b = _sub(b, *_T2_LD, 2)
    for w in ("(h + ", "(bo + ", "injp["):                 # every input-row load is guarded
        assert all("ar < MR" in ln for ln in b.split("\n") if w in ln), w
    return b


def _phase3():
    b = _sub(HP._phase3(), *_P3_REP, 1)
    for a, z in _T2_P3:
        b = _sub(b, a, z, 1)
    b = _sub(b, *_T2_RUN, 1)
    return _sub(b, *_T2_LD, 1)


@lru_cache(None)
def _kern(eps, bits=8):
    head = (f"    constexpr float EPS = {eps!r}f;\n    constexpr int NC = NT * 32;\n"
            "    threadgroup uint4 smem4[SMEM / 16];\n    threadgroup uchar* smem = (threadgroup uchar*)smem4;\n"
            "    device atomic_uint* ctr_a = (device atomic_uint*)(const device uint*)ctr_in + 4 * slot[0];\n"
            "    const int gid = int(threadgroup_position_in_grid.x);\n")
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
    if bits == 4:   # V3's 4-bit mixers: hc4r's own load / decode transforms on the same phase texts (as hc8p32)
        p1, p3 = HP.H4._to4(p1, "wq", "HCD"), HP.H4._k3_to4(p3)
    src = (HP._GBAR + head + "    if (gid < NP) {\n" + p1 + "    }\n    GBAR(1u)\n"
           + "    if (gid < 32) {\n" + HP._phase2() + "    }\n    GBAR(2u)\n    {\n" + p3 + "    }\n" + tail)
    ins = ["h", "bo", "injp", "wn", "wq", "ws", "wb", "wi", "uq", "us", "ub", "ctr_in", "slot"]
    outs = ["part_out", "ssq", "hw", "mix1", "inject", "rinv", "xsum", "mixed"]
    return mx.fast.metal_kernel(name=f"hc{bits}p16t_{eps!r}".replace(".", "_").replace("-", "m"), input_names=ins,
                                output_names=outs, source=src, header=H8._HDR)


# the barrier counters: a ring of NSLOT slots of 4 uints ([0] arrivals, [1] spin-cap fails, [2] groups out), one slot per
# call in program order, so calls that MLX runs concurrently (no data dependency: two streams, or one graph with two
# independent reads) never share a counter (11-17 09:5x: one shared counter gave 9 / 18 wrong reads in a 2-call graph)
NSLOT = 256
_SLOT = [0]


def _ctr16():
    if not _CTR16:
        c = mx.zeros((4 * NSLOT,), dtype=mx.uint32)   # 4 KB (MLX passes tiny inputs in constant space)
        mx.eval(c)
        _CTR16.append(c)
    return _CTR16[0]


def _slot():
    _SLOT[0] = (_SLOT[0] + 1) % NSLOT
    return mx.array([_SLOT[0]], dtype=mx.uint32)


def fails():
    """spin-cap fails of every read16p so far, all slots (a sync; call at report time only)"""
    return int(_ctr16().reshape(NSLOT, 4)[:, 1].sum().item()) if _CTR16 else 0


def _fits(P, NP, bits=8):
    HC, D, LR = P["hc"], P["d"], P["lr"]
    HCD = HC * D
    if HCD % NP:
        return False
    KP, NT = HCD // NP, (LR + HC + 31) // 32
    return (D % KP == 0 and KP % 64 == 0 and NP % HC == 0 and NT * 32 >= 256 and D // 32 == 2 * HP.GROUPS
            and NP <= HP.GROUPS and NT * 32 >= 2 * 32 * HC and 32 * LR * 2 + 2 * HC * 8 * 32 * 4 <= HP.SMEM
            and 2 * 32 * 64 * 2 + 2 * 32 * 4 <= HP.SMEM and LR * 2 <= HP.SMEM
            and P["wq"].shape[-1] * 32 == bits * HCD)     # the mixers' bit width (G1's HC8: 8)


def readp(P, h, bo=None, inj=None, NP=20, bits=8):
    """one launch at MR = h's rows (16: hc8r.read v4's bits; 32: hc8r32.read32's (8-bit) / hc4r.read32's (4-bit)):
    (mixed [MR, D], written [MR, HC*D], inject [MR, HC])"""
    mr = h.shape[0]
    HC, D, LR = P["hc"], P["d"], P["lr"]
    HCD = HC * D
    KP, NT = HCD // NP, (LR + HC + 31) // 32
    pend = bo is not None
    assert mr in (16, 32) and h.shape == (mr, HCD) and _fits(P, NP, bits) and (not pend or (bo.shape == (mr, D) and inj.shape == (mr, HC)))
    ins = [h, bo if pend else h, inj if pend else h, P["wn"], P["wq"], P["ws"], P["wb"], P["wi"], P["uq"], P["us"],
           P["ub"], _ctr16(), _slot()]
    shapes = [(NP, 32, NT * 32), (NP, 32), (32, HCD), (32, LR), (32, HC), (32, HC), (32, LR // 64), (32, D)]
    dts = [mx.float32, mx.float32, BF, BF, BF, mx.float32, mx.float32, BF]
    o = _kern(P["eps"], bits)(
        inputs=ins, template=[("T", BF), ("HCD", HCD), ("D", D), ("LR", LR), ("HC", HC), ("NP", NP), ("KP", KP),
                              ("NT", NT), ("PEND", int(pend)), ("P_C", HP.GROUPS), ("SPIN", HP.SPIN),
                              ("SMEM", HP.SMEM), ("MR", mr)],
        grid=(HP.GROUPS * NT * 32, 1, 1), threadgroup=(NT * 32, 1, 1), output_shapes=shapes, output_dtypes=dts)
    return o[7][:mr], o[2][:mr], o[4][:mr]


def read16p(P, h, bo=None, inj=None, NP=20):
    """hc8r.read v4 (NP) at 16 rows in one launch"""
    return readp(P, h, bo, inj, NP=NP, bits=8)


def _read(P, h, bo=None, inj=None, NP=16, v=1):
    if (HC16P[0] and v == 4 and h.shape[0] == MR and h.dtype == BF and _fits(P, NP)
            and (bo is None or (bo.dtype == BF and inj.dtype == BF))):
        COUNTS["hc16p"] += 1
        return read16p(P, h, bo, inj, NP=NP)
    return _ORIG["read"](P, h, bo, inj, NP=NP, v=v)


def _r32(bits):
    """hc8r32.read32 (bits 8) / hc4r.read32 (bits 4) -> readp at 32 rows while HC32P[0]"""
    def f(P, h, bo=None, inj=None, NP=20):
        if (HC32P[0] and h.shape[0] == 32 and h.dtype == BF and _fits(P, NP, bits)
                and (bo is None or (bo.dtype == BF and inj.dtype == BF))):
            COUNTS["hc32p"] += 1
            return readp(P, h, bo, inj, NP=NP, bits=bits)
        return _ORIG[f"r32_{bits}"](P, h, bo, inj, NP=NP)
    return f


def _gdk(q, k, v, g, beta, state, mask=None):
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if GDN16[0] and mask is None and g.ndim == 3 and 2 <= T <= MR and Dk % 32 == 0 and Dv % 8 == 0 and Hv % Hk == 0:
        import gdn_fast
        COUNTS["gdn16"] += 1
        return gdn_fast.fast(q, k, v, g, beta, state, None, C=2, U=4)
    return _ORIG["gdk"](q, k, v, g, beta, state, mask)


def install():
    if _ORIG:
        return
    import mlx_lm.models.gated_delta as GD
    import hc8r32 as R32
    _ORIG["read"], _ORIG["gdk"] = H8.read, GD.gated_delta_kernel
    _ORIG["r32_8"], _ORIG["r32_4"] = R32.read32, HP.H4.read32
    H8.read, GD.gated_delta_kernel = _read, _gdk
    R32.read32, HP.H4.read32 = _r32(8), _r32(4)


def uninstall():
    if _ORIG:
        import mlx_lm.models.gated_delta as GD
        import hc8r32 as R32
        H8.read, GD.gated_delta_kernel = _ORIG.pop("read"), _ORIG.pop("gdk")
        R32.read32, HP.H4.read32 = _ORIG.pop("r32_8"), _ORIG.pop("r32_4")


def _mixers(rng, f32, bits=8):
    HC, D, LR = 4, 2560, 320
    wd, wu = (f32(LR, HC * D) * 0.02).astype(BF), (f32(HC * D, LR) * 0.05).astype(BF)
    wq, ws, wb = mx.quantize(wd, group_size=64, bits=bits)
    uq, us, ub = mx.quantize(wu, group_size=64, bits=bits)
    P = dict(wq=wq, ws=ws, wb=wb, uq=uq, us=us, ub=ub, wi=(f32(HC, HC * D) * 0.02).astype(BF),
             wn=(1.0 + 0.1 * f32(HC * D)).astype(BF), hc=HC, d=D, lr=LR, eps=1e-6)
    mx.eval([v for v in P.values() if isinstance(v, mx.array)])
    return P


def check(seeds=3, NP=20):
    """HC16P: read16p and the installed hc8r.read (switch on) vs today's hc8r.read v4, bit for bit, on random 8-bit mixers
    in the 125B's HC shapes, with / without the pending write, M 16 / 12 / 5 real rows (zero rows after M, as ref2's
    _pad16), plus the spin-cap fail count; GDN16: the installed gated_delta_kernel (switch on, C 2 / U 4) vs mlx_lm's,
    y and state bit for bit, T 2 / 3 / 5 / 8 / 12 / 15 / 16"""
    import numpy as np
    import gdn_fast
    rng = np.random.default_rng(1716)
    f32 = lambda *s_: mx.array(rng.standard_normal(s_).astype(np.float32))
    install()
    lines, bad, n = [], 0, 0
    f0 = fails()
    for sd in range(seeds):
        P = _mixers(rng, f32)
        for pend in (False, True):
            for M in (16, 12, 5):
                z = lambda a: mx.concatenate([a, mx.zeros((MR - a.shape[0],) + a.shape[1:], a.dtype)]) if a.shape[0] < MR else a
                h = z((f32(M, 4 * 2560) * 0.8).astype(BF))
                bo = z((f32(M, 2560) * 0.5).astype(BF)) if pend else None
                inj = z((mx.sigmoid(f32(M, 4)) * 2).astype(BF)) if pend else None
                ref = _ORIG["read"](P, h, bo, inj, NP=NP, v=4)
                mx.eval(*ref)
                o = read16p(P, h, bo, inj, NP=NP)
                mx.eval(*o)
                HC16P[0] = True
                c0 = COUNTS["hc16p"]
                w = H8.read(P, h, bo, inj, NP=NP, v=4)
                hooked = COUNTS["hc16p"] == c0 + 1
                HC16P[0] = False
                mx.eval(*w)
                cc = [read16p(P, h, bo, inj, NP=NP) for _ in range(2)]   # stress: two independent reads in ONE graph
                mx.eval(*cc[0], *cc[1])
                same = [bool(mx.array_equal(o[i], ref[i])) for i in range(3)]
                same_w = [bool(mx.array_equal(w[i], ref[i])) for i in range(3)]
                same_c = all(bool(mx.array_equal(c_[i], ref[i])) for c_ in cc for i in range(3))
                ok = all(same) and all(same_w) and hooked and same_c
                n += 1
                bad += not ok
                lines.append(f"  HC16P seed {sd} pend {int(pend)} M {M:2d}: read16p (mixed, written, inject) == hc8r.read v4 "
                             f"{same}; via the hook {same_w} (engaged {hooked}); 2 concurrent calls {same_c} -> "
                             f"{'OK' if ok else 'BAD'}")
                if not ok:
                    print(lines[-1], flush=True)
    import hc8r32 as R32
    for sd in range(2):
        for bits in (8, 4):
            P = _mixers(rng, f32, bits)
            for pend in (False, True):
                for M in (32, 24, 17):
                    z = lambda a: mx.concatenate([a, mx.zeros((32 - a.shape[0],) + a.shape[1:], a.dtype)]) if a.shape[0] < 32 else a
                    h = z((f32(M, 4 * 2560) * 0.8).astype(BF))
                    bo = z((f32(M, 2560) * 0.5).astype(BF)) if pend else None
                    inj = z((mx.sigmoid(f32(M, 4)) * 2).astype(BF)) if pend else None
                    ref = _ORIG[f"r32_{bits}"](P, h, bo, inj, NP=NP)
                    mx.eval(*ref)
                    o = readp(P, h, bo, inj, NP=NP, bits=bits)
                    mx.eval(*o)
                    HC32P[0] = True
                    c0 = COUNTS["hc32p"]
                    w = (R32.read32 if bits == 8 else HP.H4.read32)(P, h, bo, inj, NP=NP)
                    hooked = COUNTS["hc32p"] == c0 + 1
                    HC32P[0] = False
                    mx.eval(*w)
                    cc = [readp(P, h, bo, inj, NP=NP, bits=bits) for _ in range(2)]
                    mx.eval(*cc[0], *cc[1])
                    same = [bool(mx.array_equal(o[i], ref[i])) for i in range(3)]
                    same_w = [bool(mx.array_equal(w[i], ref[i])) for i in range(3)]
                    same_c = all(bool(mx.array_equal(c_[i], ref[i])) for c_ in cc for i in range(3))
                    ok = all(same) and all(same_w) and hooked and same_c
                    n += 1
                    bad += not ok
                    lines.append(f"  HC32P seed {sd} {bits}-bit pend {int(pend)} M {M:2d}: readp == read32 {same}; via the hook "
                                 f"{same_w} (engaged {hooked}); 2 concurrent calls {same_c} -> {'OK' if ok else 'BAD'}")
                    if not ok:
                        print(lines[-1], flush=True)
    nf = fails() - f0
    lines.append(f"  HC16P / HC32P barrier spin-cap fails over the check: {nf} -> {'OK' if nf == 0 else 'BAD'}")
    bad += nf != 0
    import mlx_lm.models.gated_delta as GD
    for T in (2, 3, 5, 8, 12, 15, 16):
        for seed in (0, 1):
            x = gdn_fast.inputs(T, 7000 + 100 * T + seed)
            y0, s0 = _ORIG["gdk"](*x)
            GDN16[0] = True
            c0 = COUNTS["gdn16"]
            y1, s1 = GD.gated_delta_kernel(*x)
            hooked = COUNTS["gdn16"] == c0 + 1
            GDN16[0] = False
            mx.eval(y0, s0, y1, s1)
            ok = gdn_fast._same(y0, y1) and gdn_fast._same(s0, s1) and hooked
            n += 1
            bad += not ok
            lines.append(f"  GDN16 T {T:2d} seed {seed}: y, state == mlx_lm's kernel {ok and hooked} (engaged {hooked}) -> "
                         f"{'OK' if ok else 'BAD'}")
            if not ok:
                print(lines[-1], flush=True)
    uninstall()
    hdr = (f"M16 check: {n - bad} of {n} cases bit for bit (HC16P one launch vs hc8r.read v4; HC32P vs hc8r32 / hc4r read32, "
           f"8- and 4-bit; NP {NP}; GDN16 C 2 / U 4)")
    print(hdr, flush=True)
    return bad, [hdr] + lines


def bench(rounds=9, NP=20, nread=96):
    """ROUGH (share lock): ms for 96 chained 16-row reads with the pending write (h = the previous read's written), each on
    its own cold 8-bit mixers: hc8r.read v4 (3 launches each) vs read16p (1 launch)"""
    import time
    import numpy as np
    rng = np.random.default_rng(1717)
    f32 = lambda *s_: mx.array(rng.standard_normal(s_).astype(np.float32))
    Ps = [_mixers(rng, f32) for _ in range(nread)]
    h0 = (f32(MR, 4 * 2560) * 0.8).astype(BF)
    bo = (f32(MR, 2560) * 0.5).astype(BF)
    inj = (mx.sigmoid(f32(MR, 4)) * 2).astype(BF)
    mx.eval(h0, bo, inj)
    r0 = _ORIG.get("read", H8.read)

    def chain(fn):
        h, outs = h0, []
        for P in Ps:
            m, w, j = fn(P, h, bo, inj, NP=NP)
            h = w
            outs.append(m)
        return outs + [h]
    FN = {"hc8r.read v4": lambda P, h, b, i, NP: r0(P, h, b, i, NP=NP, v=4), "read16p": read16p}
    ts = {k: [] for k in FN}
    f0 = fails()
    for r in range(rounds):
        for nm in (list(FN) if r % 2 == 0 else list(FN)[::-1]):
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(chain(FN[nm]))
            mx.synchronize()
            ts[nm].append((time.perf_counter() - t0) * 1e3)
    md = {k: float(np.median(v[1:])) for k, v in ts.items()}
    a, b = md["hc8r.read v4"], md["read16p"]
    line = (f"  HC read, {nread} chained reads at 16 rows with the pending write (ROUGH, share lock), ms: hc8r.read v4 "
            f"{a:.2f} -> read16p {b:.2f} ({b - a:+.2f}, {b / a:.3f}x); launches {3 * nread} -> {nread}; spin-cap fails "
            f"{fails() - f0}")
    print(line, flush=True)
    return [line]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    p = os.path.expanduser(f"~/Leap/data/week98/check25/m16_{cmd}.txt")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if cmd == "check":
        b, lines = check()
        open(p, "w").write("\n".join(lines) + f"\nverdict: {'PASS' if not b else 'FAIL'}\n")
        print("PASS" if not b else "FAIL")
        sys.exit(1 if b else 0)
    if cmd == "bench":
        open(p, "a").write("\n".join(bench()) + "\n")
