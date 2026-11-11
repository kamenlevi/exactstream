"""N7-12 (Leap, Mon 2026-11-02 23:1x; pre-registered in ~/Leap/TABLE_350.md 'N7-12'): kernel-A register trims, bitwise to
NAX6 by construction (T1: the decode in two halves; T2: T1 + the A operand straight into the cooperative tensor), kernel A
alone and the whole call in dependent chains, S 16 / 8 / 32. Was N7-9, the ceiling of fusing NAX6's kernels A and B (Leap, Mon 2026-11-02 22:1x; before building (c), the per-group A + B
fusion with completion counters): in DEPENDENT chains (each call's x depends on the previous output, as layers are) on the
real layer-20 pools with real routing (4 traces x 4 passes, S 16 and 32): the whole C1 call (gpu_grp, rowsum64(x), A,
rowsum64(h), B, combine), kernel A alone, kernel B alone (on a pre-evaluated h / hs), and A -> B with the grouping and
row sums precomputed. call - A - B = what a single fused dispatch could remove at most (the small dispatches, the A -> B
boundary, B's ramp); A -> B - A - B = the boundary alone.
Usage: [BigMoE venv] n7_gap.py -> ~/Leap/data/n7_gap.txt"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs1/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs1/Leap_scripts"))
from tierglu import PROJ, plan_masks
import tierglu as TG
import tierglu_nax6 as N6
from tierglu_nax import rowsum64

OUT = os.path.expanduser("~/Leap/data/n7_trim.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
LAYER, PLAN = 20, "4433333333"
RD = os.path.expanduser("~/BigMoE/models/bare/records3")
idx = json.load(open(f"{RD}/index.json"))["layers"][str(LAYER)]
E, REC = idx["E"], idx["recsz"]
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
raw = np.fromfile(f"{RD}/{idx['file']}", dtype=np.uint8).reshape(E, REC)
msb, lsb = {}, {}
for p, pr in enumerate(PROJ):
    o, i = SH[pr]
    ow, os_, ob = idx["msb_piece_off"][3 * p:3 * p + 3]
    msb[pr] = {"w3": mx.array(np.ascontiguousarray(raw[:, ow:ow + o * i * 3 // 8]).view(np.uint32).reshape(E, o, i * 3 // 32)),
               "s3": mx.array(np.ascontiguousarray(raw[:, os_:os_ + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16),
               "b3": mx.array(np.ascontiguousarray(raw[:, ob:ob + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
    lw, lb = idx["lsb_piece_off"][2 * p:2 * p + 2]
    lsb[pr] = {"l1": mx.array(np.ascontiguousarray(raw[:, lw:lw + o * i // 8]).view(np.uint32).reshape(E, o, i // 32)),
               "b": mx.array(np.ascontiguousarray(raw[:, lb:lb + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
del raw
mx.eval([v for d in (msb, lsb) for pr in d for v in d[pr].values()])
say(f"N7-12 kernel-A register trims {time.strftime('%a %H:%M')} ({os.environ.get('N7_COND', 'shared GPU, rough')}); layer {LAYER}")
m4, m2, r4 = plan_masks(PLAN)
R = len(PLAN)
bb = {p: msb[p]["b3"] for p in PROJ}
g_, u_, d_ = (msb[p] for p in PROJ)
TMPL = [("T", mx.bfloat16), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0), ("GS_C", 1)]
def ka(x, xs, ms, ls, grp):
    pidx, seg, nsg_, nseg = grp
    return N6._ka4()(inputs=[x, xs, g_["w3"], u_["w3"], g_["s3"], u_["s3"], g_["b3"], u_["b3"], bb["gate_proj"], bb["up_proj"],
                             lsb["gate_proj"]["l1"], lsb["up_proj"]["l1"], lsb["gate_proj"]["b"], lsb["up_proj"]["b"], ms, ls,
                             pidx, seg, nsg_],
                     template=TMPL + [("K_C", 2560), ("N_C", 640), ("KS_C", 4)], grid=(640 // 32 * 64 * 4, nseg, 1),
                     threadgroup=(256, 1, 1), output_shapes=[(x.shape[0] * R, 640)], output_dtypes=[x.dtype])[0]
def kb(h, hs, ms, ls, grp):
    pidx, seg, nsg_, nseg = grp
    return N6._kb4()(inputs=[h, hs, d_["w3"], d_["s3"], d_["b3"], bb["down_proj"], lsb["down_proj"]["l1"], lsb["down_proj"]["b"],
                             ms, ls, pidx, seg, nsg_],
                     template=TMPL + [("N_C", 640), ("D_C", 2560), ("NSG_C", 2), ("KS_C", 2)],
                     grid=(2560 // 64 * 32 * 2 * 2, nseg, 1), threadgroup=(128, 1, 1),
                     output_shapes=[(h.shape[0], 2560)], output_dtypes=[h.dtype])[0]

DEC_OLD = N6._CORE4[N6._CORE4.index("            const uint w0 = c[0], w1 = c[1], w2 = c[2], w3 = c[3], w4 = c[4], w5 = c[5];"):
                    N6._CORE4.index("            sb[64 * j + lane] = (float)srow[g] * smul;")]
DEC_NEW = """            threadgroup uint4* dst = (threadgroup uint4*)(tb + lane * 8 * GS_C + 8 * j);
            {
                const uint w0 = c[0], w1 = c[1], w2 = c[2];
                const uint h0 = w0 & 0xFFFFFFu, h1 = (w0 >> 24) | ((w1 & 0xFFFFu) << 8), h2 = (w1 >> 16) | ((w2 & 0xFFu) << 16), h3 = w2 >> 8;
                uint4 o;
                if (tier == 4) {
                    const uint l0 = lrow[2 * g];
                    o = uint4(spread8(h0, l0 & 0xFFu), spread8(h1, (l0 >> 8) & 0xFFu), spread8(h2, (l0 >> 16) & 0xFFu), spread8(h3, (l0 >> 24) & 0xFFu));
                } else if (tier == 2) {
                    o = uint4(spread3(h0), spread3(h1), spread3(h2), spread3(h3)) & 0x66666666u;
                } else {
                    o = uint4(spread3(h0), spread3(h1), spread3(h2), spread3(h3));
                }
                dst[0] = o;
            }
            {
                const uint w3 = c[3], w4 = c[4], w5 = c[5];
                const uint h0 = w3 & 0xFFFFFFu, h1 = (w3 >> 24) | ((w4 & 0xFFFFu) << 8), h2 = (w4 >> 16) | ((w5 & 0xFFu) << 16), h3 = w5 >> 8;
                uint4 o;
                if (tier == 4) {
                    const uint l1 = lrow[2 * g + 1];
                    o = uint4(spread8(h0, l1 & 0xFFu), spread8(h1, (l1 >> 8) & 0xFFu), spread8(h2, (l1 >> 16) & 0xFFu), spread8(h3, (l1 >> 24) & 0xFFu));
                } else if (tier == 2) {
                    o = uint4(spread3(h0), spread3(h1), spread3(h2), spread3(h3)) & 0x66666666u;
                } else {
                    o = uint4(spread3(h0), spread3(h1), spread3(h2), spread3(h3));
                }
                dst[1] = o;
            }
"""
ALD_OLD = N6._CORE4[N6._CORE4.index("                const vec<T, 4> a00 = *(const device vec<T, 4>*)(ar0 + k0 + fn0);"):
                    N6._CORE4.index("                TB tbt(")]
ALD_NEW = """                { const vec<T, 4> a = *(const device vec<T, 4>*)(ar0 + k0 + fn0); for (int q = 0; q < 4; ++q) ca[q] = a[q]; }
                { const vec<T, 4> a = *(const device vec<T, 4>*)(ar1 + k0 + fn0); for (int q = 0; q < 4; ++q) ca[4 + q] = a[q]; }
                { const vec<T, 4> a = *(const device vec<T, 4>*)(ar0 + k0 + 16 + fn0); for (int q = 0; q < 4; ++q) ca[8 + q] = a[q]; }
                { const vec<T, 4> a = *(const device vec<T, 4>*)(ar1 + k0 + 16 + fn0); for (int q = 0; q < 4; ++q) ca[12 + q] = a[q]; }
"""
assert N6._CORE4.count(DEC_OLD) == 1 and N6._CORE4.count(ALD_OLD) == 1
CORE_T1 = N6._CORE4.replace(DEC_OLD, DEC_NEW)
CORE_T2 = CORE_T1.replace(ALD_OLD, ALD_NEW)
def kA(name, core):
    return mx.fast.metal_kernel(name=f"n7trim_a_{name}",
        input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["h"], source=N6._A4.replace(N6._CORE4, core), header=N6._HDR4)
KA = {"NAX6": N6._ka4(), "T1": kA("t1", CORE_T1), "T2": kA("t2", CORE_T2)}
def ka_v(kern, x, xs, ms, ls, grp):
    pidx, seg, nsg_, nseg = grp
    return kern(inputs=[x, xs, g_["w3"], u_["w3"], g_["s3"], u_["s3"], g_["b3"], u_["b3"], bb["gate_proj"], bb["up_proj"],
                        lsb["gate_proj"]["l1"], lsb["up_proj"]["l1"], lsb["gate_proj"]["b"], lsb["up_proj"]["b"], ms, ls,
                        pidx, seg, nsg_],
                template=TMPL + [("K_C", 2560), ("N_C", 640), ("KS_C", 4)], grid=(640 // 32 * 64 * 4, nseg, 1),
                threadgroup=(256, 1, 1), output_shapes=[(x.shape[0] * R, 640)], output_dtypes=[x.dtype])[0]
TR = {t: np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{t}_800_g.npz"))[f"L{LAYER}"].astype(np.int32)
      for t in ("code", "chat", "math", "bulgarian")}
mx.random.seed(0)
nb = n = 0
for trace in TR:
    for S in (8, 16, 32):
        for t in np.linspace(0, TR[trace].shape[0] - S, 2).astype(int):
            ids = TR[trace][t:t + S, :10].copy()
            ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
            ms, ls = ms_.reshape(-1), ls_.reshape(-1)
            x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
            grp = N6.gpu_grp(ms_, ls_, PLAN)
            xs = rowsum64(x)
            hs_ = [ka_v(k, x, xs, ms, ls, grp) for k in KA.values()]
            mx.eval(hs_)
            nb += all(bool(mx.all(h.view(mx.uint16) == hs_[0].view(mx.uint16)).item()) for h in hs_[1:])
            n += 1
say(f"  bits: T1 and T2 kernel A == NAX6's: {nb} / {n}")
NCH = 16
for S in (16, 8, 32):
    acc = {}
    for trace in TR:
        for t in np.linspace(0, TR[trace].shape[0] - S, 3).astype(int):
            ids = TR[trace][t:t + S, :10].copy()
            ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
            ms, ls = ms_.reshape(-1), ls_.reshape(-1)
            x0 = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
            grp = N6.gpu_grp(ms_, ls_, PLAN)
            xs = rowsum64(x0)
            mx.eval(x0, grp[:3], xs)
            h0 = ka_v(KA["NAX6"], x0, xs, ms, ls, grp)
            hs0 = rowsum64(h0)
            mx.eval(h0, hs0)
            def chainA(k):
                out = None
                for _ in range(NCH):
                    out = ka_v(k, x0 if out is None else mx.depends(x0, out), xs, ms, ls, grp)
                return out
            def chainCall(k):
                out = None
                for _ in range(NCH):
                    x = x0 if out is None else mx.depends(x0, out)
                    h = ka_v(k, x, xs, ms, ls, grp)
                    out = kb(h, rowsum64(h), ms, ls, grp)
                return out
            fns = {}
            for nm, k in KA.items():
                fns[f"A {nm}"] = (lambda k=k: chainA(k))
                fns[f"call {nm}"] = (lambda k=k: chainCall(k))
            for f in fns.values():
                mx.eval(f())
            ts = {k: [] for k in fns}
            for _ in range(9):
                for k, f in fns.items():
                    mx.synchronize()
                    t1 = time.perf_counter()
                    mx.eval(f())
                    ts[k].append((time.perf_counter() - t1) / NCH)
            for k, v in ts.items():
                acc.setdefault(k, []).append(float(np.median(v)))
    m = {k: 1e3 * float(np.mean(v)) for k, v in acc.items()}
    say(f"S {S}: kernel A NAX6 {m['A NAX6']:.3f}, T1 {m['A T1']:.3f} ({100 * (1 - m['A T1'] / m['A NAX6']):+.1f} %), "
        f"T2 {m['A T2']:.3f} ({100 * (1 - m['A T2'] / m['A NAX6']):+.1f} %); A + B NAX6 {m['call NAX6']:.3f}, T1 "
        f"{m['call T1']:.3f} ({100 * (1 - m['call T1'] / m['call NAX6']):+.1f} %), T2 {m['call T2']:.3f} "
        f"({100 * (1 - m['call T2'] / m['call NAX6']):+.1f} %)")
    if S == 16:
        g16 = max(1 - m["call T1"] / m["call NAX6"], 1 - m["call T2"] / m["call NAX6"])
v = "BLOCKED (bits)" if nb != n else "PASS" if g16 >= 0.08 else "KILL" if g16 < 0.03 else "BETWEEN"
say(f"N7-12 verdict (PASS >= +8 % at S 16 on A + B, KILL < +3 %): {v} ({100 * g16:+.1f} %)")
