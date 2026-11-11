"""N7-3 occupancy diagnostic (Leap, Mon 2026-11-02 21:4x; NAX7 groundwork): is NAX6's kernel A occupancy-bound? Its
threadgroups (8 simdgroups) use 20 KB of threadgroup memory. Arms: the same kernel with an extra threadgroup array of
0 / 4 / 8 / 12 KB (total 20 / 24 / 28 / 32 KB; touched so it is not removed), real layer-20 pools and routing at S 16,
kernel A alone, 8 chained calls per eval, median of 9 interleaved rounds. Slower with more memory -> residency matters
and a leaner (or producer / consumer) kernel has room; flat -> the limit is elsewhere. Bitwise to NAX6 (the pad holds
no data the arithmetic reads).
Usage: [BigMoE venv] n7_occ.py -> ~/Leap/data/n7_occ.txt"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/Leap_scripts"))
from tierglu import PROJ, plan_masks
import tierglu_nax6 as N6
from tierglu_nax import rowsum64

OUT = os.path.expanduser("~/Leap/data/n7_occ.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
LAYER, S, NP, PLAN = 20, 16, 4, "4433333333"
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
say(f"N7-3 kernel A occupancy {time.strftime('%a %H:%M')} ({os.environ.get('N7_COND', 'shared GPU, rough')}); layer {LAYER}, S {S}")
DECL = "    threadgroup float sb_all[2 * KS_C * 2 * 64 * GS_C];\n"
assert N6._A4.count(DECL) == 1
TAIL = "    threadgroup_barrier(mem_flags::mem_threadgroup);               // every simdgroup is done with its code buffers\n"
assert N6._A4.count(TAIL) == 1
def variant(pad_kb):
    s = N6._A4
    if pad_kb:
        n = pad_kb * 256                                              # floats
        s = s.replace(DECL, DECL + f"    threadgroup float pad_tg[{n}];\n    pad_tg[(lane + 32 * sg) % {n}] = 0.0f;\n")
        s = s.replace(TAIL, TAIL + f"    if (nseg[0] < 0) h[0] = static_cast<T>(pad_tg[lane % {n}]);\n")
    return mx.fast.metal_kernel(
        name=f"n7occ_a_{pad_kb}",
        input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["h"], source=s, header=N6._HDR4)
PADS = (0, 4, 8, 12)
KERN = {p: variant(p) for p in PADS}
m4, m2, r4 = plan_masks(PLAN)
R = len(PLAN)
bb = {p: msb[p]["b3"] for p in PROJ}
g_, u_ = msb["gate_proj"], msb["up_proj"]
def ka(kern, x, xs, ms, ls, grp):
    pidx, seg, nsg_, nseg = grp
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0),
            ("GS_C", 1), ("K_C", 2560), ("N_C", 640), ("KS_C", 4)]
    return kern(inputs=[x, xs, g_["w3"], u_["w3"], g_["s3"], u_["s3"], g_["b3"], u_["b3"], bb["gate_proj"], bb["up_proj"],
                        lsb["gate_proj"]["l1"], lsb["up_proj"]["l1"], lsb["gate_proj"]["b"], lsb["up_proj"]["b"], ms, ls,
                        pidx, seg, nsg_],
                template=tmpl, grid=(640 // 32 * 64 * 4, nseg, 1), threadgroup=(256, 1, 1),
                output_shapes=[(x.shape[0] * R, 640)], output_dtypes=[x.dtype])[0]
acc = {p: [] for p in PADS}
same = 0
mx.random.seed(0)
n = 0
for trace in ("code", "chat", "math", "bulgarian"):
    tr = np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{trace}_800_g.npz"))[f"L{LAYER}"].astype(np.int32)
    for t in np.linspace(0, tr.shape[0] - S, NP).astype(int):
        ids = tr[t:t + S, :10].copy()
        ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
        x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
        grp = N6.gpu_grp(ms_, ls_, PLAN)
        xs = rowsum64(x)
        ms, ls = ms_.astype(mx.int32).reshape(-1), ls_.astype(mx.int32).reshape(-1)
        mx.eval(grp[:3], xs, ms, ls)
        fns = {p: (lambda k=KERN[p]: ka(k, x, xs, ms, ls, grp)) for p in PADS}
        outs = [f() for f in fns.values()]
        mx.eval(outs)
        same += all(bool(mx.all(o.view(mx.uint16) == outs[0].view(mx.uint16)).item()) for o in outs[1:])
        n += 1
        for f in fns.values():
            mx.eval(*[f() for _ in range(8)])
        ts = {p: [] for p in fns}
        for _ in range(9):
            for p, f in fns.items():
                mx.synchronize()
                t1 = time.perf_counter()
                mx.eval(*[f() for _ in range(8)])
                ts[p].append((time.perf_counter() - t1) / 8)
        for p, v in ts.items():
            acc[p].append(float(np.median(v)))
say(f"  bits: every pad arm == pad 0 in {same} / {n} passes")
for p in PADS:
    say(f"  threadgroup memory {20 + p} KB: kernel A {1e3 * np.mean(acc[p]):.3f} ms ({100 * (np.mean(acc[p]) / np.mean(acc[0]) - 1):+.1f} %)")
