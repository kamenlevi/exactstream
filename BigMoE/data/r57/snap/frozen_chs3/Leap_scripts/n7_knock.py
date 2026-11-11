"""N7-1 knockouts (Leap, Mon 2026-11-02 21:4x; NAX7 groundwork, Fable mission 27 design A): what is on NAX6 kernel A's critical
path at S 16? NAX6's kernel A source (tierglu_nax6._A4 / _CORE4, unchanged otherwise) with one piece knocked out per arm,
on the real layer-20 pools with real routing (Breakthrough's expA traces, S 16, plan 4433333333), kernel A alone on
pre-evaluated inputs, 8 chained calls per eval, median of 9 interleaved rounds:
  full    NAX6 kernel A as shipped (DIAG 0)
  nodram  every lane reads row n0's codes (DIAG 3: the same instructions, ~1 line per load instead of 32)
  noops   op.run replaced by a fold of the A operand into cg (the A loads stay live; no matrix-unit work)
  noA     the A operand built from constants (no device loads of x; ops stay)
  nodec   the codes stored without spread3 / spread8 (raw words; loads stay)
Reading: noops ~ full -> the matrix unit is hidden under the other work; noops << full -> the ops are on the path, and
(full - noops) is their exposed share. Not bitwise (knockouts), timing only.
Usage: [BigMoE venv] n7_knock.py -> ~/Leap/data/n7_knock.txt"""
import json, os, sys, time
from functools import lru_cache
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs3/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs3/Leap_scripts"))
from tierglu import PROJ, plan_masks
import tierglu_nax6 as N6
from tierglu_nax import rowsum64

OUT = os.path.expanduser("~/Leap/data/n7_knock.txt")
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
say(f"N7-1 NAX6 kernel A knockouts {time.strftime('%a %H:%M')} ({os.environ.get('N7_COND', 'shared GPU, rough')}); layer {LAYER}, S {S}")

OPS = "                op.run(ca, tbt, cg);\n"
A_LD = ["                const vec<T, 4> a00 = *(const device vec<T, 4>*)(ar0 + k0 + fn0);\n",
        "                const vec<T, 4> a10 = *(const device vec<T, 4>*)(ar1 + k0 + fn0);\n",
        "                const vec<T, 4> a01 = *(const device vec<T, 4>*)(ar0 + k0 + 16 + fn0);\n",
        "                const vec<T, 4> a11 = *(const device vec<T, 4>*)(ar1 + k0 + 16 + fn0);\n"]
DEC3 = "                for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]);\n"
DEC4 = "                for (int q = 0; q < 8; ++q) nb[q] = spread8(ch[q], ((q < 4 ? l0 : l1) >> (8 * (q & 3))) & 0xFFu);\n"
DEC2 = "                for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]) & 0x66666666u;\n"
for t_ in [OPS, DEC3, DEC4, DEC2] + A_LD:
    assert N6._A4.count(t_) == 1, t_
def variant(kind):
    s = N6._A4
    if kind == "noops":
        s = s.replace(OPS, "                for (int i = 0; i < 16; ++i) cg[i] += float(ca[i]) * zf;\n")
        s = s.replace("    const uint pj = sg & 1u, kc = sg >> 1;\n",
                      "    const uint pj = sg & 1u, kc = sg >> 1;\n    const float zf = (float)(nseg[0] < 0);\n")
    elif kind == "noA":
        for j, t_ in enumerate(A_LD):
            nm = t_.split("const vec<T, 4> ")[1].split(" =")[0]
            s = s.replace(t_, f"                const vec<T, 4> {nm} = vec<T, 4>(T(0.001f * (k0 + {j} + fn0)));\n")
    elif kind == "nodec":
        s = s.replace(DEC3, "                for (int q = 0; q < 8; ++q) nb[q] = ch[q];\n")
        s = s.replace(DEC4, "                for (int q = 0; q < 8; ++q) nb[q] = ch[q] ^ l0;\n")
        s = s.replace(DEC2, "                for (int q = 0; q < 8; ++q) nb[q] = ch[q] & 0x66666666u;\n")
    return mx.fast.metal_kernel(
        name=f"n7k_a_{kind}",
        input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["h"], source=s, header=N6._HDR4)
KERN = {k: variant(k) for k in ("full", "noops", "noA", "nodec")}
m4, m2, r4 = plan_masks(PLAN)
R = len(PLAN)
bb = {p: msb[p]["b3"] for p in PROJ}
g_, u_ = msb["gate_proj"], msb["up_proj"]
def ka(kern, diag, x, xs, ms, ls, grp):
    pidx, seg, nsg_, nseg = grp
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", diag),
            ("GS_C", 1), ("K_C", 2560), ("N_C", 640), ("KS_C", 4)]
    return kern(inputs=[x, xs, g_["w3"], u_["w3"], g_["s3"], u_["s3"], g_["b3"], u_["b3"], bb["gate_proj"], bb["up_proj"],
                        lsb["gate_proj"]["l1"], lsb["up_proj"]["l1"], lsb["gate_proj"]["b"], lsb["up_proj"]["b"], ms, ls,
                        pidx, seg, nsg_],
                template=tmpl, grid=(640 // 32 * 64 * 4, nseg, 1), threadgroup=(256, 1, 1),
                output_shapes=[(x.shape[0] * R, 640)], output_dtypes=[x.dtype])[0]
ARMS = {"full": ("full", 0), "nodram": ("full", 3), "noops": ("noops", 0), "noA": ("noA", 0), "nodec": ("nodec", 0)}
acc = {a: [] for a in ARMS}
mx.random.seed(0)
ngs = []
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
        ngs.append(int(grp[2][0].item()))
        fns = {a: (lambda k=k, d=d: ka(KERN[k], d, x, xs, ms, ls, grp)) for a, (k, d) in ARMS.items()}
        for f in fns.values():
            mx.eval(*[f() for _ in range(8)])
        ts = {a: [] for a in fns}
        for _ in range(9):
            for a, f in fns.items():
                mx.synchronize()
                t1 = time.perf_counter()
                mx.eval(*[f() for _ in range(8)])
                ts[a].append((time.perf_counter() - t1) / 8)
        for a, v in ts.items():
            acc[a].append(float(np.median(v)))
G = float(np.mean(ngs))
say(f"  {len(ngs)} passes, {G:.1f} groups per call; kernel A's op set at 0.86 G op/s {G * 3200 / 0.86e9 * 1e3:.3f} ms; "
    f"its bytes (gate + up MSB {G / 1.2:.0f} experts x 1.43 MB at 284 GB/s) ~{G / 1.2 * 1.43e6 / 284e9 * 1e3:.3f} ms")
for a, v in acc.items():
    say(f"  {a:7s} {1e3 * np.mean(v):.3f} ms ({100 * (np.mean(v) / np.mean(acc['full']) - 1):+.1f} % vs full)")
