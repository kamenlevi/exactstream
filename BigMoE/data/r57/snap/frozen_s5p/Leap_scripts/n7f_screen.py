"""N7-10 screen (Leap, Mon 2026-11-02 22:1x; pre-registered in ~/Leap/TABLE_350.md "N7-10"): NAX7-F (tier_glu_nax7f, three dispatches) vs NAX6 + combine_phase. Was: N7-9, the ceiling of fusing NAX6's kernels A and B (before building (c), the per-group A + B
fusion with completion counters): in DEPENDENT chains (each call's x depends on the previous output, as layers are) on the
real layer-20 pools with real routing (4 traces x 4 passes, S 16 and 32): the whole C1 call (gpu_grp, rowsum64(x), A,
rowsum64(h), B, combine), kernel A alone, kernel B alone (on a pre-evaluated h / hs), and A -> B with the grouping and
row sums precomputed. call - A - B = what a single fused dispatch could remove at most (the small dispatches, the A -> B
boundary, B's ramp); A -> B - A - B = the boundary alone.
Usage: [BigMoE venv] n7_gap.py -> ~/Leap/data/n7_gap.txt"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5p/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5p/Leap_scripts"))
from tierglu import PROJ, plan_masks
import tierglu as TG
import tierglu_nax6 as N6
import tierglu_nax7 as N7
from tierglu_nax import rowsum64

OUT = os.path.expanduser("~/Leap/data/n7f_screen.txt")
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
say(f"N7-10 NAX7-F three dispatches {time.strftime('%a %H:%M')} ({os.environ.get('N7_COND', 'shared GPU, rough')}); layer {LAYER}")
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
TR = {t: np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{t}_800_g.npz"))[f"L{LAYER}"].astype(np.int32)
      for t in ("code", "chat", "math", "bulgarian")}
def nax6_call(x, ms_, ls_, sc, one):
    y = N6.tier_glu_nax6(x, msb, lsb, ms_, ls_, plan=PLAN, grp=N6.gpu_grp(ms_, ls_, PLAN)).reshape(-1, 2560)
    return TG.combine_phase(y, y, one, sc)
eq = lambda a, b: bool(a.shape == b.shape and mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())
mx.random.seed(0)
nb = n = 0
SS_B = [int(v) for v in os.environ.get("N7F_BITS_S", "8,16,32").split(",")]
for trace in TR:
    for S in SS_B:
        for t in np.linspace(0, TR[trace].shape[0] - S, 8).astype(int):
            ids = TR[trace][t:t + S, :10].copy()
            ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
            x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
            sc = mx.softmax(mx.random.normal((S, R)), axis=-1).astype(mx.bfloat16)
            one = mx.ones((S, R), mx.int32)
            a, b = nax6_call(x, ms_, ls_, sc, one), N7.tier_glu_nax7f(x, msb, lsb, ms_, ls_, sc, plan=PLAN)
            mx.eval(a, b)
            nb += eq(a, b)
            n += 1
say(f"1. bits NAX7-F == NAX6 + combine_phase (4 traces x S 8 / 16 / 32 x 8 passes): {nb} / {n}")
NCH = 16
SS_T = [int(v) for v in os.environ.get("N7F_S", "16,8,32").split(",")]
for S in SS_T:
    acc = {}
    for trace in TR:
        for t in np.linspace(0, TR[trace].shape[0] - S, 4).astype(int):
            ids = TR[trace][t:t + S, :10].copy()
            ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
            x0 = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
            sc = mx.softmax(mx.random.normal((S, R)), axis=-1).astype(mx.bfloat16)
            one = mx.ones((S, R), mx.int32)
            mx.eval(x0, sc, one)
            def chain(step):
                out = None
                for _ in range(NCH):
                    out = step(x0 if out is None else mx.depends(x0, out))
                return out
            fns = {"NAX6": lambda: chain(lambda x: nax6_call(x, ms_, ls_, sc, one)),
                   "NAX7-F": lambda: chain(lambda x: N7.tier_glu_nax7f(x, msb, lsb, ms_, ls_, sc, plan=PLAN))}
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
    say(f"2. S {S}: per call NAX6 {m['NAX6']:.3f} -> NAX7-F {m['NAX7-F']:.3f} ms ({100 * (1 - m['NAX7-F'] / m['NAX6']):+.1f} %)")
    if S == 16:
        g16 = 1 - m["NAX7-F"] / m["NAX6"]
v = "BLOCKED (bits)" if nb != n else "PASS" if g16 >= 0.08 else "KILL" if g16 < 0.03 else "BETWEEN"
say(f"N7-10 verdict (bits {n} / {n}; PASS whole call >= 8 % faster at S 16, KILL < 3 %): {v} ({100 * g16:.1f} %)")
