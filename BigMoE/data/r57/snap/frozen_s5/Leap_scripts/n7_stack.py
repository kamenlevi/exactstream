"""N7-14 stack screen (Leap, Mon 2026-11-02 23:3x; TABLE_350 'N7-14'): NAX6 / NAX6-N (native layout) / NAX6-H (hybrid) /
NAX6-NH (both; FAST). Was the N7-13 screen: NAX6-N (tierglu_nax6n.py: the
fragment-native record layout, NAX6's kernels with only the address arithmetic changed) vs NAX6 on the real layer-20 pools
(repacked on the CPU) with real routing: bits (y, 4 traces x S 8 / 16 x 8 passes) and the whole call + combine_phase in
dependent chains of 16, S 16 / 8 / 32, 4 traces x 4 passes, median of 9 interleaved rounds.
Lines: PASS >= +6 % at S 16; KILL < +2 %.
Usage: [BigMoE venv] n7_native.py -> ~/Leap/data/n7_native.txt"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5/Leap_scripts"))
from tierglu import PROJ, plan_masks
import tierglu as TG
import tierglu_nax6 as N6
import tierglu_nax6n as NN
import tierglu_nax6h as NH
from tierglu_nax import rowsum64

OUT = os.path.expanduser("~/Leap/data/n7_stack.txt")
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
say(f"N7-14 NAX6 / -N / -H / -NH stack {time.strftime('%a %H:%M')} ({os.environ.get('N7_COND', 'shared GPU, rough')}); layer {LAYER}")
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
t_r = time.time()
msb_n, lsb_n = NN.repack(msb, lsb)
mx.eval([v for d in (msb_n, lsb_n) for pr in d for v in d[pr].values()])
say(f"  repacked layer {LAYER}'s pools to the native layout in {time.time() - t_r:.1f} s")
TR = {t: np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{t}_800_g.npz"))[f"L{LAYER}"].astype(np.int32)
      for t in ("code", "chat", "math", "bulgarian")}
def mk(fn):
    def f(x, ms_, ls_, sc, one):
        y = fn(x, ms_, ls_).reshape(-1, 2560)
        return TG.combine_phase(y, y, one, sc)
    return f
ARMS = {"NAX6": mk(lambda x, a, b: N6.tier_glu_nax6(x, msb, lsb, a, b, plan=PLAN, grp=N6.gpu_grp(a, b, PLAN))),
        "NAX6-N": mk(lambda x, a, b: NN.tier_glu_nax6n(x, msb_n, lsb_n, a, b, plan=PLAN, grp=N6.gpu_grp(a, b, PLAN))),
        "NAX6-H": mk(lambda x, a, b: NH.tier_glu_nax6h(x, msb, lsb, a, b, plan=PLAN, grp=N6.gpu_grp(a, b, PLAN))),
        "NAX6-NH": mk(lambda x, a, b: NH.tier_glu_nax6nh(x, msb_n, lsb_n, a, b, plan=PLAN, grp=N6.gpu_grp(a, b, PLAN)))}
eq = lambda a, b: bool(a.shape == b.shape and mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())
mx.random.seed(0)
c1 = c2 = n = 0
for trace in TR:
    for S in (1, 8, 16):
        for t in np.linspace(0, TR[trace].shape[0] - S, 4).astype(int):
            ids = TR[trace][t:t + S, :10].copy()
            ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
            x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
            grp = N6.gpu_grp(ms_, ls_, PLAN)
            y6 = N6.tier_glu_nax6(x, msb, lsb, ms_, ls_, plan=PLAN, grp=grp)
            yh = NH.tier_glu_nax6h(x, msb, lsb, ms_, ls_, plan=PLAN, grp=grp)
            ynh0 = NH.tier_glu_nax6nh(x, msb_n, lsb_n, ms_, ls_, plan=PLAN, grp=grp, hybrid=False)
            ynh = NH.tier_glu_nax6nh(x, msb_n, lsb_n, ms_, ls_, plan=PLAN, grp=grp)
            mx.eval(y6, yh, ynh0, ynh)
            c1 += eq(ynh0, y6)
            c2 += eq(ynh, yh)
            n += 1
say(f"1. bits: NAX6-NH hybrid off == NAX6 {c1} / {n}; NAX6-NH == NAX6-H {c2} / {n}")
NCH = 16
res = {}
for S in (16, 8, 32, 1):
    acc = {}
    for trace in TR:
        for t in np.linspace(0, TR[trace].shape[0] - S, 4).astype(int):
            ids = TR[trace][t:t + S, :10].copy()
            ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
            x0 = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
            sc = mx.softmax(mx.random.normal((S, R)), axis=-1).astype(mx.bfloat16)
            one = mx.ones((S, R), mx.int32)
            mx.eval(x0, sc, one)
            def chain(f):
                out = None
                for _ in range(NCH):
                    out = f(x0 if out is None else mx.depends(x0, out), ms_, ls_, sc, one)
                return out
            fns = {k: (lambda f=f: chain(f)) for k, f in ARMS.items()}
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
    res[S] = m
    say(f"2. S {S:2d}: per call " + ", ".join(f"{k} {m[k]:.3f}" + ("" if k == "NAX6" else f" ({100 * (1 - m[k] / m['NAX6']):+.1f} %)")
                                              for k in ARMS))
m = res[16]
g = 1 - m["NAX6-NH"] / m["NAX6"]
v = "PASS" if g >= 0.22 else "KILL" if g < 0.12 else "BETWEEN"
say(f"N7-14 verdict (the orchestrator's: PASS NAX6-NH >= +22 % over NAX6 at S 16, KILL < +12 %): {v} ({100 * g:+.1f} %; "
    f"NAX6-N alone {100 * (1 - m['NAX6-N'] / m['NAX6']):+.1f} %, NAX6-H alone {100 * (1 - m['NAX6-H'] / m['NAX6']):+.1f} %)")
