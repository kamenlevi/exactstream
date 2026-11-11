"""H-1i screen (Leap, Mon 2026-11-02 23:1x; pre-registered in ~/Leap/TABLE_350.md 'H-1i'): NAX6-H, the in-kernel hybrid
(tierglu_nax6h.py: one-pair groups on the SIMD ALUs inside NAX6's kernels), the engine form of H-1c for BigMoE's fastBEh
arm. Real layer-20 pools and routing (Breakthrough's expA traces):
1. Bits: tier_glu_nax6h(hybrid=False) == tier_glu_nax6 (4 traces x S 1 / 8 / 16 / 32 x 4 passes).
2. Quality vs fp64 (K-62g's deq), one S 16 pass per trace: NAX6, NAX6-H, 3c.
3. Speed, the whole call + combine_phase in dependent chains of 16: NAX6, NAX6-H, H-1c's compact split (3c via
   tier_glu_pairs + _kc), at S 16 / 8 / 32 / 4 / 2 / 1.
Lines (FAST, speed): PASS NAX6-H >= +8 % at S 16 vs NAX6; KILL < +3 %.
Usage: [BigMoE venv] h1i_screen.py -> ~/Leap/data/h1i_screen.txt"""
import json, os, sys, time
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4"); os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs3/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs3/Leap_scripts"))
from tierglu import tier_glu, PROJ
import tierglu as TG
import tierglu_nax6 as N6
import tierglu_nax6h as NH

OUT = os.path.expanduser("~/Leap/data/h1i_screen.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
LAYER, PLAN = 20, "4433333333"
RD = os.path.expanduser("~/BigMoE/models/bare/records3")
idx = json.load(open(f"{RD}/index.json"))["layers"][str(LAYER)]
E, REC = idx["E"], idx["recsz"]
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
raw = np.fromfile(f"{RD}/{idx['file']}", dtype=np.uint8).reshape(E, REC)
msb, lsb, npc = {}, {}, {}
for p, pr in enumerate(PROJ):
    o, i = SH[pr]
    ow, os_, ob = idx["msb_piece_off"][3 * p:3 * p + 3]
    lw, lb = idx["lsb_piece_off"][2 * p:2 * p + 2]
    npc[pr] = (raw[:, ow:ow + o * i * 3 // 8], raw[:, os_:os_ + o * i // 32].view(np.uint16), raw[:, ob:ob + o * i // 32].view(np.uint16),
               raw[:, lw:lw + o * i // 8], raw[:, lb:lb + o * i // 32].view(np.uint16))
    msb[pr] = {"w3": mx.array(np.ascontiguousarray(npc[pr][0]).view(np.uint32).reshape(E, o, i * 3 // 32)),
               "s3": mx.array(np.ascontiguousarray(npc[pr][1]).reshape(E, o, i // 64)).view(mx.bfloat16),
               "b3": mx.array(np.ascontiguousarray(npc[pr][2]).reshape(E, o, i // 64)).view(mx.bfloat16)}
    lsb[pr] = {"l1": mx.array(np.ascontiguousarray(npc[pr][3]).view(np.uint32).reshape(E, o, i // 32)),
               "b": mx.array(np.ascontiguousarray(npc[pr][4]).reshape(E, o, i // 64)).view(mx.bfloat16)}
mx.eval([v for d in (msb, lsb) for pr in d for v in d[pr].values()])
say(f"H-1i screen {time.strftime('%a %H:%M')} ({os.environ.get('H1I_COND', 'shared GPU, rough')}); layer {LAYER}, plan {PLAN}")
def bf16(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
def deq(pr, e, tier):
    """K-62g's fp64 dequant of one expert's projection"""
    o, i = SH[pr]
    w3, s3, b3, l1, bl = (a[e] for a in npc[pr])
    q = np.unpackbits(np.ascontiguousarray(w3).reshape(o, -1), axis=1, bitorder="little").reshape(o, i, 3)
    q = (q[..., 0] + 2 * q[..., 1] + 4 * q[..., 2]).astype(np.float64)
    s = np.repeat(bf16(np.ascontiguousarray(s3).reshape(o, -1)), 64, axis=1)
    if tier == 4:
        l = np.unpackbits(np.ascontiguousarray(l1).reshape(o, -1), axis=1, bitorder="little").astype(np.float64)
        b = np.repeat(bf16(np.ascontiguousarray(bl).reshape(o, -1)), 64, axis=1)
        return s * 0.5 * (2 * q + l) + b
    b = np.repeat(bf16(np.ascontiguousarray(b3).reshape(o, -1)), 64, axis=1)
    return s * q + b
TR = {t: np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{t}_800_g.npz"))[f"L{LAYER}"].astype(np.int32)
      for t in ("code", "chat", "math", "bulgarian")}
R = len(PLAN)
eq = lambda a, b: bool(a.shape == b.shape and mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())
mx.random.seed(0)
# ---------------------------------------------------------------- 1. hybrid off == NAX6
nb = n = 0
for trace in TR:
    for S in (1, 8, 16, 32):
        for t in np.linspace(0, TR[trace].shape[0] - S, 4).astype(int):
            ids = TR[trace][t:t + S, :10].copy()
            ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
            x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
            grp = N6.gpu_grp(ms_, ls_, PLAN)
            a = N6.tier_glu_nax6(x, msb, lsb, ms_, ls_, plan=PLAN, grp=grp)
            b = NH.tier_glu_nax6h(x, msb, lsb, ms_, ls_, plan=PLAN, grp=grp, hybrid=False)
            mx.eval(a, b)
            nb += eq(a, b)
            n += 1
say(f"1. bits: NAX6-H with the hybrid off == NAX6: {nb} / {n}")
# ---------------------------------------------------------------- 2. quality vs fp64
S = 16
for trace in TR:
    ids = TR[trace][300:300 + S, :10].copy()
    x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
    ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
    grp = N6.gpu_grp(ms_, ls_, PLAN)
    outs = {"NAX6-H": NH.tier_glu_nax6h(x, msb, lsb, ms_, ls_, plan=PLAN, grp=grp),
            "NAX6": N6.tier_glu_nax6(x, msb, lsb, ms_, ls_, plan=PLAN, grp=grp),
            "3c": tier_glu(x, msb, lsb, ms_, ls_, plan=PLAN)}
    mx.eval(list(outs.values()))
    outs = {k: np.array(v.astype(mx.float32)).astype(np.float64) for k, v in outs.items()}
    xf = np.array(x.astype(mx.float32)).astype(np.float64)
    ref = np.zeros_like(outs["NAX6"])
    for m in range(S):
        for r in range(R):
            t_ = 4 if PLAN[r] == "4" else 3
            g = deq("gate_proj", ids[m, r], t_) @ xf[m]
            u = deq("up_proj", ids[m, r], t_) @ xf[m]
            ref[m, r] = deq("down_proj", ids[m, r], t_) @ (g / (1 + np.exp(-g)) * u)
    rms = float(np.sqrt((ref ** 2).mean()))
    err = {k: np.abs(v - ref) for k, v in outs.items()}
    dif = np.abs(outs["NAX6-H"] - outs["NAX6"])
    say(f"2. {trace:9s}: " + "; ".join(f"{k} max {e.max() / rms:.2e} mean {e.mean() / rms:.2e}" for k, e in err.items())
        + f"; NAX6-H vs NAX6 mean {dif.mean() / rms:.2e} max {dif.max() / rms:.2e}, rows (pairs) differing "
        f"{100 * (dif.reshape(S * R, -1).max(axis=1) > 0).mean():.0f} %")
# ---------------------------------------------------------------- 3. speed
NCH = 16
def keys(ids, ls):
    k = np.zeros(ids.shape, np.int64)
    for r in range(R):
        k[:, r] = (ids[:, r].astype(np.int64) << 11) | (((2 << 9) | (ls[:, r].astype(np.int64) + 1)) if PLAN[r] == "4" else (1 << 9))
    return k.reshape(-1)
def call6(x, ms_, ls_, sc, one, aux):
    y = N6.tier_glu_nax6(x, msb, lsb, ms_, ls_, plan=PLAN, grp=N6.gpu_grp(ms_, ls_, PLAN)).reshape(-1, 2560)
    return TG.combine_phase(y, y, one, sc)
def callh(x, ms_, ls_, sc, one, aux):
    y = NH.tier_glu_nax6h(x, msb, lsb, ms_, ls_, plan=PLAN, grp=N6.gpu_grp(ms_, ls_, PLAN)).reshape(-1, 2560)
    return TG.combine_phase(y, y, one, sc)
def callc(x, ms_, ls_, sc, one, aux):
    ph, small_idx, loc = aux
    y1 = N6.tier_glu_nax6(x, msb, lsb, ms_, ls_, plan=PLAN, grp=N6.gpu_grp_phase(ms_, ls_, PLAN, ph, 1)).reshape(-1, 2560)
    y2 = TG.tier_glu_pairs(x, msb, lsb, ms_, ls_, small_idx, plan=PLAN) if small_idx.size else mx.zeros((1, 2560), x.dtype)
    (out,) = TG._kc()(inputs=[y1, y2, loc, sc.reshape(-1)], template=[("T", x.dtype), ("D_C", 2560), ("NR_C", R), ("LY_C", 8)],
                      grid=(2560, x.shape[0], 1), threadgroup=(256, 1, 1), output_shapes=[(x.shape[0], 2560)],
                      output_dtypes=[x.dtype])
    return out
res = {}
for S in (16, 8, 32, 4, 2, 1):
    acc = {}
    for trace in TR:
        for t in np.linspace(0, TR[trace].shape[0] - S, 4).astype(int):
            ids = TR[trace][t:t + S, :10].copy()
            ls = ids[:, :2].copy()
            ms_, ls_ = mx.array(ids), mx.array(ls)
            x0 = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
            sc = mx.softmax(mx.random.normal((S, R)), axis=-1).astype(mx.bfloat16)
            one = mx.ones((S, R), mx.int32)
            _, inv, cnt = np.unique(keys(ids, ls), return_inverse=True, return_counts=True)
            size = cnt[inv]
            sm = np.nonzero(size <= 1)[0].astype(np.int32)
            loc = np.arange(S * R, dtype=np.int32)
            loc[sm] = -(np.arange(sm.size) + 1)
            aux = (mx.array(np.where(size <= 1, 2, 1).astype(np.int32)), mx.array(sm), mx.array(loc))
            mx.eval(x0, sc, one, *aux)
            def chain(f):
                out = None
                for _ in range(NCH):
                    out = f(x0 if out is None else mx.depends(x0, out), ms_, ls_, sc, one, aux)
                return out
            fns = {"NAX6": lambda: chain(call6), "NAX6-H": lambda: chain(callh), "H-1c": lambda: chain(callc)}
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
    say(f"3. S {S:2d}: per call NAX6 {m['NAX6']:.3f} ms; NAX6-H {m['NAX6-H']:.3f} ({100 * (1 - m['NAX6-H'] / m['NAX6']):+.1f} %); "
        f"H-1c {m['H-1c']:.3f} ({100 * (1 - m['H-1c'] / m['NAX6']):+.1f} %)")
g16 = 1 - res[16]["NAX6-H"] / res[16]["NAX6"]
v = "PASS" if g16 >= 0.08 else "KILL" if g16 < 0.03 else "BETWEEN"
say(f"H-1i verdict (FAST speed: PASS NAX6-H >= +8 % vs NAX6 at S 16, KILL < +3 %): {v} ({100 * g16:+.1f} %)")
