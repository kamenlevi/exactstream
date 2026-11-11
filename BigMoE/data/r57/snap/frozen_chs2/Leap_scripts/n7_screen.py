"""N7-2 screen (Leap, Mon 2026-11-02 21:4x; pre-registered in ~/Leap/TABLE_350.md 'N7-2'): NAX7's pipelined kernel A
(tierglu_nax7.py) vs NAX6 on the real layer-20 pools with real routing (Breakthrough's expA traces code / chat / math /
bulgarian, plan 4433333333):
1. Bits: 4 traces x S 8 / 16 / 32 x 8 passes = 96 cases: kernel A's h and the whole call's y (tier_glu_nax7 vs
   tier_glu_nax6, both on the GPU grouping) bitwise.
2. Time: kernel A alone (pre-evaluated inputs) and the whole call (+ combine, ref2's C1 form), NAX6 vs NAX7, 8 chained
   calls per eval, median of 9 interleaved rounds, at S 16 (and 8 / 32), 4 passes per trace.
Lines: bits 96 / 96, else BLOCKED; PASS kernel A >= 20 % faster at S 16; KILL < 5 %.
Usage: [BigMoE venv] n7_screen.py -> ~/Leap/data/n7_screen.txt"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs2/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs2/Leap_scripts"))
from tierglu import PROJ, plan_masks
import tierglu as TG
import tierglu_nax6 as N6
import tierglu_nax7 as N7
from tierglu_nax import rowsum64

OUT = os.path.expanduser("~/Leap/data/n7_screen.txt")
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
ARM = os.environ.get("N7_ARM", "p")                     # p = N7-2's pipelined kernel A, ws = N7-5's loader / op simdgroups
if ARM == "ws":
    N7.kernel_a7 = N7.kernel_a7w                         # tier_glu_nax7 calls kernel_a7 by name
TAG = {"p": "N7-2 NAX7 pipelined kernel A", "ws": "N7-5 NAX7-WS kernel A (loader / op simdgroups)"}[ARM]
say(f"{TAG} {time.strftime('%a %H:%M')} ({os.environ.get('N7_COND', 'shared GPU, rough')}); layer {LAYER}")
m4, m2, r4 = plan_masks(PLAN)
R = len(PLAN)
TR = {t: np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{t}_800_g.npz"))[f"L{LAYER}"].astype(np.int32)
      for t in ("code", "chat", "math", "bulgarian")}
eq = lambda a, b: bool(a.shape == b.shape and mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())

def case(trace, S, t):
    ids = TR[trace][t:t + S, :10].copy()
    ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
    x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
    grp = N6.gpu_grp(ms_, ls_, PLAN)
    xs = rowsum64(x)
    ms, ls = ms_.astype(mx.int32).reshape(-1), ls_.astype(mx.int32).reshape(-1)
    mx.eval(x, grp[:3], xs, ms, ls)
    return ids, ms_, ls_, x, grp, xs, ms, ls

def ka6(x, xs, ms, ls, grp):
    pidx, seg, nsg_, nseg = grp
    g, u = msb["gate_proj"], msb["up_proj"]
    bb = {p: msb[p]["b3"] for p in PROJ}
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0), ("GS_C", 1),
            ("K_C", 2560), ("N_C", 640), ("KS_C", 4)]
    return N6._ka4()(inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                             lsb["gate_proj"]["l1"], lsb["up_proj"]["l1"], lsb["gate_proj"]["b"], lsb["up_proj"]["b"], ms, ls,
                             pidx, seg, nsg_],
                     template=tmpl, grid=(640 // 32 * 64 * 4, nseg, 1), threadgroup=(256, 1, 1),
                     output_shapes=[(x.shape[0] * R, 640)], output_dtypes=[x.dtype])[0]

# ---------------------------------------------------------------- 1. bits, 96 cases
mx.random.seed(0)
nh = ny = n = 0
for trace in TR:
    for S in (8, 16, 32):
        for t in np.linspace(0, TR[trace].shape[0] - S, 8).astype(int):
            ids, ms_, ls_, x, grp, xs, ms, ls = case(trace, S, t)
            h6, h7 = ka6(x, xs, ms, ls, grp), N7.kernel_a7(x, xs, msb, lsb, ms, ls, grp, PLAN)
            y6 = N6.tier_glu_nax6(x, msb, lsb, ms_, ls_, plan=PLAN, grp=grp)
            y7 = N7.tier_glu_nax7(x, msb, lsb, ms_, ls_, plan=PLAN, grp=grp)
            mx.eval(h6, h7, y6, y7)
            nh += eq(h6, h7)
            ny += eq(y6, y7)
            n += 1
say(f"1. bits NAX7 == NAX6 (4 traces x S 8 / 16 / 32 x 8 passes): kernel A h {nh} / {n}, whole call y {ny} / {n}")

# ---------------------------------------------------------------- 2. time
one = {}
for S in (16, 8, 32):
    acc = {}
    for trace in TR:
        for t in np.linspace(0, TR[trace].shape[0] - S, 4).astype(int):
            ids, ms_, ls_, x, grp, xs, ms, ls = case(trace, S, t)
            sc = mx.softmax(mx.random.normal((S, R)), axis=-1).astype(mx.bfloat16)
            if S not in one:
                one[S] = mx.ones((S, R), mx.int32)
            o1 = one[S]
            fns = {
                "A NAX6": lambda: ka6(x, xs, ms, ls, grp),
                "A NAX7": lambda: N7.kernel_a7(x, xs, msb, lsb, ms, ls, grp, PLAN),
                "call NAX6": lambda: TG.combine_phase(*(lambda y: (y, y))(N6.tier_glu_nax6(
                    x, msb, lsb, ms_, ls_, plan=PLAN, grp=N6.gpu_grp(ms_, ls_, PLAN)).reshape(S * R, -1)), o1, sc),
                "call NAX7": lambda: TG.combine_phase(*(lambda y: (y, y))(N7.tier_glu_nax7(
                    x, msb, lsb, ms_, ls_, plan=PLAN, grp=N6.gpu_grp(ms_, ls_, PLAN)).reshape(S * R, -1)), o1, sc)}
            for f in fns.values():
                mx.eval(*[f() for _ in range(8)])
            ts = {k: [] for k in fns}
            for _ in range(9):
                for k, f in fns.items():
                    mx.synchronize()
                    t1 = time.perf_counter()
                    mx.eval(*[f() for _ in range(8)])
                    ts[k].append((time.perf_counter() - t1) / 8)
            for k, v in ts.items():
                acc.setdefault(k, []).append(float(np.median(v)))
    m = {k: 1e3 * float(np.mean(v)) for k, v in acc.items()}
    say(f"2. S {S}: kernel A NAX6 {m['A NAX6']:.3f} -> NAX7 {m['A NAX7']:.3f} ms ({100 * (1 - m['A NAX7'] / m['A NAX6']):+.1f} %); "
        f"whole call NAX6 {m['call NAX6']:.3f} -> NAX7 {m['call NAX7']:.3f} ms ({100 * (1 - m['call NAX7'] / m['call NAX6']):+.1f} %)")
    if S == 16:
        gain16 = 1 - m["A NAX7"] / m["A NAX6"]
ok = nh == n and ny == n
v = ("BLOCKED (bits)" if not ok else "PASS" if gain16 >= 0.20 else "KILL" if gain16 < 0.05 else "BETWEEN")
say(f"{TAG.split()[0]} verdict (bits 96 / 96; PASS kernel A >= 20 % faster at S 16, KILL < 5 %): {v} ({100 * gain16:.1f} %)")
