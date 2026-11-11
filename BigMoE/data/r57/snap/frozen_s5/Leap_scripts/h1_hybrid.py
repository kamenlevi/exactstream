"""H-1, the hybrid on the FAST track (Leap, Mon 2026-11-02 22:0x; pre-registered in ~/Leap/TABLE_350.md 'H-1'; Fable
mission 27 design H, orchestrator 22:0x under Kamen's two-track rule): NAX6 pays its full 16-row op set per (expert, tier)
group; 3c pays per pair on the SIMD ALUs. Route the small groups' pairs to 3c and the rest to NAX6, then the LY = 8
combine:
  NAX6  all pairs on NAX6 (ref2's C1 call: tier_glu_nax6 on the GPU grouping + combine_phase)
  H1    pairs in one-pair groups -> 3c (tierglu.tier_glu_phase), the rest -> NAX6 (gpu_grp_phase), combine_phase
  H2    the same with groups of <= 2 pairs on 3c
The split is decided on the host here (group sizes from the same key NAX6's grouping uses); the engine would split on the
GPU. Real layer-20 pools, real routing (4 traces), S 16 / 8 / 32, 4 passes per trace; each arm as a DEPENDENT chain of 8
calls (x of call i + 1 depends on call i's output, as layers are), median of 9 interleaved rounds. Also: the rows whose
output differs from NAX6's (bits) and the max relative difference. FAST track: speed here, quality by EvalSuite later.
Lines (S 16): PASS H1 or H2 >= 15 % faster than NAX6; KILL < 5 %.
Usage: [BigMoE venv] h1_hybrid.py -> ~/Leap/data/h1_hybrid.txt"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/MoonShots/patches/planes_engine"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5/Leap_scripts"))
import tierglu as TG
from tierglu import PROJ
import tierglu_nax6 as N6

OUT = os.path.expanduser("~/Leap/data/h1_hybrid.txt")
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
say(f"H-1 hybrid (FAST track) {time.strftime('%a %H:%M')} ({os.environ.get('H1_COND', 'shared GPU, rough')}); layer {LAYER}; "
    f"3c from {TG.__file__}")
R = len(PLAN)
R4 = PLAN.count("4")
TR = {t: np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{t}_800_g.npz"))[f"L{LAYER}"].astype(np.int32)
      for t in ("code", "chat", "math", "bulgarian")}

def keys(ids, ls):
    """NAX6's grouping key per pair (_GRP): slot << 11 | (tier - 2) << 9 | (sidecar slot + 1 for tier 4)"""
    M = ids.shape[0]
    k = np.zeros((M, R), np.int64)
    for r in range(R):
        if PLAN[r] == "4":
            k[:, r] = (ids[:, r].astype(np.int64) << 11) | (2 << 9) | (ls[:, r].astype(np.int64) + 1)
        else:
            k[:, r] = (ids[:, r].astype(np.int64) << 11) | (1 << 9)
    return k.reshape(-1)

def nax6_call(x, ms_, ls_, sc, one):
    y = N6.tier_glu_nax6(x, msb, lsb, ms_, ls_, plan=PLAN, grp=N6.gpu_grp(ms_, ls_, PLAN)).reshape(-1, 2560)
    return TG.combine_phase(y, y, one, sc)

def hybrid_call(x, ms_, ls_, sc, ph):
    y1 = N6.tier_glu_nax6(x, msb, lsb, ms_, ls_, plan=PLAN, grp=N6.gpu_grp_phase(ms_, ls_, PLAN, ph, 1)).reshape(-1, 2560)
    y2 = TG.tier_glu_phase(x, msb, lsb, ms_, ls_, ph, 2, plan=PLAN).reshape(-1, 2560)
    return TG.combine_phase(y1, y2, ph, sc)

def hybrid_compact(x, ms_, ls_, sc, ph, small_idx, loc):
    """best case of the split: 3c on a compact list of the small groups' pairs (tier_glu_pairs) and the OOO split's combine
    (_kc: each pair's row from NAX6's full y or 3c's compact y)"""
    y1 = N6.tier_glu_nax6(x, msb, lsb, ms_, ls_, plan=PLAN, grp=N6.gpu_grp_phase(ms_, ls_, PLAN, ph, 1)).reshape(-1, 2560)
    y2 = TG.tier_glu_pairs(x, msb, lsb, ms_, ls_, small_idx, plan=PLAN)
    (out,) = TG._kc()(inputs=[y1, y2, loc, sc.reshape(-1)], template=[("T", x.dtype), ("D_C", 2560), ("NR_C", R), ("LY_C", 8)],
                      grid=(2560, x.shape[0], 1), threadgroup=(256, 1, 1), output_shapes=[(x.shape[0], 2560)],
                      output_dtypes=[x.dtype])
    return out

def chain(f, x0, n=8):
    out, x = None, x0
    for _ in range(n):
        x = x0 if out is None else mx.depends(x0, out)
        out = f(x)
    return out

mx.random.seed(0)
for S in (16, 8, 32):
    acc = {}
    stats = {"H1": [], "H2": []}
    small = {"H1": [], "H2": []}
    for trace in TR:
        for t in np.linspace(0, TR[trace].shape[0] - S, 4).astype(int):
            ids = TR[trace][t:t + S, :10].copy()
            ls = ids[:, :R4].copy()
            ms_, ls_ = mx.array(ids), mx.array(ls)
            x0 = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
            sc = mx.softmax(mx.random.normal((S, R)), axis=-1).astype(mx.bfloat16)
            one = mx.ones((S, R), mx.int32)
            k = keys(ids, ls)
            _, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
            size = cnt[inv]                                          # pairs in each pair's group
            ph = {"H1": mx.array(np.where(size <= 1, 2, 1).astype(np.int32)),
                  "H2": mx.array(np.where(size <= 2, 2, 1).astype(np.int32))}
            mx.eval(x0, sc, one, ph["H1"], ph["H2"])
            cmp = {}
            for a, thr in (("H1", 1), ("H2", 2)):
                sm = np.nonzero(size <= thr)[0].astype(np.int32)
                loc = np.arange(S * R, dtype=np.int32)
                loc[sm] = -(np.arange(sm.size) + 1)
                cmp[a] = (mx.array(sm), mx.array(loc))
            mx.eval([v for t_ in cmp.values() for v in t_])
            ref = nax6_call(x0, ms_, ls_, sc, one)
            for a in ("H1", "H2"):
                yh = hybrid_call(x0, ms_, ls_, sc, ph[a])
                mx.eval(ref, yh)
                rb = np.array(ref.view(mx.uint16)) != np.array(yh.view(mx.uint16))
                rf = np.array(ref.astype(mx.float32))
                d = np.abs(np.array(yh.astype(mx.float32)) - rf)
                stats[a].append((rb.any(axis=1).mean(), float(d.max() / (np.abs(rf).max() + 1e-30))))
                small[a].append(float((np.array(ph[a]) == 2).mean()))
            fns = {"NAX6": lambda: chain(lambda x: nax6_call(x, ms_, ls_, sc, one), x0),
                   "H1": lambda: chain(lambda x: hybrid_call(x, ms_, ls_, sc, ph["H1"]), x0),
                   "H2": lambda: chain(lambda x: hybrid_call(x, ms_, ls_, sc, ph["H2"]), x0),
                   "H1c": lambda: chain(lambda x: hybrid_compact(x, ms_, ls_, sc, ph["H1"], *cmp["H1"]), x0),
                   "H2c": lambda: chain(lambda x: hybrid_compact(x, ms_, ls_, sc, ph["H2"], *cmp["H2"]), x0)}
            for f in fns.values():
                mx.eval(f())
            ts = {kk: [] for kk in fns}
            for _ in range(9):
                for kk, f in fns.items():
                    mx.synchronize()
                    t1 = time.perf_counter()
                    mx.eval(f())
                    ts[kk].append((time.perf_counter() - t1) / 8)
            for kk, v in ts.items():
                acc.setdefault(kk, []).append(float(np.median(v)))
    m = {kk: 1e3 * float(np.mean(v)) for kk, v in acc.items()}
    say(f"S {S}: per call NAX6 {m['NAX6']:.3f} ms; H1 {m['H1']:.3f} ({100 * (1 - m['H1'] / m['NAX6']):+.1f} %), "
        f"H2 {m['H2']:.3f} ({100 * (1 - m['H2'] / m['NAX6']):+.1f} %); pairs on 3c H1 {100 * np.mean(small['H1']):.0f} %, "
        f"H2 {100 * np.mean(small['H2']):.0f} %; rows differing from NAX6 H1 {100 * np.mean([s[0] for s in stats['H1']]):.0f} %, "
        f"H2 {100 * np.mean([s[0] for s in stats['H2']]):.0f} %; max |diff| / max |y| H1 {max(s[1] for s in stats['H1']):.1e}, "
        f"H2 {max(s[1] for s in stats['H2']):.1e}; compact 3c: H1c {m['H1c']:.3f} ({100 * (1 - m['H1c'] / m['NAX6']):+.1f} %), "
        f"H2c {m['H2c']:.3f} ({100 * (1 - m['H2c'] / m['NAX6']):+.1f} %)")
    if S == 16:
        g16 = max(1 - m[a] / m["NAX6"] for a in ("H1", "H2", "H1c", "H2c"))
v = "PASS" if g16 >= 0.15 else "KILL" if g16 < 0.05 else "BETWEEN"
say(f"H-1 verdict (S 16: PASS >= 15 % faster than NAX6, KILL < 5 %; FAST track, quality later): {v} ({100 * g16:.1f} %)")
