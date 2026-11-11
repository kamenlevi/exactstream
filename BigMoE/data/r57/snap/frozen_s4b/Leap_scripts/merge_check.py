"""TIER-MERGE check + screen (Leap, Wed 2026-11-04; TABLE_350 'TIER-MERGE'): tierglu_merge.tier_glu_merge vs NAX6 on ONE
REAL 125B layer (records3 pools of all 512 experts, slot = expert id) with REAL routing (Breakthrough's expA traces; a
pass = S consecutive tokens' top-10, plan 4433333333; the 4-bit ranks' sidecar slot = expert id).
Activations: 'rand' N(0, 0.5) bf16, and 'tap' real hidden states (TrainLab d02a taps, RMS-normalized per row).
 (1) MERGE-X == NAX6 bitwise on every pass (both activation sets);
 (2) MERGE-B vs NAX6: share of bf16 outputs that differ, on 4-bit pairs vs 3-bit pairs, max ulp;
 (3) INV: a pass's first row alone == inside the pass (X and B); the phase form (random split, two calls) == one call;
 (4) ms per call (gpu grouping included), NAX6 / X / B, 8 calls per eval, median of 7 interleaved rounds; groups.
usage: merge_check.py [layer=20] [S=8,16,32] [passes=8] -> ~/Leap/data/merge_check_L{layer}.txt (GPU, MEM ~2.5 GB)"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Leap/third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
from tierglu import PROJ
import tierglu_nax6 as N6
import tierglu_merge as TM

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SS = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "8,16,32").split(",")]
NP = int(sys.argv[3]) if len(sys.argv) > 3 else 8
PLAN = "4433333333"
OUT = os.path.expanduser(f"~/Leap/data/merge_check_L{LAYER}.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nTIER-MERGE check {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}, layer {LAYER}, S {SS}, {NP} passes per trace")
RD = os.path.expanduser("~/BigMoE/models/bare/records3")
assert all(v.get("layout", "row-major") == "row-major" for v in json.load(open(f"{RD}/index.json"))["layers"].values()), \
    "records3 is converted (native+p11+pad, Thu 11-05): this script reads row-major bytes only"   # Leap 18:5x guard
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
tap = np.load(os.path.expanduser("~/TrainLab/data/d02a/agent_bash17.npz"))["f"][:, 2, :].astype(np.float32)
tap = tap / np.sqrt((tap ** 2).mean(1, keepdims=True) + 1e-6)

def bits(a):
    return np.array(a.view(mx.uint16)).astype(np.int64)
def ulp(a, b):
    """bf16 ulp distance per element (same-sign order via the sign-magnitude map)"""
    f = lambda v: np.where(v & 0x8000, -(v & 0x7FFF), v & 0x7FFF)
    return np.abs(f(bits(a)) - f(bits(b)))

n6 = lambda x, ms_, ls_: N6.tier_glu_nax6(x, msb, lsb, ms_, ls_, plan=PLAN, grp=N6.gpu_grp(ms_, ls_, PLAN))
mg = lambda mode: (lambda x, ms_, ls_: TM.tier_glu_merge(x, msb, lsb, ms_, ls_, plan=PLAN, grp=TM.gpu_grp_merge(ms_, ls_, PLAN), mode=mode))
ARMS = {"NAX6": n6, "X": mg(1), "B": mg(2)} if os.environ.get("MERGE_ARMS", "XB") == "XB" else {"NAX6": n6, "X": mg(1), "X2": mg(3)}
KX = list(ARMS)[1:]
rng = np.random.default_rng(4)
mx.random.seed(0)
okX = okINV = okPH = True
dB = {"4": [0, 0], "3": [0, 0]}
umax = 0
for trace in ("code", "chat", "math", "bulgarian"):
    tr = np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{trace}_800_g.npz"))[f"L{LAYER}"].astype(np.int32)
    for S in SS:
        starts = np.linspace(0, tr.shape[0] - S, NP).astype(int)
        res = []
        for t in starts:
            ids = tr[t:t + S, :10].copy()
            ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
            for kind in ("rand", "tap"):
                if kind == "rand":
                    x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
                else:
                    r0 = int(rng.integers(0, tap.shape[0] - S))
                    x = mx.array(tap[r0:r0 + S]).astype(mx.bfloat16)
                ys = {k: f(x, ms_, ls_) for k, f in ARMS.items()}
                mx.eval(*ys.values())
                y6, yX, yB = ys["NAX6"], ys["X"], ys.get("B", ys.get("X2"))
                okX &= bool((bits(yX) == bits(y6)).all()) and ("X2" not in ys or bool((bits(ys["X2"]) == bits(y6)).all()))
                b6, bB = bits(y6).reshape(S, 10, -1), bits(yB).reshape(S, 10, -1)
                for k_, sl in (("4", slice(0, 2)), ("3", slice(2, 10))):
                    dB[k_][0] += int((b6[:, sl] != bB[:, sl]).sum()); dB[k_][1] += b6[:, sl].size
                umax = max(umax, int(ulp(yB, y6).max()))
                for mode in ((1, 2) if "B" in ys else (1, 3)):     # INV: row 0 alone == inside the pass
                    y1 = TM.tier_glu_merge(x[:1], msb, lsb, ms_[:1], ls_[:1], plan=PLAN, mode=mode)
                    okINV &= bool((bits(y1[0]) == bits((yX if mode == 1 else yB)[0])).all())
                ph = mx.array(rng.integers(1, 3, (S, 10)).astype(np.int32))   # the phase form: two calls == one
                for mode, yref in ((1, yX), (2 if "B" in ys else 3, yB)):
                    ya = TM.tier_glu_merge(x, msb, lsb, ms_, ls_, plan=PLAN, mode=mode, grp=TM.gpu_grp_phase_merge(ms_, ls_, PLAN, ph, 1))
                    yb = TM.tier_glu_merge(x, msb, lsb, ms_, ls_, plan=PLAN, mode=mode, grp=TM.gpu_grp_phase_merge(ms_, ls_, PLAN, ph, 2))
                    yp = mx.where((ph == 1)[..., None], ya, yb)
                    okPH &= bool((bits(yp) == bits(yref)).all())
            g6 = int(N6.gpu_grp(ms_, ls_, PLAN)[2].item()); gm = int(TM.gpu_grp_merge(ms_, ls_, PLAN)[2].item())
            fns = [lambda f=f: f(x, ms_, ls_) for f in ARMS.values()]
            for f in fns:
                mx.eval(*[f() for _ in range(8)])
            ts = [[] for _ in fns]
            for rnd in range(7):
                order = list(range(len(fns))) if rnd % 2 == 0 else list(range(len(fns)))[::-1]
                for i in order:
                    mx.synchronize(); t1 = time.perf_counter(); mx.eval(*[fns[i]() for _ in range(8)]); mx.synchronize()
                    ts[i].append(time.perf_counter() - t1)
            res.append([len(np.unique(ids)), g6, gm] + [float(np.median(v)) / 8 for v in ts])
        r = np.array(res)
        say(f"{trace:9s} S {S:2d}: distinct {r[:, 0].mean():5.1f}, groups NAX6 {r[:, 1].mean():5.1f} merged {r[:, 2].mean():5.1f} "
            f"({100 * (r[:, 2].mean() / r[:, 1].mean() - 1):+.0f} %); ms NAX6 {1e3 * r[:, 3].mean():.3f}, X {1e3 * r[:, 4].mean():.3f} "
            f"({100 * (r[:, 4].mean() / r[:, 3].mean() - 1):+.1f} %), {KX[1]} {1e3 * r[:, 5].mean():.3f} ({100 * (r[:, 5].mean() / r[:, 3].mean() - 1):+.1f} %)")
say(f"(1) {' and '.join(k for k in KX if k != 'B')} == NAX6 bitwise on every pass: {okX}")
if "B" in ARMS: say(f"(2) MERGE-B vs NAX6: bf16 outputs differing on 4-bit pairs {100 * dB['4'][0] / max(dB['4'][1], 1):.4f} % ({dB['4'][0]} of {dB['4'][1]}), "
    f"on 3-bit pairs {100 * dB['3'][0] / max(dB['3'][1], 1):.4f} %; max ulp {umax}")
say(f"(3) INV row 0 ({' and '.join(KX)}): {okINV}; phase form == one call: {okPH}")
say(f"TIER-MERGE check: {'PASS' if okX and okINV and okPH else 'FAIL'} (MERGE-X exactness, INV, phases)")
