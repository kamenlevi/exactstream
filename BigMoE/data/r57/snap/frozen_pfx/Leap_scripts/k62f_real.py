"""K-62f (Leap, Sun 2026-11-01; shared GPU, rough): NAX4 vs 3c on ONE REAL 125B layer (BigMoE v3 records3 pools: MSB-3 + LSB-1
of all 512 experts in RAM, slot = expert id; loader as MoonShots' wide/microbench.py) with REAL routing (Breakthrough's
expA 125B decode traces: top-20 per token per layer, 800 tokens; a check = S consecutive tokens' top-10, plan 4433333333).
Per pass: tier_glu vs tier_glu_nax4 (groups precomputed), 8 chained calls per eval, median of 7 interleaved rounds.
Also per trace: NAX4 vs 3c max |diff| / RMS, and INV (a pass's first row alone == inside the pass, bitwise).
usage: k62f_real.py [layer=20] [S=16,32] [passes=12] -> ~/Leap/data/k62f_real_L{layer}.txt"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_scripts"))
from tierglu import tier_glu, PROJ
from tierglu_grouped import make_grp
from tierglu_nax4 import tier_glu_nax4

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SS = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "16,32").split(",")]
NP = int(sys.argv[3]) if len(sys.argv) > 3 else 12
PLAN = "4433333333"
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
RD = os.path.expanduser("~/BigMoE/models/bare/records3")
idx = json.load(open(f"{RD}/index.json"))["layers"][str(LAYER)]
E, REC = idx["E"], idx["recsz"]
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
t0 = time.time()
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
say(f"K-62f: MLX {mx.__version__}; layer {LAYER}: {E} experts in {time.time() - t0:.1f}s; plan {PLAN}")

def bits(a):
    return np.array(a.view(mx.uint16)).astype(np.int64)

mx.random.seed(0)
for trace in ("code", "chat", "math", "bulgarian"):
    tr = np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{trace}_800_g.npz"))[f"L{LAYER}"].astype(np.int32)
    for S in SS:
        starts = np.linspace(0, tr.shape[0] - S, NP).astype(int)
        res = []
        for t in starts:
            ids = tr[t:t + S, :10].copy()
            ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
            x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
            grp = make_grp(ms_, ls_, PLAN, S, ids, ids[:, :2], 16)
            fns = [lambda: tier_glu(x, msb, lsb, ms_, ls_, plan=PLAN),
                   lambda: tier_glu_nax4(x, msb, lsb, ms_, ls_, plan=PLAN, grp=grp)]
            for f in fns:
                mx.eval(*[f() for _ in range(8)])
            ts = [[], []]
            for _ in range(7):
                for i, f in enumerate(fns):
                    mx.synchronize(); t1 = time.perf_counter(); mx.eval(*[f() for _ in range(8)]); ts[i].append(time.perf_counter() - t1)
            res.append((len(np.unique(ids)), grp[3], float(np.median(ts[0])) / 8, float(np.median(ts[1])) / 8))
            if t == starts[0]:
                y3, y4 = fns[0](), fns[1]()
                rms = float(mx.sqrt(mx.mean(y3.astype(mx.float32) ** 2)).item())
                dmax = float(mx.max(mx.abs(y3.astype(mx.float32) - y4.astype(mx.float32))).item())
                y1 = tier_glu_nax4(x[:1], msb, lsb, ms_[:1], ls_[:1], plan=PLAN, ms_np=ids[:1], ls_np=ids[:1, :2])
                inv = bool((bits(y1[0]) == bits(y4[0])).all())
        r = np.array(res)
        say(f"{trace:9s} S {S:2d}: distinct {r[:, 0].mean():5.1f}, groups {r[:, 1].mean():5.1f}; tier_glu {1e3 * r[:, 2].mean():.3f} ms, "
            f"NAX4 {1e3 * r[:, 3].mean():.3f} ms -> {r[:, 2].mean() / r[:, 3].mean():.2f}x (per pass {np.min(r[:, 2] / r[:, 3]):.2f}-"
            f"{np.max(r[:, 2] / r[:, 3]):.2f}); NAX4 vs 3c max |diff| / RMS {dmax / rms:.2e}; INV row 0 {'PASS' if inv else 'FAIL'}")
say("K-62f done")
open(os.path.expanduser(f"~/Leap/data/k62f_real_L{LAYER}.txt"), "w").write("\n".join(lines) + "\n")
