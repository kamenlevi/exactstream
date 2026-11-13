"""K-62i (Leap, Sun 2026-11-01; shared GPU, rough): NAX5 (NAX4 with its rowsum dispatches folded in) vs NAX4 vs 3c, both on GPU grouping; NAX5 INV (row 0 alone == inside the pass). Was: NAX4 with the pair grouping on the GPU (tierglu_nax4.gpu_grp, no host
sync; grid sized for P groups, empty ones exit) vs the host grouping (make_grp) on one real 125B layer + real routing:
bitwise equal outputs (a pair's bits do not depend on its group), and the time cost of the GPU grouping.
usage: k62h_gpugrp.py [layer=20]"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_scripts"))
from tierglu import tier_glu, PROJ
from tierglu_grouped import make_grp
from tierglu_nax4 import tier_glu_nax4, gpu_grp
import tierglu_nax5 as N5
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 20
PLAN = "4433333333"
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
mx.random.seed(0)
ok_all = True
for trace in ("code", "chat", "math", "bulgarian"):
    tr = np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{trace}_800_g.npz"))[f"L{LAYER}"].astype(np.int32)
    for S in (1, 3, 16, 32):
        res = []
        for t in np.linspace(0, tr.shape[0] - S, 6).astype(int):
            ids = tr[t:t + S, :10].copy()
            ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
            x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
            fns = [lambda: tier_glu(x, msb, lsb, ms_, ls_, plan=PLAN),
                   lambda: tier_glu_nax4(x, msb, lsb, ms_, ls_, plan=PLAN, grp=gpu_grp(ms_, ls_, PLAN)),
                   lambda: N5.tier_glu_nax5(x, msb, lsb, ms_, ls_, plan=PLAN, grp=N5.gpu_grp(ms_, ls_, PLAN))]
            b = fns[2]()
            y1 = N5.tier_glu_nax5(x[:1], msb, lsb, ms_[:1], ls_[:1], plan=PLAN, grp=N5.gpu_grp(ms_[:1], ls_[:1], PLAN))
            same = bool(mx.all(y1[0].view(mx.uint16) == b[0].view(mx.uint16)).item())
            ok_all &= same
            for f in fns:
                mx.eval(*[f() for _ in range(8)])
            ts = [[], [], []]
            for _ in range(7):
                for i, f in enumerate(fns):
                    mx.synchronize(); t1 = time.perf_counter(); mx.eval(*[f() for _ in range(8)]); ts[i].append(time.perf_counter() - t1)
            res.append([float(np.median(v)) / 8 for v in ts] + [same])
        r = np.array(res, dtype=float)
        print(f"{trace:9s} S {S:2d}: tier_glu {1e3 * r[:, 0].mean():.3f} ms, NAX4 {1e3 * r[:, 1].mean():.3f} "
              f"({r[:, 0].mean() / r[:, 1].mean():.2f}x), NAX5 {1e3 * r[:, 2].mean():.3f} ({r[:, 0].mean() / r[:, 2].mean():.2f}x); "
              f"NAX5 INV row 0: {bool(r[:, 3].all())}", flush=True)
print("NAX5 INV PASS" if ok_all else "NAX5 INV FAIL")
