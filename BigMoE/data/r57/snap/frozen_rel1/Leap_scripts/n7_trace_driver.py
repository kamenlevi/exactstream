"""N7-0b driver (Leap, Mon 2026-11-02 21:4x): a short GPU workload for an Instruments 'Metal GPU Counters' trace of NAX6 at
S 16 (layer 20 real pools, one real routing pass from the code trace): 1 s idle, then 150 chained NAX6 calls (gpu_grp,
rowsum64, kernel A, rowsum64, kernel B, combine) with an eval every 10, 1 s idle, then 150 x 3c (tier_glu with scores)
for comparison. Run under: xcrun xctrace record --template 'Metal System Trace' --instrument 'Metal GPU Counters'
--output <trace> --launch -- <venv python> n7_trace_driver.py"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_scripts"))
from tierglu import PROJ, tier_glu
import tierglu as TG
import tierglu_nax6 as N6

LAYER, S, PLAN = 20, 16, "4433333333"
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
tr = np.load(os.path.expanduser("~/Breakthrough/data/expA/traces125/code_800_g.npz"))[f"L{LAYER}"].astype(np.int32)
ids = tr[200:200 + S, :10].copy()
ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
mx.random.seed(0)
x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
sc = mx.softmax(mx.random.normal((S, 10)), axis=-1).astype(mx.bfloat16)
one = mx.ones((S, 10), mx.int32)
mx.eval(x, sc, one)
def nax6():
    y = N6.tier_glu_nax6(x, msb, lsb, ms_, ls_, plan=PLAN, grp=N6.gpu_grp(ms_, ls_, PLAN)).reshape(S * 10, -1)
    return TG.combine_phase(y, y, one, sc)
def c3():
    return tier_glu(x, msb, lsb, ms_, ls_, plan=PLAN, scores=sc)
for f in (nax6, c3):
    mx.eval([f() for _ in range(5)])
mx.synchronize()
print(f"groups {int(N6.gpu_grp(ms_, ls_, PLAN)[2][0].item())}, distinct {len(np.unique(ids))}", flush=True)
for name, f in (("nax6", nax6), ("3c", c3)):
    time.sleep(1.0)
    t0 = time.perf_counter()
    for i in range(15):
        mx.eval([f() for _ in range(10)])
    mx.synchronize()
    print(f"{name}: {1e3 * (time.perf_counter() - t0) / 150:.3f} ms per call ({time.strftime('%H:%M:%S')})", flush=True)
time.sleep(0.5)
