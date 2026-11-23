"""K-62g (Leap, Sun 2026-11-01): accuracy of NAX4 vs 3c on ONE REAL 125B layer against an fp64 reference from the same
dequantized weights (records3 pools, real routing from Breakthrough's expA traces, one S-row pass per trace, random x).
Per output element: |err| = |y - ref|; reports max |err| / RMS, mean |err| / RMS, the error relative to each element's own
magnitude (median and p99 of |err| / max(|ref|, RMS/64)), and how often NAX4's error exceeds 3c's.
usage: k62g_acc_real.py [layer=20] [S=16]   (CPU numpy fp64 <= 4 threads + a little GPU)"""
import json, os, sys
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4"); os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Leap/third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
from tierglu import tier_glu, PROJ
from tierglu_nax4 import tier_glu_nax4
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 20
S = int(sys.argv[2]) if len(sys.argv) > 2 else 16
PLAN = "4433333333"
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
def bf16(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
_cache = {}
def deq(pr, e, tier):
    key = (pr, e, tier)
    if key not in _cache:
        o, i = SH[pr]
        w3, s3, b3, l1, bl = (a[e] for a in npc[pr])
        q = np.unpackbits(np.ascontiguousarray(w3).reshape(o, -1), axis=1, bitorder="little").reshape(o, i, 3)
        q = (q[..., 0] + 2 * q[..., 1] + 4 * q[..., 2]).astype(np.float64)
        s = np.repeat(bf16(np.ascontiguousarray(s3).reshape(o, -1)), 64, axis=1)
        if tier == 4:
            l = np.unpackbits(np.ascontiguousarray(l1).reshape(o, -1), axis=1, bitorder="little").astype(np.float64)
            b = np.repeat(bf16(np.ascontiguousarray(bl).reshape(o, -1)), 64, axis=1)
            _cache[key] = s * 0.5 * (2 * q + l) + b
        else:
            b = np.repeat(bf16(np.ascontiguousarray(b3).reshape(o, -1)), 64, axis=1)
            _cache[key] = s * q + b
    return _cache[key]
mx.random.seed(0)
print(f"K-62g: layer {LAYER}, S {S}, plan {PLAN}", flush=True)
for trace in ("code", "chat", "math", "bulgarian"):
    tr = np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{trace}_800_g.npz"))[f"L{LAYER}"].astype(np.int32)
    ids = tr[300:300 + S, :10].copy()
    x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
    ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
    y3 = np.array(tier_glu(x, msb, lsb, ms_, ls_, plan=PLAN).astype(mx.float32)).astype(np.float64)
    y4 = np.array(tier_glu_nax4(x, msb, lsb, ms_, ls_, plan=PLAN, ms_np=ids, ls_np=ids[:, :2]).astype(mx.float32)).astype(np.float64)
    xf = np.array(x.astype(mx.float32)).astype(np.float64)
    ref = np.zeros_like(y3)
    for m in range(S):
        for r in range(10):
            t = 4 if PLAN[r] == "4" else 3
            g, u = deq("gate_proj", ids[m, r], t) @ xf[m], deq("up_proj", ids[m, r], t) @ xf[m]
            ref[m, r] = deq("down_proj", ids[m, r], t) @ (g / (1 + np.exp(-g)) * u)
    rms = float(np.sqrt((ref ** 2).mean()))
    e3, e4 = np.abs(y3 - ref), np.abs(y4 - ref)
    rel = lambda e: e / np.maximum(np.abs(ref), rms / 64)
    print(f"{trace:9s}: |ref| max/RMS {np.abs(ref).max() / rms:.1f}; 3c max {e3.max() / rms:.2e} mean {e3.mean() / rms:.2e} "
          f"rel p50 {np.median(rel(e3)):.2e} p99 {np.quantile(rel(e3), 0.99):.2e} | NAX4 max {e4.max() / rms:.2e} mean "
          f"{e4.mean() / rms:.2e} rel p50 {np.median(rel(e4)):.2e} p99 {np.quantile(rel(e4), 0.99):.2e} | NAX4 err > 3c err "
          f"{100 * (e4 > e3).mean():.1f} %, < {100 * (e4 < e3).mean():.1f} %", flush=True)
