"""N7-6 residency (Leap, Mon 2026-11-02 21:5x; NAX7 groundwork): how many NAX6 kernel-A threadgroups does a core run at
once, and what is one threadgroup's latency? Synthetic routings with exactly G one-pair groups (G rows, plan '3', G distinct
experts of the real layer-20 pools), so kernel A launches 20 x G threadgroups (20 column tiles) of identical work;
16 calls in a DEPENDENT chain (each call's x depends on the previous h through mx.depends: no overlap between calls),
ms per call, median of 9. Time flat in G up to R per core, then rising, means R resident threadgroups per core; the
flat level is one threadgroup's latency (10 quant groups per simdgroup). Also the same for the whole NAX6 call (A, B and
the small dispatches) to see the per-call fixed cost.
Usage: [BigMoE venv] n7_resid.py -> ~/Leap/data/n7_resid.txt"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_e5/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_e5/Leap_scripts"))
from tierglu import PROJ, plan_masks
import tierglu_nax6 as N6
from tierglu_nax import rowsum64

OUT = os.path.expanduser("~/Leap/data/n7_resid.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
LAYER, PLAN = 20, "3"
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
say(f"N7-6 kernel A residency {time.strftime('%a %H:%M')} ({os.environ.get('N7_COND', 'shared GPU, rough')}); layer {LAYER}, "
    f"plan '{PLAN}', one-pair groups")
m4, m2, r4 = plan_masks(PLAN)
R = len(PLAN)
bb = {p: msb[p]["b3"] for p in PROJ}
g_, u_ = msb["gate_proj"], msb["up_proj"]
L = {p: {"l1": msb[p]["w3"], "b": msb[p]["b3"]} for p in PROJ}           # unread: no 4-bit rank
def ka(x, xs, ms, ls, grp):
    pidx, seg, nsg_, nseg = grp
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0),
            ("GS_C", 1), ("K_C", 2560), ("N_C", 640), ("KS_C", 4)]
    return N6._ka4()(inputs=[x, xs, g_["w3"], u_["w3"], g_["s3"], u_["s3"], g_["b3"], u_["b3"], bb["gate_proj"], bb["up_proj"],
                             L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls,
                             pidx, seg, nsg_],
                     template=tmpl, grid=(640 // 32 * 64 * 4, nseg, 1), threadgroup=(256, 1, 1),
                     output_shapes=[(x.shape[0] * R, 640)], output_dtypes=[x.dtype])[0]
rng = np.random.default_rng(3)
mx.random.seed(0)
res = []
for G in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 72):
    ids = rng.choice(E, G, replace=False).astype(np.int32).reshape(G, 1)
    ms_ = mx.array(ids)
    ls_ = mx.zeros((G, 1), mx.int32)
    x0 = (mx.random.normal((G, 2560)) * 0.5).astype(mx.bfloat16)
    grp = N6.gpu_grp(ms_, ms_, PLAN)
    ms, ls = ms_.reshape(-1), mx.zeros((1,), mx.int32)
    xs = rowsum64(x0)
    mx.eval(x0, grp[:3], xs, ms)
    assert int(grp[2][0].item()) == G
    def chain_a(n=16):
        h, x = None, x0
        for _ in range(n):
            x = x0 if h is None else mx.depends(x0, h)
            h = ka(x, xs, ms, ls, grp)
        return h
    def chain_call(n=16):
        y, x = None, x0
        for _ in range(n):
            x = x0 if y is None else mx.depends(x0, y)
            y = N6.tier_glu_nax6(x, msb, None, ms_, None, plan=PLAN, grp=N6.gpu_grp(ms_, ms_, PLAN))
        return y
    out = {}
    for name, f in (("A", chain_a), ("call", chain_call)):
        for _ in range(2):
            mx.eval(f())
        ts = []
        for _ in range(9):
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(f())
            ts.append((time.perf_counter() - t0) / 16)
        out[name] = 1e3 * float(np.median(ts))
    res.append((G, out["A"], out["call"]))
    say(f"  G {G:3d} ({20 * G:4d} kernel-A threadgroups, {20 * G / 20:.1f} per core): kernel A {out['A']:.3f} ms, "
        f"whole call {out['call']:.3f} ms per call; per group {1e3 * out['A'] / G:.1f} / {1e3 * out['call'] / G:.1f} us")
