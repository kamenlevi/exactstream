"""N7-0, NAX6's time by dispatch (Leap, Mon 2026-11-02 21:4x; before the NAX7 build, Fable mission 27 design A; orchestrator
21:2x): one real 125B layer (layer 20, BigMoE records3 pools of all 512 experts, k62f_real.py's loader) with real routing
(Breakthrough's expA traces code / chat / math / bulgarian, S consecutive tokens' top-10, plan 4433333333). Per pass, the
engine's C1 call (ref2._nax6_glu with scores: gpu_grp, rowsum64(x), kernel A, rowsum64(h), kernel B, combine_phase) as a
whole, and each of its six dispatches alone on pre-evaluated inputs; 8 chained calls per eval, median of 9 interleaved
rounds. Also groups per call, the op-set time at M27-P's measured 16 x 32 x 32 rate (4,800 ops per group at 0.86 G op/s)
and the byte time (distinct experts' records at 284 GB/s).
Usage: [BigMoE venv] n7_breakdown.py [layer=20] [S=16] [passes=6] -> ~/Leap/data/n7_breakdown.txt"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_scripts"))
from tierglu import PROJ, plan_masks
import tierglu as TG
import tierglu_nax6 as N6
from tierglu_nax import rowsum64

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SS = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "16").split(",")]
NP = int(sys.argv[3]) if len(sys.argv) > 3 else 6
PLAN = "4433333333"
OUT = os.path.expanduser("~/Leap/data/n7_breakdown.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
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
MSB_REC = sum(int(np.prod(v.shape[1:])) * (4 if v.dtype == mx.uint32 else 2) for pr in msb for v in msb[pr].values())
LSB_REC = sum(int(np.prod(v.shape[1:])) * (4 if v.dtype == mx.uint32 else 2) for pr in lsb for v in lsb[pr].values())
say(f"N7-0 NAX6 breakdown {time.strftime('%a %H:%M')} ({os.environ.get('N7_COND', 'shared GPU, rough')}); layer {LAYER}: "
    f"{E} experts in {time.time() - t0:.1f}s; record MSB {MSB_REC / 1e6:.2f} MB + LSB {LSB_REC / 1e6:.2f} MB; plan {PLAN}")
m4, m2, r4 = plan_masks(PLAN)
R = len(PLAN)
bb = {p: msb[p]["b3"] for p in PROJ}
g_, u_, d_ = (msb[p] for p in PROJ)

def pieces(x, ms_, ls_, sc):
    """the six dispatches of ref2._nax6_glu(scores=...), each as a closure on the previous pieces' (evaluated) outputs"""
    M = x.shape[0]
    ms = ms_.astype(mx.int32).reshape(-1)
    ls = ls_.astype(mx.int32).reshape(-1)
    grp = N6.gpu_grp(ms_, ls_, PLAN)
    pidx, seg, nsg_, nseg = grp
    xs = rowsum64(x)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0), ("GS_C", 1)]
    def ka(xs_, pidx_, seg_, nsg__):
        return N6._ka4()(inputs=[x, xs_, g_["w3"], u_["w3"], g_["s3"], u_["s3"], g_["b3"], u_["b3"], bb["gate_proj"],
                                 bb["up_proj"], lsb["gate_proj"]["l1"], lsb["up_proj"]["l1"], lsb["gate_proj"]["b"],
                                 lsb["up_proj"]["b"], ms, ls, pidx_, seg_, nsg__],
                         template=tmpl + [("K_C", 2560), ("N_C", 640), ("KS_C", 4)],
                         grid=(640 // 32 * 64 * 4, nseg, 1), threadgroup=(64 * 4, 1, 1),
                         output_shapes=[(M * R, 640)], output_dtypes=[x.dtype])[0]
    def kb(h_, hs_, pidx_, seg_, nsg__):
        return N6._kb4()(inputs=[h_, hs_, d_["w3"], d_["s3"], d_["b3"], bb["down_proj"], lsb["down_proj"]["l1"],
                                 lsb["down_proj"]["b"], ms, ls, pidx_, seg_, nsg__],
                         template=tmpl + [("N_C", 640), ("D_C", 2560), ("NSG_C", 2), ("KS_C", 2)],
                         grid=(2560 // 64 * 32 * 2 * 2, nseg, 1), threadgroup=(32 * 2 * 2, 1, 1),
                         output_shapes=[(M * R, 2560)], output_dtypes=[x.dtype])[0]
    h = ka(xs, pidx, seg, nsg_)
    hs = rowsum64(h)
    y = kb(h, hs, pidx, seg, nsg_)
    out = TG.combine_phase(y, y, mx.ones((M, R), mx.int32), sc)
    mx.eval(pidx, seg, nsg_, xs, h, hs, y, out)
    full = lambda: TG.combine_phase(*(lambda y_: (y_, y_))(N6.tier_glu_nax6(x, msb, lsb, ms_, ls_, plan=PLAN,
                                                                             grp=N6.gpu_grp(ms_, ls_, PLAN)).reshape(M * R, -1)),
                                    mx.ones((M, R), mx.int32), sc)
    return nsg_, out, {
        "full call": full,
        "gpu_grp": lambda: N6.gpu_grp(ms_, ls_, PLAN)[:3],
        "rowsum64(x)": lambda: rowsum64(x),
        "kernel A": lambda: ka(xs, pidx, seg, nsg_),
        "rowsum64(h)": lambda: rowsum64(h),
        "kernel B": lambda: kb(h, hs, pidx, seg, nsg_),
        "combine": lambda: TG.combine_phase(y, y, mx.ones((M, R), mx.int32), sc)}

mx.random.seed(0)
for S in SS:
    acc = {}
    ngs, nds = [], []
    for trace in ("code", "chat", "math", "bulgarian"):
        tr = np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{trace}_800_g.npz"))[f"L{LAYER}"].astype(np.int32)
        for t in np.linspace(0, tr.shape[0] - S, NP).astype(int):
            ids = tr[t:t + S, :10].copy()
            ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
            x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
            sc = mx.softmax(mx.random.normal((S, R)), axis=-1).astype(mx.bfloat16)
            nsg_, out, P = pieces(x, ms_, ls_, sc)
            ok = bool(mx.all(out.view(mx.uint16) == P["full call"]().view(mx.uint16)).item())
            assert ok, "the pieces do not reproduce the full call"
            ngs.append(int(nsg_[0].item()))
            nds.append(len(np.unique(ids)))
            for f in P.values():
                mx.eval(*[f() for _ in range(8)])
            ts = {k: [] for k in P}
            for _ in range(9):
                for k, f in P.items():
                    mx.synchronize()
                    t1 = time.perf_counter()
                    mx.eval(*[f() for _ in range(8)])
                    ts[k].append((time.perf_counter() - t1) / 8)
            for k, v in ts.items():
                acc.setdefault(k, []).append(float(np.median(v)))
    G, Dn = float(np.mean(ngs)), float(np.mean(nds))
    say(f"S {S}: {len(ngs)} passes; groups {G:.1f}, distinct experts {Dn:.1f}; pieces reproduce the full call bitwise")
    for k, v in acc.items():
        say(f"  {k:12s} {1e3 * np.mean(v):.3f} ms per call (per pass {1e3 * min(v):.3f}-{1e3 * max(v):.3f})")
    parts = sum(np.mean(acc[k]) for k in acc if k != "full call")
    say(f"  sum of the six alone {1e3 * parts:.3f} ms vs the full call {1e3 * np.mean(acc['full call']):.3f}; op set at 0.86 G "
        f"op/s {G * 4800 / 0.86e9 * 1e3:.3f} ms; bytes {Dn * MSB_REC / 284e9 * 1e3:.3f} ms (MSB of the distinct experts) + "
        f"LSB of the 4-bit ones")
