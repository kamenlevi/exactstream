"""NAX-OCC probe (Leap, Wed 2026-11-04 16:0x; TABLE_350 'NAX-OCC'): is the NAX6 expert call limited by how many threadgroups
stay resident per GPU core? After TIER-MERGE's KILL (merged groups serialize work that NAX6 runs as parallel threadgroups:
the call looks latency-bound per threadgroup).
 (1) kernel A's threadgroup memory padded from 20 KB to 24 / 28 / 32 KB (kernel B 10 KB unchanged) and kernel B's from
     10 KB to 20 / 32 KB (A unchanged): same arithmetic, so the same bits (checked); only residency can change. A call
     that slows as the footprint grows says residency binds, and a smaller footprint (single-buffered codes, split
     chunk sums: the same bits) would pay.
 (2) two independent calls on two GPU streams vs back to back: spare capacity the call leaves idle.
Layer 20's real records, real routing (code / chat traces), S 8 / 16 / 32, random x; ms per call, 8 calls per eval, median
of 9 interleaved rounds.
usage: [BigMoE venv] nax_occ_probe.py -> ~/Leap/data/nax_occ_probe.txt (GPU, MEM ~2.5 GB; share lock)"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs2/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs2/Leap_scripts"))
from tierglu import PROJ, plan_masks
from tierglu_nax import rowsum64
import tierglu_nax6 as N6

OUT = os.path.expanduser("~/Leap/data/nax_occ_probe.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nNAX-OCC probe {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}")
LAYER, PLAN = 20, "4433333333"
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

PADA = "    threadgroup float sb_all[2 * KS_C * 2 * 64 * GS_C];\n"
PADB = "    threadgroup float sb_all[NSG_C * KS_C * 2 * 64 * GS_C];\n"
TOUCH = "    threadgroup float pad_[PAD_C];\n    if (nseg[0] < 0) { pad_[lane] = 1.0f; OUTV[0] = static_cast<T>(pad_[(lane + 1) & 31]); }\n"
assert N6._A4.count(PADA) == 1 and N6._B4.count(PADB) == 1
_KA, _KB = {}, {}
def ka(pad):
    if pad not in _KA:
        _KA[pad] = N6._ka4() if pad == 0 else mx.fast.metal_kernel(
            name=f"nax6_a_pad{pad}", input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g",
                                                  "l1u", "bg", "bu", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
            output_names=["h"], source=N6._A4.replace(PADA, PADA + TOUCH.replace("OUTV", "h")), header=N6._HDR4)
    return _KA[pad]
def kb(pad):
    if pad not in _KB:
        _KB[pad] = N6._kb4() if pad == 0 else mx.fast.metal_kernel(
            name=f"nax6_b_pad{pad}", input_names=["h", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot",
                                                  "pidx", "seg", "nseg"],
            output_names=["y"], source=N6._B4.replace(PADB, PADB + TOUCH.replace("OUTV", "y")), header=N6._HDR4)
    return _KB[pad]

def call(x, ms_, ls_, pa=0, pb=0, ks=4, ks_b=2, nsg_b=2, gs=1):
    """tier_glu_nax6's body with kernel A / B padded by pa / pb floats of threadgroup memory"""
    M, K = x.shape
    R = len(PLAN)
    m4, m2, r4 = plan_masks(PLAN)
    N, D = msb["gate_proj"]["s3"].shape[1], msb["down_proj"]["s3"].shape[1]
    pidx, seg, nsg_, nseg = N6.gpu_grp(ms_, ls_, PLAN)
    bb = {p: msb[p]["b3"] for p in PROJ}
    ls = ls_.astype(mx.int32).reshape(-1)
    ms = ms_.astype(mx.int32).reshape(-1)
    g, u, d = (msb[p] for p in PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0), ("GS_C", gs)]
    xs = rowsum64(x)
    (h,) = ka(pa)(inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                          lsb["gate_proj"]["l1"], lsb["up_proj"]["l1"], lsb["gate_proj"]["b"], lsb["up_proj"]["b"], ms, ls,
                          pidx, seg, nsg_],
                  template=tmpl + [("K_C", K), ("N_C", N), ("KS_C", ks)] + ([("PAD_C", pa)] if pa else []),
                  grid=(N // 32 * 64 * ks, nseg, 1), threadgroup=(64 * ks, 1, 1),
                  output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    hs = rowsum64(h)
    (y,) = kb(pb)(inputs=[h, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], lsb["down_proj"]["l1"], lsb["down_proj"]["b"],
                          ms, ls, pidx, seg, nsg_],
                  template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)] + ([("PAD_C", pb)] if pb else []),
                  grid=(D // (32 * nsg_b) * 32 * nsg_b * ks_b, nseg, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
                  output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D)

ARMS = {"A 20 KB / B 10 KB (NAX6)": (0, 0), "A 24 KB": (1024, 0), "A 28 KB": (2048, 0), "A 32 KB": (3072, 0),
        "B 20 KB": (0, 2560), "B 32 KB": (0, 5632)}
s2 = mx.new_stream(mx.gpu)
mx.random.seed(0)
ok = True
for trace in ("code", "chat"):
    tr = np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{trace}_800_g.npz"))[f"L{LAYER}"].astype(np.int32)
    for S in (8, 16, 32):
        res = {k: [] for k in list(ARMS) + ["2 calls back to back", "2 calls on 2 streams"]}
        for t in np.linspace(0, tr.shape[0] - 2 * S, 6).astype(int):
            ids = tr[t:t + S, :10].copy(); ids2 = tr[t + S:t + 2 * S, :10].copy()
            ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
            ms2, ls2 = mx.array(ids2), mx.array(ids2[:, :2].copy())
            x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
            x2 = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
            y0 = call(x, ms_, ls_)
            ref = N6.tier_glu_nax6(x, msb, lsb, ms_, ls_, plan=PLAN, grp=N6.gpu_grp(ms_, ls_, PLAN))
            mx.eval(y0, ref)
            ok &= bool(mx.all(y0.view(mx.uint16) == ref.view(mx.uint16)).item())
            for k, (pa, pb) in ARMS.items():
                yk = call(x, ms_, ls_, pa, pb); mx.eval(yk)
                ok &= bool(mx.all(yk.view(mx.uint16) == ref.view(mx.uint16)).item())
            def two_seq():
                return [call(x, ms_, ls_), call(x2, ms2, ls2)] * 4
            def two_par():
                a = [call(x, ms_, ls_) for _ in range(4)]
                with mx.stream(s2):
                    b = [call(x2, ms2, ls2) for _ in range(4)]
                return a + b
            fns = {k: (lambda pa=pa, pb=pb: [call(x, ms_, ls_, pa, pb) for _ in range(8)]) for k, (pa, pb) in ARMS.items()}
            fns["2 calls back to back"] = two_seq
            fns["2 calls on 2 streams"] = two_par
            for f in fns.values():
                mx.eval(*f())
            ts = {k: [] for k in fns}
            keys = list(fns)
            for rnd in range(9):
                for k in (keys if rnd % 2 == 0 else keys[::-1]):
                    mx.synchronize(); t1 = time.perf_counter(); mx.eval(*fns[k]()); mx.synchronize()
                    ts[k].append((time.perf_counter() - t1) / 8)
            for k in fns:
                res[k].append(float(np.median(ts[k])))
        base = np.mean(res[list(ARMS)[0]])
        say(f"{trace} S {S:2d}: " + ", ".join(f"{k} {1e3 * np.mean(v):.3f} ms ({100 * (np.mean(v) / base - 1):+.1f} %)"
                                            for k, v in res.items()))
say(f"bits == NAX6 for every padded arm: {ok}")
