"""NAX-MICRO (Leap, Wed 2026-11-04 16:1x; TABLE_350 'NAX-ABL'): where NAX6-N's time goes, by ablation (timing only; the
ablated arms compute garbage). NAX6-N (tierglu_nax6n: NAX6 on the fragment-native layout) is the expert kernel after
Friday's conversion. TIER-MERGE and NAX-OCC say the call is instruction-bound in its hot loop; this measures each part's
exposed share: arm time vs full = what removing that part saves (parts overlap, so shares need not add up).
  no-dec    codes -> nibbles is the raw words (no spread3 / spread8, no 24-bit extraction)
  no-epi    acc += P (no scale / bias reads, no FMAs)
  no-op     no matrix op (the A loads and their marshalling become dead too)
  no-A      the A operand from constants (no device loads of activations)
  no-codes  code / LSB words from constants (no device loads of codes)
  skeleton  no-dec + no-op + no-epi + no-codes (loops, barriers, stores, chunk sums, launches)
Layer 20's real records repacked native, real routing (code / chat), S 8 / 16 / 32, random x; ms per call, 8 calls per
eval, median of 7 interleaved rounds, 6 passes per (trace, S).
usage: [BigMoE venv] nax_ablate.py -> ~/Leap/data/nax_ablate.txt (GPU, MEM ~3 GB; share lock)"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs2/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs2/Leap_scripts"))
from tierglu import PROJ, plan_masks
from tierglu_nax import rowsum64
import tierglu_nax6 as N6
import tierglu_nax6n as NN

OUT = os.path.expanduser("~/Leap/data/nax_micro.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nNAX-MICRO {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}")
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
msb_n, lsb_n = NN.repack(msb, lsb)
del msb, lsb
mx.eval([v for d in (msb_n, lsb_n) for pr in d for v in d[pr].values()])


C = NN._CORE_N
EPI = ("""            const float x0 = as0[g], x1 = as1[g];
            for (int i = 0; i < 16; ++i) {
                const int col = 16 * (i >> 3) + fn0 + (i & 3);
                acc[i] += sb[64 * j + col] * cg[i] + sb[64 * j + 32 + col] * (((i >> 2) & 1) ? x1 : x0);
            }""", """            const float x0 = as0[g], x1 = as1[g];
            const float4 sv0 = *(threadgroup const float4*)(sb + 64 * j + fn0), sv1 = *(threadgroup const float4*)(sb + 64 * j + 16 + fn0);
            const float4 bv0 = *(threadgroup const float4*)(sb + 64 * j + 32 + fn0), bv1 = *(threadgroup const float4*)(sb + 64 * j + 48 + fn0);
            for (int i = 0; i < 16; ++i) {
                const float sv = (i >> 3) ? sv1[i & 3] : sv0[i & 3];
                const float bv = (i >> 3) ? bv1[i & 3] : bv0[i & 3];
                acc[i] += sv * cg[i] + bv * (((i >> 2) & 1) ? x1 : x0);
            }""")
assert C.count(EPI[0]) == 1
_K = {}
def kern(v4):
    if v4 not in _K:
        c = C.replace(*EPI) if v4 else C
        a_src, b_src = NN._A_N.replace(C, c), NN._B_N.replace(C, c)
        _K[v4] = (mx.fast.metal_kernel(name=f"nmic_a_{int(v4)}", input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u",
                                       "b2g", "b2u", "l1g", "l1u", "bg", "bu", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
                                       output_names=["h"], source=a_src, header=N6._HDR4),
                  mx.fast.metal_kernel(name=f"nmic_b_{int(v4)}", input_names=["h", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd",
                                       "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
                                       output_names=["y"], source=b_src, header=N6._HDR4))
    return _K[v4]

def call(v4, ny, x, ms_, ls_, grp, ks=4, ks_b=2, nsg_b=2):
    """NAX6-N's call; v4: vectorized epilogue loads; ny: the grid's y (None = P, as today)"""
    M, K = x.shape
    R = len(PLAN)
    m4, m2, r4 = plan_masks(PLAN)
    N, D = msb_n["gate_proj"]["s3"].shape[1], msb_n["down_proj"]["s3"].shape[1]
    pidx, seg, nsg_, nseg = grp if grp is not None else N6.gpu_grp(ms_, ls_, PLAN)
    Y = nseg if ny is None else ny
    ls = ls_.astype(mx.int32).reshape(-1)
    ms = ms_.astype(mx.int32).reshape(-1)
    g, u, d = (msb_n[p] for p in PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0), ("GS_C", 1)]
    ka, kb = kern(v4)
    xs = rowsum64(x)
    (h,) = ka(inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], g["b3"], u["b3"],
                      lsb_n["gate_proj"]["l1"], lsb_n["up_proj"]["l1"], lsb_n["gate_proj"]["b"], lsb_n["up_proj"]["b"], ms, ls,
                      pidx, seg, nsg_],
              template=tmpl + [("K_C", K), ("N_C", N), ("KS_C", ks)],
              grid=(N // 32 * 64 * ks, Y, 1), threadgroup=(64 * ks, 1, 1),
              output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    hs = rowsum64(h)
    (y,) = kb(inputs=[h, hs, d["w3"], d["s3"], d["b3"], d["b3"], lsb_n["down_proj"]["l1"], lsb_n["down_proj"]["b"],
                      ms, ls, pidx, seg, nsg_],
              template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)],
              grid=(D // (32 * nsg_b) * 32 * nsg_b * ks_b, Y, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
              output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D)

mx.random.seed(0)
ok = True
ARMS = ("full", "epi-v4", "tight", "epi-v4 + tight")
for trace in ("code", "chat"):
    tr = np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{trace}_800_g.npz"))[f"L{LAYER}"].astype(np.int32)
    for S in (8, 16, 32):
        res = {k: [] for k in ARMS}
        fill = []
        for t in np.linspace(0, tr.shape[0] - S, 6).astype(int):
            ids = tr[t:t + S, :10].copy()
            ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
            x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
            grp = N6.gpu_grp(ms_, ls_, PLAN)
            ng = int(grp[2].item())
            fill.append(ng / grp[3])
            yr = NN.tier_glu_nax6n(x, msb_n, lsb_n, ms_, ls_, plan=PLAN)
            ys = [call(False, None, x, ms_, ls_, None), call(True, None, x, ms_, ls_, None), call(True, ng, x, ms_, ls_, grp)]
            mx.eval(yr, *ys)
            ok &= all(bool(mx.all(v.view(mx.uint16) == yr.view(mx.uint16)).item()) for v in ys)
            fns = {"full": lambda: [call(False, None, x, ms_, ls_, None) for _ in range(8)],
                   "epi-v4": lambda: [call(True, None, x, ms_, ls_, None) for _ in range(8)],
                   "tight": lambda: [call(False, ng, x, ms_, ls_, grp) for _ in range(8)],
                   "epi-v4 + tight": lambda: [call(True, ng, x, ms_, ls_, grp) for _ in range(8)]}
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
        base = np.mean(res["full"])
        say(f"{trace} S {S:2d} (groups / P {100 * np.mean(fill):.0f} %): " + ", ".join(
            f"{k} {1e3 * np.mean(v):.3f} ms ({100 * (np.mean(v) / base - 1):+.1f} %)" for k, v in res.items()))
say(f"epi-v4 and the tight grid == tier_glu_nax6n bitwise: {ok}")
say("note: 'tight' and 'epi-v4 + tight' reuse one grouping per pass (the others group per call); grid y = the real count")
