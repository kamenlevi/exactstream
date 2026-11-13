"""NAX-ABL (Leap, Wed 2026-11-04 16:1x; TABLE_350 'NAX-ABL'): where NAX6-N's time goes, by ablation (timing only; the
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
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_scripts"))
from tierglu import PROJ, plan_masks
from tierglu_nax import rowsum64
import tierglu_nax6 as N6
import tierglu_nax6n as NN

OUT = os.path.expanduser("~/Leap/data/nax_ablate.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nNAX-ABL {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}")
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
SW = {
    "dec": [("uint ch[8] = {w0 & 0xFFFFFFu, (w0 >> 24) | ((w1 & 0xFFFFu) << 8), (w1 >> 16) | ((w2 & 0xFFu) << 16), w2 >> 8,\n"
             "                          w3 & 0xFFFFFFu, (w3 >> 24) | ((w4 & 0xFFFFu) << 8), (w4 >> 16) | ((w5 & 0xFFu) << 16), w5 >> 8};",
             "uint ch[8] = {w0, w1, w2, w3, w4, w5, w0 ^ w3, w1 ^ w4};"),
            ("for (int q = 0; q < 8; ++q) nb[q] = spread8(ch[q], ((q < 4 ? l0 : l1) >> (8 * (q & 3))) & 0xFFu);",
             "for (int q = 0; q < 8; ++q) nb[q] = ch[q] ^ (q < 4 ? l0 : l1);"),
            ("for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]) & 0x66666666u;", "for (int q = 0; q < 8; ++q) nb[q] = ch[q];"),
            ("for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]);", "for (int q = 0; q < 8; ++q) nb[q] = ch[q];")],
    "epi": [("acc[i] += sb[64 * j + col] * cg[i] + sb[64 * j + 32 + col] * (((i >> 2) & 1) ? x1 : x0);", "acc[i] += cg[i];")],
    "op": [("op.run(ca, tbt, cg);", "")],
    "A": [("const vec<T, 4> a00 = *(const device vec<T, 4>*)(ar0 + k0 + fn0);", "const vec<T, 4> a00 = vec<T, 4>(T(0.5f));"),
          ("const vec<T, 4> a10 = *(const device vec<T, 4>*)(ar1 + k0 + fn0);", "const vec<T, 4> a10 = vec<T, 4>(T(0.25f));"),
          ("const vec<T, 4> a01 = *(const device vec<T, 4>*)(ar0 + k0 + 16 + fn0);", "const vec<T, 4> a01 = vec<T, 4>(T(0.75f));"),
          ("const vec<T, 4> a11 = *(const device vec<T, 4>*)(ar1 + k0 + 16 + fn0);", "const vec<T, 4> a11 = vec<T, 4>(T(1.5f));")],
    "codes": [("const uint w0 = c[0], w1 = c[1], w2 = c[2], w3 = c[3], w4 = c[4], w5 = c[5];",
               "const uint w0 = (uint)g * 2654435761u + lane, w1 = w0 ^ 0x5bd1e995u, w2 = w0 + 7u * lane, w3 = w1 + 3u, w4 = w2 ^ 77u, w5 = w3 ^ 99u;"),
              ("const uint l0 = lrow[64 * g], l1 = lrow[64 * g + 1];", "const uint l0 = (uint)g * 40503u + lane, l1 = l0 ^ 0xA5A5A5A5u;")],
}
ARMS = {"full": [], "no-dec": ["dec"], "no-epi": ["epi"], "no-op": ["op"], "no-A": ["A"], "no-codes": ["codes"],
        "skeleton": ["dec", "op", "epi", "codes"]}
_K = {}
def kern(arm):
    if arm not in _K:
        c = C
        for part in ARMS[arm]:
            c = NN._swap(c, SW[part])
        a_src, b_src = NN._A_N.replace(C, c), NN._B_N.replace(C, c)
        _K[arm] = (mx.fast.metal_kernel(name=f"nabl_a_{arm.replace('-', '_')}",
                                        input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g",
                                                     "l1u", "bg", "bu", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
                                        output_names=["h"], source=a_src, header=N6._HDR4),
                   mx.fast.metal_kernel(name=f"nabl_b_{arm.replace('-', '_')}",
                                        input_names=["h", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot",
                                                     "lsb_slot", "pidx", "seg", "nseg"],
                                        output_names=["y"], source=b_src, header=N6._HDR4))
    return _K[arm]

def call(arm, x, ms_, ls_, ks=4, ks_b=2, nsg_b=2):
    M, K = x.shape
    R = len(PLAN)
    m4, m2, r4 = plan_masks(PLAN)
    N, D = msb_n["gate_proj"]["s3"].shape[1], msb_n["down_proj"]["s3"].shape[1]
    pidx, seg, nsg_, nseg = N6.gpu_grp(ms_, ls_, PLAN)
    ls = ls_.astype(mx.int32).reshape(-1)
    ms = ms_.astype(mx.int32).reshape(-1)
    g, u, d = (msb_n[p] for p in PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0), ("GS_C", 1)]
    ka, kb = kern(arm)
    xs = rowsum64(x)
    (h,) = ka(inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], g["b3"], u["b3"],
                      lsb_n["gate_proj"]["l1"], lsb_n["up_proj"]["l1"], lsb_n["gate_proj"]["b"], lsb_n["up_proj"]["b"], ms, ls,
                      pidx, seg, nsg_],
              template=tmpl + [("K_C", K), ("N_C", N), ("KS_C", ks)],
              grid=(N // 32 * 64 * ks, nseg, 1), threadgroup=(64 * ks, 1, 1),
              output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    hs = rowsum64(h)
    (y,) = kb(inputs=[h, hs, d["w3"], d["s3"], d["b3"], d["b3"], lsb_n["down_proj"]["l1"], lsb_n["down_proj"]["b"],
                      ms, ls, pidx, seg, nsg_],
              template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)],
              grid=(D // (32 * nsg_b) * 32 * nsg_b * ks_b, nseg, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
              output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D)

mx.random.seed(0)
ok = True
for trace in ("code", "chat"):
    tr = np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{trace}_800_g.npz"))[f"L{LAYER}"].astype(np.int32)
    for S in (8, 16, 32):
        res = {k: [] for k in ARMS}
        for t in np.linspace(0, tr.shape[0] - S, 6).astype(int):
            ids = tr[t:t + S, :10].copy()
            ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
            x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
            yf, yr = call("full", x, ms_, ls_), NN.tier_glu_nax6n(x, msb_n, lsb_n, ms_, ls_, plan=PLAN)
            mx.eval(yf, yr)
            ok &= bool(mx.all(yf.view(mx.uint16) == yr.view(mx.uint16)).item())
            fns = {k: (lambda k=k: [call(k, x, ms_, ls_) for _ in range(8)]) for k in ARMS}
            for f in fns.values():
                mx.eval(*f())
            ts = {k: [] for k in fns}
            keys = list(fns)
            for rnd in range(7):
                for k in (keys if rnd % 2 == 0 else keys[::-1]):
                    mx.synchronize(); t1 = time.perf_counter(); mx.eval(*fns[k]()); mx.synchronize()
                    ts[k].append((time.perf_counter() - t1) / 8)
            for k in fns:
                res[k].append(float(np.median(ts[k])))
        base = np.mean(res["full"])
        say(f"{trace} S {S:2d}: " + ", ".join(f"{k} {1e3 * np.mean(v):.3f} ms ({100 * (np.mean(v) / base - 1):+.1f} %)"
                                            for k, v in res.items()))
say(f"full arm == tier_glu_nax6n bitwise: {ok}")
