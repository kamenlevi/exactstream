"""K-62 NAX3 screen (Leap, Sun 2026-11-01; shared GPU, rough timing): tierglu_nax.tier_glu_nax vs MoonShots' 3c tier_glu
(frozen copy) at the 125B's expert shapes, synthetic pools as in K-61 (base 128 slots, sidecar 64; each base expert has
one sidecar slot). Lines in TABLE_350 (K-62).
1. accuracy: both kernels against an fp64 reference from the same dequantized weights (M 4, plan 4433333333): max and
   mean |err| / output RMS; plus NAX vs 3c in bf16 ulps
2. INV: a probe row's outputs bitwise across contexts: alone (M 1), as row 0 of 16, as row 9 of 16 with other
   companions, and in a 16-row call where every row routes like it (groups of 16)
3. timing at M 16 (U 100 / 70 / 24): tier_glu vs tier_glu_nax (+ its parts: diag 1 const codes, diag 2 no MMA),
   interleaved rounds, 8 chained calls per
   eval, median of 25 rounds -> ms per call
Usage: [BigMoE venv] k62_nax_test.py -> ~/Leap/data/k62_nax.txt"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_fu/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_fu/Leap_scripts"))
from tierglu import tier_glu
from tierglu_grouped import make_grp
from tierglu_nax import tier_glu_nax
from tierglu_nax4 import tier_glu_nax4
if os.environ.get("K62_FN") == "nax4":
    tier_glu_nax = lambda *a, diag=0, **k: tier_glu_nax4(*a, **k)

D_ = os.path.expanduser("~/Leap/data")
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
H, I, CM, CL, NR = 2560, 640, 128, 64, 10
PLAN = "4433333333"
mx.random.seed(0)
def ru32(*shape):
    return mx.random.randint(0, 2 ** 31 - 1, shape).astype(mx.uint32)
def rbf(*shape, s=0.01):
    return (mx.random.normal(shape) * s).astype(mx.bfloat16)
def msb_part(o, k):
    return {"w3": ru32(CM, o, k * 3 // 32), "s3": rbf(CM, o, k // 64), "b3": rbf(CM, o, k // 64, s=0.03)}
def lsb_part(o, k):
    return {"l1": ru32(CL, o, k // 32), "b": rbf(CL, o, k // 64, s=0.03)}
msb = {"gate_proj": msb_part(I, H), "up_proj": msb_part(I, H), "down_proj": msb_part(H, I)}
lsb = {"gate_proj": lsb_part(I, H), "up_proj": lsb_part(I, H), "down_proj": lsb_part(H, I)}
mx.eval(msb, lsb)
rng = np.random.default_rng(1)
side_of = rng.permutation(CM) % CL

def route(M, U):
    ms = np.stack([rng.choice(U, NR, replace=False) for _ in range(M)]).astype(np.int32)
    return ms, side_of[ms[:, :2]].astype(np.int32)

def bits(a):
    return np.array(a.view(mx.uint16)).astype(np.int64)

say(f"K-62 NAX3: MLX {mx.__version__}")

# ---- 1. accuracy against fp64
_np = {}
def npy(a):
    k = id(a)
    if k not in _np:
        _np[k] = np.array(a.astype(mx.float32)) if a.dtype == mx.bfloat16 else np.array(a)
    return _np[k]
def deq(p, e, tier, el):
    """fp64 weights [o, k] of projection p for base slot e (sidecar slot el for tier 4)"""
    w3 = npy(msb[p]["w3"])[e].view(np.uint8)
    o = w3.shape[0]
    q = np.unpackbits(w3, axis=1, bitorder="little").reshape(o, -1, 3)
    q = (q[..., 0] + 2 * q[..., 1] + 4 * q[..., 2]).astype(np.float64)
    s = np.repeat(npy(msb[p]["s3"])[e].astype(np.float64), 64, axis=1)
    if tier == 4:
        l = np.unpackbits(npy(lsb[p]["l1"])[el].view(np.uint8), axis=1, bitorder="little").astype(np.float64)
        b = np.repeat(npy(lsb[p]["b"])[el].astype(np.float64), 64, axis=1)
        return s * 0.5 * (2 * q + l) + b
    b = np.repeat(npy(msb[p]["b3"])[e].astype(np.float64), 64, axis=1)
    return s * q + b

M = 4
ms_np, ls_np = route(M, 100)
x = (mx.random.normal((M, H)) * 0.5).astype(mx.bfloat16)
ms_, ls_ = mx.array(ms_np), mx.array(ls_np)
y3 = tier_glu(x, msb, lsb, ms_, ls_, plan=PLAN)
yn = tier_glu_nax(x, msb, lsb, ms_, ls_, plan=PLAN, ms_np=ms_np, ls_np=ls_np)
mx.eval(y3, yn)
xf = np.array(x.astype(mx.float32)).astype(np.float64)
ref = np.zeros((M, NR, H))
for m in range(M):
    for r in range(NR):
        t = 4 if PLAN[r] == "4" else 3
        e, el = int(ms_np[m, r]), int(ls_np[m, min(r, 1)])
        gt, up, dn = (deq(p, e, t, el) for p in ("gate_proj", "up_proj", "down_proj"))
        g, u = gt @ xf[m], up @ xf[m]
        ref[m, r] = dn @ (g / (1 + np.exp(-g)) * u)
rms = float(np.sqrt((ref ** 2).mean()))
for name, y in (("3c", y3), ("NAX", yn)):
    err = np.abs(np.array(y.astype(mx.float32)).astype(np.float64) - ref)
    say(f"1. {name}: |err| vs fp64 max {err.max() / rms:.3e} mean {err.mean() / rms:.3e} (of output RMS {rms:.3e})")
for name, y in (("NAX", yn),):
    d = np.abs(bits(y) - bits(y3))
    say(f"   {name} vs 3c: bf16 ulps max {int(d.max())}, outputs differing {100 * (d > 0).mean():.1f} %")

# ---- 2. INV
xp = (mx.random.normal((1, H)) * 0.5).astype(mx.bfloat16)
mp, lp = route(1, 100)
def probe(ctx):
    if ctx == "alone":
        X, Ms, Ls, row = xp, mp, lp, 0
    else:
        Mo, Lo = route(16, 70 if ctx != "same" else 100)
        Xo = (mx.random.normal((16, H)) * 0.5).astype(mx.bfloat16)
        row = 0 if ctx == "row0" else 9
        if ctx == "same":
            Mo, Lo = np.repeat(mp, 16, 0), np.repeat(lp, 16, 0)
            Xo = mx.concatenate([xp] * 16)
        Ms, Ls = Mo.copy(), Lo.copy()
        Ms[row], Ls[row] = mp[0], lp[0]
        X = mx.concatenate([Xo[:row], xp, Xo[row + 1:]])
    y = tier_glu_nax(X, msb, lsb, mx.array(Ms), mx.array(Ls), plan=PLAN, ms_np=Ms, ls_np=Ls)
    mx.eval(y)
    return bits(y[row])
outs = {c: probe(c) for c in ("alone", "row0", "row9", "same")}
inv = all((o == outs["alone"]).all() for o in outs.values())
say(f"2. INV (probe row alone / row 0 of 16 / row 9 of 16 / 16 rows routed alike): {'PASS bitwise' if inv else 'FAIL'}")
if not inv:
    for c, o in outs.items():
        say(f"   {c}: differs from alone in {int((o != outs['alone']).sum())} of {o.size}")

# ---- 3. timing
def bench(fns, rounds=25, chain=8):
    for f in fns:
        for _ in range(3):
            mx.eval(*[f() for _ in range(chain)])
    ts = [[] for _ in fns]
    for _ in range(rounds):
        for i, f in enumerate(fns):
            mx.synchronize(); t0 = time.perf_counter(); mx.eval(*[f() for _ in range(chain)]); ts[i].append(time.perf_counter() - t0)
    return [float(np.median(t)) / chain for t in ts]

M = 16
for U in (100, 70, 24):
    ms_np, ls_np = route(M, U)
    ms_, ls_ = mx.array(ms_np), mx.array(ls_np)
    x = (mx.random.normal((M, H)) * 0.5).astype(mx.bfloat16)
    grp = make_grp(ms_, ls_, PLAN, M, ms_np, ls_np, 16)
    fns = [lambda: tier_glu(x, msb, lsb, ms_, ls_, plan=PLAN),
           lambda: tier_glu_nax(x, msb, lsb, ms_, ls_, plan=PLAN, grp=grp),
           lambda: tier_glu_nax(x, msb, lsb, ms_, ls_, plan=PLAN, grp=grp, diag=1),
           lambda: tier_glu_nax(x, msb, lsb, ms_, ls_, plan=PLAN, grp=grp, diag=2)]
    if os.environ.get("K62_FN") == "nax4":
        import tierglu_nax as _v2
        fns[2:] = [lambda: _v2.tier_glu_nax(x, msb, lsb, ms_, ls_, plan=PLAN, grp=grp), fns[0]]
    t = bench(fns)
    tail = (f"NAX {1e3 * t[1]:.3f} ({t[0] / t[1]:.2f}x); parts: const codes {1e3 * t[2]:.3f}, no MMA {1e3 * t[3]:.3f}"
            if os.environ.get("K62_FN") != "nax4" else
            f"NAX4 {1e3 * t[1]:.3f} ({t[0] / t[1]:.2f}x); v2 {1e3 * t[2]:.3f} ({t[0] / t[2]:.2f}x); tier_glu again {1e3 * t[3]:.3f}")
    say(f"3. M 16, U {U}: {len(np.unique(ms_np))} distinct experts, {grp[3]} groups; tier_glu {1e3 * t[0]:.3f} ms; " + tail)
say("K-62 done")
open(os.path.join(D_, os.environ.get("K62_OUT", "k62_nax.txt")), "w").write("\n".join(lines) + "\n")
