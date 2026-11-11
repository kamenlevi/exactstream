"""M-11b LOWRANKCORR (MoonShots, Sun 2026-11-01; published: LQER 2024, QERA 2024, MiLo 2025): M-11's mean correction is the
rank-1 case of an activation-aware low-rank correction of the tier error. For a 3-bit tier use of expert e at layer L:
  gate / up   + (W4 - W3) U_r U_r^T x     U_r = top-r eigenvectors of the layer's E[x x^T] (calibration tokens, uncentered),
                                         so U_r^T x is computed once per token per layer and shared by all 10 experts;
                                         per expert (W4 - W3) U_r costs 2.56 KB x r (bf16, gate + up)
  down        + (W4 - W3) mean(h4_e)     (M-11), or rank 8 on the expert's own h basis (+51 KB)
Same inputs, split (calibration 0-4095, scored 4096-8191), tiers (BigMoE DESIGN_3bit.md) and block metric as M-11
(scripts/biascorr16.py). Plans: 4433333333 (REF2-T), 4333333333, 3333333333 (no LSB planes at all).
Variants: none | mean (M-11 corrE) | r16 / r32 / r64 (gate / up rank r + down mean) | r32d8 (gate / up r32 + down rank 8).
Lines (QUEUE M-11b, written first): PASS = 3333333333 or 4333333333, corrected at r <= 64, at <= 1.00x uncorrected REF2-T's
block error out of sample; KILL = no cheaper corrected plan <= 1.2x uncorrected REF2-T at r <= 64; BETWEEN otherwise.
usage: [BigMoE venv] taskpolicy -b python lowrank16.py (CPU, <= 4 threads, ~5 min) -> stdout + data/lowrank16.txt"""
import os
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "4"
import json, struct, sys, time
import numpy as np

MD = os.path.expanduser("~/BigMoE/models/bare")
XD = os.path.expanduser("~/Fit12/data/q2")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
CAL, EV = (0, 4096), (4096, 8192)
KSHR = 32
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


if os.path.exists(PAUSE):
    sys.exit("PAUSE set: not running")
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
heads = {}


def tensor(name):
    f = os.path.join(MD, idx[name])
    if f not in heads:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            heads[f] = (8 + n, json.loads(fh.read(n)))
    start, h = heads[f]
    m = h[name]
    dt = {"U32": np.uint32, "BF16": np.uint16, "F32": np.float32}[m["dtype"]]
    return np.memmap(f, dtype=dt, mode="r", offset=start + m["data_offsets"][0], shape=tuple(m["shape"]))


def f32(u16):
    return (np.asarray(u16).astype(np.uint32) << 16).view(np.float32)


def bf16r(a):
    u = np.asarray(a, np.float32).view(np.uint32).astype(np.uint64)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16).astype(np.uint32).view(np.float32)


def codes(w, bits):
    sh = np.arange(0, 32, bits, dtype=np.uint32)
    q = (np.asarray(w)[..., None] >> sh) & np.uint32((1 << bits) - 1)
    return q.reshape(*w.shape[:-1], -1)


def deq(prefix, e=None, tiers=(4,)):
    """dequantized fp32 weights [out, in] for the requested tiers of one matrix (expert e or a dense layer)"""
    w, s, b = (tensor(prefix + "." + k) for k in ("weight", "scales", "biases"))
    if e is not None:
        w, s, b = w[e], s[e], b[e]
    bits = w.shape[-1] * 32 // (s.shape[-1] * 64)
    q = codes(w, bits).astype(np.float32)
    o, n = q.shape
    q = q.reshape(o, n // 64, 64)
    s, b = f32(s)[..., None], f32(b)[..., None]
    out = {}
    for t in tiers:
        if t == 4 or bits != 4:
            out[t] = (q * s + b).reshape(o, n)
        else:
            k = 4 - t                                                     # low bits dropped
            hi = np.floor(q / (1 << k))
            lo = q - hi * (1 << k)
            bt = bf16r(b + s * lo.mean(-1, keepdims=True))
            out[t] = ((1 << k) * s * hi + bt).reshape(o, n)
    return out


PLANS = ("4433333333", "4333333333", "3333333333")
VARS = ("none", "mean", "r16", "r32", "r64", "r32d8")
RMAX = 64


def silu(a):
    return a / (1.0 + np.exp(-a))


LAYERS = [L for L in (2, 12, 24, 36, 46) if os.path.exists(os.path.join(XD, f"x_L{L}.npy"))]
say(f"M-11b LOWRANKCORR: REF2-T tiers with low-rank activation-aware corrections; layers {LAYERS}; calibration tokens "
    f"{CAL[0]}-{CAL[1] - 1}, scored on {EV[0]}-{EV[1] - 1}; CPU fp32")
t0 = time.time()
acc = {(p, v): [0.0, 0.0] for p in PLANS for v in VARS}
for L in LAYERS:
    while os.path.exists(PAUSE):
        time.sleep(20)
    pre = f"language_model.model.layers.{L}.mlp."
    X = np.load(os.path.join(XD, f"x_L{L}.npy")).astype(np.float32)
    Wr = deq(pre + "gate")[4]
    lg = X @ Wr.T
    lg -= lg.max(1, keepdims=True)
    p = np.exp(lg); p /= p.sum(1, keepdims=True)
    ids = np.argsort(-p, axis=1)[:, :10]
    w10 = np.take_along_axis(p, ids, 1); w10 /= w10.sum(1, keepdims=True)
    sg, su, sd = (deq(pre + "shared_expert." + m)[4] for m in ("gate_proj", "up_proj", "down_proj"))
    sgate = deq(pre + "shared_expert_gate")[4]
    S = (1.0 / (1.0 + np.exp(-(X @ sgate.T)))) * (silu(X @ sg.T) * (X @ su.T)) @ sd.T
    xc = X[CAL[0]:CAL[1]]
    mu = xc.mean(0)
    C = (xc.T @ xc) / len(xc)
    ev, U = np.linalg.eigh(C.astype(np.float64))
    ev, U = ev[::-1], U[:, ::-1].astype(np.float32)
    frac = {r: float(ev[:r].sum() / ev.sum()) for r in (1, 4, 16, 32, 64)}
    Z = X @ U[:, :RMAX]
    ne = EV[1] - EV[0]
    Y = {"ref": np.zeros((ne, 2560), np.float32)}
    for pl in PLANS:
        for v in VARS:
            Y[(pl, v)] = np.zeros((ne, 2560), np.float32)
    sw = pre + "switch_mlp."
    for e in range(512):
        tok, rk = np.nonzero(ids == e)
        if tok.size == 0:
            continue
        ev_ = (tok >= EV[0]) & (tok < EV[1])
        if not ev_.any():
            continue
        G = deq(sw + "gate_proj", e, (4, 3)); Uu = deq(sw + "up_proj", e, (4, 3)); D = deq(sw + "down_proj", e, (4, 3))
        dG, dU, dD = G[4] - G[3], Uu[4] - Uu[3], D[4] - D[3]
        cal = tok[(tok >= CAL[0]) & (tok < CAL[1])]
        Uh = None
        if cal.size:
            xcal = X[cal]
            h4c = silu(xcal @ G[4].T) * (xcal @ Uu[4].T)
            muh = h4c.mean(0)
            mue = (cal.size * xcal.mean(0) + KSHR * mu) / (cal.size + KSHR)
            if cal.size >= 16:
                eh, vh = np.linalg.eigh((h4c.T @ h4c).astype(np.float64) / cal.size)
                Uh = vh[:, ::-1][:, :8].astype(np.float32)
        else:
            muh = np.zeros(640, np.float32); mue = mu
        te, re = tok[ev_], rk[ev_]
        xe, z = X[te], Z[te]
        wt = w10[te, re][:, None]
        y4 = (silu(xe @ G[4].T) * (xe @ Uu[4].T)) @ D[4].T
        g3, u3 = xe @ G[3].T, xe @ Uu[3].T
        Ag, Au = dG @ U[:, :RMAX], dU @ U[:, :RMAX]
        cd = dD @ muh
        o3 = {"none": (silu(g3) * u3) @ D[3].T,
              "mean": (silu(g3 + dG @ mue) * (u3 + dU @ mue)) @ D[3].T + cd}
        for R in (16, 32, 64):
            h = silu(g3 + z[:, :R] @ Ag[:, :R].T) * (u3 + z[:, :R] @ Au[:, :R].T)
            o3[f"r{R}"] = h @ D[3].T + cd
            if R == 32:
                o3["r32d8"] = h @ D[3].T + ((h @ Uh) @ (dD @ Uh).T if Uh is not None else cd)
        r0 = te - EV[0]
        Y["ref"][r0] += wt * y4                                        # a token routes to e at most once
        for pl in PLANS:
            tier = np.array([int(c) for c in pl])[re]
            s4, s3 = tier == 4, tier == 3
            for v in VARS:
                if s4.any():
                    Y[(pl, v)][r0[s4]] += wt[s4] * y4[s4]
                if s3.any():
                    Y[(pl, v)][r0[s3]] += wt[s3] * o3[v][s3]
    Sev = S[EV[0]:EV[1]]
    ref = Y["ref"] + Sev
    den = float((ref * ref).sum())
    lay = {}
    for pl in PLANS:
        for v in VARS:
            err = float(((Y[(pl, v)] + Sev - ref) ** 2).sum())
            acc[(pl, v)][0] += err; acc[(pl, v)][1] += den
            lay[(pl, v)] = err / den
    say(f"  layer {L:2d}: input energy in the top 1 / 4 / 16 / 32 / 64 directions " +
        " / ".join(f"{100 * frac[r]:.0f}" for r in (1, 4, 16, 32, 64)) + " %; 3333333333 none " +
        f"{lay[('3333333333', 'none')]:.5f} -> r64 {lay[('3333333333', 'r64')]:.5f} ({time.time() - t0:.0f} s)")
val = {k: a[0] / a[1] for k, a in acc.items()}
base = val[("4433333333", "none")]
say("block error, 5 layers pooled, out of sample (vs all-4-bit); x = multiple of uncorrected REF2-T:")
say("  plan          " + "  ".join(f"{v:>14s}" for v in VARS))
for pl in PLANS:
    say(f"  {pl}  " + "  ".join(f"{val[(pl, v)]:.5f} ({val[(pl, v)] / base:4.2f}x)" for v in VARS))
say("  correction bytes per expert (bf16): gate / up r16 41 KB, r32 82 KB, r64 164 KB; down mean 5 KB, rank 8 51 KB "
    "(MSB-3 record 2,162,688 B; LSB part 720,896 B)")
cheap = min(val[(pl, v)] for pl in PLANS[1:] for v in VARS[1:])
best = min((val[(pl, v)], pl, v) for pl in PLANS[1:] for v in VARS[1:])
call = "PASS" if cheap <= base else ("KILL" if cheap > 1.2 * base else "BETWEEN")
say(f"Lines: best cheaper corrected plan {best[1]} {best[2]} at {best[0] / base:.2f}x uncorrected REF2-T (PASS <= 1.00, "
    f"KILL > 1.20) -> {call}")
say(f"({time.time() - t0:.0f} s)")
open(os.path.expanduser("~/MoonShots/data/lowrank16.txt"), "w").write("\n".join(lines) + "\n")
