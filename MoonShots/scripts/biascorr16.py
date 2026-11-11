"""M-11 BIASCORR (MoonShots, Sun 2026-11-01; published: DFQ bias correction, Nagel et al. 2019; KBVQ-MoE, ICLR 2026): does
a per-expert bias correction shrink the error of REF2-T's low-bit tiers enough to buy KLD headroom for byte cuts?
REF2-T's tiers (BigMoE DESIGN_3bit.md, ~/Leap/third_party/ms_gdn/tierglu.py):
  4-bit  W4 = s q + b                        (q = 4-bit code, group 64)
  MSB-3  W3 = 2s (q >> 1) + b3,  b3 = bf16(b + s mean_group(q & 1))
  MSB-2  W2 = 4s (q >> 2) + b2,  b2 = bf16(b + s mean_group(q & 3))
b3 / b2 already remove the per-group mean of the weight error. Left: the systematic part (W4 - Wt) mean(x), carried by the
MoE input's mean direction (strong here: rows at cosine ~0.6, M-10). Correction for a tier-t use of expert e:
  gate / up pre-activations + (W4 - Wt) mu_x          corr-L: mu_x = the layer's calibration mean
                                                       corr-E: the expert's routed calibration tokens, shrunk to mu_x (k 32)
  down output              + (W4 - Wt) mean(h4_e)     the expert's routed calibration tokens (4-bit h)
Inputs (read only): Fit12's MoE inputs ~/Fit12/data/q2/x_L*.npy (8192 tokens; layers 2 / 12 / 24 / 36 / 46), the real router
(8-bit, softmax over 512, top-10 renormalized), the real 4-bit experts and gated shared expert of ~/BigMoE/models/bare, in
fp32 numpy (a model of MLX's arithmetic, fine for error ratios). Calibration on tokens 0-4095, scored on 4096-8191.
Metric (L1-17 / M-09): MoE block error sum ||y_plan - y_4bit||^2 / sum ||y_4bit||^2, all 5 layers pooled.
Plans (tier per routing rank 0..9; 0 = dropped, no renormalization): 4433333333 (REF2-T), 4333333333, 4433332222, 4433333300.
Lines (QUEUE M-11, written first): PASS = correction cuts REF2-T's block error by >= 25 % out of sample AND a cheaper
corrected plan lands at or under uncorrected REF2-T's error; KILL = < 10 % cut on REF2-T and no cheaper corrected plan within
1.2x of it; BETWEEN otherwise.
usage: [BigMoE venv] taskpolicy -b python biascorr16.py (CPU, <= 4 threads, a few minutes) -> stdout + data/biascorr16.txt"""
import os
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "4"
import json, struct, sys, time
import numpy as np

MD = os.path.expanduser("~/BigMoE/models/bare")
XD = os.path.expanduser("~/Fit12/data/q2")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
CAL, EV = (0, 4096), (4096, 8192)
PLANS = ("4433333333", "4333333333", "4433332222", "4433333300")
VARS = ("none", "corrL", "corrE")
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


def silu(a):
    return a / (1.0 + np.exp(-a))


LAYERS = [L for L in (2, 12, 24, 36, 46) if os.path.exists(os.path.join(XD, f"x_L{L}.npy"))]
say(f"M-11 BIASCORR: REF2-T tiers with / without bias correction; layers {LAYERS}; calibration tokens {CAL[0]}-{CAL[1] - 1}, "
    f"scored on {EV[0]}-{EV[1] - 1}; CPU fp32")
t0 = time.time()
acc = {(p, v): [0.0, 0.0] for p in PLANS for v in VARS}
perL = {}
for L in LAYERS:
    while os.path.exists(PAUSE):
        time.sleep(20)
    pre = f"language_model.model.layers.{L}.mlp."
    X = np.load(os.path.join(XD, f"x_L{L}.npy")).astype(np.float32)
    T = X.shape[0]
    Wr = deq(pre + "gate")[4]
    lg = X @ Wr.T
    lg -= lg.max(1, keepdims=True)
    p = np.exp(lg); p /= p.sum(1, keepdims=True)
    ids = np.argsort(-p, axis=1)[:, :10]
    w10 = np.take_along_axis(p, ids, 1); w10 /= w10.sum(1, keepdims=True)
    sg, su, sd = (deq(pre + "shared_expert." + m)[4] for m in ("gate_proj", "up_proj", "down_proj"))
    sgate = deq(pre + "shared_expert_gate")[4]
    S = (1.0 / (1.0 + np.exp(-(X @ sgate.T)))) * (silu(X @ sg.T) * (X @ su.T)) @ sd.T
    mu = X[CAL[0]:CAL[1]].mean(0)
    xc = X[CAL[0]:CAL[1]]
    dc = float(mu @ mu / (xc * xc).sum(1).mean())
    mug = mu.reshape(-1, 64); mug = mug - mug.mean(1, keepdims=True)
    xg = xc.reshape(len(xc), -1, 64); xg = xg - xg.mean(2, keepdims=True)
    dcw = float((mug * mug).sum() / (xg * xg).sum((1, 2)).mean())
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
        G = deq(sw + "gate_proj", e, (4, 3, 2)); U = deq(sw + "up_proj", e, (4, 3, 2)); D = deq(sw + "down_proj", e, (4, 3, 2))
        cal = tok[(tok >= CAL[0]) & (tok < CAL[1])]
        if cal.size:
            xcal = X[cal]
            h4c = silu(xcal @ G[4].T) * (xcal @ U[4].T)
            muh = h4c.mean(0)
            mue = (cal.size * xcal.mean(0) + KSHR * mu) / (cal.size + KSHR)
        else:
            muh = np.zeros(640, np.float32); mue = mu
        ev = (tok >= EV[0]) & (tok < EV[1])
        if not ev.any():
            continue
        te, re = tok[ev], rk[ev]
        xe = X[te]
        wt = w10[te, re][:, None]
        y4 = (silu(xe @ G[4].T) * (xe @ U[4].T)) @ D[4].T
        outs = {(4, v): y4 for v in VARS}
        for t in (3, 2):
            dG, dU, dD = G[4] - G[t], U[4] - U[t], D[4] - D[t]
            g, u = xe @ G[t].T, xe @ U[t].T
            outs[(t, "none")] = (silu(g) * u) @ D[t].T
            cd = dD @ muh
            for v, m in (("corrL", mu), ("corrE", mue)):
                outs[(t, v)] = (silu(g + dG @ m) * (u + dU @ m)) @ D[t].T + cd
        r0 = te - EV[0]
        Y["ref"][r0] += wt * y4                                        # a token routes to e at most once
        for pl in PLANS:
            tier = np.array([int(c) for c in pl])[re]
            for t in (4, 3, 2):
                sel = tier == t
                if sel.any():
                    for v in VARS:
                        Y[(pl, v)][r0[sel]] += wt[sel] * outs[(t, v)][sel]
    Sev = S[EV[0]:EV[1]]
    ref = Y["ref"] + Sev
    den = float((ref * ref).sum())
    perL[L] = {}
    for pl in PLANS:
        for v in VARS:
            err = float(((Y[(pl, v)] + Sev - ref) ** 2).sum())
            acc[(pl, v)][0] += err; acc[(pl, v)][1] += den
            perL[L][(pl, v)] = err / den
    say(f"  layer {L:2d}: input mean energy {100 * dc:.1f} % (within-group part {100 * dcw:.1f} %); REF2-T block error "
        f"{perL[L][('4433333333', 'none')]:.5f} -> corrL {perL[L][('4433333333', 'corrL')]:.5f}, corrE "
        f"{perL[L][('4433333333', 'corrE')]:.5f} ({time.time() - t0:.0f} s)")
val = {k: a[0] / a[1] for k, a in acc.items()}
say("block error, 5 layers pooled, out of sample (vs all-4-bit):")
say("  plan          none      corrL     corrE")
for pl in PLANS:
    say(f"  {pl}  " + "  ".join(f"{val[(pl, v)]:.5f}" for v in VARS) +
        f"   (x REF2-T none: " + " / ".join(f"{val[(pl, v)] / val[('4433333333', 'none')]:.2f}" for v in VARS) + ")")
base = val[("4433333333", "none")]
best = min(val[("4433333333", "corrL")], val[("4433333333", "corrE")])
cut = 1 - best / base
cheap = min(val[(pl, v)] for pl in PLANS[1:] for v in ("corrL", "corrE"))
call = "PASS" if (cut >= 0.25 and cheap <= base) else ("KILL" if (cut < 0.10 and cheap > 1.2 * base) else "BETWEEN")
say(f"Lines: correction cuts REF2-T's block error by {100 * cut:.1f} % (PASS >= 25, KILL < 10); best cheaper corrected plan "
    f"{cheap / base:.2f}x uncorrected REF2-T (PASS <= 1.00, KILL > 1.20) -> {call}")
say(f"({time.time() - t0:.0f} s)")
open(os.path.expanduser("~/MoonShots/data/biascorr16.txt"), "w").write("\n".join(lines) + "\n")
