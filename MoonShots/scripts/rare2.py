"""M-12 RARE2 step 1 (MoonShots, Sun 2026-11-01; SSD-miss wall; published pieces: static per-expert bit widths by usage,
MC-MoE / MxMoE / AlphaQ): can rarely used experts take a 2-bit tier at ranks 2-9, paid for by M-11's mean correction?
Rule (per token, so the fast loop stays exact against its own one-token decoding): per layer, the rare set = the bottom 20 / 40 %
of experts by routing frequency (any rank) on the calibration tokens. Rare experts at ranks 2-9 use MSB-2
(W2 = 4s (q >> 2) + b2, b2 = bf16(b + s mean_group(q & 3)), BigMoE DESIGN_3bit.md); ranks 0-1 stay 4-bit, other ranks 2-9 MSB-3.
Correction (M-11 corrE): gate / up + (W4 - Wt) mu_e (the expert's routed calibration mean, shrunk to the layer mean, k 32);
down + (W4 - Wt) mean(h4_e). Same data, split (calibration 0-4095, scored 4096-8191), router, shared expert and block metric
as M-11 (scripts/biascorr16.py).
Lines (QUEUE M-12, written first): PASS = rare-40 % with the correction at <= 1.00x uncorrected REF2-T's block error;
KILL = rare-20 % with the correction > 1.20x; BETWEEN otherwise.
usage: [BigMoE venv] taskpolicy -b python rare2.py (CPU, <= 4 threads, ~5 min) -> stdout + data/rare2.txt"""
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


PLANS = ("ref2t", "rare20", "rare40")
VARS = ("none", "corr")


def silu(a):
    return a / (1.0 + np.exp(-a))


LAYERS = [L for L in (2, 12, 24, 36, 46) if os.path.exists(os.path.join(XD, f"x_L{L}.npy"))]
say(f"M-12 RARE2 step 1: rare experts at ranks 2-9 on a 2-bit tier, with / without M-11's mean correction; layers {LAYERS}; "
    f"calibration tokens {CAL[0]}-{CAL[1] - 1}, scored on {EV[0]}-{EV[1] - 1}; CPU fp32")
t0 = time.time()
acc = {(p, v): [0.0, 0.0] for p in PLANS for v in VARS}
share = {20: [0, 0], 40: [0, 0]}
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
    freq = np.bincount(ids[CAL[0]:CAL[1]].ravel(), minlength=512)
    order = np.argsort(freq, kind="stable")
    rare = {20: np.zeros(512, bool), 40: np.zeros(512, bool)}
    rare[20][order[:102]] = True; rare[40][order[:205]] = True
    ide = ids[EV[0]:EV[1], 2:]
    for k in (20, 40):
        share[k][0] += int(rare[k][ide].sum()); share[k][1] += ide.size
    sg, su, sd = (deq(pre + "shared_expert." + m)[4] for m in ("gate_proj", "up_proj", "down_proj"))
    sgate = deq(pre + "shared_expert_gate")[4]
    S = (1.0 / (1.0 + np.exp(-(X @ sgate.T)))) * (silu(X @ sg.T) * (X @ su.T)) @ sd.T
    mu = X[CAL[0]:CAL[1]].mean(0)
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
        G = deq(sw + "gate_proj", e, (4, 3, 2)); U = deq(sw + "up_proj", e, (4, 3, 2)); D = deq(sw + "down_proj", e, (4, 3, 2))
        cal = tok[(tok >= CAL[0]) & (tok < CAL[1])]
        if cal.size:
            xcal = X[cal]
            muh = (silu(xcal @ G[4].T) * (xcal @ U[4].T)).mean(0)
            mue = (cal.size * xcal.mean(0) + KSHR * mu) / (cal.size + KSHR)
        else:
            muh = np.zeros(640, np.float32); mue = mu
        te, re = tok[ev_], rk[ev_]
        xe = X[te]
        wt = w10[te, re][:, None]
        y4 = (silu(xe @ G[4].T) * (xe @ U[4].T)) @ D[4].T
        outs = {(4, v): y4 for v in VARS}
        for t in (3, 2):
            g, u = xe @ G[t].T, xe @ U[t].T
            outs[(t, "none")] = (silu(g) * u) @ D[t].T
            outs[(t, "corr")] = (silu(g + (G[4] - G[t]) @ mue) * (u + (U[4] - U[t]) @ mue)) @ D[t].T + (D[4] - D[t]) @ muh
        r0 = te - EV[0]
        Y["ref"][r0] += wt * y4                                        # a token routes to e at most once
        for pl in PLANS:
            low = 2 if (pl != "ref2t" and rare[int(pl[4:])][e]) else 3
            tier = np.where(re < 2, 4, low)
            for t in (4, 3, 2):
                sel = tier == t
                if sel.any():
                    for v in VARS:
                        Y[(pl, v)][r0[sel]] += wt[sel] * outs[(t, v)][sel]
    Sev = S[EV[0]:EV[1]]
    ref = Y["ref"] + Sev
    den = float((ref * ref).sum())
    lay = {}
    for pl in PLANS:
        for v in VARS:
            err = float(((Y[(pl, v)] + Sev - ref) ** 2).sum())
            acc[(pl, v)][0] += err; acc[(pl, v)][1] += den
            lay[(pl, v)] = err / den
    say(f"  layer {L:2d}: " + "  ".join(f"{pl} {lay[(pl, 'none')]:.5f} / corr {lay[(pl, 'corr')]:.5f}" for pl in PLANS) +
        f" ({time.time() - t0:.0f} s)")
val = {k: a[0] / a[1] for k, a in acc.items()}
base = val[("ref2t", "none")]
say("block error, 5 layers pooled, out of sample (vs all-4-bit); x = multiple of uncorrected REF2-T:")
for pl in PLANS:
    say(f"  {pl:7s} none {val[(pl, 'none')]:.5f} ({val[(pl, 'none')] / base:.2f}x)   corrected {val[(pl, 'corr')]:.5f} "
        f"({val[(pl, 'corr')] / base:.2f}x)")
say("  share of rank 2-9 picks (scored tokens) that go 2-bit: rare-20 % " +
    f"{100 * share[20][0] / share[20][1]:.1f} %, rare-40 % {100 * share[40][0] / share[40][1]:.1f} %")
r40, r20 = val[("rare40", "corr")] / base, val[("rare20", "corr")] / base
call = "PASS" if r40 <= 1.0 else ("KILL" if r20 > 1.2 else "BETWEEN")
say(f"Lines: rare-40 % corrected {r40:.2f}x (PASS <= 1.00); rare-20 % corrected {r20:.2f}x (KILL > 1.20) -> {call}")
say(f"({time.time() - t0:.0f} s)")
open(os.path.expanduser("~/MoonShots/data/rare2.txt"), "w").write("\n".join(lines) + "\n")
