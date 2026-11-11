"""M-16 RUNAHEAD step 1 (MoonShots, Sun 2026-11-01; QUEUE M-16, lines written first): how far is a check computed with its
non-resident experts SKIPPED (the runahead pass) from the same check computed exactly?
Runahead = the target itself over the drafted rows, no SSD waits: experts not resident at the check's start are left out
(their reads are queued and land for the next pass, where the rows are re-checked exactly). Here, per row-layer:
  skipped routing mass (top-10 weights, softmax over 512, renormalized as the model does), skips by rank, and the share of
  misses some row needs at rank 0-1 (the hybrid variant waits for those);
  the MoE block error of runahead variants vs the served REF2-T block (tiers 4433333333: ranks 0-1 4-bit, ranks 2-9
  MSB-3), in units of REF2-T's own tier error vs all-4-bit on the same rows (M-11: 0.0121 on Fit12's tokens):
    skip    = skipped experts contribute 0
    renorm  = skip, then the kept weights renormalized to sum 1
    mean    = skip + w x the expert's mean output (calibrated on Fit12's 8,336 tokens per layer, a resident 5 KB vector)
    hybrid  = misses needed at rank 0-1 by any row of the check are waited for; the rest skipped (+ mean stand-in)
Data (read only): BigMoE's cap45 (C 160 live REF2b loop, 16-row checks, captured MoE inputs x per row and layer), the real
8-bit routers and 4-bit experts + gated shared expert of ~/BigMoE/models/bare (fp32 numpy, a model of MLX's arithmetic, fine
for error ratios), Fit12's MoE inputs ~/Fit12/data/q2/x_L*.npy (layers 2 / 12 / 24 / 36 / 46) for the mean stand-ins.
Residency: BigMoE's live rule LRU + 30 x popularity (leave-one-prompt-out), chained over cap45's six prompts (res0 forced in
per prompt; prompts 2-6 counted) at C 224 / 256 = K 246 / 287 resident slots per layer (as rare2ssd.py / evict45ws.py).
Lines (QUEUE M-16 step 1): PASS = the best variant's block error <= 2x REF2-T's tier error with >= 70 % of the misses still
skipped -> step 2 (live dual pass) tonight; KILL = every variant > 5x; BETWEEN otherwise.
usage: [BigMoE venv] taskpolicy -b python runahead_mass.py (CPU, <= 4 threads, ~5-10 min) -> stdout + data/runahead_mass.txt"""
import os
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "4"
import heapq, json, struct, sys, time
import numpy as np

CAP = os.path.expanduser("~/BigMoE/data/cap45")
MD = os.path.expanduser("~/BigMoE/models/bare")
XD = os.path.expanduser("~/Fit12/data/q2")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
NL, E = 48, 512
ORDER = ("humaneval_HumanEval_68", "math500_test_prealgebra_1640", "humaneval_HumanEval_91", "math500_test_algebra_694",
         "math500_test_algebra_1282", "gsm8k_708")
KS = ((224, 246), (256, 287))
PRIOR = 30.0
LAYERS = (2, 12, 24, 36, 46)
PLAN = "4433333333"
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def wait_pause():
    while os.path.exists(PAUSE):
        time.sleep(20)


if os.path.exists(PAUSE):
    sys.exit("PAUSE set: not running")
t0 = time.time()
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


def deq(prefix, e=None, tiers=(4,)):
    """dequantized fp32 [out, in] for the requested tiers (4 = the 4-bit grid, 3 = MSB-3 with b3), as biascorr16.py"""
    w, s, b = (tensor(prefix + "." + k) for k in ("weight", "scales", "biases"))
    if e is not None:
        w, s, b = w[e], s[e], b[e]
    bits = w.shape[-1] * 32 // (s.shape[-1] * 64)
    sh = np.arange(0, 32, bits, dtype=np.uint32)
    q = ((np.asarray(w)[..., None] >> sh) & np.uint32((1 << bits) - 1)).reshape(w.shape[0], -1).astype(np.float32)
    o, n = q.shape
    q = q.reshape(o, n // 64, 64)
    s, b = f32(s)[..., None], f32(b)[..., None]
    out = {}
    for t in tiers:
        if t == 4 or bits != 4:
            out[t] = (q * s + b).reshape(o, n)
        else:
            k = 4 - t
            hi = np.floor(q / (1 << k))
            lo = q - hi * (1 << k)
            out[t] = ((1 << k) * s * hi + bf16r(b + s * lo.mean(-1, keepdims=True))).reshape(o, n)
    return out


def silu(a):
    return a / (1.0 + np.exp(-a))


WR = [deq(f"language_model.model.layers.{l}.mlp.gate")[4] for l in range(NL)]
P, agree = [], [0, 0]
for name in ORDER:
    z = np.load(os.path.join(CAP, name + "_gate_0.npz"))
    seq, L, rows, roff, xoff, ids, X = z["seq"], z["L"], z["rows"], z["row_off"], z["x_off"], z["ids"], z["x"]
    checks, order = {}, []
    for n, (s, l, r) in enumerate(zip(seq, L, rows)):
        s, l, r = int(s), int(l), int(r)
        c = checks.setdefault(s, {"need": {}, "rid": {}, "w": {}, "rk": {}, "x": {}, "kept": None})
        if s not in order:
            order.append(s)
        if l in c["need"]:
            c["kept"] = r
            continue
        rid = ids[roff[n]:roff[n] + r].astype(np.int64)                   # [r, 10] captured ids (order not trusted)
        xr = f32(X[xoff[n]:xoff[n] + r])
        lg = xr @ WR[l].T                                                  # [r, 512]
        lg -= lg.max(1, keepdims=True)
        p = np.exp(lg); p /= p.sum(1, keepdims=True)
        w = np.take_along_axis(p, rid, 1)
        o = np.argsort(-w, axis=1, kind="stable")                          # rank order among the captured ids
        rid, w = np.take_along_axis(rid, o, 1), np.take_along_axis(w, o, 1)
        agree[0] += int((np.argsort(-p, axis=1)[:, :10][:, :, None] == rid[:, None, :]).any(2).sum()); agree[1] += rid.size
        c["need"][l] = frozenset(np.unique(rid).tolist())
        c["rid"][l], c["w"][l] = rid, (w / w.sum(1, keepdims=True)).astype(np.float32)
        if l in LAYERS:
            c["x"][l] = xr
    cks = [checks[s] for s in order if len(checks[s]["need"]) == NL and checks[s]["kept"] is not None]
    P.append((name, cks, [frozenset(np.nonzero(z["res0"][l] >= 0)[0].tolist()) for l in range(NL)]))
POP = []
for i in range(len(P)):
    pop, n = np.zeros((NL, E)), 0
    for j, (_, cks, _) in enumerate(P):
        if j != i:
            for c in cks:
                for l in range(NL):
                    pop[l, list(c["need"][l])] += 1
            n += len(cks)
    POP.append(pop / n)
NCK = sum(len(c) for _, c, _ in P[1:])
say(f"M-16 RUNAHEAD step 1: cap45, {NCK} live 16-row checks over prompts 2-6 (chained from prompt 1); live rule LRU + "
    f"{PRIOR:.0f} x popularity; recomputed router top-10 holds {100 * agree[0] / agree[1]:.2f} % of the captured ids "
    f"({time.time() - t0:.0f} s)")


def replay(K):
    """skip[p][t][l] = experts the check needs at layer l that are not resident at the check's start (its misses)"""
    skip = [[[None] * NL for _ in cks] for _, cks, _ in P]
    for l in range(NL):
        st, i = {}, 0
        for p, (_, cks, res0) in enumerate(P):
            pop = POP[p][l]
            for t, need in [(-1, res0[l])] + [(t, c["need"][l]) for t, c in enumerate(cks)]:
                i += 1
                if t >= 0:
                    skip[p][t][l] = frozenset(e for e in need if e not in st)
                for e in need:
                    st[e] = float(i)
                if len(st) > K:
                    for e in heapq.nsmallest(len(st) - K, (e for e in st if e not in need),
                                             key=lambda e: st[e] + PRIOR * pop[e]):
                        del st[e]
    return skip


def masks(c, l, sk):
    """per row [r, 10]: skipped (plain runahead) and skipped under the hybrid rule (misses needed at rank 0-1 are waited for)"""
    rid = c["rid"][l]
    m = np.isin(rid, list(sk)) if sk else np.zeros(rid.shape, bool)
    low = set(rid[:, :2][m[:, :2]].tolist())                               # misses some row needs at rank 0-1
    mh = m & ~np.isin(rid, list(low)) if low else m
    return m, mh, len(low)


SK = {}
for C, K in KS:
    wait_pause()
    sk = replay(K)
    SK[C] = sk
    mass, massh, rank0, anyskip, mtot, mlow, accum, accumh = [], [], 0, 0, 0, 0, [], []
    byrank = np.zeros(10)
    for p in range(1, len(P)):
        for t, c in enumerate(P[p][1]):
            acc_r = np.zeros(c["rid"][0].shape[0]); acc_rh = np.zeros_like(acc_r)
            for l in range(NL):
                m, mh, nlow = masks(c, l, sk[p][t][l])
                w = c["w"][l]
                ms, msh = (w * m).sum(1), (w * mh).sum(1)
                mass.append(ms); massh.append(msh)
                acc_r += ms; acc_rh += msh
                rank0 += int(m[:, 0].sum()); anyskip += int(m.any(1).sum())
                byrank += m.sum(0)
                mtot += len(sk[p][t][l]); mlow += nlow
            accum.append(acc_r); accumh.append(acc_rh)
    mass, massh = np.concatenate(mass), np.concatenate(massh)
    accum, accumh = np.concatenate(accum), np.concatenate(accumh)
    nrl = mass.size
    say(f"C {C} (K {K}): misses per 16-row check {mtot / NCK:.0f}; {100 * mlow / mtot:.1f} % of them are needed at rank 0-1 by "
        f"some row (hybrid waits for these, skips {100 * (1 - mlow / mtot):.1f} %)")
    say(f"  skipped routing mass per row-layer: mean {100 * mass.mean():.2f} %, P50 {100 * np.median(mass):.2f}, P90 "
        f"{100 * np.quantile(mass, .9):.2f}, P99 {100 * np.quantile(mass, .99):.2f}; row-layers with a skip "
        f"{100 * anyskip / nrl:.1f} %, with rank 0 skipped {100 * rank0 / nrl:.2f} %")
    say(f"  hybrid: mean {100 * massh.mean():.2f} %, P90 {100 * np.quantile(massh, .9):.2f}, P99 {100 * np.quantile(massh, .99):.2f}; "
        f"summed over 48 layers per row: plain {accum.mean():.2f}, hybrid {accumh.mean():.2f} (of 48)")
    say("  skipped uses by rank 0..9 (% of that rank's uses): " +
        " ".join(f"{100 * byrank[k] / nrl:.1f}" for k in range(10)))
say(f"({time.time() - t0:.0f} s)")

# ---- block error of the runahead variants on layers 2 / 12 / 24 / 36 / 46 (rows of prompts 2-6, all 16 rows per check)
VAR = ("skip", "renorm", "mean", "hybrid", "hybrid+mean")
num = {(C, v): 0.0 for C, _ in KS for v in VAR}
den, tier_num, tier_den = 0.0, 0.0, 0.0
for L in LAYERS:
    wait_pause()
    pre = f"language_model.model.layers.{L}.mlp."
    sw = pre + "switch_mlp."
    Xc = np.load(os.path.join(XD, f"x_L{L}.npy")).astype(np.float32)          # calibration tokens for the mean stand-ins
    lg = Xc @ WR[L].T
    idc = np.argpartition(-lg, 10, axis=1)[:, :10]
    rows = [(p, t) for p in range(1, len(P)) for t in range(len(P[p][1]))]
    Xr = np.concatenate([P[p][1][t]["x"][L] for p, t in rows])                 # [n, 2560]
    rid = np.concatenate([P[p][1][t]["rid"][L] for p, t in rows])
    w = np.concatenate([P[p][1][t]["w"][L] for p, t in rows])
    n = Xr.shape[0]
    M, MH = {}, {}
    for C, _ in KS:
        ms, mhs = [], []
        for p, t in rows:
            m, mh, _ = masks(P[p][1][t], L, SK[C][p][t][L])
            ms.append(m); mhs.append(mh)
        M[C], MH[C] = np.concatenate(ms), np.concatenate(mhs)
    sg, su, sd = (deq(pre + "shared_expert." + m)[4] for m in ("gate_proj", "up_proj", "down_proj"))
    sgate = deq(pre + "shared_expert_gate")[4]
    S = (1.0 / (1.0 + np.exp(-(Xr @ sgate.T)))) * (silu(Xr @ sg.T) * (Xr @ su.T)) @ sd.T
    Y4 = np.zeros((n, 10, 2560), np.float32)                                  # per-use outputs, 4-bit
    YT = np.zeros((n, 10, 2560), np.float32)                                  # per-use outputs at REF2-T's tier
    MU = np.zeros((n, 10, 2560), np.float32)                                  # per-use mean stand-ins
    for e in np.unique(rid):
        rr, kk = np.nonzero(rid == e)
        G = deq(sw + "gate_proj", e, (4, 3)); U = deq(sw + "up_proj", e, (4, 3)); D = deq(sw + "down_proj", e, (4, 3))
        xe = Xr[rr]
        y4 = (silu(xe @ G[4].T) * (xe @ U[4].T)) @ D[4].T
        y3 = (silu(xe @ G[3].T) * (xe @ U[3].T)) @ D[3].T
        Y4[rr, kk] = y4
        lo = np.array([int(PLAN[k]) == 4 for k in kk])
        YT[rr, kk] = np.where(lo[:, None], y4, y3)
        tc = np.nonzero((idc == e).any(1))[0]
        if tc.size:
            xc = Xc[tc]
            MU[rr, kk] = ((silu(xc @ G[4].T) * (xc @ U[4].T)) @ D[4].T).mean(0)
    y4 = (w[..., None] * Y4).sum(1) + S
    yT = (w[..., None] * YT).sum(1) + S
    d = (yT * yT).sum()
    den += d
    tier_num += ((yT - y4) ** 2).sum(); tier_den += (y4 * y4).sum()
    for C, _ in KS:
        keep, keeph = ~M[C], ~MH[C]
        ys = (w * keep)[..., None] * YT
        num[(C, "skip")] += ((ys.sum(1) + S - yT) ** 2).sum()
        wr = (w * keep) / np.maximum((w * keep).sum(1, keepdims=True), 1e-9)
        num[(C, "renorm")] += (((wr[..., None] * YT).sum(1) + S - yT) ** 2).sum()
        num[(C, "mean")] += ((ys.sum(1) + ((w * M[C])[..., None] * MU).sum(1) + S - yT) ** 2).sum()
        yh = ((w * keeph)[..., None] * YT).sum(1) + S
        num[(C, "hybrid")] += ((yh - yT) ** 2).sum()
        num[(C, "hybrid+mean")] += ((yh + ((w * MH[C])[..., None] * MU).sum(1) - yT) ** 2).sum()
    say(f"  layer {L}: {n} rows; REF2-T tier error so far {tier_num / tier_den:.5f}; skip at C 224 so far "
        f"{num[(224, 'skip')] / den:.5f} ({time.time() - t0:.0f} s)")
tier = tier_num / tier_den
say(f"block error vs the REF2-T block, 5 layers pooled (unit: REF2-T's own tier error vs 4-bit on these rows = {tier:.5f}):")
say("  C     " + "  ".join(f"{v:>12}" for v in VAR))
best = None
for C, _ in KS:
    errs = [num[(C, v)] / den for v in VAR]
    say(f"  {C}  " + "  ".join(f"{e:.5f} {e / tier:4.1f}x" for e in errs))
    if C == 224:
        best = min(zip(errs, VAR))
say(f"Lines (QUEUE M-16 step 1, C 224): best variant {best[1]} at {best[0] / tier:.2f}x REF2-T's tier error (PASS <= 2x with >= 70 % "
    f"of the misses skipped; KILL every variant > 5x)")
say(f"({time.time() - t0:.0f} s)")
open(os.path.expanduser("~/MoonShots/data/runahead_mass.txt"), "w").write("\n".join(lines) + "\n")
