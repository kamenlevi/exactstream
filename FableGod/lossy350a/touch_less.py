"""LOSSY350-A replay 2 (Fable A, Thu 5 Nov 2026): which weights of an expert can go untouched, on REAL weights and REAL
routed inputs. CPU, numpy, one process, 4 threads, nice 15. Weights: numpy mmap of ~/BigMoE/models/bare safetensors, read
only (orchestrator's ruling 16:1x: <= ~200 MB touched, no MLX, no GPU). Inputs: Fit12's Q2 capture x_L*.npy (MoE inputs).
Experts: Leap's L1-16 experts (layers 2/12/24/36/46, rng(0).choice(512, 8) per layer) so numbers line up with act16.txt.
Rows: each expert is evaluated on the tokens the layer's own router sends to it (top-10 of the router logits), with rank.
Metric: whole-expert OUTPUT error  sum ||y~ - y||^2 / sum ||y||^2 , y = Wd (silu(Wg x) * (Wu x)) at the 4-bit weights.
Reference scale: the same metric for the MSB-3 expert (all three matrices at 3 bits) = "1.0x".
Variants (h = silu(g) * u, 640 units; K = 2560 inputs):
  TEALH-n   oracle: keep the n largest |h| per row (needs g and u everywhere; the upper bound for unit sparsity)
  GF-n      gate-first: keep the n units with the largest |silu(g_i)| * ubar_i (ubar = static rms of u_i); up and down
            are then needed only for those units
  GFS-n     gate-first on |silu(g_i)| alone
  STAT-n    static unit order (by mean h^2 over the layer's tokens), prefix n
  XS-f      input sparsity: keep the f*2560 largest |x_j| per row for gate and up (TEAL on x)
  XD-f      deflated input sparsity: exact response to the top-128 principal directions + the largest |x_perp| coords
  ROTK-r    K-prefix in the rotated basis (train split), held-out rows only
  CAS-n     cascade: select units from a gate computed on XS-0.5 inputs, then exact h on the selected units
  STK-f-n   stack: XS-f gate -> select n units -> XS-f up on them -> down on them
Costs (share of the expert's weights touched, one-row group): GF (1 + 2a)/3; XS (2f + 1)/3; CAS (0.5 + 3a)/3;
STK (f + a f + a)/3 ; a = n / 640.
"""
import json, os, struct, sys, time
import numpy as np
MD = os.path.expanduser("~/BigMoE/models/bare")
XD = os.path.expanduser("~/Fit12/data/q2")
LAYERS = (2, 12, 24, 36, 46)
NTR = 5836
t0 = time.time()
out = []
def say(s=""):
    print(s, flush=True); out.append(s)
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
_h = {}
touched = 0
def tensor(name, sl=None):
    """memmap one tensor (or expert slice sl along axis 0); returns a numpy array copy of what is read"""
    global touched
    f = idx[name]; p = os.path.join(MD, f)
    if f not in _h:
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; _h[f] = (n, json.loads(fh.read(n)))
    n, h = _h[f]; t = h[name]
    dt = {"U32": "<u4", "BF16": "<u2"}[t["dtype"]]
    mm = np.memmap(p, dtype=dt, mode="r", offset=8 + n + t["data_offsets"][0], shape=tuple(t["shape"]))
    a = np.array(mm if sl is None else mm[sl]); touched += a.nbytes
    del mm
    if t["dtype"] == "BF16":
        a = (a.astype(np.uint32) << 16).view(np.float32)
    return a
def deq(w, s, b, bits):
    per = 32 // bits; m = (1 << bits) - 1
    codes = np.stack([(w >> (bits * i)) & m for i in range(per)], -1).reshape(w.shape[0], -1).astype(np.float32)
    S = np.repeat(s, 64, -1); B = np.repeat(b, 64, -1)
    W4 = S * codes + B
    W3 = 2 * S * np.floor(codes / 2) + B + S / 2 if bits == 4 else None
    return W4, W3
def expert(L, mat, e):
    pre = f"language_model.model.layers.{L}.mlp.switch_mlp.{mat}."
    return deq(tensor(pre + "weight", e), tensor(pre + "scales", e), tensor(pre + "biases", e), 4)
def silu(a):
    return a / (1.0 + np.exp(-a))
def topmask(score, n):
    """boolean mask of the n largest entries per row"""
    if n >= score.shape[1]:
        return np.ones(score.shape, bool)
    ix = np.argpartition(-score, n - 1, axis=1)[:, :n]
    m = np.zeros(score.shape, bool); np.put_along_axis(m, ix, True, 1); return m
NS = (80, 160, 240, 320, 480)           # units kept of 640
FS = (0.5, 0.625, 0.75)                  # inputs kept of 2560
acc = {}                                 # variant -> [num, den] ; also per rank class
def add(tag, num, den, cls=None):
    a = acc.setdefault(tag, [0.0, 0.0]); a[0] += num; a[1] += den
def e2(a):
    return float(np.sum(a.astype(np.float64) ** 2))
rng = np.random.default_rng(0)
xcheck = {"gate": [], "up": []}
unions = {}
conc = []
rows_total = 0
for L in LAYERS:
    X = np.load(os.path.join(XD, f"x_L{L}.npy")).astype(np.float32).reshape(-1, 2560)
    T = X.shape[0]
    es = sorted(rng.choice(512, 8, replace=False).tolist())
    # router (8-bit g64): top-10 experts per token
    pre = f"language_model.model.layers.{L}.mlp.gate."
    R, _ = deq(tensor(pre + "weight"), tensor(pre + "scales"), tensor(pre + "biases"), 8)
    logits = X @ R.T
    order = np.argsort(-logits, axis=1)[:, :10]                     # [T, 10] expert ids by rank
    top = np.take_along_axis(logits, order, 1); pw = np.exp(top - top.max(1, keepdims=True)); pw /= pw.sum(1, keepdims=True)
    use = np.bincount(order.ravel(), minlength=512)
    # rotated basis from the train split
    Xtr = X[:NTR].astype(np.float64); lam, V = np.linalg.eigh(Xtr.T @ Xtr); V = V[:, ::-1].astype(np.float32); del Xtr
    V128 = V[:, :128]
    say(f"\nlayer {L}: {T} tokens; Leap's experts {es}; routed rows per expert {[int(use[e]) for e in es]} (mean over 512: {use.mean():.0f});"
        f" top-10 router weight profile {np.round(pw.mean(0), 3).tolist()}")
    for e in es:
        Wg, Wg3 = expert(L, "gate_proj", e); Wu, Wu3 = expert(L, "up_proj", e); Wd, Wd3 = expert(L, "down_proj", e)
        # cross-check against Leap's act16 (all tokens, per matrix): tr(D H D^T) / tr(W H W^T)
        Ga = X @ Wg.T; Ua = X @ Wu.T
        xcheck["gate"].append(e2(X @ (Wg3 - Wg).T) / e2(Ga)); xcheck["up"].append(e2(X @ (Wu3 - Wu).T) / e2(Ua))
        Ha = silu(Ga) * Ua
        ubar = np.sqrt((Ua.astype(np.float64) ** 2).mean(0)).astype(np.float32)
        stat_order = np.argsort(-(Ha.astype(np.float64) ** 2).mean(0))
        rows, rank = np.nonzero(order == e)
        if len(rows) < 8:
            continue
        rows_total += len(rows)
        Xe = X[rows]; g = Ga[rows]; u = Ua[rows]; s = silu(g); h = s * u; y = h @ Wd.T
        den = e2(y)
        cls = np.where(rank <= 1, 0, np.where(rank <= 4, 1, 2))
        def put(tag, yt):
            d = (yt - y).astype(np.float64); pe = (d * d).sum(1); ye = (y.astype(np.float64) ** 2).sum(1)
            add(tag, pe.sum(), ye.sum())
            for c in (0, 1, 2):
                if (cls == c).any():
                    add(f"{tag}|c{c}", pe[cls == c].sum(), ye[cls == c].sum())
        # reference scale: MSB-3 whole expert, and its three parts
        g3 = Xe @ Wg3.T; u3 = Xe @ Wu3.T
        put("MSB3 whole", (silu(g3) * u3) @ Wd3.T)
        put("MSB3 gate only", (silu(g3) * u) @ Wd.T); put("MSB3 up only", (s * u3) @ Wd.T); put("MSB3 down only", h @ Wd3.T)
        # unit side
        ah = np.abs(h); hs = np.sort(ah.astype(np.float64) ** 2, axis=1)[:, ::-1]; cs = np.cumsum(hs, 1) / hs.sum(1, keepdims=True)
        conc.append([float(cs[:, n - 1].mean()) for n in NS] + [float((g < 0).mean()), float(g.std()), float(u.std())])
        for n in NS:
            put(f"TEALH-{n}", (h * topmask(ah, n)) @ Wd.T)
            mgf = topmask(np.abs(s) * ubar[None], n)
            put(f"GF-{n}", (h * mgf) @ Wd.T)
            put(f"GFS-{n}", (h * topmask(np.abs(s), n)) @ Wd.T)
            ms = np.zeros(640, bool); ms[stat_order[:n]] = True
            put(f"STAT-{n}", (h * ms[None]) @ Wd.T)
            if len(rows) >= 2:                                   # union of two rows' kept units (2-row groups)
                a_ = np.arange(len(rows)); b_ = np.roll(a_, 1)
                unions.setdefault(f"GF-{n}", []).append(float((mgf[a_] | mgf[b_]).sum(1).mean() / 640))
        # input side
        ax = np.abs(Xe)
        Xs = {}
        for f in FS:
            k = int(round(f * 2560)); mk = topmask(ax, k); Xs[f] = Xe * mk
            gt = Xs[f] @ Wg.T; ut = Xs[f] @ Wu.T
            put(f"XS-{f}", (silu(gt) * ut) @ Wd.T)
            put(f"XS-{f} gate only", (silu(gt) * u) @ Wd.T)
            if len(rows) >= 2:
                a_ = np.arange(len(rows)); b_ = np.roll(a_, 1)
                unions.setdefault(f"XS-{f}", []).append(float((mk[a_] | mk[b_]).sum(1).mean() / 2560))
            # deflated: exact top-128 part + sparse residual with the same number of touched columns
            ct = Xe @ V128; xt = ct @ V128.T; xp = Xe - xt
            kp = k - 128; xps = xp * topmask(np.abs(xp), kp); Xd = xt + xps
            put(f"XD-{f}", (silu(Xd @ Wg.T) * (Xd @ Wu.T)) @ Wd.T)

        # HALF-K: keep the top-c members of each 64-wide quant group (one matrix-unit op per quant group when c <= 32)
        def gmask(score, c):
            n_, K_ = score.shape; sc = score.reshape(n_, K_ // 64, 64)
            ix = np.argpartition(-sc, c - 1, axis=2)[:, :, :c]
            m = np.zeros(sc.shape, bool); np.put_along_axis(m, ix, True, 2); return m.reshape(n_, K_)
        cng = np.sqrt((Wg.astype(np.float64) ** 2).sum(0)).astype(np.float32); cnu = np.sqrt((Wu.astype(np.float64) ** 2).sum(0)).astype(np.float32)
        for c in (32, 24, 16):
            mx_ = gmask(ax, c); Xk = Xe * mx_
            gk = Xk @ Wg.T; uk = Xk @ Wu.T; hk = silu(gk) * uk
            put(f"HK-x-{c}", hk @ Wd.T)                                   # gate + up on half the inputs, down full
            mh_ = gmask(ah, c)
            put(f"HK-h-{c}", (h * mh_) @ Wd.T)                            # down on half the units, gate / up full
            put(f"HK-xh-{c}", (hk * gmask(np.abs(hk), c)) @ Wd.T)        # all three matrices
            if c == 32:
                put("HK-xh-32 gate full", ((s * uk) * gmask(np.abs(s * uk), c)) @ Wd.T)
                # column-norm weighted scores, per matrix
                gk2 = (Xe * gmask(ax * cng[None], c)) @ Wg.T; uk2 = (Xe * gmask(ax * cnu[None], c)) @ Wu.T; hk2 = silu(gk2) * uk2
                put("HK-xh-32 colnorm", (hk2 * gmask(np.abs(hk2), c)) @ Wd.T)
                # on the MSB-3 weights (the FAST preset's ranks 2-9), error vs the 4-bit expert
                gk3 = Xk @ Wg3.T; uk3 = Xk @ Wu3.T; hk3 = silu(gk3) * uk3
                put("HK-xh-32 on MSB3", (hk3 * gmask(np.abs(hk3), c)) @ Wd3.T)
                # 2-row groups: one joint member set per quant group (summed energy of the two rows), row a's error
                if len(rows) >= 2:
                    b_ = np.roll(np.arange(len(rows)), 1)
                    mj = gmask(ax ** 2 + ax[b_] ** 2, c); Xj = Xe * mj
                    hj = silu(Xj @ Wg.T) * (Xj @ Wu.T); ahj = np.abs(hj)
                    put("HK-xh-32 joint2", (hj * gmask(ahj ** 2 + ahj[b_] ** 2, c)) @ Wd.T)
                    unions.setdefault("HK-x-32 union>32 share of quant groups", []).append(float(((mx_ | mx_[b_]).reshape(len(rows), 40, 64).sum(2) > 32).mean()))
                    unions.setdefault("HK-x-32 mean union size of 64", []).append(float((mx_ | mx_[b_]).reshape(len(rows), 40, 64).sum(2).mean()))
        # rotated K-prefix, held-out rows only
        ho = rows >= NTR
        if ho.sum() >= 4:
            for r in (1280, 1920, 2240):
                Vr = V[:, :r]; Xr = (Xe[ho] @ Vr) @ Vr.T
                yt = (silu(Xr @ Wg.T) * (Xr @ Wu.T)) @ Wd.T
                add(f"ROTK-{r} (held out)", e2(yt - y[ho]), e2(y[ho]))
            add("MSB3 whole (held out rows)", e2(((silu(g3) * u3) @ Wd3.T)[ho] - y[ho]), e2(y[ho]))
        # cascade and stack
        gs = Xs[0.5] @ Wg.T
        for n in (160, 240, 320):
            put(f"CAS-{n}", (h * topmask(np.abs(silu(gs)) * ubar[None], n)) @ Wd.T)
            for f in (0.625, 0.75):
                gt = Xs[f] @ Wg.T; st = silu(gt); m = topmask(np.abs(st) * ubar[None], n)
                put(f"STK-{f}-{n}", ((st * (Xs[f] @ Wu.T)) * m) @ Wd.T)
    say(f"  layer {L} done at {time.time() - t0:.0f} s; weights touched so far {touched / 1e6:.0f} MB")
say(f"\ncross-check vs Leap act16 (all tokens, per matrix, mean of the 40 experts): MSB-3 gate {np.mean(xcheck['gate']):.4f} (Leap 0.0168), up {np.mean(xcheck['up']):.4f} (Leap 0.0223)")
say(f"routed (row, expert) pairs evaluated: {rows_total}")
c = np.array(conc)
say("h energy in the top-n units per routed row (mean over experts): " + "  ".join(f"n{n} {c[:, i].mean():.3f}" for i, n in enumerate(NS)) +
    f" ; share of gate pre-activations < 0: {c[:, -3].mean():.3f}; std g {c[:, -2].mean():.2f}, std u {c[:, -1].mean():.2f}")
ref = acc["MSB3 whole"][0] / acc["MSB3 whole"][1]
def cost(tag):
    p = tag.split("-")
    if p[0] in ("GF", "GFS", "STAT"): return (1 + 2 * int(p[1]) / 640) / 3
    if p[0] == "TEALH": return (2 + int(p[1]) / 640) / 3
    if p[0] in ("XS", "XD"): return (2 * float(p[1].split()[0]) + 1) / 3 if "gate only" not in tag else (float(p[1].split()[0]) + 2) / 3
    if p[0] == "HK": return {"x": lambda c: (2 * c + 1) / 3, "h": lambda c: (2 + c) / 3, "xh": lambda c: c}[p[1]](int(p[2].split()[0]) / 64)
    if p[0] == "CAS": return (0.5 + 3 * int(p[1]) / 640) / 3
    if p[0] == "STK": f = float(p[1]); a = int(p[2]) / 640; return (f + a * f + a) / 3
    if p[0] == "ROTK": return (2 * int(p[1].split()[0]) / 2560 + 1) / 3
    return 1.0
say(f"\nvariant | whole-expert error | x MSB-3 whole ({ref:.4f}) | weights touched (1-row group) | by rank class 0-1 / 2-4 / 5-9 (x MSB-3 of the class)")
for tag in [k for k in acc if "|" not in k]:
    v = acc[tag][0] / acc[tag][1]
    cl = []
    for cc in (0, 1, 2):
        k = f"{tag}|c{cc}"; rk = f"MSB3 whole|c{cc}"
        cl.append(f"{(acc[k][0] / acc[k][1]) / (acc[rk][0] / acc[rk][1]):.2f}" if k in acc and rk in acc else "-")
    say(f"  {tag:28s} | {v:.4f} | {v / ref:6.2f}x | {cost(tag):.3f} | {' / '.join(cl)}")
say("\n2-row groups, union of the two rows' kept sets (share of the axis): " + "  ".join(f"{k} {np.mean(v):.3f}" for k, v in unions.items()))
say(f"weights touched {touched / 1e6:.0f} MB; done in {time.time() - t0:.0f} s")
open(os.path.expanduser("~/FableGod/lossy350a/touch_less.out"), "w").write("\n".join(out) + "\n")
