"""LOSSY350-A replay 6 (each expert's own input directions; per-layer view; loader as replay 3) (Fable A, Thu 5 Nov 2026): can a cheap sketch pick the units of an expert that matter BEFORE the
expert runs, and does deflating the dominant input directions help input sparsity on HELD-OUT rows?
Same weights, experts, routing and metric as touch_less.py (whole-expert output error vs the 4-bit expert; MSB-3 whole
expert = 1.0x). Basis V from the train split (tokens 0..5835); every number below is on held-out routed rows (>= 5836).
  TEALH-n     oracle: n largest |h| per row
  SKj-n       sketch: x^ = V_j V_j^T x (j principal directions; per expert this is two [640 x j] response matrices,
              2*640*j weights = j/3840 of the expert), score |silu(Wg x^) * (Wu x^)|, keep the top n units, then the
              exact h on those units only (gate, up and down touched for n units)
  SKjF-n      the same, and the sketch's own h^ fills the dropped units' gate / up (down still needs their columns: reported
              only to see how much of the dropped units' output the sketch carries)
  XS / XD     input sparsity without / with deflation of j directions (global top-k of |x| or |x_perp|)
  HK / HKD    per-64-group top-32 without / with deflation; HKDxh adds the down on the top-32 units of each 64-group
"""
import json, os, struct, time
import numpy as np
MD = os.path.expanduser("~/BigMoE/models/bare"); XD_ = os.path.expanduser("~/Fit12/data/q2")
LAYERS = (2, 12, 24, 36, 46); NTR = 5836
t0 = time.time(); out = []
def say(s=""):
    print(s, flush=True); out.append(s)
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]; _h = {}; touched = 0
def tensor(name, sl=None):
    global touched
    f = idx[name]; p = os.path.join(MD, f)
    if f not in _h:
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; _h[f] = (n, json.loads(fh.read(n)))
    n, h = _h[f]; t = h[name]; dt = {"U32": "<u4", "BF16": "<u2"}[t["dtype"]]
    mm = np.memmap(p, dtype=dt, mode="r", offset=8 + n + t["data_offsets"][0], shape=tuple(t["shape"]))
    a = np.array(mm if sl is None else mm[sl]); touched += a.nbytes; del mm
    return (a.astype(np.uint32) << 16).view(np.float32) if t["dtype"] == "BF16" else a
def deq(w, s, b, bits):
    per = 32 // bits; m = (1 << bits) - 1
    codes = np.stack([(w >> (bits * i)) & m for i in range(per)], -1).reshape(w.shape[0], -1).astype(np.float32)
    S = np.repeat(s, 64, -1); B = np.repeat(b, 64, -1)
    return S * codes + B, (2 * S * np.floor(codes / 2) + B + S / 2 if bits == 4 else None)
def expert(L, mat, e):
    pre = f"language_model.model.layers.{L}.mlp.switch_mlp.{mat}."
    return deq(tensor(pre + "weight", e), tensor(pre + "scales", e), tensor(pre + "biases", e), 4)
def silu(a): return a / (1.0 + np.exp(-a))
def topmask(score, n):
    if n >= score.shape[1]: return np.ones(score.shape, bool)
    ix = np.argpartition(-score, n - 1, axis=1)[:, :n]; m = np.zeros(score.shape, bool); np.put_along_axis(m, ix, True, 1); return m
def gmask(score, c):
    n_, K_ = score.shape; sc = score.reshape(n_, K_ // 64, 64); ix = np.argpartition(-sc, c - 1, axis=2)[:, :, :c]
    m = np.zeros(sc.shape, bool); np.put_along_axis(m, ix, True, 2); return m.reshape(n_, K_)
def e2(a): return float(np.sum(a.astype(np.float64) ** 2))
acc = {}; per_layer = {}
def add(tag, num, den):
    a = acc.setdefault(tag, [0.0, 0.0]); a[0] += num; a[1] += den
rng = np.random.default_rng(0); npairs = 0; stats = []
for L in LAYERS:
    X = np.load(os.path.join(XD_, f"x_L{L}.npy")).astype(np.float32).reshape(-1, 2560)
    es = sorted(rng.choice(512, 8, replace=False).tolist())
    pre = f"language_model.model.layers.{L}.mlp.gate."
    R, _ = deq(tensor(pre + "weight"), tensor(pre + "scales"), tensor(pre + "biases"), 8)
    order = np.argsort(-(X @ R.T), axis=1)[:, :10]
    Xtr = X[:NTR].astype(np.float64); lam, V = np.linalg.eigh(Xtr.T @ Xtr); V32 = V[:, ::-1][:, :32].astype(np.float32); del Xtr, V
    for e in es:
        rows, rank = np.nonzero(order == e)
        tr = rows[rows < NTR]; ho = rows[rows >= NTR]
        if len(tr) < 60 or len(ho) < 20: continue
        npairs += len(ho)
        Wg, Wg3 = expert(L, "gate_proj", e); Wu, Wu3 = expert(L, "up_proj", e); Wd, Wd3 = expert(L, "down_proj", e)
        Xe = X[ho]; g = Xe @ Wg.T; u = Xe @ Wu.T; h = silu(g) * u; y = h @ Wd.T; den = e2(y)
        def put(tag, yt):
            add(tag, e2(yt - y), den)
            a = per_layer.setdefault((tag, L), [0.0, 0.0]); a[0] += e2(yt - y); a[1] += den
        put("MSB3 whole", (silu(Xe @ Wg3.T) * (Xe @ Wu3.T)) @ Wd3.T)
        # layer-wide deflation (32), then the expert's OWN directions from its train rows (after the layer deflation)
        xh = (Xe @ V32) @ V32.T; xp = Xe - xh
        Xt = X[tr]; Xtp = (Xt - (Xt @ V32) @ V32.T).astype(np.float64)
        mu = Xtp.mean(0); Ue, se, Vte = np.linalg.svd(Xtp, full_matrices=False)       # rows < 2560: the expert's own subspace
        e_res = e2(xp)
        row = [L, e, len(tr), len(ho), 1 - e_res / e2(Xe)]
        Xd = xh + xp * gmask(np.abs(xp), 32); hd = silu(Xd @ Wg.T) * (Xd @ Wu.T)
        put("layer 32 | x32 h32", (hd * gmask(np.abs(hd), 32)) @ Wd.T)
        for j in (4, 16, 48):
            if j > len(tr) - 1: continue
            Pj = Vte[:j].T.astype(np.float32); xo = (xp @ Pj) @ Pj.T; xr = xp - xo
            row.append(1 - e2(xr) / e_res)
            Xd = xh + xo + xr * gmask(np.abs(xr), 32); hd = silu(Xd @ Wg.T) * (Xd @ Wu.T)
            put(f"layer 32 + own {j} | x32 h32", (hd * gmask(np.abs(hd), 32)) @ Wd.T)
        stats.append(row)
    say(f"layer {L} done at {time.time() - t0:.0f} s")
ref = acc["MSB3 whole"][0] / acc["MSB3 whole"][1]
say(f"\nexperts with >= 60 train and >= 20 held-out routed rows: {len(stats)}; held-out pairs {npairs}")
say("energy the layer's 32 directions remove from an expert's own held-out inputs (mean over experts): " + f"{np.mean([r[4] for r in stats]):.3f}")
say("extra share of the REMAINING energy removed by the expert's own top-j directions (held out): " +
    "  ".join(f"j{j} {np.mean([r[5 + i] for r in stats if len(r) > 5 + i]):.3f}" for i, j in enumerate((4, 16, 48))))
say("variant | error x MSB-3 whole")
for tag, (a, b) in acc.items(): say(f"  {tag:28s} | {a / b / ref:5.2f}")
say("per layer, x32 h32 with the layer's 32 directions (x MSB-3 of that layer): " +
    "  ".join(f"L{L} {per_layer[('layer 32 | x32 h32', L)][0] / per_layer[('layer 32 | x32 h32', L)][1] / (per_layer[('MSB3 whole', L)][0] / per_layer[('MSB3 whole', L)][1]):.2f}" for L in LAYERS if ('MSB3 whole', L) in per_layer))
say("per layer MSB-3 whole-expert error: " + "  ".join(f"L{L} {per_layer[('MSB3 whole', L)][0] / per_layer[('MSB3 whole', L)][1]:.4f}" for L in LAYERS if ('MSB3 whole', L) in per_layer))
open(os.path.expanduser("~/FableGod/lossy350a/own_dirs.out"), "w").write("\n".join(out) + "\n")
