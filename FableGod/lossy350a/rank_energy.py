"""LOSSY350-A replay 7 (output energy by router rank; loader as replay 3) (Fable A, Thu 5 Nov 2026): can a cheap sketch pick the units of an expert that matter BEFORE the
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
rng = np.random.default_rng(0)
en = np.zeros(10); cnt = np.zeros(10); pe = np.zeros(10); en3 = np.zeros(10); pw_all = np.zeros(10); xs_e = np.zeros(10)
for L in LAYERS:
    X = np.load(os.path.join(XD_, f"x_L{L}.npy")).astype(np.float32).reshape(-1, 2560)
    es = sorted(rng.choice(512, 8, replace=False).tolist())
    pre = f"language_model.model.layers.{L}.mlp.gate."
    R, _ = deq(tensor(pre + "weight"), tensor(pre + "scales"), tensor(pre + "biases"), 8)
    lg = X @ R.T; order = np.argsort(-lg, axis=1)[:, :10]
    top = np.take_along_axis(lg, order, 1); z = np.exp(lg - lg.max(1, keepdims=True)); z /= z.sum(1, keepdims=True)
    p = np.take_along_axis(z, order, 1); p /= p.sum(1, keepdims=True)                   # renormalized top-10 weights
    for e in es:
        rows, rank = np.nonzero(order == e)
        if len(rows) < 8: continue
        Wg, Wg3 = expert(L, "gate_proj", e); Wu, Wu3 = expert(L, "up_proj", e); Wd, Wd3 = expert(L, "down_proj", e)
        Xe = X[rows]; h = silu(Xe @ Wg.T) * (Xe @ Wu.T); y = h @ Wd.T
        ye = (y.astype(np.float64) ** 2).sum(1); xe = (Xe.astype(np.float64) ** 2).sum(1)
        y3 = (silu(Xe @ Wg3.T) * (Xe @ Wu3.T)) @ Wd3.T; e3 = ((y3 - y).astype(np.float64) ** 2).sum(1)
        w = p[rows, rank]
        np.add.at(en, rank, ye / xe); np.add.at(cnt, rank, 1); np.add.at(pe, rank, w * w * ye / xe); np.add.at(en3, rank, e3 / xe); np.add.at(pw_all, rank, w)
say("by router rank 0..9, over the routed pairs of Leap's 40 experts:")
say("  pairs:                 " + "  ".join(f"{int(c)}" for c in cnt))
say("  mean router weight p:  " + "  ".join(f"{v:.3f}" for v in pw_all / cnt))
say("  mean ||y||^2 / ||x||^2: " + "  ".join(f"{v:.4f}" for v in en / cnt))
m = pe / cnt
say("  share of the block's routed energy, p^2 ||y||^2 (normalised): " + "  ".join(f"{v:.3f}" for v in m / m.sum()))
say(f"  ranks 0-1 share: {m[:2].sum() / m.sum():.3f}; ranks 2-4: {m[2:5].sum() / m.sum():.3f}; ranks 5-9: {m[5:].sum() / m.sum():.3f}")
say("  MSB-3 whole-expert error by rank (err / ||y||^2): " + "  ".join(f"{v:.3f}" for v in en3 / en))
open(os.path.expanduser("~/FableGod/lossy350a/rank_energy.out"), "w").write("\n".join(out) + "\n")
