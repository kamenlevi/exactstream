"""LOSSY350-A replay 4 (the test arms' own combos; loader as replay 3) (Fable A, Thu 5 Nov 2026): can a cheap sketch pick the units of an expert that matter BEFORE the
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
acc = {}
def add(tag, num, den):
    a = acc.setdefault(tag, [0.0, 0.0]); a[0] += num; a[1] += den
def gthr(score, c):
    """adaptive (TEAL-style threshold): one magnitude threshold per row set so the MEAN kept count per 64-group is c"""
    n_, K_ = score.shape; k = K_ * c // 64
    thr = -np.partition(-score, k - 1, axis=1)[:, k - 1:k]
    return score >= thr
rng = np.random.default_rng(0); npairs = 0
J = 0
for L in LAYERS:
    X = np.load(os.path.join(XD_, f"x_L{L}.npy")).astype(np.float32).reshape(-1, 2560)
    es = sorted(rng.choice(512, 8, replace=False).tolist())
    pre = f"language_model.model.layers.{L}.mlp.gate."
    R, _ = deq(tensor(pre + "weight"), tensor(pre + "scales"), tensor(pre + "biases"), 8)
    order = np.argsort(-(X @ R.T), axis=1)[:, :10]
    Xtr = X[:NTR].astype(np.float64); lam, V = np.linalg.eigh(Xtr.T @ Xtr); del Xtr, V
    for e in es:
        rows, rank = np.nonzero(order == e); keep = rows >= NTR; rows = rows[keep]; rank = rank[keep]
        if len(rows) < 4: continue
        npairs += len(rows)
        Wg, Wg3 = expert(L, "gate_proj", e); Wu, Wu3 = expert(L, "up_proj", e); Wd, Wd3 = expert(L, "down_proj", e)
        Xe = X[rows]; g = Xe @ Wg.T; u = Xe @ Wu.T; h = silu(g) * u; y = h @ Wd.T
        ye = (y.astype(np.float64) ** 2).sum(1); cls = np.where(rank <= 1, 0, 1)
        def put(tag, yt):
            pe = ((yt - y).astype(np.float64) ** 2).sum(1); add(tag, pe.sum(), ye.sum())
            for c in (0, 1):
                if (cls == c).any(): add(f"{tag}|c{c}", pe[cls == c].sum(), ye[cls == c].sum())
        y3 = (silu(Xe @ Wg3.T) * (Xe @ Wu3.T)) @ Wd3.T
        put("MSB3 whole", y3)
        xh = np.zeros_like(Xe); xp = Xe; axp = np.abs(xp)
        for cx in (48, 40, 32):
            Xd = xh + xp * gmask(axp, cx); hd = silu(Xd @ Wg.T) * (Xd @ Wu.T)
            put(f"D32 x{cx}", hd @ Wd.T)
            for ch in (40, 32, 24):
                put(f"D32 x{cx} h{ch}", (hd * gmask(np.abs(hd), ch)) @ Wd.T)
            if cx in (40, 32):
                # the same on the MSB-3 weights (the preset's ranks 2-9): total error vs the 4-bit expert
                hd3 = silu(Xd @ Wg3.T) * (Xd @ Wu3.T)
                put(f"D32 x{cx} h32 on MSB3", (hd3 * gmask(np.abs(hd3), 32)) @ Wd3.T)
                # threshold form (adaptive count per group, same mean)
                Xa = xh + xp * gthr(axp, cx); ha = silu(Xa @ Wg.T) * (Xa @ Wu.T)
                put(f"D32 x{cx} h32 threshold form", (ha * gthr(np.abs(ha), 32)) @ Wd.T)
    say(f"layer {L} done at {time.time() - t0:.0f} s")
ref = {c: acc[f"MSB3 whole{c}"][0] / acc[f"MSB3 whole{c}"][1] for c in ("", "|c0", "|c1")}
say(f"\nheld-out routed pairs: {npairs}; deflation j = {J}; MSB-3 whole-expert error all / ranks 0-1 / ranks 2-9: {ref['']:.4f} / {ref['|c0']:.4f} / {ref['|c1']:.4f}")
say("arm (x = members kept of each 64 inputs for gate and up, on the deflated input; h = units kept of each 64 for down) | weights touched | error x MSB-3: all / ranks 0-1 / ranks 2-9")
for tag in [k for k in acc if "|" not in k]:
    p = tag.split(); cx = int(p[1][1:]) if len(p) > 1 and p[1][0] == "x" else 64; ch = int(p[2][1:]) if len(p) > 2 and p[2][0] == "h" else 64
    cost = (2 * cx / 64 + ch / 64) / 3 + (2 * 640 * J) / 4915200 if tag != "MSB3 whole" else 1.0
    say(f"  {tag:30s} | {cost:.3f} | " + " / ".join(f"{acc[tag + c][0] / acc[tag + c][1] / ref[c]:.2f}" for c in ("", "|c0", "|c1")))
say(f"weights touched {touched / 1e6:.0f} MB; done in {time.time() - t0:.0f} s")
open(os.path.expanduser("~/FableGod/lossy350a/arms_plain.out"), "w").write("\n".join(out) + "\n")
