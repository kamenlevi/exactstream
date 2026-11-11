"""LOSSY350-A replay 3 (Fable A, Thu 5 Nov 2026): can a cheap sketch pick the units of an expert that matter BEFORE the
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
rng = np.random.default_rng(0); npairs = 0; hold_tail = {}
for L in LAYERS:
    X = np.load(os.path.join(XD_, f"x_L{L}.npy")).astype(np.float32).reshape(-1, 2560)
    es = sorted(rng.choice(512, 8, replace=False).tolist())
    pre = f"language_model.model.layers.{L}.mlp.gate."
    R, _ = deq(tensor(pre + "weight"), tensor(pre + "scales"), tensor(pre + "biases"), 8)
    order = np.argsort(-(X @ R.T), axis=1)[:, :10]
    Xtr = X[:NTR].astype(np.float64); lam, V = np.linalg.eigh(Xtr.T @ Xtr); V = V[:, ::-1].astype(np.float32); del Xtr
    Xho = X[NTR:]; tot = e2(Xho)
    for j in (32, 64, 128, 256):
        hold_tail.setdefault(j, []).append(1 - e2(Xho @ V[:, :j]) / tot)
    for e in es:
        rows, rank = np.nonzero(order == e); rows = rows[rows >= NTR]
        if len(rows) < 4: continue
        npairs += len(rows)
        Wg, Wg3 = expert(L, "gate_proj", e); Wu, Wu3 = expert(L, "up_proj", e); Wd, Wd3 = expert(L, "down_proj", e)
        Xe = X[rows]; g = Xe @ Wg.T; u = Xe @ Wu.T; s = silu(g); h = s * u; y = h @ Wd.T; den = e2(y); ah = np.abs(h); ax = np.abs(Xe)
        def put(tag, yt): add(tag, e2(yt - y), den)
        put("MSB3 whole", (silu(Xe @ Wg3.T) * (Xe @ Wu3.T)) @ Wd3.T)
        for n in (160, 240, 320):
            put(f"TEALH-{n}", (h * topmask(ah, n)) @ Wd.T)
        for j in (32, 64, 128, 256):
            Vj = V[:, :j]; xh = (Xe @ Vj) @ Vj.T; gh = xh @ Wg.T; uh = xh @ Wu.T; hh = silu(gh) * uh
            for n in (160, 240, 320):
                m = topmask(np.abs(hh), n)
                put(f"SK{j}-{n}", (h * m) @ Wd.T)
                if j == 128: put(f"SK{j}F-{n}", np.where(m, h, hh) @ Wd.T)
            # a second score: sketch magnitude plus the residual's expected size (unit i's exposure to x_perp)
            if j == 128:
                xp = Xe - xh; rp = np.sqrt((xp.astype(np.float64) ** 2).sum(1, keepdims=True) / 2560).astype(np.float32)
                ng = np.sqrt((Wg.astype(np.float64) ** 2).sum(1)).astype(np.float32); nu = np.sqrt((Wu.astype(np.float64) ** 2).sum(1)).astype(np.float32)
                sc2 = (np.abs(silu(gh)) + 0.5 * rp * ng[None]) * (np.abs(uh) + rp * nu[None])
                for n in (240, 320):
                    put(f"SK128B-{n}", (h * topmask(sc2, n)) @ Wd.T)
            # input sparsity with deflation
            xp = Xe - xh
            for f in (0.5, 0.625):
                k = int(round(f * 2560)) - j; Xd = xh + xp * topmask(np.abs(xp), k)
                put(f"XD{j}-{f}", (silu(Xd @ Wg.T) * (Xd @ Wu.T)) @ Wd.T)
            Xd = xh + xp * gmask(np.abs(xp), 32); gd = Xd @ Wg.T; ud = Xd @ Wu.T; hd = silu(gd) * ud
            put(f"HKD{j}-x-32", hd @ Wd.T)
            put(f"HKD{j}-xh-32", (hd * gmask(np.abs(hd), 32)) @ Wd.T)
            if j in (32, 128):
                Xd24 = xh + xp * gmask(np.abs(xp), 24); h24 = silu(Xd24 @ Wg.T) * (Xd24 @ Wu.T)
                put(f"HKD{j}-x-24", h24 @ Wd.T)
        for f in (0.5, 0.625):
            Xs = Xe * topmask(ax, int(round(f * 2560))); put(f"XS-{f}", (silu(Xs @ Wg.T) * (Xs @ Wu.T)) @ Wd.T)
        Xk = Xe * gmask(ax, 32); hk = silu(Xk @ Wg.T) * (Xk @ Wu.T)
        put("HK-x-32", hk @ Wd.T); put("HK-h-32", (h * gmask(ah, 32)) @ Wd.T); put("HK-xh-32", (hk * gmask(np.abs(hk), 32)) @ Wd.T)
    say(f"layer {L} done at {time.time() - t0:.0f} s")
ref = acc["MSB3 whole"][0] / acc["MSB3 whole"][1]
say(f"\nheld-out routed pairs: {npairs}; held-out input energy outside the top-j train directions: " +
    "  ".join(f"j{j} {np.mean(v):.3f}" for j, v in hold_tail.items()))
say(f"variant | whole-expert error | x MSB-3 whole ({ref:.4f})")
for tag, (a, b) in acc.items():
    say(f"  {tag:18s} | {a / b:.4f} | {a / b / ref:5.2f}x")
say(f"weights touched {touched / 1e6:.0f} MB; done in {time.time() - t0:.0f} s")
open(os.path.expanduser("~/FableGod/lossy350a/sketch_units.out"), "w").write("\n".join(out) + "\n")
