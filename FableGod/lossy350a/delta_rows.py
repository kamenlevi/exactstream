"""LOSSY350-A replay 5: is the next row's MoE input close to an earlier row's (so an expert could reuse its last result
and compute only on the difference)? Inputs only (Fit12 Q2) + the routers (8-bit, 1.3 MB per layer). CPU, numpy.
Reports per layer: cos(x_t, x_{t+1}) raw and after removing the top-32 train directions; the same for consecutive USES of
one expert (the rows the router sends to it, in token order); and the energy of the difference relative to the row's own
energy after deflation (a delta scheme pays only if this is well below 1)."""
import json, os, struct, time, numpy as np
MD = os.path.expanduser("~/BigMoE/models/bare"); XD_ = os.path.expanduser("~/Fit12/data/q2"); NTR = 5836
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]; _h = {}
def tensor(name):
    f = idx[name]; p = os.path.join(MD, f)
    if f not in _h:
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; _h[f] = (n, json.loads(fh.read(n)))
    n, h = _h[f]; t = h[name]; dt = {"U32": "<u4", "BF16": "<u2"}[t["dtype"]]
    a = np.array(np.memmap(p, dtype=dt, mode="r", offset=8 + n + t["data_offsets"][0], shape=tuple(t["shape"])))
    return (a.astype(np.uint32) << 16).view(np.float32) if t["dtype"] == "BF16" else a
out = []
def say(s):
    print(s, flush=True); out.append(s)
def cosr(A, B):
    return (A * B).sum(1) / np.sqrt((A * A).sum(1) * (B * B).sum(1) + 1e-30)
for L in (2, 12, 24, 36, 46):
    X = np.load(os.path.join(XD_, f"x_L{L}.npy")).astype(np.float64).reshape(-1, 2560)
    w, s, b = (tensor(f"language_model.model.layers.{L}.mlp.gate.{k}") for k in ("weight", "scales", "biases"))
    codes = np.stack([(w >> (8 * i)) & 255 for i in range(4)], -1).reshape(512, -1).astype(np.float64)
    R = np.repeat(s, 64, -1) * codes + np.repeat(b, 64, -1)
    order = np.argsort(-(X @ R.T), axis=1)[:, :10]
    lam, V = np.linalg.eigh(X[:NTR].T @ X[:NTR]); V = V[:, ::-1][:, :32]
    Xp = X - (X @ V) @ V.T
    c_raw = cosr(X[1:], X[:-1]); c_def = cosr(Xp[1:], Xp[:-1])
    d_def = ((Xp[1:] - Xp[:-1]) ** 2).sum(1) / (Xp[1:] ** 2).sum(1)
    # consecutive uses of the same expert (any rank), and the nearest of the last 16 tokens that used it
    cs, ds, gaps = [], [], []
    last = {}
    for t in range(X.shape[0]):
        for e in order[t]:
            if e in last:
                p = last[e]; cs.append(float(Xp[t] @ Xp[p] / np.sqrt((Xp[t] @ Xp[t]) * (Xp[p] @ Xp[p]))))
                ds.append(float(((Xp[t] - Xp[p]) ** 2).sum() / (Xp[t] ** 2).sum())); gaps.append(t - p)
            last[e] = t
    cs = np.array(cs); ds = np.array(ds); gaps = np.array(gaps)
    near = gaps <= 16
    say(f"layer {L}: adjacent tokens cos raw {c_raw.mean():.3f}, deflated {c_def.mean():.3f}, difference energy / row energy (deflated) {d_def.mean():.2f};"
        f" same-expert previous use: {len(cs)} pairs, {near.mean():.2f} within 16 tokens; cos deflated {cs.mean():.3f} (near {cs[near].mean():.3f});"
        f" difference energy {ds.mean():.2f} (near {ds[near].mean():.2f}, p10 {np.quantile(ds[near], 0.1):.2f})")
open(os.path.expanduser("~/FableGod/lossy350a/delta_rows.out"), "w").write("\n".join(out) + "\n")
