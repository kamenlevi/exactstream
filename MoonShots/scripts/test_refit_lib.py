"""CPU check of refit_lib.fit_proj (MLX, used by refit35.py on the GPU) against the numpy fit validated in actrefit35.py (held-out
error), on real 35B routed inputs (NewIdeas' cap35/L3) for the most-routed expert, tiers 3 and 2, gate and down.
usage: test_refit_lib.py   (CPU only)"""
import glob, json, os, sys
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refit_lib import fit_proj, codes, solve

mx.set_default_device(mx.cpu)
MODEL = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MLX-8bit/snapshots/*/"))[0]
cap = np.load(os.path.expanduser("~/NewIdeas/data/cap35/L3.npz"))
X, I, seg = cap["X"].astype(np.float32), cap["I"].astype(np.int64), cap["seg"]
pre = "language_model.model.layers.3.mlp.switch_mlp."
W = mx.load(MODEL + json.load(open(MODEL + "model.safetensors.index.json"))["weight_map"][pre + "gate_proj.weight"])
e = int(np.argmax(np.bincount(I.ravel(), minlength=256)))
rows = np.nonzero((I == e).any(1))[0]
tr, te = rows[seg[rows] % 2 == 0], rows[seg[rows] % 2 == 1]


def q4(name):
    w = mx.dequantize(W[pre + name + ".weight"][e], W[pre + name + ".scales"][e], W[pre + name + ".biases"][e], group_size=64, bits=8)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    return wq, s.astype(mx.float32), b.astype(mx.float32)


def numpy_fit(q, s, b, Xtr, n):
    """actrefit35.fits 'd act' for tier n (ridge 1e-3 * tr / 2G + 1e-6 toward the truncation)"""
    out, K = q.shape
    G, sh = K // 64, 4 - n
    m = np.floor(q / 2 ** sh).reshape(out, G, 64)
    w = s[..., None] * q.reshape(out, G, 64) + b[..., None]
    th0 = np.concatenate([s * 2 ** sh, b + s * (q.reshape(out, G, 64) - 2 ** sh * m).mean(-1)], -1)
    Xg = Xtr.reshape(len(Xtr), G, 64)
    Phi = np.concatenate([np.einsum("sgk,ngk->nsg", Xg, m, optimize=True), np.broadcast_to(Xg.sum(-1)[None], (out, len(Xtr), G))], -1)
    t = np.einsum("sgk,ngk->ns", Xg, w, optimize=True)
    A = np.einsum("nsi,nsj->nij", Phi, Phi, optimize=True)
    lam = 1e-3 * np.trace(A, axis1=1, axis2=2)[:, None, None] / (2 * G) + 1e-6
    th = np.linalg.solve(A + lam * np.eye(2 * G)[None], (np.einsum("nsi,ns->ni", Phi, t) + lam[..., 0] * th0)[..., None])[..., 0]
    return th[:, :G], th[:, G:], m, w


ok = True
wg = {}
for name in ("gate_proj", "up_proj", "down_proj"):
    wq, s, b = q4(name)
    qe = codes(wq[None])[0]
    q, sn, bn = np.array(qe).astype(np.float32), np.array(s), np.array(b)
    wg[name] = (sn[..., None] * q.reshape(*sn.shape, 64) + bn[..., None]).reshape(q.shape)
    if name == "up_proj":
        continue
    if name == "down_proj":
        h = lambda r: (lambda g, u: g / (1 + np.exp(-g)) * u)(X[r] @ wg["gate_proj"].T, X[r] @ wg["up_proj"].T)
        Xtr, Xte = h(tr), h(te)
    else:
        Xtr, Xte = X[tr], X[te]
    for n in (3, 2):
        A, rhs, G = fit_proj(mx.array(Xtr), qe, s, b, n)
        th = solve(np.array(A), np.array(rhs))
        a_m, b_m = th[:, :G], th[:, G:]
        a_n, b_n, m, w = numpy_fit(q, sn, bn, Xtr, n)
        ra = np.abs(a_m - a_n).max() / np.abs(a_n).max()
        rb = np.abs(b_m - b_n).max() / np.abs(b_n).max()
        err = lambda a, bb: np.linalg.norm(Xte @ ((a[..., None] * m + bb[..., None]) - w).reshape(len(w), -1).T) / np.linalg.norm(Xte @ w.reshape(len(w), -1).T)
        e_m, e_n = err(a_m, b_m), err(a_n, b_n)
        good = ra < 1e-2 and rb < 1e-2 and abs(e_m - e_n) < 1e-3
        ok &= good
        print(f"expert {e} {name} tier {n}: max rel diff alpha {ra:.2e} beta {rb:.2e}; held-out rel error MLX {e_m:.4f} numpy {e_n:.4f} "
              f"-> {'ok' if good else 'MISMATCH'}", flush=True)
print("PASS" if ok else "FAIL", flush=True)
