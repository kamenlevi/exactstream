"""Activation-aware refit of the truncated 2-bit tier on REAL routed inputs (35B proxy, NewIdeas' capture data/cap35/L<l>.npz):
per expert, the 2-bit codes stay m2 = q >> 2 (nested layout); only the per-group scale/bias change. Methods:
  a trunc     stored truncation: 4s, b + s * mean(q & 3)                       (today's tier 2)
  b ls        per-group least squares in weight space                          (scripts/lsrefit2.py)
  c diag      per-group least squares weighted by the layer's channel second moments E[x_k^2]
  d act       per-row joint least squares of the outputs on the expert's own routed inputs (ridge toward a)
Fit on the even segments, scored on the odd ones: relative output error ||(W_hat - W) x|| / ||W x|| per projection
(gate/up on x; down on h = silu(gate x) * up x from the 4-bit weights). Experts: the NE most-routed of the layer.
Weights: the 8-bit unsloth 35B requantized to 4-bit g64 exactly as capture35.py did (on the CPU stream here).
usage: actrefit35.py [layer=3] [NE=24] [block]   (CPU only; ~minutes; "block": the whole MoE block, all 256
experts, plan 42222222 with truncated vs refit 2-bit ranks against the 4-bit block on held-out tokens)"""
import glob, os, sys, time
import numpy as np
import mlx.core as mx

L = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NE = int(sys.argv[2]) if len(sys.argv) > 2 else 24
mx.set_default_device(mx.cpu)
MODEL = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MLX-8bit/snapshots/*/"))[0]
t0 = time.time()
cap = np.load(os.path.expanduser(f"~/NewIdeas/data/cap35/L{L}.npz"))
X, I, seg = cap["X"].astype(np.float32), cap["I"].astype(np.int64), cap["seg"]
SPLIT = os.environ.get("REFIT_SPLIT", "parity")                         # parity | xdomain:<prose|gsm|code> (held-out source)
SRC = {"prose": 0, "gsm": 1, "code": 2}                                   # capture35: 4 segments per source, in this order
train = (seg % 2) == 0 if SPLIT == "parity" else (seg // 4) != SRC[SPLIT.split(":")[1]]
sig2 = (X[train] ** 2).mean(0)                                            # channel second moments (layer, all tokens)
pre = f"language_model.model.layers.{L}.mlp.switch_mlp."
import json
W = mx.load(MODEL + json.load(open(MODEL + "model.safetensors.index.json"))["weight_map"][pre + "gate_proj.weight"])
counts = np.bincount(I.ravel(), minlength=256)
experts = np.argsort(-counts)[:NE]
print(f"layer {L}: {len(X)} tokens ({train.sum()} train), experts {NE} most routed ({counts[experts].min()}-{counts[experts].max()} "
      f"rows each)", flush=True)


def q4(name, e):
    """4-bit g64 codes q [out, K], scale s, bias b [out, K/64] (float32) of expert e, requantized from the 8-bit weights"""
    w = mx.dequantize(W[pre + name + ".weight"][e], W[pre + name + ".scales"][e], W[pre + name + ".biases"][e], group_size=64, bits=8)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    mx.eval(wq, s, b)
    wq = np.array(wq)
    q = ((wq[..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 15).reshape(wq.shape[0], -1).astype(np.float32)
    return q, np.array(s.astype(mx.float32)), np.array(b.astype(mx.float32))


def bf16(a):
    return np.array(mx.array(a.astype(np.float32)).astype(mx.bfloat16).astype(mx.float32))


def fits(q, s, b, Xtr):
    """per-method (alpha, beta) [out, G] for the 2-bit tier of one projection"""
    out, K = q.shape
    G = K // 64
    m = np.floor(q / 4).reshape(out, G, 64)
    w = (s[..., None] * q.reshape(out, G, 64) + b[..., None])             # 4-bit weights [out, G, 64]
    res = {"a trunc": (s * 4, b + s * (q.reshape(out, G, 64) - 4 * m).mean(-1))}
    for name, wt in (("b ls", np.ones(K, np.float32)), ("c diag", sig2 if K == X.shape[1] else None)):
        if wt is None:
            continue
        wt = wt.reshape(1, G, 64)
        sw = wt.sum(-1)
        mm, wm = (wt * m).sum(-1) / sw, (wt * w).sum(-1) / sw
        var = (wt * (m - mm[..., None]) ** 2).sum(-1)
        cov = (wt * (m - mm[..., None]) * (w - wm[..., None])).sum(-1)
        alpha = np.where(var > 0, cov / np.maximum(var, 1e-30), 4 * s)
        res[name] = (alpha, wm - alpha * mm)
    # d act: per row, outputs y = sum_g alpha_g (x_g . m_g) + beta_g sum(x_g) fitted to x . w on the routed inputs
    if Xtr is None:
        res["d act"] = res["a trunc"]
        return {k: (bf16(a), bf16(bb)) for k, (a, bb) in res.items()}, m, w
    S = Xtr.shape[0]
    Xg = Xtr.reshape(S, G, 64)
    phi_a = np.einsum("sgk,ngk->nsg", Xg, m, optimize=True)               # [out, S, G]
    phi_b = np.broadcast_to(Xg.sum(-1)[None], (out, S, G))
    Phi = np.concatenate([phi_a, phi_b], -1)                              # [out, S, 2G]
    t = np.einsum("sgk,ngk->ns", Xg, w, optimize=True)                    # [out, S]
    th0 = np.concatenate(res["a trunc"], -1)                              # [out, 2G]
    A = np.einsum("nsi,nsj->nij", Phi, Phi, optimize=True)
    lam = float(os.environ.get("REFIT_LAMBDA", "1e-3")) * np.trace(A, axis1=1, axis2=2)[:, None, None] / (2 * G) + 1e-6
    rhs = np.einsum("nsi,ns->ni", Phi, t) + lam[..., 0] * th0
    th = np.linalg.solve(A + lam * np.eye(2 * G)[None], rhs[..., None])[..., 0]
    res["d act"] = (th[:, :G], th[:, G:])
    return {k: (bf16(a), bf16(bb)) for k, (a, bb) in res.items()}, m, w


def err(alpha, beta, m, w, Xte):
    out, G, _ = m.shape
    wh = alpha[..., None] * m + beta[..., None]
    y = Xte @ w.reshape(out, -1).T
    return np.linalg.norm(Xte @ (wh - w).reshape(out, -1).T) / np.linalg.norm(y)


if len(sys.argv) > 3 and sys.argv[3] == "block":
    S_ = cap["S"].astype(np.float32)
    te_rows = np.nonzero(~train)[0]
    pos = {t: i for i, t in enumerate(te_rows)}
    rank = np.argsort(np.argsort(-S_, axis=1), axis=1)                    # 0 = highest score in the row
    Yv = {k: np.zeros((len(te_rows), X.shape[1]), np.float32) for k in ("4bit", "a trunc", "d act")}
    silu = lambda g: g / (1 + np.exp(-g))
    for e in range(256):
        rows = np.nonzero((I == e).any(1))[0]
        tr, te = rows[train[rows]], rows[~train[rows]]
        if len(te) == 0:
            continue
        j = np.argmax(I[te] == e, axis=1)                                 # column of e in each test row
        sc, r1 = S_[te, j], rank[te, j] == 0
        W4, F = {}, {}
        for name in ("gate_proj", "up_proj", "down_proj"):
            q, s4, b4 = q4(name, e)
            if name == "down_proj":
                hh = lambda r: silu(X[r] @ W4["gate_proj"].T) * (X[r] @ W4["up_proj"].T)
                Xtr = hh(tr) if len(tr) >= 32 else np.zeros((0, q.shape[1]), np.float32)
            else:
                Xtr = X[tr] if len(tr) >= 32 else np.zeros((0, q.shape[1]), np.float32)
            fz, m, w = fits(q, s4, b4, Xtr if len(Xtr) else None)        # no calibration rows: truncation only
            W4[name] = w.reshape(w.shape[0], -1)
            F[name] = {k: (fz[k][0][..., None] * m + fz[k][1][..., None]).reshape(w.shape[0], -1) for k in ("a trunc", "d act")}
        xt = X[te]
        y4 = (silu(xt @ W4["gate_proj"].T) * (xt @ W4["up_proj"].T)) @ W4["down_proj"].T
        ii = np.array([pos[t] for t in te])
        Yv["4bit"][ii] += sc[:, None] * y4
        for k in ("a trunc", "d act"):
            y2 = (silu(xt @ F["gate_proj"][k].T) * (xt @ F["up_proj"][k].T)) @ F["down_proj"][k].T
            Yv[k][ii] += sc[:, None] * np.where(r1[:, None], y4, y2)
        if e % 32 == 31:
            print(f"  experts 0-{e} done ({time.time() - t0:.0f} s)", flush=True)
    ycap = cap["Y"][te_rows].astype(np.float32)
    nrm = np.linalg.norm(Yv["4bit"])
    print(f"layer {L} MoE block, plan 42222222 (rank 1 4-bit, ranks 2-8 2-bit), {len(te_rows)} held-out tokens: 4-bit block vs "
          f"captured Y rel {np.linalg.norm(Yv['4bit'] - ycap) / np.linalg.norm(ycap):.4f}", flush=True)
    for k in ("a trunc", "d act"):
        d = Yv[k] - Yv["4bit"]
        print(f"  {k}: block rel error {np.linalg.norm(d) / nrm:.4f} (MSE x{(np.linalg.norm(d) / np.linalg.norm(Yv['a trunc'] - Yv['4bit'])) ** 2:.3f})", flush=True)
    sys.exit(0)

tot = {}
for e in experts:
    rows = np.nonzero((I == e).any(1))[0]
    tr, te = rows[train[rows]], rows[~train[rows]]
    if len(tr) < 200 or len(te) < 100:
        continue
    line = f"e{e:3d} ({len(tr)}/{len(te)}):"
    acts = {}
    for name in ("gate_proj", "up_proj", "down_proj"):
        q, s, b = q4(name, e)
        if name == "down_proj":
            gq, uq = acts["gate_proj"], acts["up_proj"]
            h = lambda r: (lambda g, u: g / (1 + np.exp(-g)) * u)(X[r] @ gq.T, X[r] @ uq.T)
            Xtr, Xte = h(tr), h(te)
        else:
            Xtr, Xte = X[tr], X[te]
        F, m, w = fits(q, s, b, Xtr)
        acts[name] = w.reshape(w.shape[0], -1)
        es = {k: err(a, bb, m, w, Xte) for k, (a, bb) in F.items()}
        for k, v in es.items():
            tot.setdefault((name, k), []).append(v)
        line += f"  {name[:4]} " + " ".join(f"{k.split()[0]}{v:.4f}" for k, v in es.items())
    print(line, flush=True)
print(f"--- mean relative output error over experts (test segments), {time.time() - t0:.0f} s", flush=True)
for name in ("gate_proj", "up_proj", "down_proj"):
    base = np.mean(tot[(name, "a trunc")])
    print(f"{name:9s} " + "  ".join(f"{k}: {np.mean(tot[(name, k)]):.4f} (MSE x{(np.mean(tot[(name, k)]) / base) ** 2:.3f})"
                                    for k in ("a trunc", "b ls", "c diag", "d act") if (name, k) in tot), flush=True)
