"""ROLLOUT (results/ROLLOUT.md, lines written first): a learned transition W_{t+1} ~ f(W_t, emb(tok_{t+1})) of the target's
top state (the head's input: the post-norm final hidden), rolled out from fresh anchors and fed to s30 as its stand-in.
  fit    linear ridge on [W_t / sW, emb(tok_{t+1}) / sE] -> W_{t+1} (each half scaled to unit RMS, so one ridge penalty
         treats them alike), base answers' answer rows; lambda by held-out direction error -> ckpt/rollout_lin.npz
  mlp    a residual MLP on top (4096 -> 2048 -> 2048, zero-init output), direction loss on one-step pairs -> ckpt/rollout_mlp.safetensors
  curve  the rollout's direction error by depth k = 1..30 on MBPP 100 (and one-step on val) -> results/rollout_curve.json
The stand-in hooks are levers/stale.py, levers/rollout_lin.py, levers/rollout_mlp.py (rollout tables built per window here).
usage: rollout.py fit | mlp | curve"""
import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
import json, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from sb import core as C

LIN = os.path.join(C.CKPT, "rollout_lin.npz")
MLPW = os.path.join(C.CKPT, "rollout_mlp.safetensors")


def split_files():
    s = json.load(open(os.path.join(C.DATA, "split.json")))
    return ([os.path.join(C.DATA, "mc", k + ".npz") for k in s["base"]], [os.path.join(C.DATA, "mc", k + ".npz") for k in s["val"]])


def states(f):
    """W [S, 2048] (post-norm final hidden), E [S, 2048] (token embeddings), first answer anchor"""
    z = np.load(f)
    W = np.array(C.load_W(z)[0].astype(mx.float32))
    E = np.array(C.embed(mx.array(np.asarray(z["ids"], np.int32))))
    return W, E, C.first_anchor(z["asst"])


def direr(P, Y):
    """sums for the RMS-normalized relative error (fitmap's rel_err_normalized)"""
    pn = P / np.sqrt((P ** 2).mean(-1, keepdims=True) + 1e-12); yn = Y / np.sqrt((Y ** 2).mean(-1, keepdims=True) + 1e-12)
    return float(((pn - yn) ** 2).sum()), float((yn ** 2).sum())


def cmd_fit():
    tr, va = split_files()
    t0 = time.time()
    sW = sE = n0 = 0.0
    for f in tr[:40]:                                    # the two halves' RMS, from a subset
        W, E, a = states(f)
        sW += float((W[a:] ** 2).sum()); sE += float((E[a:] ** 2).sum()); n0 += W[a:].size
    sW, sE = (sW / n0) ** 0.5, (sE / n0) ** 0.5
    din = 4096
    Sxx = np.zeros((din, din)); Sxy = np.zeros((din, 2048)); sx = np.zeros(din); sy = np.zeros(2048); n = 0; nrm = []
    for f in tr:
        W, E, a = states(f)
        X = np.concatenate([W[a:-1] / sW, E[a + 1:] / sE], -1); Y = W[a + 1:]
        Sxx += (X.T @ X).astype(np.float64); Sxy += (X.T @ Y).astype(np.float64); sx += X.sum(0); sy += Y.sum(0); n += len(X)
        nrm.append(np.linalg.norm(Y, axis=-1))
    xm, ym = sx / n, sy / n
    XtX = Sxx - n * np.outer(xm, xm); XtY = Sxy - n * np.outer(xm, ym)
    V = []
    for f in va:
        W, E, a = states(f)
        V.append((np.concatenate([W[a:-1] / sW, E[a + 1:] / sE], -1), W[a + 1:], W[a:-1]))
    num0 = den0 = 0.0
    for X, Y, Wp in V:                                   # "no change" (W_{t+1} = W_t): the stale one-step baseline
        a_, b_ = direr(Wp, Y); num0 += a_; den0 += b_
    print(f"[ro] {n} training pairs from {len(tr)} files; sW {sW:.3f} sE {sE:.4f}; stale one-step direction error "
          f"{(num0 / den0) ** 0.5:.4f}; sums in {time.time() - t0:.0f}s", flush=True)
    dg = float(np.mean(np.diag(XtX))); best = None; grid = {}
    for lf in (1e-4, 1e-3, 1e-2, 1e-1):
        A = np.linalg.solve(XtX + lf * dg * np.eye(din), XtY).astype(np.float32)
        num = den = 0.0
        for X, Y, _ in V:
            a_, b_ = direr((X - xm) @ A + ym, Y); num += a_; den += b_
        e = (num / den) ** 0.5; grid[lf] = e
        print(f"[ro]   lambda {lf:g}: one-step held-out direction error {e:.4f}", flush=True)
        if best is None or e < best[0]:
            best = (e, lf, A)
    np.savez(LIN, A=best[2], xm=xm.astype(np.float32), ym=ym.astype(np.float32), sW=sW, sE=sE,
             norm=float(np.mean(np.concatenate(nrm))))
    out = {"pairs": n, "stale_one_step": (num0 / den0) ** 0.5, "lin_one_step": best[0], "lambda": best[1], "grid": grid}
    json.dump(out, open(os.path.join(C.RES, "rollout_fit.json"), "w"), indent=1)
    print(f"[ro] linear f: one-step direction error {best[0]:.4f} (stale {(num0 / den0) ** 0.5:.4f}) in {time.time() - t0:.0f}s", flush=True)


class Res(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(4096, 2048)
        self.down = nn.Linear(2048, 2048)
        self.down.weight = mx.zeros((2048, 2048)); self.down.bias = mx.zeros((2048,))

    def __call__(self, x):
        return self.down(nn.silu(self.up(x)))


_F = {}
def load_f(kind):
    if kind not in _F:
        L = np.load(LIN)
        lin = (mx.array(L["A"]), mx.array(L["xm"]), mx.array(L["ym"]), float(L["sW"]), float(L["sE"]), float(L["norm"]))
        res = None
        if kind == "mlp":
            res = Res(); res.load_weights(MLPW); mx.eval(res.parameters())
        _F[kind] = (lin, res)
    return _F[kind]


def fstep(kind, Wt, Et):
    """one step of f on rows: Wt [n, 2048] (current state), Et [n, 2048] (next token's embedding) -> the next state,
    rescaled to the training rows' mean norm"""
    (A, xm, ym, sW, sE, nm), res = load_f(kind)
    X = mx.concatenate([Wt / sW, Et / sE], -1)
    Y = (X - xm) @ A + ym
    if res is not None:
        Y = Y + res(X)
    return Y * (nm / mx.maximum(mx.sqrt((Y ** 2).sum(-1, keepdims=True)), 1e-6))


def rollout_table(kind, W, E, K=31):
    """R[k] [S - k, 2048]: f rolled k steps from the TRUE anchor state at each row s along the true tokens (k = 0: W)"""
    S = W.shape[0]
    R, cur = [W], W
    for k in range(1, K):
        if S - k <= 0:
            break
        cur = fstep(kind, cur[:S - k], E[k:S])
        R.append(cur)
    mx.eval(R)
    return R


def cmd_mlp():
    tr, va = split_files()
    (A, xm, ym, sW, sE, nm), _ = load_f("lin")
    rng = np.random.default_rng(1114)
    Xs, Ys = [], []
    for f in tr:
        W, E, a = states(f)
        X = np.concatenate([W[a:-1] / sW, E[a + 1:] / sE], -1); Y = W[a + 1:]
        k = rng.choice(len(X), size=min(len(X), 110), replace=False)
        Xs.append(X[k].astype(np.float16)); Ys.append(Y[k].astype(np.float16))
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    res = Res(); mx.eval(res.parameters())
    opt = optim.Adam(learning_rate=1e-3)

    def loss_fn(xb, yb):
        p = (xb - xm) @ A + ym + res(xb)
        pn = p / mx.sqrt((p ** 2).mean(-1, keepdims=True) + 1e-12); yn = yb / mx.sqrt((yb ** 2).mean(-1, keepdims=True) + 1e-12)
        return ((pn - yn) ** 2).sum(-1).mean() / 2048
    lg = nn.value_and_grad(res, loss_fn)
    t0 = time.time()
    for ep in range(4):
        perm = rng.permutation(len(X))
        for i in range(0, len(X) - 255, 256):
            b = perm[i:i + 256]
            l, g = lg(mx.array(X[b].astype(np.float32)), mx.array(Y[b].astype(np.float32)))
            opt.update(res, g); mx.eval(res.parameters(), opt.state)
        print(f"[ro] mlp epoch {ep + 1}: train loss {float(l):.4f} ({time.time() - t0:.0f}s)", flush=True)
    mx.save_safetensors(MLPW, dict(__import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(res.parameters())))
    _F.pop("mlp", None)
    num = den = 0.0
    for f in va:
        W, E, a = states(f)
        P = np.array(fstep("mlp", mx.array(W[a:-1]), mx.array(E[a + 1:])))
        a_, b_ = direr(P, W[a + 1:]); num += a_; den += b_
    fj = os.path.join(C.RES, "rollout_fit.json"); out = json.load(open(fj)); out["mlp_one_step"] = (num / den) ** 0.5
    json.dump(out, open(fj, "w"), indent=1)
    print(f"[ro] linear + MLP f: one-step held-out direction error {(num / den) ** 0.5:.4f} ({len(X)} pairs, {time.time() - t0:.0f}s)", flush=True)


def cmd_curve():
    fs = C.files("mbpp")
    out = {}
    for kind in ("lin", "mlp"):
        if kind == "mlp" and not os.path.exists(MLPW):
            continue
        num, den = np.zeros(31), np.zeros(31)
        for f in fs:
            z = np.load(f)
            W = C.load_W(z)[0].astype(mx.float32); E = C.embed(mx.array(np.asarray(z["ids"], np.int32)))
            a = max(0, int(z["gen"]) - 2)
            R = rollout_table(kind, W, E)
            Wn = np.array(W)
            for k in range(len(R)):
                P = np.array(R[k][a:]); Y = Wn[a + k:a + k + len(P)]
                P = P[:len(Y)]
                if len(P) == 0:
                    continue
                a_, b_ = direr(P, Y); num[k] += a_; den[k] += b_
        out[kind] = [float((num[k] / den[k]) ** 0.5) if den[k] else None for k in range(31)]
        print(f"[ro] {kind} rollout direction error by depth k: " + " ".join(f"{k}:{v:.3f}" for k, v in enumerate(out[kind]) if v is not None and k in (0, 1, 2, 4, 8, 12, 16, 22, 29, 30)), flush=True)
    json.dump(out, open(os.path.join(C.RES, "rollout_curve.json"), "w"), indent=1)


if __name__ == "__main__":
    C.limit_memory(4.5); C.load_target(); C.drop_trunk()
    {"fit": cmd_fit, "mlp": cmd_mlp, "curve": cmd_curve}[sys.argv[1]]()
