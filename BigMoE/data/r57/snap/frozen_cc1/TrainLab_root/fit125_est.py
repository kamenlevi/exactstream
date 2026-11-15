"""TrainLab: can a learned estimator name the 125B's true top-2 experts one layer ahead with >= ~85 % precision?
(NewIdeas 19:40; the training-free tap names them ~61-62 %.) CPU fits on data/cap125est (cap125_est.py), held out by
document. Per target layer L in {8, 16, 24, 32, 39}, predictors of layer L's router logits from layer L-1's MoE input:
  tap      router_L(x_{L-1})  (BigMoE's d=1 look-ahead)
  ridge    tap + ridge((g - tap) ~ x)                (MoEPrefetch's / SlotStream's correction, full rank)
  ridgeH   tap + ridge on [x, history]               history = multi-hot of L-1's true top-10 (same token) and of
                                                     L's true top-10 at the previous token (Speculating Experts, 2609.17940)
  mlpH     tap + MLP([x, history]) -> 512, trained with KL to the true router + CE on the true top-2 (MLX, CPU)
Metrics: prec2 = |top2(pred) & top2(true)| / 2, r0 = true rank 0 in top2(pred), rec4 = |top2(true) & top4(pred)| / 2.
usage: fit125_est.py [--mlp-epochs 6]"""
import argparse, glob, os, time
import numpy as np

D0 = os.environ.get("CAP125EST", os.path.expanduser("~/TrainLab/data/cap125est"))
TARGETS = (8, 16, 24, 32, 39)
VAL = {"py5", "wiki9", "chat7", "gsm_d", "note3"}
E = 512


def multi_hot(top):
    h = np.zeros((top.shape[0], E), np.float32)
    np.put_along_axis(h, top.astype(np.int64), 1.0, axis=1)
    return h


def load(L, names):
    X, G, TP, H = [], [], [], []
    for n in names:
        z = np.load(os.path.join(D0, n + ".npz"))
        x, g, tp = z[f"x{L - 1}"].astype(np.float32), z[f"g{L}"].astype(np.float32), z[f"tap{L}"].astype(np.float32)
        h1 = multi_hot(z[f"top{L - 1}"])
        prev = np.concatenate([np.zeros((1, 10), z[f"top{L}"].dtype), z[f"top{L}"][:-1]])
        h2 = multi_hot(prev); h2[0] = 0
        X.append(x); G.append(g); TP.append(tp); H.append(np.concatenate([h1, h2], 1))
    return np.concatenate(X), np.concatenate(G), np.concatenate(TP), np.concatenate(H)


def metrics(pred, g):
    t2 = np.argpartition(-g, 2, axis=1)[:, :2]
    r0 = np.argmax(g, axis=1)
    p2 = np.argpartition(-pred, 2, axis=1)[:, :2]
    p4 = np.argpartition(-pred, 4, axis=1)[:, :4]
    prec2 = np.mean([len(set(a) & set(b)) for a, b in zip(p2, t2)]) / 2
    r0in = np.mean([r in set(a) for r, a in zip(r0, p2)])
    rec4 = np.mean([len(set(a) & set(b)) for a, b in zip(p4, t2)]) / 2
    return prec2, r0in, rec4


def ridge(F, Y, lam):
    mu = F.mean(0)
    Fc = F - mu
    W = np.linalg.solve(Fc.T @ Fc + lam * len(F) * np.eye(F.shape[1], dtype=np.float32), Fc.T @ Y)
    return lambda Fv: (Fv - mu) @ W + (Y.mean(0))


def fit_mlp(F, G, TP, Fv, epochs):
    import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
    mx.set_default_device(mx.cpu)
    mu, sd = F.mean(0), F.std(0) + 1e-3
    net = nn.Sequential(nn.Linear(F.shape[1], 1024), nn.GELU(), nn.Linear(1024, E))
    net.layers[-1].weight = net.layers[-1].weight * 0.0
    opt = optim.Adam(learning_rate=1e-3)
    Fx = mx.array((F - mu) / sd); Gx = mx.array(G); Tx = mx.array(TP)
    t2 = mx.array(np.argpartition(-G, 2, axis=1)[:, :2])

    def loss(m, f, g, tp, tt):
        lg = tp + m(f)
        lq = lg - mx.logsumexp(lg, -1, keepdims=True)
        lp = g - mx.logsumexp(g, -1, keepdims=True)
        kl = (mx.exp(lp) * (lp - lq)).sum(-1).mean()
        ce = -mx.take_along_axis(lq, tt, -1).mean()
        return kl + 0.5 * ce

    lf = nn.value_and_grad(net, loss)
    n = len(F)
    for ep in range(epochs):
        perm = np.random.default_rng(ep).permutation(n)
        for i in range(0, n, 512):
            b = mx.array(perm[i:i + 512])
            l, g = lf(net, Fx[b], Gx[b], Tx[b], t2[b])
            opt.update(net, g); mx.eval(net.parameters(), opt.state)
    return lambda Fvv, TPv: np.array(mx.array(TPv) + net(mx.array((Fvv - mu) / sd)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mlp-epochs", type=int, default=6)
    a = ap.parse_args()
    names = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(D0, "*.npz")))
    tr = [n for n in names if n not in VAL]; va = [n for n in names if n in VAL]
    print(f"train docs {len(tr)}, val docs {len(va)}", flush=True)
    rows = {}
    for L in TARGETS:
        t0 = time.time()
        X, G, TP, H = load(L, tr); Xv, Gv, TPv, Hv = load(L, va)
        res = {"tap": metrics(TPv, Gv)}
        best = None
        for lam in (1e-3, 1e-2, 1e-1):
            f = ridge(X, G - TP, lam); m = metrics(TPv + f(Xv), Gv)
            if best is None or m[0] > best[1][0]: best = (lam, m)
        res["ridge"] = best[1]
        F, Fv = np.concatenate([X, H], 1), np.concatenate([Xv, Hv], 1)
        best = None
        for lam in (1e-3, 1e-2, 1e-1):
            f = ridge(F, G - TP, lam); m = metrics(TPv + f(Fv), Gv)
            if best is None or m[0] > best[1][0]: best = (lam, m)
        res["ridgeH"] = best[1]
        pm = fit_mlp(F, G, TP, Fv, a.mlp_epochs)
        res["mlpH"] = metrics(pm(Fv, TPv), Gv)
        rows[L] = res
        print(f"L{L:2d} (train {len(X)}, val {len(Xv)}, {time.time() - t0:.0f}s): " + " | ".join(
            f"{k} prec2 {v[0]:.3f} r0 {v[1]:.3f} rec4 {v[2]:.3f}" for k, v in res.items()), flush=True)
    for k in rows[TARGETS[0]]:
        m = np.mean([rows[L][k] for L in TARGETS], 0)
        print(f"MEAN {k:7s} prec2 {m[0]:.3f} r0 {m[1]:.3f} rec4 {m[2]:.3f}")


if __name__ == "__main__":
    main()
