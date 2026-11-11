"""CPU analysis of captured MoE inputs on the 35B proxy (NewIdeas I2 + I3, 2026-10-21). Runs on the CPU device only.
Expert weights = the 8-bit checkpoint requantized to 4-bit g64 (as in capture35.py), dequantized to f32.
Eval rows = segments with odd index (held out); fit rows (I2 bases) = even segments.

sparsity L [NTOK]: I3a oracle. Per eval token: each routed expert's neurons c_i = h_i v_i, h = silu(Wg x) * (Wu x),
   ranked by |h_i| ||v_i||. (i) per expert: share of neurons needed for 90 / 96 / 99 % of ||y_e||^2 (NMSE <= 0.10 /
   0.04 / 0.01); (ii) the gate-weighted routed sum: neurons of all 8 experts ranked by w_e |h_i| ||v_i||, share of the
   4096 needed for the same NMSE of ||sum_e w_e y_e||^2; (iii) static hot set: per expert the neurons ranked by mean
   contribution on FIT rows, NMSE on eval rows at 10..50 %; (iv) a cheap selector: h estimated from a 2-bit (g64)
   copy of Wg, Wu, ranked the same way, true NMSE of the chosen set.
subspace L: I2. Per expert with >= 64 fit rows: PCA of its routed fit inputs; on its routed eval inputs, energy outside
   the top-r subspace and the expert's output NMSE when evaluated on the projected input (r = 16..256); plus one
   layer-wide basis from all fit rows (r = 64..1024).
"""
import glob, os, sys, time
import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)
CAP = os.path.expanduser("~/NewIdeas/data/cap35")
OUT = os.path.expanduser("~/NewIdeas/data/ana35"); os.makedirs(OUT, exist_ok=True)
SRC = os.environ.get("NI_SRC", "probe35")
if SRC == "probe35":    # MoEPrefetch's capture: RMS-normalised pre-MLP residuals of its 4-bit 35B, 5 domains x 2500 tokens
    MODEL = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit")
    PROBE = sorted(glob.glob(os.path.expanduser("~/MoEPrefetch/traces/probe35/*.npz")))
else:
    MODEL = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MLX-8bit/snapshots/*/"))[0]
_M = {}


def model():
    if "m" not in _M:
        from mlx_lm import load
        _M["m"] = load(MODEL, lazy=True)[0]
    return _M["m"]


def experts(L, bits=4):
    """f32 numpy Wg, Wu [E, 512, 2048], Wd [E, 2048, 512] for layer L, requantized from the 8-bit checkpoint."""
    sw = model().language_model.model.layers[L].mlp.switch_mlp
    out = []
    for p in (sw.gate_proj, sw.up_proj, sw.down_proj):
        w = mx.dequantize(p.weight, p.scales, p.biases, group_size=p.group_size, bits=p.bits)
        if not (p.bits == bits and p.group_size == 64):
            w = mx.dequantize(*mx.quantize(w, group_size=64, bits=bits), group_size=64, bits=bits)
        out.append(np.array(w.astype(mx.float32)))
    return out


def silu(z):
    return z / (1 + np.exp(-z))


def load_cap(L):
    if SRC != "probe35":
        d = np.load(os.path.join(CAP, f"L{L}.npz"))
        return d["X"].astype(np.float32), d["I"].astype(np.int64), d["S"], d["seg"]
    lay = model().language_model.model.layers[L]
    w = np.array(lay.post_attention_layernorm.weight.astype(mx.float32))
    X, I, seg = [], [], []
    for k, f in enumerate(PROBE):                          # per domain: first half fit (even seg), second half eval (odd)
        d = np.load(f)
        h = d[f"H{L}"].astype(np.float32); n = len(h)
        X.append(h * w[None]); I.append(d[f"A{L}"].astype(np.int64))
        seg.append(np.where(np.arange(n) < n // 2, 2 * k, 2 * k + 1))
    X, I, seg = np.concatenate(X), np.concatenate(I), np.concatenate(seg)
    g = lay.mlp.gate
    logits = np.array((mx.array(X).astype(mx.bfloat16) @ mx.dequantize(g.weight, g.scales, g.biases, group_size=g.group_size,
                       bits=g.bits).astype(mx.bfloat16).T).astype(mx.float32))
    p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    S = np.take_along_axis(p, I, axis=1); S /= S.sum(1, keepdims=True)
    return X, I, S, seg


def need(err_curve, grid, thr):
    """smallest grid value whose NMSE <= thr (err_curve [..., len(grid)]); len(grid)+ -> inf"""
    ok = err_curve <= thr
    idx = np.where(ok.any(-1), ok.argmax(-1), len(grid))
    g = np.append(np.asarray(grid, float), np.inf)
    return g[idx]


def sparsity(L, ntok):
    t0 = time.time()
    X, I, S, seg = load_cap(L)
    Wg, Wu, Wd = experts(L)
    Wg2, Wu2, _ = experts(L, bits=2)
    E, F, D = Wg.shape
    vn = np.linalg.norm(Wd, axis=1)                       # [E, F] ||v_i||
    fit, ev = np.where(seg % 2 == 0)[0], np.where(seg % 2 == 1)[0]
    rng = np.random.default_rng(0)
    ev = np.sort(rng.choice(ev, min(ntok, len(ev)), replace=False))
    grid = np.array([8, 16, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 448, 512])
    # static hot sets from fit rows: mean |h_i| ||v_i|| per expert
    hot = np.zeros((E, F))
    cnt = np.zeros(E)
    for e in range(E):
        rows = fit[(I[fit] == e).any(1)]
        if len(rows) == 0:
            continue
        h = silu(X[rows] @ Wg[e].T) * (X[rows] @ Wu[e].T)
        hot[e] = np.abs(h).mean(0) * vn[e]; cnt[e] = len(rows)
    hot_rank = np.argsort(-hot, axis=1)
    per_exp, per_exp_hot, per_exp_sel = [], [], []
    glob_need = []
    gridg = np.array([32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096])
    fr_static = [0.1, 0.2, 0.3, 0.4, 0.5]
    for t in ev:
        x = X[t]; ids = I[t]; w = S[t]
        Cs, sc = [], []
        for k, e in enumerate(ids):
            h = silu(Wg[e] @ x) * (Wu[e] @ x)
            C = Wd[e] * h[None, :]                         # [D, F] contributions as columns
            y = C.sum(1); yy = (y * y).sum() + 1e-30
            r = np.abs(h) * vn[e]
            o = np.argsort(-r)
            cum = np.cumsum(C[:, o], axis=1)               # [D, F]
            err = ((y[:, None] - cum) ** 2).sum(0) / yy    # NMSE after keeping the top n neurons, n = 1..F
            per_exp.append(err[grid - 1])
            oh = hot_rank[e]
            per_exp_hot.append([((y - C[:, oh[:int(f * F)]].sum(1)) ** 2).sum() / yy for f in fr_static])
            h2 = silu(Wg2[e] @ x) * (Wu2[e] @ x)
            o2 = np.argsort(-(np.abs(h2) * vn[e]))
            cum2 = np.cumsum(C[:, o2], axis=1)
            per_exp_sel.append((((y[:, None] - cum2) ** 2).sum(0) / yy)[grid - 1])
            Cs.append(C * w[k]); sc.append(r * w[k])
        Call = np.concatenate(Cs, 1); scall = np.concatenate(sc)
        yt = Call.sum(1); yyt = (yt * yt).sum() + 1e-30
        o = np.argsort(-scall)
        cum = np.cumsum(Call[:, o], axis=1)
        errg = ((yt[:, None] - cum) ** 2).sum(0) / yyt
        glob_need.append(errg[gridg - 1])
    per_exp, per_exp_hot, per_exp_sel, glob_need = map(np.array, (per_exp, per_exp_hot, per_exp_sel, glob_need))
    lines = [f"L{L} sparsity: {len(ev)} eval tokens x {I.shape[1]} experts ({time.time() - t0:.0f}s); F = {F} neurons"]
    for thr in (0.10, 0.04, 0.01):
        n = need(per_exp, grid, thr); n2 = need(per_exp_sel, grid, thr); ng = need(glob_need, gridg, thr)
        lines.append(f"  NMSE <= {thr:.2f}: per expert oracle median {np.median(n) / F:.3f} of neurons (P90 "
                     f"{np.percentile(n, 90) / F:.3f}); 2-bit selector median {np.median(n2) / F:.3f} (P90 "
                     f"{np.percentile(n2, 90) / F:.3f}); routed sum, global oracle median {np.median(ng) / (8 * F):.3f} "
                     f"(P90 {np.percentile(ng, 90) / (8 * F):.3f}) of the 8x{F}")
    lines.append("  mean per-expert NMSE at kept share   " + "  ".join(
        f"{g / F:.2f}:{per_exp[:, j].mean():.3f}/{per_exp_sel[:, j].mean():.3f}" for j, g in enumerate(grid))
        + "   (oracle/2-bit selector)")
    lines.append("  static hot set (fit rows) mean NMSE  " + "  ".join(
        f"{f:.1f}:{per_exp_hot[:, j].mean():.3f}" for j, f in enumerate(fr_static)))
    lines.append("  routed sum, global oracle mean NMSE  " + "  ".join(
        f"{g / (8 * F):.3f}:{glob_need[:, j].mean():.4f}" for j, g in enumerate(gridg)))
    return lines


def subspace(L):
    t0 = time.time()
    X, I, S, seg = load_cap(L)
    Wg, Wu, Wd = experts(L)
    E = Wg.shape[0]
    fit, ev = seg % 2 == 0, seg % 2 == 1
    rs = [16, 32, 64, 128, 256]
    out_e, en_e, nfit = [], [], []

    def f(e, Z):
        return (silu(Z @ Wg[e].T) * (Z @ Wu[e].T)) @ Wd[e].T
    for e in range(E):
        m = (I == e).any(1)
        A, B = X[m & fit], X[m & ev]
        if len(A) < 64 or len(B) < 16:
            continue
        mu = 0                                             # uncentred: the projection must keep the mean direction
        _, _, Vt = np.linalg.svd(A - mu, full_matrices=False)
        y = f(e, B); yy = (y * y).sum(1) + 1e-30
        bb = (B * B).sum(1)
        en_e.append([(((B - (B @ Vt[:r].T) @ Vt[:r]) ** 2).sum(1) / bb).mean() for r in rs if r <= len(Vt)]
                    + [np.nan] * sum(r > len(Vt) for r in rs))
        out_e.append([(((y - f(e, (B @ Vt[:r].T) @ Vt[:r])) ** 2).sum(1) / yy).mean() if r <= len(Vt) else np.nan
                      for r in rs])
        nfit.append(len(A))
    out_e, en_e = np.array(out_e), np.array(en_e)
    lines = [f"L{L} subspace: {len(out_e)} experts with >= 64 fit rows (median {np.median(nfit):.0f}) "
             f"({time.time() - t0:.0f}s)",
             "  per-expert routed-input basis, r: input energy outside / expert output NMSE (median over experts)  "
             + "  ".join(f"{r}:{np.nanmedian(en_e[:, j]):.3f}/{np.nanmedian(out_e[:, j]):.3f}" for j, r in enumerate(rs))]
    _, _, Vt = np.linalg.svd(X[fit][::2], full_matrices=False)
    rl = [64, 128, 256, 512, 768, 1024, 1536]
    rows = np.where(ev)[0][::8]
    bb = (X[rows] ** 2).sum(1)
    res = []
    for r in rl:
        P = Vt[:r]
        Z = (X[rows] @ P.T) @ P
        eo = []
        for t, z in zip(rows, Z):
            for e in I[t][:3]:                             # the top routed experts of the token (argpartition order)
                y = f(e, X[t][None])[0]; yz = f(e, z[None])[0]
                eo.append(((y - yz) ** 2).sum() / ((y * y).sum() + 1e-30))
        res.append((r, (((X[rows] - Z) ** 2).sum(1) / bb).mean(), np.median(eo)))
    lines.append("  layer-wide basis, r: input energy outside / routed expert output NMSE (median)  "
                 + "  ".join(f"{r}:{a:.3f}/{b:.3f}" for r, a, b in res))
    return lines


if __name__ == "__main__":
    mode, L = sys.argv[1], int(sys.argv[2])
    lines = sparsity(L, int(sys.argv[3]) if len(sys.argv) > 3 else 400) if mode == "sparsity" else subspace(L)
    for l in lines:
        print(l, flush=True)
    open(os.path.join(OUT, "results.txt"), "a").write("\n".join(lines) + "\n")
