"""TrainLab (NewIdeas 10-22 09:16): the 35B kill test for TRAINED STICKY ROUTING, layer-local (no backprop through the
model).
Train, online: windows of --seq tokens from data/train_tokens.npy (the last 4K tokens held out). One frozen forward of
the 35B with its original routers (exact flow) captures every MoE layer's input x; then, for each variant (one --betas
value) and layer, an fp32 router W (init = the dequantized original W0) takes ReMoE's (2605.27081) router objective
WITHOUT its language-model term, with ReMoE's weights relative to reuse (0.2 : 0.05 : 0.05):
    KL(P || P_ref) + beta * (L_reuse + 0.25 L_smooth + 0.25 L_lag),   P = softmax(W x), P_ref = softmax(W0 x),
    L_reuse  = -log(mean_t (1/K) sum_{e in E_(t-1)} P_t(e)),  E_(t-1) = top-K of P_(t-1) under stop_gradient,
    L_smooth = mean_t SymKL(P_t, P_(t-1)),   L_lag = mean over d in {1,2,4,8,16} of mean_t SymKL(P_t, P_(t-d)).
beta = 0.44 is ReMoE's own ratio (lambda_reuse / lambda_KL = 0.2 / 0.45); "8r" = beta 8 with the reuse term only (a
control: on random routers the SymKL terms fall by flattening P, not by stickiness). Each layer's loss sees only its own router:
no cross-layer gradient. Adam at lr = 3e-3 x the routers' weight rms (scale-free), warm-up over 30 steps, then linear
decay to 10 % over the --train-min budget.
Eval (the reference is the original model): the full model with a variant's routers (train35.RouterFull) at natural
top-k and capped at m = 2 / 3 (train35.cap_select); end-to-end KLD per held-out segment, switch rate, U16-64, LRU hits
(Chat1000's union_metrics), next to zero-shot caps m = 1..6 and an untrained control (RouterFull of W0). A second
zero-shot baseline, soft stickiness without training: the previous token's experts get +delta (0.5..2) on their logits
before the top-k (the training-free form of "bias the router toward recently selected experts"). A third, the per-pass
union budget (16-row blocks, U = 16-40 experts by summed w^2, rows take their top-k within the set): on the 125B traces
the exact block optimum has 1.7-2.6x less error energy than the hard caps at equal U(16) (sticky/oracle125.py). And the
published batch-aware rule it refines, OEA (2511.02237: pass set = union of every row's top-k0, rows fill to k within
it; on Qwen3-30B at batch 16, 48.8 -> 25.1 experts per layer with unchanged AIME / GPQA / LiveCodeBench / MATH-500), at
16 rows (k0 2-4) and 4 rows (k0 3-5).
Per-layer error (kill rule 2): in every evaluation each layer's MoE output is compared with the original routing's
output at the same input, ||y - y0||^2 / ||y0||^2 (3 frozen segments); err = its mean over layers. The zero-shot caps
fit a law KLD = a * err^b; a trained point far above the law compounds worse than its per-layer error predicts.
Compounding, second view: band additivity, KLD with the change in all 40 layers vs the sum over 4 bands of 10 layers,
for the zero-shot cap 2 and the best trained point.
Kill (NewIdeas 10-22 09:16): at equal U(16) the trained routers must cut KLD by >= ~2x vs the zero-shot caps, and the
end-to-end KLD must not compound far above the per-layer prediction.
usage: sticky35.py run [--betas 0.44,2,8,32,8r] [--train-min 18] [--steps 3000] [--seq 1024] [--lr-rel 3e-3 | --lr X]
       -> data/sticky35_results.txt (+ .json), ckpt/sticky35_b<beta>.safetensors"""
import argparse, json, os, re, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train35 as T

RES = os.path.join(T.TL, "data/sticky35_results.txt")
LAGS = (1, 2, 4, 8, 16)                                   # ReMoE's lag set
G_SMOOTH = G_LAG = 0.25                                   # ReMoE: lambda_smooth = lambda_lag = 0.05 vs lambda_reuse 0.2
EXACT = T.Flow("exact")
CAP = {"on": False}
ERR = {"on": False, "num": None, "den": None}
GATES0 = []                                               # the original (8-bit) router modules
_moe = T.moe


def _moe_hook(m, L, x, flow, pre_logits):
    """Captures every MoE input (training) and measures each layer's MoE output error vs the original routing at the
    same input (evaluations, on the segments evaluate() traces: prose / gsm / code)."""
    if CAP["on"]:
        CAP[L] = x
    if not (ERR["on"] and T.Trace.rec is not None) or (m.gate is GATES0[L] and (flow.cap is None or not flow.on(L))):
        return _moe(m, L, x, flow, pre_logits)
    g = m.gate
    m.gate = GATES0[L]
    y0 = _moe(m, L, x, EXACT, None)[0]                    # first: the real call below overwrites Trace.rec[L]
    m.gate = g
    out = _moe(m, L, x, flow, pre_logits)
    y0 = y0.astype(mx.float32)
    d = out[0].astype(mx.float32) - y0
    ERR["num"][L] += float((d * d).sum())
    ERR["den"][L] += float((y0 * y0).sum())
    return out


T.moe = _moe_hook                                         # layer_body calls T.moe
BONUS = {"delta": None}
_cap_select = T.cap_select


def bonus_select(lg, k, delta):
    """zero-shot soft stickiness: the previous token's experts get +delta on their logits before the top-k (a new expert
    enters only if it beats the best left-out old one by delta); rank order by the true logits"""
    out = np.zeros((lg.shape[0], k), np.int64)
    prev = None
    for t in range(lg.shape[0]):
        lt = lg[t].copy()
        if prev is not None:
            lt[prev] += delta
        top = np.argpartition(-lt, k)[:k]
        out[t] = prev = top[np.argsort(-lg[t][top])]
    return out


def budget_select(lg, k, U, B=16):
    """zero-shot per-pass union budget (a wide pass knows all its rows' router outputs before the experts run): in each
    block of B consecutive rows the U experts with the largest sum over rows of w^2 (w = the row's top-k gate weights,
    normalised over its top-k; the exact energy-optimal set when dropped experts cost w^2) form the pass set A; each row
    takes its top-k within A. Batch-aware routing in the line of Lynx / Opportunistic Expert Activation."""
    T_, E = lg.shape
    out = np.zeros((T_, k), np.int64)
    for b in range(0, T_, B):
        lb = lg[b:b + B]
        pb = np.exp(lb - lb.max(-1, keepdims=True))
        top = np.argpartition(-lb, k, axis=-1)[:, :k]
        w = np.take_along_axis(pb, top, -1)
        w = w / w.sum(-1, keepdims=True)
        c = np.zeros(E)
        np.add.at(c, top.ravel(), (w ** 2).ravel())
        lr = np.full_like(lb, -np.inf)
        A = np.argsort(-c)[:max(U, k)]
        lr[:, A] = lb[:, A]
        sel = np.argpartition(-lr, k, axis=-1)[:, :k]
        out[b:b + B] = np.take_along_axis(sel, np.argsort(-np.take_along_axis(lb, sel, -1), -1), -1)
    return out


def oea_select(lg, k, k0, B=16):
    """Opportunistic Expert Activation (OEA, 2511.02237), simplified form: in each block of B rows the pass set is the
    union of every row's top-k0; each row keeps its top-k0 and fills up to k with its best-scored members of the set
    (= its top-k within the set); weights renormalised over the selection (moe() does that)"""
    T_, E = lg.shape
    out = np.zeros((T_, k), np.int64)
    for b in range(0, T_, B):
        lb = lg[b:b + B]
        A = np.unique(np.argpartition(-lb, k0, axis=-1)[:, :k0])
        if len(A) < k:                                    # rare (near-identical rows): fill the set to k by summed logit
            rest = [e for e in np.argsort(-lb.sum(0)) if e not in set(A.tolist())]
            A = np.concatenate([A, np.array(rest[:k - len(A)], A.dtype)])
        lr = np.full_like(lb, -np.inf)
        lr[:, A] = lb[:, A]
        sel = np.argpartition(-lr, k, axis=-1)[:, :k]
        out[b:b + B] = np.take_along_axis(sel, np.argsort(-np.take_along_axis(lb, sel, -1), -1), -1)
    return out


def _cap_select_hook(lg, k, m):                           # moe() calls T.cap_select for flows capM
    if BONUS.get("k0"):
        return oea_select(lg, k, BONUS["k0"], BONUS.get("B", 16))
    if BONUS.get("U"):
        return budget_select(lg, k, BONUS["U"], BONUS.get("B", 16))
    return _cap_select(lg, k, m) if BONUS["delta"] is None else bonus_select(lg, k, BONUS["delta"])


T.cap_select = _cap_select_hook
_locality = T.locality


def _locality_u4(traces):                                 # + the union of 4-row passes (today's MTP verify width)
    import union_metrics as U                             # (train35.locality put Chat1000's scripts on the path)
    return _locality(traces) + f" U4 {np.mean([U.union(ids, 4) for ids in traces]):.1f}"


T.locality = _locality_u4


def log(line):
    print(line, flush=True)
    open(RES, "a").write(line + "\n")


def orig_routers(layers):
    W = []
    for layer in layers:
        g = layer.mlp.gate
        w = mx.dequantize(g.weight, g.scales, g.biases, group_size=g.group_size, bits=g.bits) \
            if isinstance(g, nn.QuantizedLinear) else g.weight
        W.append(w.astype(mx.float32))
    mx.eval(W)
    return W


def log_softmax(lg):
    return lg - mx.logsumexp(lg, axis=-1, keepdims=True)


def symkl(a, b):                                          # mean over rows of SymKL between log-distributions a, b
    return 0.5 * ((mx.exp(a) - mx.exp(b)) * (a - b)).sum(-1).mean()


def router_loss(W, xf, lpr, K, beta, g=G_SMOOTH):
    lp = log_softmax(xf @ W.T)
    P = mx.exp(lp)
    trust = (P * (lp - lpr)).sum(-1).mean()
    prev = mx.stop_gradient(mx.argpartition(-lp[:-1], kth=K - 1, axis=-1)[:, :K])
    rho = (mx.take_along_axis(P[1:], prev, axis=-1).sum(-1) / K).mean()
    smooth = symkl(lp[1:], lp[:-1])
    lag = sum(symkl(lp[d:], lp[:-d]) for d in LAGS) / len(LAGS)
    return trust + beta * (-mx.log(rho + 1e-8) + g * smooth + g * lag), (trust, rho, smooth)


def total_loss(params, XF, LPR, K, beta, g=G_SMOOTH):
    """mean over layers; every layer's term depends on its own router only"""
    tot, parts = 0.0, (0.0, 0.0, 0.0)
    for L, W in enumerate(params["w"]):
        l, p = router_loss(W, XF[L], LPR[L], K, beta, g)
        tot = tot + l
        parts = tuple(a + b for a, b in zip(parts, p))
    return tot / len(XF), tuple(p / len(XF) for p in parts)


def capture(model, ids, nl):
    CAP["on"] = True
    T.forward(model, ids, EXACT, False)
    CAP["on"] = False
    X = [CAP[L].reshape(-1, CAP[L].shape[-1]).astype(mx.float32) for L in range(nl)]
    mx.eval(X)
    return X


def local_metrics(W, XH, LPRH, K):
    """the routers on held-out captures (the original model upstream, no compounding): trust KL, reuse mass, top-K
    switch rate and U16 (Chat1000's definitions)"""
    sys.path.insert(0, os.path.expanduser("~/Chat1000/scripts"))
    import union_metrics as U
    kl, rho, ids = [], [], []
    for L in range(len(W)):
        lp = log_softmax(XH[L] @ W[L].T)
        top = mx.argpartition(-lp, kth=K - 1, axis=-1)[:, :K]
        r = (mx.take_along_axis(mx.exp(lp[1:]), top[:-1], axis=-1).sum(-1) / K).mean()
        k = (mx.exp(lp) * (lp - LPRH[L])).sum(-1).mean()
        mx.eval(top, r, k)
        kl.append(float(k)); rho.append(float(r)); ids.append(np.array(top))
    ids = np.stack(ids, axis=1).astype(np.int64)
    return (f"trust KL {np.mean(kl):.4f} (max layer {np.max(kl):.4f}) reuse {np.mean(rho):.3f} switch "
            f"{U.switch_rate(ids):.3f} U16 {U.union(ids, 16):.1f}")


def install(layers, W, band=None):
    for L, layer in enumerate(layers):
        layer.mlp.gate = T.RouterFull(W[L]) if band is None or band[0] <= L < band[1] else GATES0[L]
    mx.eval([layer.mlp.gate.parameters() for layer in layers])


def u16(r, key="U16"):
    m = re.search(key + r" ([0-9.]+)", r.get("cache") or "")
    return float(m.group(1)) if m else float("nan")


def zs_at(u, Z):
    """zero-shot caps' KLD at U16 u: log-linear interpolation between the bracketing caps (None outside their range)"""
    pts = sorted(Z)
    for (u0, k0), (u1, k1) in zip(pts, pts[1:]):
        if u0 <= u <= u1:
            f = (u - u0) / (u1 - u0) if u1 > u0 else 0.0
            return float(np.exp(np.log(k0) + f * (np.log(k1) - np.log(k0))))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--betas", default="0.44,2,8,32,8r")
    ap.add_argument("--train-min", type=float, default=18); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seq", type=int, default=1024); ap.add_argument("--lr", type=float, default=0)
    ap.add_argument("--lr-rel", type=float, default=3e-3)     # lr = lr_rel x the routers' weight rms (unless --lr)
    a = ap.parse_args()
    T.limit_memory()
    t0 = time.time()
    limit = float(os.environ.get("TL_LIMIT", "1e9"))
    budget = min(a.train_min * 60, 0.45 * limit)
    model, tok = T.setup()
    layers = model.language_model.model.layers
    nl, K = len(layers), layers[0].mlp.top_k
    GATES0.extend(layer.mlp.gate for layer in layers)
    W0 = orig_routers(layers)
    betas = [float(b.rstrip("r")) for b in a.betas.split(",")]          # "8r": reuse only (no smooth / lag)
    gs = [0.0 if b.endswith("r") else G_SMOOTH for b in a.betas.split(",")]
    names = [f"beta {b}" + ("r" if g == 0 else "") for b, g in zip(betas, gs)]
    params = [{"w": list(W0)} for _ in betas]
    wstat = float(np.sqrt(np.mean([float(mx.mean(w * w)) for w in W0])))
    lr0 = a.lr or a.lr_rel * wstat
    opts = [optim.Adam(learning_rate=lr0) for _ in betas]
    vg = mx.value_and_grad(total_loss)
    stream = np.load(os.path.join(T.TL, "data/train_tokens.npy"), mmap_mode="r")
    hold = mx.array(np.asarray(stream[len(stream) - 2049:len(stream) - 1], np.int32)[None])
    XH = capture(model, hold, nl)
    LPRH = [log_softmax(XH[L] @ W0[L].T) for L in range(nl)]
    mx.eval(LPRH)
    rng = np.random.default_rng(7)
    log(f"{time.strftime('%m-%d %H:%M')} sticky35 train: betas {betas}, <= {a.steps} steps x {a.seq} tokens, "
        f"{budget / 60:.0f} min, lr {lr0:.2e}, K {K}, {nl} layers, router weight rms {wstat:.4f}")
    log(f"  held-out 2048 tokens, original routers: {local_metrics(W0, XH, LPRH, K)}")
    step, t_tr = 0, time.time()
    while step < a.steps and time.time() - t_tr < budget:
        o = int(rng.integers(0, len(stream) - 4096 - a.seq))
        XF = capture(model, mx.array(np.asarray(stream[o:o + a.seq], np.int32)[None]), nl)
        LPR = [log_softmax(XF[L] @ W0[L].T) for L in range(nl)]
        mx.eval(LPR)
        frac = (time.time() - t_tr) / budget
        lr = lr0 * min(1.0, (step + 1) / 30) * (1 - 0.9 * min(1.0, frac))
        for v, beta in enumerate(betas):
            opts[v].learning_rate = lr
            (loss, parts), gr = vg(params[v], XF, LPR, K, beta, gs[v])
            params[v] = opts[v].apply_gradients(gr, params[v])
            mx.eval(params[v], opts[v].state, loss, *parts)
            if step % 50 == 0:
                print(f"step {step} {names[v]}: loss {float(loss):.4f} trust KL {float(parts[0]):.4f} reuse "
                      f"{float(parts[1]):.3f} smooth {float(parts[2]):.3f} lr {lr:.2e}  {time.time() - t0:.0f}s",
                      flush=True)
        if step % 200 == 199:
            for v, beta in enumerate(betas):
                print(f"  held-out @ step {step + 1}, {names[v]}: {local_metrics(params[v]['w'], XH, LPRH, K)}",
                      flush=True)
        step += 1
    WV = [p["w"] for p in params]
    for v, beta in enumerate(betas):
        mx.save_safetensors(os.path.join(T.TL, "ckpt", f"sticky35_b{names[v][5:]}.safetensors"),
                            {f"w{L}": w for L, w in enumerate(WV[v])})
        log(f"  held-out after {step} steps ({step * a.seq} tokens, {time.time() - t_tr:.0f}s), {names[v]}: "
            f"{local_metrics(WV[v], XH, LPRH, K)}")
    del XH, LPRH
    # ---- evaluation (reference: the original model's distribution, data/ref_heldout.npz)
    segs = T.heldout(tok)
    R = T.reference(model, segs)
    out = {}

    def ev(tag, flow):
        t1 = time.time()
        ERR.update(on=True, num=np.zeros(nl), den=np.zeros(nl))
        r = T.evaluate(model, segs, R, T.Flow(flow))
        ERR["on"] = False
        el = np.where(ERR["den"] > 0, ERR["num"] / np.maximum(ERR["den"], 1e-30), 0.0)
        r.update(flow=flow, U16=u16(r), U4=u16(r, "U4"), err=float(el.mean()), err_layers=[float(e) for e in el])
        out[tag] = r
        log(f"{time.strftime('%m-%d %H:%M')} sticky35 {tag:24s} {T.fmt(r)} | per-layer err {r['err']:.5f}  "
            f"({time.time() - t1:.0f}s)")
        json.dump(out, open(RES.replace(".txt", ".json"), "w"))
        return r
    install(layers, W0, band=(0, 0))                      # the original routers everywhere
    for f in ("exact", "cap1", "cap2", "cap3", "cap4", "cap5", "cap6"):
        ev(f"zero-shot {f}", f)
    for d in (0.5, 1.0, 2.0):                             # zero-shot soft stickiness (cap8 = the cap never binds)
        BONUS["delta"] = d
        ev(f"zero-shot bonus {d}", "cap8")
    BONUS["delta"] = None
    for U in (16, 24, 32, 40):                            # zero-shot per-pass union budget (16-row passes)
        BONUS["U"] = U
        ev(f"zero-shot budget {U}", "cap8")
    BONUS["B"] = 4
    for U in (12, 16, 20):                                # ... and 4-row passes
        BONUS["U"] = U
        ev(f"zero-shot budget4 {U}", "cap8")
    BONUS["U"] = None
    for B_, k0s in ((16, (2, 3, 4)), (4, (3, 4, 5))):    # OEA, the published batch-aware rule, at both widths
        BONUS["B"] = B_
        for k0 in k0s:
            BONUS["k0"] = k0
            ev(f"zero-shot oea{B_} {k0}", "cap8")
    BONUS["k0"], BONUS["B"] = None, 16
    install(layers, W0)
    ev("control (fp32 W0) exact", "exact")
    for v, nm in enumerate(names):
        install(layers, WV[v])
        for f in ("exact", "cap3", "cap2"):
            ev(f"{nm} {f}", f)
    # ---- kill rule 1: KLD at equal U16 vs the zero-shot caps
    Z = [(out[f"zero-shot cap{m}"]["U16"], out[f"zero-shot cap{m}"]["kld"]) for m in range(1, 7)]
    floor = max(out["control (fp32 W0) exact"]["kld"], 1e-4)        # natural routing at the numerics floor
    Z.append((out["zero-shot exact"]["U16"], floor))
    B = [(out[f"zero-shot bonus {d}"]["U16"], out[f"zero-shot bonus {d}"]["kld"]) for d in (0.5, 1.0, 2.0)
         if out[f"zero-shot bonus {d}"]["kld"] > 0]
    Z4 = [(out[f"zero-shot cap{m}"]["U4"], out[f"zero-shot cap{m}"]["kld"]) for m in range(1, 7)] + \
        [(out["zero-shot exact"]["U4"], floor)]
    for tag in [f"bonus {d}" for d in (0.5, 1.0, 2.0)] + [f"budget {U}" for U in (16, 24, 32, 40)] + \
            [f"budget4 {U}" for U in (12, 16, 20)] + [f"oea16 {k0}" for k0 in (2, 3, 4)] + [f"oea4 {k0}" for k0 in (3, 4, 5)]:
        r = out[f"zero-shot {tag}"]
        key = "U4" if tag.startswith(("budget4", "oea4")) else "U16"
        z = zs_at(r[key], Z4 if key == "U4" else Z)
        log(f"  frontier: zero-shot {tag}: {key} {r[key]:.1f} KLD {r['kld']:.4f} vs zero-shot caps "
            + (f"{z:.4f} at equal {key} -> {z / r['kld']:.2f}x" if z and r["kld"] > 0 else f"(outside the caps' {key} range)"))
    best, best_ratio = None, 0.0
    for v, nm in enumerate(names):
        for f in ("exact", "cap3", "cap2"):
            r = out[f"{nm} {f}"]
            z = zs_at(r["U16"], Z)
            ratio = z / r["kld"] if z is not None and r["kld"] > 0 else None
            r["zs_ratio"] = ratio
            zb = zs_at(r["U16"], B)
            log(f"  frontier: {nm} {f}: U16 {r['U16']:.1f} KLD {r['kld']:.4f} vs zero-shot caps "
                + (f"{z:.4f} at equal U16 -> {ratio:.2f}x" if ratio else "(outside the caps' U16 range)")
                + (f"; vs zero-shot bonus {zb:.4f} -> {zb / r['kld']:.2f}x" if zb and r["kld"] > 0 else ""))
            if ratio and r["U16"] <= out["zero-shot cap4"]["U16"] + 0.5 and ratio > best_ratio:
                best, best_ratio = (v, nm, f), ratio
    # ---- kill rule 2: the zero-shot law KLD = a * err^b, and band additivity
    e = np.array([out[f"zero-shot cap{m}"]["err"] for m in range(1, 7)])
    k = np.array([out[f"zero-shot cap{m}"]["kld"] for m in range(1, 7)])
    b_, a_ = np.polyfit(np.log(e), np.log(k), 1)
    log(f"  zero-shot law: KLD = {np.exp(a_):.3f} * err^{b_:.2f}  (caps 1-6)")
    for tag, r in out.items():
        if tag.startswith("beta") or tag.startswith("control"):
            if r["err"] > 0:
                r["law_ratio"] = r["kld"] / float(np.exp(a_) * r["err"] ** b_)
                log(f"  law: {tag}: err {r['err']:.5f} -> predicted KLD {np.exp(a_) * r['err'] ** b_:.4f}, actual "
                    f"{r['kld']:.4f} ({r['law_ratio']:.2f}x)")
    if best is None:                                       # nothing in the useful U16 range: band-test the stickiest
        best = min(((v, nm, f) for v, nm in enumerate(names) for f in ("exact", "cap3", "cap2")),
                   key=lambda t: out[f"{t[1]} {t[2]}"]["U16"])
    v, nm, f = best
    log(f"  band test: zero-shot cap2 and {nm} {f} (best at U16 <= cap4's: {best_ratio:.2f}x)")
    for name, W, fl in (("zero-shot", None, "cap2"), (nm, WV[v], f)):
        sb = 0.0
        for i in range(4):
            band = (i * nl // 4, (i + 1) * nl // 4)
            install(layers, W if W is not None else W0, band=band if W is not None else (0, 0))
            bf = f"{fl}_B{band[0]}-{band[1]}" if fl != "exact" else "exact"
            sb += ev(f"{name} {fl} band {band[0]}-{band[1]}", bf)["kld"]
        tot = out[f"{name} {fl}"]["kld"]
        log(f"  additivity: {name} {fl}: all layers {tot:.4f} vs sum of 4 bands {sb:.4f} -> {tot / sb:.2f}x")
    json.dump(out, open(RES.replace(".txt", ".json"), "w"))
    log(f"sticky35 done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
