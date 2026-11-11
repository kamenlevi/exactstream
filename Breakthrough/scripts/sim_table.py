"""Expert output-similarity tables for the 125B (Breakthrough D1b, SERE-style substitution; 2026-10-20 22:20).
  sim_table.py capture   : exact prefill (fork125bt streamed runtime, C=128) of calibration sequences = the Eval suite's
                           own-output tf sets NOT used by tf_frugal_var.py (it scores every 4th ifeval/multi sequence;
                           here: the others, plus code and gsm), hook every routed MoE block, keep NTOK rows per layer
                           (uniform over all prefilled rows) -> data/simtab/X.npz (L<l>: [NTOK, 2560] f16)
  sim_table.py build     : per layer, read all 512 experts (9 tensors) straight from the safetensors (no model load),
                           Y_e = down(silu(gate X) * up X) for every expert on X, then per pair ||Y_u - Y_v||_F and ||Y_u||
                           -> data/simtab/D<l>.npy (512 x 512 f16 distances), N<l>.npy (512 norms). 68 GB read once.
Use: a missing expert u is replaced by the resident v minimising D[u, v] if D[u, v] < N[u] (better than skipping),
at u's own gate weight (SERE keeps router weights)."""
import json, os, sys, time
import numpy as np
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
OUT = os.path.expanduser("~/Breakthrough/data/simtab"); os.makedirs(OUT, exist_ok=True)
MD = os.path.expanduser("~/BigMoE/models/bare")
NTOK = int(os.environ.get("BT_NTOK", "512"))
MODE = sys.argv[1]

if MODE == "capture":
    os.environ["EVALSUITE_FRUGAL"] = "1"; os.environ["BIGMOE_SPARE"] = "0"; os.environ["BIGMOE_PF_NOPUBLISH"] = "1"
    for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
        os.environ.setdefault(k, "0")
    os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
    sys.path.insert(0, os.path.expanduser("~/Breakthrough/fork125bt"))
    import mlx.core as mx
    from bigmoe.loader import load_streamed
    from bigmoe import decode as D
    sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
    from evalsuite import stages as SG
    rt, st = load_streamed(MD, 128, mtp=False)
    from mtplx.models import qwen4_exp
    orig = qwen4_exp.SparseMoeBlock.__call__
    rows = {}

    def hook(self, x):
        bm = getattr(self.switch_mlp, "_bm", None)
        if bm is not None and x.shape[1] > 1:
            rows.setdefault(bm["L"], []).append(np.array(x[0].astype(mx.float16)))
        return orig(self, x)
    qwen4_exp.SparseMoeBlock.__call__ = hook
    items = SG.tf_items()
    pick = []
    for s in ("tf_ifeval", "tf_multi"):
        idx = [i for i, it in enumerate(items) if it["set"] == s]
        held = set(idx[::4])                                   # tf_frugal_var.py's subset: excluded
        pick += [i for i in idx if i not in held][:8]
    for s in ("tf_code", "tf_gsm"):
        pick += [i for i, it in enumerate(items) if it["set"] == s][:4]
    t0 = time.time()
    for i in pick:                                             # silent night: ~24 short prefills (SSD-bound, light GPU)
        ids = [int(t) for t in items[i]["ids"]][:512]
        D.greedy(rt.model, ids, 1)
    rng = np.random.default_rng(0)
    X = {}
    for L, parts in rows.items():
        a = np.concatenate(parts)
        X[f"L{L}"] = a[np.sort(rng.choice(len(a), min(NTOK, len(a)), replace=False))]
    np.savez(os.path.join(OUT, "X.npz"), **X)
    print(f"capture: {len(pick)} sequences, {len(X)} layers x {NTOK} rows in {time.time() - t0:.0f}s", flush=True)

elif MODE == "build":
    import mlx.core as mx
    sys.path.insert(0, os.path.expanduser("~/Breakthrough/fork125bt"))
    from bigmoe_wt.stream import read_headers, DT
    H = read_headers(MD)
    X = np.load(os.path.join(OUT, "X.npz"))
    layers = sorted(int(k[1:]) for k in X.files)
    pre = "language_model.model.layers.{}.mlp.switch_mlp.{}.{}"

    def load(L, p, f):
        path, off, dt, shape = H[pre.format(L, p, f)]
        npdt, mxdt = DT[dt]
        n = int(np.prod(shape))
        a = np.fromfile(path, dtype=npdt, count=n, offset=off).reshape(shape)
        return mx.array(a).view(mxdt) if npdt is np.uint16 and mxdt != mx.uint16 else mx.array(a)
    for L in layers:
        if os.path.exists(os.path.join(OUT, f"D{L}.npy")):
            continue
        t0 = time.time()
        W = {(p, f): load(L, p, f) for p in ("gate_proj", "up_proj", "down_proj") for f in ("weight", "scales", "biases")}
        x = mx.array(X[f"L{L}"]).astype(mx.bfloat16)                     # [N, 2560]
        E = W[("gate_proj", "weight")].shape[0]
        Ys = []
        for e in range(E):
            q = lambda p, h: mx.quantized_matmul(h, W[(p, "weight")][e], W[(p, "scales")][e], W[(p, "biases")][e],
                                                 transpose=True, group_size=64, bits=4)
            g, u = q("gate_proj", x), q("up_proj", x)
            Ys.append(q("down_proj", (g * mx.sigmoid(g)) * u).astype(mx.float32).reshape(-1))
            if e % 64 == 63:
                mx.eval(Ys)
        Y = mx.stack(Ys)                                                   # [E, N*2560]
        G = Y @ Y.T
        n2 = mx.diag(G)
        D2 = mx.maximum(n2[:, None] + n2[None, :] - 2 * G, 0)
        mx.eval(D2, n2)
        np.save(os.path.join(OUT, f"D{L}.npy"), np.sqrt(np.array(D2)).astype(np.float16))
        np.save(os.path.join(OUT, f"N{L}.npy"), np.sqrt(np.array(n2)).astype(np.float32))
        # zero-byte proxies (SPICE-style shared-expert base; a resident mean-weights expert; the mean of all outputs as
        # the best any single common proxy could do): per expert, optimal scale alpha = <Y_u, P> / ||P||^2 and the
        # relative residual ||Y_u - alpha P|| / ||Y_u|| (1.0 = no better than dropping the expert)
        def proxy(P):
            P = P.reshape(-1); s2 = (P * P).sum(); d = Y @ P
            return np.array(d / s2), np.sqrt(np.maximum(np.array(n2 - d * d / s2), 0) / np.array(n2))
        sh = "language_model.model.layers.{}.mlp.shared_expert.{}.{}"
        def loads(p, f):
            path, off, dt, shape = H[sh.format(L, p, f)]
            npdt, mxdt = DT[dt]
            a = np.fromfile(path, dtype=npdt, count=int(np.prod(shape)), offset=off).reshape(shape)
            return mx.array(a).view(mxdt) if npdt is np.uint16 and mxdt != mx.uint16 else mx.array(a)
        SW = {(p, f): loads(p, f) for p in ("gate_proj", "up_proj", "down_proj") for f in ("weight", "scales", "biases")}
        qs = lambda p, h: mx.quantized_matmul(h, SW[(p, "weight")], SW[(p, "scales")], SW[(p, "biases")], transpose=True,
                                              group_size=64, bits=4)
        g, u = qs("gate_proj", x), qs("up_proj", x)
        a_sh, r_sh = proxy(qs("down_proj", (g * mx.sigmoid(g)) * u).astype(mx.float32))
        Wm = {}
        for p in ("gate_proj", "up_proj", "down_proj"):
            acc = None
            for e0 in range(0, E, 64):
                dq = mx.dequantize(W[(p, "weight")][e0:e0 + 64], W[(p, "scales")][e0:e0 + 64], W[(p, "biases")][e0:e0 + 64],
                                   group_size=64, bits=4).astype(mx.float32).sum(axis=0)
                acc = dq if acc is None else acc + dq
            Wm[p] = (acc / E).astype(mx.bfloat16)
        gm, um = x @ Wm["gate_proj"].T, x @ Wm["up_proj"].T
        a_mw, r_mw = proxy((((gm * mx.sigmoid(gm)) * um) @ Wm["down_proj"].T).astype(mx.float32))
        a_mo, r_mo = proxy(Y.mean(axis=0))
        np.save(os.path.join(OUT, f"P{L}.npy"), np.stack([a_sh, r_sh, a_mw, r_mw, a_mo, r_mo]).astype(np.float32))
        print(f"L{L}: residual / norm with the best scaled proxy, median over experts: shared expert {np.median(r_sh):.3f}, "
              f"mean-weights expert {np.median(r_mw):.3f}, mean of all outputs {np.median(r_mo):.3f}", flush=True)
        Dn = np.sqrt(np.array(D2)); np.fill_diagonal(Dn, np.inf); nn = Dn.min(1) / np.sqrt(np.array(n2))
        print(f"L{L}: {time.time() - t0:.1f}s; nearest-other distance / own norm: median {np.median(nn):.3f}, "
              f"P10 {np.percentile(nn, 10):.3f}, P90 {np.percentile(nn, 90):.3f}", flush=True)
        del W, Y, G, D2, Ys
        mx.clear_cache()
