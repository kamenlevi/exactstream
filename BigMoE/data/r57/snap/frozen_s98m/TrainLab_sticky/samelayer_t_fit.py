"""SAMELAYER-T final fit (MTP Head, Wed 11-04 15:0x; BigMoE's ask 14:5x for lever 2 = build #2): the trained same-layer
forecast router fit on ALL 18 prompts of BigMoE's pr2_160 dump, saved as weights for the engine. CPU only.
  lambda per layer: the LOPO grid. Each LAMS value is scored leave-one-prompt-out on all 18 folds (pooled per-row recall@10
  of the held-out prompts, code and chat together); the best value per layer is refit on all prompts. (The SAMELAYER-T
  verdict, REPORT :2828, picked lambda on inner prompts per fold: 0.823 / 0.851. This grid picks on the outer folds, so its
  LOPO recall is slightly optimistic.)
  forecast (folded, the same GEMV as the router): logits_L = X W_L^T + b_L with W_L = R_L + dW_L^T [512 x 2560] and
  b_L = e0 - mu dW_L [512]; the forecast is its top-10. The bias is needed: a per-expert offset moves the top-10.
  X = the dump's X: layer L's MLP-side read of its pre-attention stream (~/BigMoE/scripts/r57_screen.py:582,
  pre = self.mlp_hyper_connection(hidden)[0], after the PLE add at :578; pr2's corrected 'full'). R_L = L's 8-bit router
  dequantized (fp32).
Saves SLT_W (~/TrainLab/data/samelayer_t_W.npz): W_bf16 (bf16 bits as uint16, [47, 512, 2560]; MLX:
mx.array(W_bf16).view(mx.bfloat16)), b (float32 [47, 512]), layers, lams, lam_rel / lam (chosen, relative to G's mean
eigenvalue / absolute), lopo [47, 4] (untrained code, trained code, untrained chat, trained chat), lopo_grid [47, 5] (pooled
LOPO recall per lams value), insample [47, 3] (fp64 folded W, the saved bf16 W, R exact + dW in bf16), prompts, note.
Report: SLT_OUT (~/TrainLab/data/samelayer_t_fit.txt, appended).
"""
import os, glob, json, struct, time, resource
TH = os.environ.get("SLT_THREADS", "8")
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = TH
import numpy as np
DIR = os.path.expanduser(os.environ.get("SLT_DIR", "~/BigMoE/data/r57/pr_dump"))
TAG = os.environ.get("SLT_TAG", "pr2_160")
OUT = os.path.expanduser(os.environ.get("SLT_OUT", "~/TrainLab/data/samelayer_t_fit.txt"))
WOUT = os.path.expanduser(os.environ.get("SLT_W", "~/TrainLab/data/samelayer_t_W.npz"))
MD = os.path.expanduser("~/BigMoE/models/bare")
LAMS = (0.1, 0.3, 1.0, 3.0, 10.0)
NOTE = ("SAMELAYER-T final fit (MTP Head 11-04): forecast logits_L = X @ W_L.T + b_L, top-10. X = layer L's MLP-side read of "
        "its pre-attention stream: ~/BigMoE/scripts/r57_screen.py:582 pre = self.mlp_hyper_connection(hidden)[0], after the "
        "PLE add at :578 ('full'). W_L = R_L + dW_L^T, R_L = L's 8-bit router dequantized. W_bf16 = bf16 bits as uint16. "
        f"Fit on all prompts of {TAG}; lambda per layer from the 18-fold LOPO grid (pooled per-row recall@10).")
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)


idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
heads = {}
def tensor(name):                                       # MoonShots' laysz45 safetensors reader (copy, as samelayer_t.py)
    f = os.path.join(MD, idx[name])
    if f not in heads:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            heads[f] = (8 + n, json.loads(fh.read(n)))
    start, h = heads[f]; m = h[name]
    dt = {"U32": np.uint32, "BF16": np.uint16, "F32": np.float32}[m["dtype"]]
    return np.memmap(f, dtype=dt, mode="r", offset=start + m["data_offsets"][0], shape=tuple(m["shape"]))
def f32(u16):
    return (np.asarray(u16).astype(np.uint32) << 16).view(np.float32)
def router(l):                                          # layer l's 8-bit router, dequantized fp32 [512, 2560]
    w, s, b = (tensor(f"language_model.model.layers.{l}.mlp.gate.{k}") for k in ("weight", "scales", "biases"))
    bits = w.shape[-1] * 32 // (s.shape[-1] * 64)
    sh = np.arange(0, 32, bits, dtype=np.uint32)
    q = ((np.asarray(w)[..., None] >> sh) & np.uint32((1 << bits) - 1)).reshape(w.shape[0], -1).astype(np.float32)
    o, n = q.shape
    return (q.reshape(o, n // 64, 64) * f32(s)[..., None] + f32(b)[..., None]).reshape(o, n)
def top10(lg):
    return np.argpartition(-lg, 9, axis=1)[:, :10]
def recall(pred, true):                                 # per-row recall@10 of the true top-10
    m = np.zeros((len(true), 512), bool)
    np.put_along_axis(m, true, True, axis=1)
    return np.take_along_axis(m, pred, axis=1).sum(1) / 10.0
def bf16_bits(a):                                       # float -> bf16 bit pattern, round to nearest even (finite values)
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
def from_bits(b16):
    return (b16.astype(np.uint32) << 16).view(np.float32)


t0 = time.time()
files = sorted(glob.glob(os.path.join(DIR, f"{TAG}_*.npz")))
assert files, f"no dump at {DIR}/{TAG}_*.npz"
P = []
for f in files:
    z = np.load(f)
    key = str(z["key"]); sub = key.split("/")[0]
    P.append(dict(key=key, sub=sub, dom="code" if sub in ("humaneval", "mbpp") else "chat", L=z["L"].astype(np.int64),
                  X=z["X"], Y=z["Y"]))
layers = sorted(set(np.concatenate([p["L"] for p in P]).tolist()))
say(f"SAMELAYER-T FIT {time.strftime('%m-%d %H:%M')} ({DIR}/{TAG}_*): {len(P)} prompts "
    f"({sum(p['dom'] == 'code' for p in P)} code, {sum(p['dom'] == 'chat' for p in P)} chat), "
    f"{sum(len(p['L']) for p in P)} row-layers, layers {layers[0]}..{layers[-1]}; loaded in {time.time() - t0:.0f} s; "
    f"lambdas {LAMS} x mean eigenvalue, picked per layer on the 18-fold LOPO grid")

ACC = {}                                                # (key) -> [sum, n]
def add(k, r):
    a = ACC.setdefault(k, [0.0, 0]); a[0] += float(np.sum(r)); a[1] += int(np.size(r))
def g(k):
    a = ACC.get(k); return a[0] / a[1] if a and a[1] else float("nan")
Wbs, bs, LAMR, LAMA, LOPO, GRID, INS = [], [], [], [], [], [], []
for L in layers:
    tl = time.time()
    R = router(L).astype(np.float64)
    D = []                                              # per prompt at this layer: n, sx, se, X, E, base logits, truth
    for p in P:
        s_ = p["L"] == L
        X = p["X"][s_].astype(np.float64); Y = p["Y"][s_].astype(np.float64)
        B = X @ R.T; E = Y @ R.T - B
        D.append((len(X), X.sum(0), E.sum(0), X, E, B, top10(Y @ R.T)) if len(X) else None)
    have = [i for i, d in enumerate(D) if d is not None]
    tot = [sum(D[i][k] for i in have) for k in range(3)] + [sum(D[i][3].T @ D[i][3] for i in have),
                                                            sum(D[i][3].T @ D[i][4] for i in have)]

    def stats(excl):                                    # centered Gram / cross terms of all prompts but excl
        n, sx, se = (tot[k] - sum(D[i][k] for i in excl) for k in range(3))
        gxx = tot[3] - sum(D[i][3].T @ D[i][3] for i in excl)
        gxe = tot[4] - sum(D[i][3].T @ D[i][4] for i in excl)
        mu, e0 = sx / n, se / n
        G = gxx - n * np.outer(mu, mu); C = gxe - n * np.outer(mu, e0)
        return mu, e0, G, C, np.trace(G) / G.shape[0]

    if L == layers[0]:                                  # independent check of the leave-one-out totals (fold 0)
        tr0 = have[1:]
        Xt = np.concatenate([D[j][3] for j in tr0]); Xt -= Xt.mean(0)
        Et = np.concatenate([D[j][4] for j in tr0]); Et -= Et.mean(0)
        _, _, G_, C_, _ = stats(have[:1])
        eg = np.abs(G_ - Xt.T @ Xt).max() / np.abs(G_).max(); ec = np.abs(C_ - Xt.T @ Et).max() / np.abs(C_).max()
        say(f"  check (layer {L}, fold 0): leave-one-out Gram / cross terms vs direct, max rel err {eg:.1e} / {ec:.1e}")
        assert eg < 1e-6 and ec < 1e-6, "leave-one-out totals disagree with the direct computation"
        del Xt, Et

    I = np.eye(R.shape[1])
    S = {lam: {"code": [0.0, 0], "chat": [0.0, 0]} for lam in LAMS}
    U = {"code": [0.0, 0], "chat": [0.0, 0]}
    for i in have:                                      # the LOPO grid: every lambda on every held-out prompt
        mu, e0, G, C, sc = stats([i])
        dom = P[i]["dom"]
        X, B, T = D[i][3], D[i][5], D[i][6]
        ru = recall(top10(B), T); U[dom][0] += ru.sum(); U[dom][1] += len(ru)
        Xc = X - mu
        for lam in LAMS:
            dW = np.linalg.solve(G + lam * sc * I, C)
            r = recall(top10(B + Xc @ dW + e0), T)
            S[lam][dom][0] += r.sum(); S[lam][dom][1] += len(r)
    pooled = {lam: (S[lam]["code"][0] + S[lam]["chat"][0]) / (S[lam]["code"][1] + S[lam]["chat"][1]) for lam in LAMS}
    best = max(LAMS, key=lambda lam: pooled[lam])
    rr = lambda d: d[0] / d[1] if d[1] else float("nan")
    LOPO.append([rr(U["code"]), rr(S[best]["code"]), rr(U["chat"]), rr(S[best]["chat"])])
    GRID.append([pooled[lam] for lam in LAMS])
    for dom in ("code", "chat"):
        ACC.setdefault((dom, "untrained"), [0.0, 0]); ACC.setdefault((dom, "lopo"), [0.0, 0])
        ACC[(dom, "untrained")][0] += U[dom][0]; ACC[(dom, "untrained")][1] += U[dom][1]
        ACC[(dom, "lopo")][0] += S[best][dom][0]; ACC[(dom, "lopo")][1] += S[best][dom][1]

    mu, e0, G, C, sc = stats([])                        # the final fit: all prompts, the chosen lambda
    dW = np.linalg.solve(G + best * sc * I, C)          # [2560, 512]
    Wf = R + dW.T                                       # [512, 2560], the router's orientation
    bf = e0 - mu @ dW                                   # [512]
    Wb = bf16_bits(Wf)
    Wq = from_bits(Wb).astype(np.float64)
    Wd = R + from_bits(bf16_bits(dW.T)).astype(np.float64)
    ins = []
    for nm, Wv in (("fp64", Wf), ("bf16", Wq), ("Rdw", Wd)):
        tot_r = 0.0; tot_n = 0
        for i in have:
            r = recall(top10(D[i][3] @ Wv.T + bf), D[i][6])
            add((P[i]["dom"], "in_" + nm), r); tot_r += r.sum(); tot_n += len(r)
        ins.append(tot_r / tot_n)
    INS.append(ins)
    Wbs.append(Wb); bs.append(bf.astype(np.float32)); LAMR.append(best); LAMA.append(best * sc)
    if L in (layers[0], layers[len(layers) // 2], layers[-1]):
        say(f"  layer {L}: {sum(D[i][0] for i in have)} rows, {time.time() - tl:.1f} s; lambda {best:g} "
            f"(pooled LOPO {', '.join(f'{lam:g}: {pooled[lam]:.4f}' for lam in LAMS)}); in-sample fp64 / bf16 / R+dW "
            f"{ins[0]:.4f} / {ins[1]:.4f} / {ins[2]:.4f}")

LO = np.array(LOPO)
say(f"  LOPO recall at the chosen lambdas (pooled rows x layers): code {g(('code', 'untrained')):.3f} -> "
    f"{g(('code', 'lopo')):.3f}, chat {g(('chat', 'untrained')):.3f} -> {g(('chat', 'lopo')):.3f} "
    f"(SAMELAYER-T's nested pick: code 0.706 -> 0.823, chat 0.821 -> 0.851)")
say(f"  lambda picks over layers: " + ", ".join(f"{lam:g} x{sum(1 for x in LAMR if x == lam)}" for lam in LAMS))
say(f"  in-sample recall (all prompts, optimistic): fp64 W code {g(('code', 'in_fp64')):.4f} chat {g(('chat', 'in_fp64')):.4f}; "
    f"saved bf16 W code {g(('code', 'in_bf16')):.4f} chat {g(('chat', 'in_bf16')):.4f}; R exact + dW bf16 code "
    f"{g(('code', 'in_Rdw')):.4f} chat {g(('chat', 'in_Rdw')):.4f}")
np.savez(WOUT, W_bf16=np.stack(Wbs), b=np.stack(bs), layers=np.array(layers, np.int16), lams=np.array(LAMS),
         lam_rel=np.array(LAMR, np.float32), lam=np.array(LAMA, np.float64), lopo=LO.astype(np.float32),
         lopo_grid=np.array(GRID, np.float32), insample=np.array(INS, np.float32),
         prompts=np.array([p["key"] for p in P]), note=np.array(NOTE))
z = np.load(WOUT)                                       # read-back check: the file holds what was computed
assert z["W_bf16"].shape == (len(layers), 512, 2560) and z["W_bf16"].dtype == np.uint16 and z["b"].shape == (len(layers), 512)
assert np.array_equal(z["W_bf16"][-1], Wbs[-1]) and np.array_equal(z["b"][0], bs[0])
say(f"  saved {WOUT}: {os.path.getsize(WOUT) / 2**20:.0f} MB (W_bf16 {z['W_bf16'].shape}, b {z['b'].shape}); read back OK; "
    f"{time.time() - t0:.0f} s in all; max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.1f} GB")
open(OUT, "a").write("\n".join(lines) + "\n\n")
