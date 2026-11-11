"""SAMELAYER-T (MTP Head, Wed 11-04 06:0x; pre-registered in REPORT 'PR160 read check + SAMELAYER-T', 4ecbb6f; CPU only): a
trained same-layer (pre-attention) forecast router (arXiv 2511.10676 adapted) on BigMoE's corrected pre-route dump (WED_TESTS
3b, R57_PR_DUMP=1: ~/BigMoE/data/r57/pr_dump/<TAG>_NN.npz, one file per prompt: key, L, chk, row, X, Y; per decode verify
row (rejected draft rows included) and streamed layer L = 1..47: X = L's MLP-side read of its pre-attention stream (pr2's
'full' input, PLE included), Y = L's true MoE input, float32 x 2560).
  truth     = top10(Y R_L^T), R_L = L's 8-bit router dequantized to fp32 (MoonShots' laysz45 reader, copied read-only)
  untrained = top10(X R_L^T)   (cross-check: BigMoE's pr2 'full' per-row recall, expect within ~0.01-0.02)
  trained   = top10(X R_L^T + (X - mu) dW + e0): ridge on the logit residual E = (Y - X) R_L^T (residual form: dW = 0 is
              the untrained forecast). At run time it folds into the router (W = R_L + dW^T, bias e0 - mu dW): the same GEMV.
Leave-one-prompt-out over the dump's prompts. Per layer and fold, lambda (relative to G's mean eigenvalue) from LAMS on two
inner held-out prompts (one code, one chat, when present), then a refit on all training prompts.
Metric: per-row recall@10 of the truth, rows x layers pooled per domain (code = humaneval / mbpp; chat = the rest).
LINE (pre-registered): PASS >= 0.85 code AND >= 0.90 chat; KILL < 0.78 code; BETWEEN otherwise.
env: SLT_DIR (~/BigMoE/data/r57/pr_dump), SLT_TAG (pr2_160), SLT_THREADS (8), SLT_OUT (~/TrainLab/data/samelayer_t.txt).
"""
import os, glob, json, struct, time
TH = os.environ.get("SLT_THREADS", "8")
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = TH
import numpy as np
DIR = os.path.expanduser(os.environ.get("SLT_DIR", "~/BigMoE/data/r57/pr_dump"))
TAG = os.environ.get("SLT_TAG", "pr2_160")
OUT = os.path.expanduser(os.environ.get("SLT_OUT", "~/TrainLab/data/samelayer_t.txt"))
MD = os.path.expanduser("~/BigMoE/models/bare")
LAMS = (0.1, 0.3, 1.0, 3.0, 10.0)
BANDS = ((1, 7), (8, 39), (40, 47))
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)


idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
heads = {}
def tensor(name):                                       # MoonShots' laysz45 safetensors reader (copy)
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
say(f"SAMELAYER-T {time.strftime('%m-%d %H:%M')} ({DIR}/{TAG}_*): {len(P)} prompts "
    f"({', '.join(f'{s} {sum(p['sub'] == s for p in P)}' for s in sorted({p['sub'] for p in P}))}), "
    f"{sum(len(p['L']) for p in P)} row-layer samples, layers {layers[0]}..{layers[-1]} ({len(layers)}); "
    f"loaded in {time.time() - t0:.0f} s; lambdas {LAMS} x mean eigenvalue")

ACC = {}                                                # (dom or sub, variant, band) -> [sum, n]
def add(keys, r):
    for k in keys:
        a = ACC.setdefault(k, [0.0, 0]); a[0] += float(r.sum()); a[1] += len(r)
LAMC = {lam: 0 for lam in LAMS}
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

    def stats(excl):                                    # centered Gram / cross terms of all prompts but excl; the
        n, sx, se = (tot[k] - sum(D[i][k] for i in excl) for k in range(3))    # excluded prompts' Grams are rebuilt
        gxx = tot[3] - sum(D[i][3].T @ D[i][3] for i in excl)                 # on the fly (no 18 x 62 MB store:
        gxe = tot[4] - sum(D[i][3].T @ D[i][4] for i in excl)                 # peak ~3 GB, under the 5 GB day line)
        mu, e0 = sx / n, se / n
        G = gxx - n * np.outer(mu, mu); C = gxe - n * np.outer(mu, e0)
        return mu, e0, G, C, np.trace(G) / G.shape[0]

    def solve(G, C, sc, lam):
        return np.linalg.solve(G + lam * sc * np.eye(G.shape[0]), C)

    def score(ids, mu, e0, dW):
        r = np.concatenate([recall(top10(D[i][5] + (D[i][3] - mu) @ dW + e0), D[i][6]) for i in ids])
        return r.mean()

    if L == layers[0]:                                  # independent check of the leave-one-out totals (fold 0)
        tr0 = have[1:]
        Xt = np.concatenate([D[j][3] for j in tr0]); Xt -= Xt.mean(0)
        Et = np.concatenate([D[j][4] for j in tr0]); Et -= Et.mean(0)
        _, _, G_, C_, _ = stats(have[:1])
        eg = np.abs(G_ - Xt.T @ Xt).max() / np.abs(G_).max(); ec = np.abs(C_ - Xt.T @ Et).max() / np.abs(C_).max()
        say(f"  check (layer {L}, fold 0): leave-one-out Gram / cross terms vs direct, max rel err {eg:.1e} / {ec:.1e}")
        assert eg < 1e-6 and ec < 1e-6,"leave-one-out totals disagree with the direct computation"
        del Xt, Et

    band = next(f"{a}-{b}" for a, b in BANDS if a <= L <= b)
    for i in have:
        tr = [j for j in have if j != i]
        inner = []
        for dom in ("code", "chat"):
            c = [j for j in tr if P[j]["dom"] == dom]
            if c and len(tr) - len(inner) > 1:
                inner.append(c[(i + L) % len(c)])     # rotate the inner prompt over folds and layers
        if inner:
            mu, e0, G, C, sc = stats([i] + inner)
            best = max(LAMS, key=lambda lam: score(inner, mu, e0, solve(G, C, sc, lam)))
        else:
            best = LAMS[1]
        LAMC[best] += 1
        mu, e0, G, C, sc = stats([i])
        dW = solve(G, C, sc, best)
        n, X, B, T = D[i][0], D[i][3], D[i][5], D[i][6]
        ru = recall(top10(B), T); rt = recall(top10(B + (X - mu) @ dW + e0), T)
        for v, r in (("untrained", ru), ("trained", rt)):
            add([(P[i]["dom"], v, "all"), (P[i]["sub"], v, "all"), (P[i]["dom"], v, band), ("all", v, "all")], r)
    if L in (layers[0], layers[len(layers) // 2], layers[-1]):
        say(f"  layer {L}: {sum(D[i][0] for i in have)} rows over {len(have)} prompts, {time.time() - tl:.1f} s; running "
            f"recall untrained / trained: " + ", ".join(
                f"{d} {ACC[(d, 'untrained', 'all')][0] / ACC[(d, 'untrained', 'all')][1]:.3f} / "
                f"{ACC[(d, 'trained', 'all')][0] / ACC[(d, 'trained', 'all')][1]:.3f}"
                for d in ("code", "chat") if (d, "trained", "all") in ACC))


def g(k):
    a = ACC.get(k); return a[0] / a[1] if a and a[1] else float("nan")
for d in ["code", "chat"] + sorted({p["sub"] for p in P}) + ["all"]:
    if (d, "trained", "all") in ACC:
        say(f"  {d:9s}: per-row recall@10 untrained {g((d, 'untrained', 'all')):.3f} -> trained {g((d, 'trained', 'all')):.3f} "
            f"({100 * (g((d, 'trained', 'all')) - g((d, 'untrained', 'all'))):+.1f} points; {ACC[(d, 'trained', 'all')][1]} row-layers)"
            + ("; bands " + ", ".join(f"{a}-{b} {g((d, 'untrained', f'{a}-{b}')):.3f}/{g((d, 'trained', f'{a}-{b}')):.3f}"
                                      for a, b in BANDS if (d, 'trained', f'{a}-{b}') in ACC) if d in ("code", "chat") else ""))
import resource
say(f"  lambda picks: " + ", ".join(f"{lam:g} x{c}" for lam, c in LAMC.items()) + f"; {time.time() - t0:.0f} s in all; "
    f"max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.1f} GB")
c_, h_ = g(("code", "trained", "all")), g(("chat", "trained", "all"))
v = "PASS" if (c_ >= 0.85 and h_ >= 0.90) else ("KILL" if c_ < 0.78 else "BETWEEN")
say(f"  LINE (PASS >= 0.85 code AND >= 0.90 chat; KILL < 0.78 code) -> {v} (code {c_:.3f}, chat {h_:.3f})")
open(OUT, "a").write("\n".join(lines) + "\n\n")
