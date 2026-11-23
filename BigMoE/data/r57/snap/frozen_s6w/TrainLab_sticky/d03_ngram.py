"""D-03 NGRAM (NewIdeas Wed ~15:40, drafter track D98; pre-registered in REPORT 'D-03 NGRAM'): does the 125B's own hashed
n-gram (PLE) table carry next-token information? A ridge probe from a position's 16 n-gram rows (the 2-gram and 3-gram
ending at token i, 8 hash heads each, 160-d, concatenated) to the LM-head row of the 125B's top-1 for token i + 1,
against the same probe on token i's input embedding (control). Halves are whole samples (sorted, even / odd), two folds
pooled; lambda by grouped 3-fold CV MSE inside the training half. CPU only, numpy with 4 BLAS threads; the table, the
checkpoint and mtplx's source are read-only (the row arithmetic is exec'd from qwen4_exp.py, not imported).
Usage: taskpolicy -b nice -n 10 [BigMoE .venv] python sticky/d03_ngram.py [code,gsm]  -> data/d03_ngram.txt / .json"""
import os, sys
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import ast, glob, json, math, mmap, struct, time
import numpy as np

T0 = time.time()
MODEL = os.path.expanduser("~/BigMoE/models/bare")
SHARD = f"{MODEL}/model-00016-of-00017.safetensors"
TABLE = f"{MODEL}/ngram-table.safetensors"
QWEN = os.path.expanduser("~/BigMoE/.venv324/lib/python3.12/site-packages/mtplx/models/qwen4_exp.py")
LEAP = os.path.expanduser("~/Leap/data")
OUT = os.environ.get("D03_OUT", os.path.expanduser("~/TrainLab/data/d03_ngram"))
SETS = (sys.argv[1] if len(sys.argv) > 1 else "code,gsm").split(",")
PFX = "language_model.model.layers.1.ple.ple_embedding."
ALPHAS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
LINES, RES = [], {}


def say(s):
    s = f"[{time.time() - T0:6.1f}s] {s}"
    print(s, flush=True)
    LINES.append(s)


# ---- safetensors (read-only memmaps) and MLX's 4-bit affine layout ----------------------------------------------------
DT = {"U32": np.uint32, "BF16": np.uint16, "F16": np.float16, "F32": np.float32, "I64": np.int64, "I32": np.int32}


def st_open(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))

    def get(name):
        info = h[name]
        mm = np.memmap(path, mode="r", dtype=DT[info["dtype"]], offset=8 + n + info["data_offsets"][0],
                       shape=tuple(info["shape"]))
        try:
            mm._mmap.madvise(mmap.MADV_RANDOM)
        except (AttributeError, OSError, ValueError):
            pass
        return mm
    return h, get


def bf16(u):
    return (np.asarray(u).astype(np.uint32) << 16).view(np.float32)


def deq(w, s, b, gs):
    """MLX affine dequantize, bits 4: element j of a uint32 sits in bits 4j..4j+3; value = q * scale + bias per group."""
    w = np.asarray(w)
    n = w.shape[0]
    q = (w[:, :, None] >> (np.arange(8, dtype=np.uint32) * 4)) & np.uint32(15)
    q = q.reshape(n, -1, gs).astype(np.float32)
    return (q * bf16(s)[:, :, None] + bf16(b)[:, :, None]).reshape(n, -1)


# ---- mtplx's exact n-gram row arithmetic (source exec'd, nothing imported) --------------------------------------------
def mtplx_defs():
    tree = ast.parse(open(QWEN).read())
    want = {"_splitmix64", "_build_layer_multipliers", "_is_prime", "_head_vocab_layout", "_ngram_rows_np"}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want]
    body += [n for n in tree.body if isinstance(n, ast.Assign) and all(
        isinstance(t, ast.Name) and t.id.startswith(("_MASK64", "_SPLITMIX", "_PRIME_1")) for t in n.targets)]
    body.sort(key=lambda n: n.lineno)
    ns = {"math": math}
    exec(compile(ast.Module(body=body, type_ignores=[]), QWEN, "exec"), ns)
    assert want <= set(ns), want - set(ns)
    return ns


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cfg = json.load(open(f"{MODEL}/config.json"))["text_config"]
    eos = cfg["eos_token_id"][0] if isinstance(cfg["eos_token_id"], list) else cfg["eos_token_id"]
    NS, HPN, V, D = cfg["ngram_size"], cfg["heads_per_ngram"], cfg["vocab_size"], cfg["hidden_size"]
    M = mtplx_defs()
    hs, gs_ = st_open(SHARD)
    mult, sizes, offs = (np.asarray(gs_(PFX + k)).astype(np.int64) for k in
                         ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"))
    d_mult = M["_build_layer_multipliers"](V, NS, 0, cfg.get("seed", 1234))
    d_sizes, d_offs, _ = M["_head_vocab_layout"](cfg["ngram_vocab_size_base"], (NS - 1) * HPN, 0)
    same = (list(mult) == d_mult, list(sizes) == d_sizes, list(offs) == d_offs)
    say(f"hash constants from the checkpoint: mult {list(mult)}, {len(sizes)} heads, sizes {sizes[0]}..{sizes[-1]}; "
        f"= derived (ple_index 0, seed {cfg.get('seed', 1234)}): {same}; eos {eos}")

    ht, gt = st_open(TABLE)
    meta = ht["__metadata__"]
    TW, TS, TB = gt("ngram.weight"), gt("ngram.scales"), gt("ngram.biases")
    tgs, tdim = int(meta["ngram_group_size"]), int(meta["dim"])
    assert int(meta["ngram_bits"]) == 4 and tdim * (NS - 1) * HPN == D, meta

    # LM head (4-bit g64) dequantized whole, f32; input embeddings gathered per needed token
    t = time.time()
    LW, LS, LB = (gs_(f"language_model.lm_head.{k}") for k in ("weight", "scales", "biases"))
    lgs = D // LS.shape[1]
    W = np.empty((V, D), np.float32)
    for a in range(0, V, 16384):
        W[a:a + 16384] = deq(LW[a:a + 16384], LS[a:a + 16384], LB[a:a + 16384], lgs)
    wn2 = np.einsum("ij,ij->i", W, W)
    EW, ES, EB = (gs_(f"language_model.model.embed_tokens.{k}") for k in ("weight", "scales", "biases"))
    egs = D // ES.shape[1]
    say(f"lm_head dequantized ({V} x {D}, group {lgs}) in {time.time() - t:.1f}s; embed group {egs}; table group {tgs}")

    # self-check of the numpy dequantize against MLX's (CPU device), 8 rows each
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    for nm, (w_, s_, b_, g_) in {"lm_head": (LW, LS, LB, lgs), "ngram": (TW, TS, TB, tgs)}.items():
        r = np.arange(0, w_.shape[0], w_.shape[0] // 8)[:8]
        ref = mx.dequantize(mx.array(np.asarray(w_[r])), mx.array(bf16(s_[r])), mx.array(bf16(b_[r])),
                            group_size=g_, bits=4)
        ref = np.array(ref.astype(mx.float32))
        mine = deq(w_[r], s_[r], b_[r], g_)
        err = np.abs(mine - ref).max() / np.abs(ref).max()
        say(f"dequantize check {nm}: max |numpy - mlx| / max |mlx| = {err:.2e}")
        assert err < 1e-5, err

    for kind in SETS:
        run_set(kind, M, eos, mult, sizes, offs, NS, HPN, (TW, TS, TB, tgs), (EW, ES, EB, egs), W, wn2)
    open(OUT + ".txt", "w").write("\n".join(LINES) + "\n")
    json.dump(RES, open(OUT + ".json", "w"), indent=1)
    say(f"wrote {OUT}.txt / .json")


def load_set(kind):
    fs = sorted(glob.glob(f"{LEAP}/l118_tgt/*.npz" if kind == "code" else f"{LEAP}/l120/gsm_*.npz"))
    S = []
    for f in fs:
        z = np.load(f)
        ids, tk, lp, asst = z["ids"].astype(np.int64), z["tk"], z["lp"].astype(np.float32), z["asst"]
        n = len(ids) - 1
        assert tk.shape[0] == n and asst.shape[0] == n + 1, f
        if kind == "code":   # the labels' ids are dfl2d_code's (the 125B's greedy code)
            assert np.array_equal(np.load(f"{LEAP}/dfl2d_code/{os.path.basename(f)}")["ids"], z["ids"]), f
        S.append(dict(name=os.path.basename(f), ids=ids, n=n, y=tk[:, 0].astype(np.int64), p=np.exp(lp[:, 0]),
                      comp=asst[1:].astype(bool)))
    return S


def ridge(X, Y, alphas):
    mx_, my_ = X.mean(0), Y.mean(0)
    Xc = X - mx_
    G = (Xc.T @ Xc).astype(np.float64)
    B = (Xc.T @ (Y - my_)).astype(np.float64)
    lam0 = np.trace(G) / G.shape[0]
    out = []
    for a in alphas:
        Gl = G.copy()
        Gl[np.diag_indices_from(Gl)] += a * lam0
        out.append((np.linalg.solve(Gl, B).astype(np.float32), mx_, my_))
    return out


def fit(X, Y, grp):
    ug = np.unique(grp)
    mse = np.zeros(len(ALPHAS))
    for k in range(3):
        va = np.isin(grp, ug[k::3])
        for j, (Wk, mx_, my_) in enumerate(ridge(X[~va], Y[~va], ALPHAS)):
            mse[j] += float((((X[va] - mx_) @ Wk + my_ - Y[va]) ** 2).sum())
    j = int(np.argmin(mse))
    return ridge(X, Y, (ALPHAS[j],))[0], ALPHAS[j]


def score(P, y, W, wn2, chunk=512):
    """Nearest LM-head row (Euclidean; argmax 2 p.w - |w|^2) top-1 / top-4, and the dot-product (logit) top-1."""
    e1, e4, d1 = (np.zeros(len(P), bool) for _ in range(3))
    for a in range(0, len(P), chunk):
        S = P[a:a + chunk] @ W.T
        d1[a:a + chunk] = S.argmax(1) == y[a:a + chunk]
        S *= 2
        S -= wn2
        e1[a:a + chunk] = S.argmax(1) == y[a:a + chunk]
        t4 = np.argpartition(-S, 4, axis=1)[:, :4]
        e4[a:a + chunk] = (t4 == y[a:a + chunk, None]).any(1)
    return e1, e4, d1


def run_set(kind, M, eos, mult, sizes, offs, NS, HPN, tab, emb, W, wn2):
    t = time.time()
    S = load_set(kind)
    nrows = sum(s["n"] for s in S)
    ncomp = sum(int(s["comp"].sum()) for s in S)
    agree = np.mean(np.concatenate([s["y"][s["comp"]] == s["ids"][1:][s["comp"]] for s in S]))
    say(f"== {kind}: {len(S)} samples, {nrows} rows, {ncomp} completion rows; the 125B's top-1 = the next token on "
        f"{agree * 100:.2f} % of completion rows")
    for s in S:
        rows, _ = M["_ngram_rows_np"](s["ids"][None, :], np.full((1, NS - 1), eos, np.int64), mult=mult, sizes=sizes,
                                      offs=offs, eos=eos, ngram_size=NS, heads_per_ngram=HPN)
        s["rows"] = rows[0, :s["n"]]
    allr = np.concatenate([s["rows"].ravel() for s in S])
    U = np.unique(allr)
    TW, TS, TB, tgs = tab
    t1 = time.time()
    RU = deq(TW[U], TS[U], TB[U], tgs)
    say(f"{len(U)} distinct n-gram rows of {len(allr)} gathered + dequantized in {time.time() - t1:.1f}s "
        f"(row ids {U.min()}..{U.max()})")
    EW, ES, EB, egs = emb
    toks = np.unique(np.concatenate([s["ids"][:s["n"]] for s in S]))
    EU = deq(EW[toks], ES[toks], EB[toks], egs)
    for s in S:
        s["N"] = RU[np.searchsorted(U, s["rows"])].reshape(s["n"], -1)
        s["C"] = EU[np.searchsorted(toks, s["ids"][:s["n"]])]
    half = np.array([i % 2 for i in range(len(S))])
    out = {}
    for arm in ("N", "C", "N+C"):
        e1, e4, d1, conf, al = [], [], [], [], []
        for test in (0, 1):
            tr = [s for s, h in zip(S, half) if h != test]
            te = [s for s, h in zip(S, half) if h == test]
            X = np.concatenate([np.hstack([s[a] for a in arm.split("+")]) for s in tr])
            Y = W[np.concatenate([s["y"] for s in tr])]
            grp = np.concatenate([np.full(s["n"], i) for i, s in enumerate(tr)])
            (Wr, mx_, my_), alpha = fit(X, Y, grp)
            Xt = np.concatenate([np.hstack([s[a] for a in arm.split("+")])[s["comp"]] for s in te])
            yt = np.concatenate([s["y"][s["comp"]] for s in te])
            r = score((Xt - mx_) @ Wr + my_, yt, W, wn2)
            for lst, v in zip((e1, e4, d1), r):
                lst.append(v)
            conf.append(np.concatenate([s["p"][s["comp"]] for s in te]))
            al.append(alpha)
        e1, e4, d1, conf = (np.concatenate(v) for v in (e1, e4, d1, conf))
        hi = conf >= 0.9
        out[arm] = dict(top1=e1.mean(), top4=e4.mean(), dot1=d1.mean(), top1_p90=e1[hi].mean(), top4_p90=e4[hi].mean(),
                        top1_lo=e1[~hi].mean(), n=int(len(e1)), n_p90=int(hi.sum()), alphas=al)
        o = out[arm]
        say(f"{kind} {arm:4s}: held-out completion rows top-1 {o['top1'] * 100:5.1f} % top-4 {o['top4'] * 100:5.1f} % "
            f"(dot top-1 {o['dot1'] * 100:5.1f}) | p >= 0.9 ({o['n_p90']}): top-1 {o['top1_p90'] * 100:5.1f} % "
            f"top-4 {o['top4_p90'] * 100:5.1f} % | p < 0.9: top-1 {o['top1_lo'] * 100:5.1f} % | alpha {al}")
    RES[kind] = out
    n1, c1 = out["N"]["top1_p90"] * 100, out["C"]["top1_p90"] * 100
    if kind == "code":
        v = ("PASS" if n1 >= 30 and n1 >= c1 + 10 else "KILL" if n1 < 10 or n1 <= c1 + 3 else "BETWEEN")
        RES["verdict"] = v
        say(f"D-03 verdict (code, p >= 0.9 held-out top-1): N {n1:.1f} % vs C {c1:.1f} % ({n1 - c1:+.1f} points); "
            f"PASS >= 30 and >= C + 10, KILL < 10 or <= C + 3 -> {v}")
    else:
        say(f"{kind} (information): N {n1:.1f} % vs C {c1:.1f} % ({n1 - c1:+.1f} points) on p >= 0.9 rows")
    say(f"{kind} done in {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
