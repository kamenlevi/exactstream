"""WEEK98 S13 QATSCREEN read-out (MTP Head; REPORT 'WEEK98 S13 QATSCREEN'), sandbox (2B target): per run (or
run@lever, read as evalx reads it) on MBPP-50 / HE-30, the teacher-forced chain from every anchor to depth 31 with the
head's top-8 per cell, then kept per check under the walk (cc_readout's: from row max(0, gen - 1 - a0), +1 per check):
- chain-32: the leading right depths + 1 (equals cc_readout's exact kept/32; printed as the self-check);
- tree-16 / tree-32: S9b's EAGLE-2 proxy (tree_kprice.py's proxy_check): best-first over path probabilities, K - 1
  pops; on-path children are exact from the true path's top-8, off-path children = the true path's top-8 at the same
  depth ^ gamma (1 central; band 0.5 .. 2, S11's calibration on the 125B);
- top-1 agreement with the FIRST run (the bf16 reference) per depth band, on the same cells.
Self-check: each run's first window's 'right' equals core.chain_table with the run's lever fn.
Pairs (PAIRS='run:ctl,...'): kept ratio run / ctl per set and K, window bootstrap 95 % CI.
Usage: cd ~/Sandbox98 && PY ~/TrainLab/sb98/tree8.py OUT.json RUN[@LEVER] ..."""
import os, sys, glob, json, heapq
sys.path.insert(0, os.path.expanduser("~/Sandbox98"))
import numpy as np
import mlx.core as mx
from sb import core as C

C.limit_memory(4.5)
OUT, SPECS = sys.argv[1], sys.argv[2:]
RUNS = [sp.partition("@")[0] for sp in SPECS]
KEYS = [r + ("@" + os.path.basename(sp.partition("@")[2]) if RUNS.count(r) > 1 and "@" in sp else "")
        for r, sp in zip(RUNS, SPECS)]                       # a run read twice (e.g. s30 and s30@qat4) keys by lever
SETS = (("mbpp", 50), ("he", 30))
DEP, K8, KS, GAMMAS = 31, 8, (16, 32), (0.5, 1.0, 2.0)
BANDS = ((0, 0), (1, 2), (3, 6), (7, 14), (15, 22), (23, 30))
PAIRS = [tuple(x.split(":")) for x in os.environ.get("PAIRS", "").split(",") if ":" in x]


def files(d, n):
    return sorted(f for f in glob.glob(os.path.join(C.ROOT, "data", d, "*.npz")) if not f.endswith(".taps.npz"))[:n]


def cells(D, f, fn):
    """-> R [n, 31] the true token's slot in the top-8 (8 = below, -1 = no target), Q [n, 31, 8] top-8 probs,
    T1 [n, 31] top-1 ids, right [n, 31], gen, a0"""
    z = np.load(f)
    ids = np.asarray(z["ids"], np.int32); gen = int(z["gen"])
    W = C.load_W(z)
    a0 = max(0, gen - 2); S = len(ids); n = S - 1 - a0
    R = np.full((n, DEP), -1, np.int8); Q = np.zeros((n, DEP, K8), np.float32); T1 = np.full((n, DEP), -1, np.int64)
    C.REPL.update(R=None, K=1, fn=fn)
    C.CUR["f"], C.CUR["train"], C.CUR["x"] = f, False, None
    hs = []

    def on_depth(j, h, anc):
        m = min(h.shape[1], S - 1 - j - a0)
        if m > 0:
            hs.append((j, m, D.norm(h[0, :m]).astype(mx.bfloat16)))
    C.chain(D, W, mx.array(ids), DEP, on_depth, a0)
    Hs = mx.concatenate([x for _, _, x in hs], 0)
    top, val = [], []
    for c0 in range(0, Hs.shape[0], 512):
        lg = C.lm_logits(Hs[c0:c0 + 512])
        lse = mx.logsumexp(lg, -1, keepdims=True)
        i8 = mx.argpartition(-lg, K8 - 1, axis=-1)[:, :K8]
        v8 = mx.take_along_axis(lg, i8, -1)
        o = mx.argsort(-v8, -1)
        i8 = mx.take_along_axis(i8, o, -1); v8 = mx.take_along_axis(v8, o, -1)
        am = mx.argmax(lg, -1)
        fix = i8[:, :1] != am[:, None]                            # ties: slot 0 = core's argmax (chain_table's)
        i8 = mx.where(fix, mx.concatenate([am[:, None], i8[:, :K8 - 1]], -1), i8)
        v8 = mx.where(fix, mx.concatenate([mx.take_along_axis(lg, am[:, None], -1), v8[:, :K8 - 1]], -1), v8)
        top.append(i8); val.append(mx.exp(v8 - lse)); mx.eval(top[-1], val[-1])
    top = np.array(mx.concatenate(top)).astype(np.int64); val = np.array(mx.concatenate(val)).astype(np.float32)
    o = 0
    for j, m, _ in hs:
        t8, p8 = top[o:o + m], val[o:o + m]; o += m
        true = ids[a0 + j + 1:a0 + j + 1 + m].astype(np.int64)
        isT = t8 == true[:, None]
        R[:m, j - 1] = np.where(isT.any(1), isT.argmax(1), K8).astype(np.int8)
        Q[:m, j - 1] = p8; T1[:m, j - 1] = t8[:, 0]
    del hs, Hs, W
    mx.clear_cache()
    return R, Q, T1, (R == 0).astype(np.int8) - (R < 0).astype(np.int8) * 1, gen, a0


def proxy_check(Q, R, N, gamma):
    hp = [(-float(Q[0, r]), 0, bool(R[0] == r)) for r in range(K8) if Q[0, r] > 0]
    heapq.heapify(hp); acc = cnt = 0
    while hp and cnt < N - 1:
        ns, j, on = heapq.heappop(hp); cnt += 1
        if on:
            acc = j + 1
        if j + 1 >= DEP:
            continue
        for r in range(K8):
            q = float(Q[j + 1, r])
            if q > 0:
                heapq.heappush(hp, (ns * (q if on else q ** gamma), j + 1, bool(on and R[j + 1] == r)))
    return acc


def walk(n, r0, accept):
    ks, nc, p = 0, 0, r0
    while p < n:
        a = accept(p); ks += a + 1; nc += 1; p += a + 1
    return ks, nc


RES, T1REF = {}, {}
for ri, (run, spec, key) in enumerate(zip(RUNS, SPECS, KEYS)):
    lev = spec.partition("@")[2]
    D = C.load_drafter(run, strict=not lev); fn = None
    if lev:
        from sb.train import load_lever
        L = load_lever(lev)
        if hasattr(L, "setup"):
            L.setup(D, None)
            D.load_weights(os.path.join(C.CKPT, run + ".safetensors"), strict=False)
        fn = getattr(L, "standin", None)
    for st, nw in SETS:
        per = {k: [] for k in ["chain"] + [f"K{N}|g{g}" for N in KS for g in GAMMAS]}
        agree = np.zeros((2, DEP))
        for wi, f in enumerate(files(st, nw)):
            R, Q, T1, right, gen, a0 = cells(D, f, fn)
            if wi == 0:
                ref = C.chain_table(D, f, DEP, fn=fn)[0]
                mx.clear_cache()
                ok = ref.shape == right.shape and bool((ref == right).all())
                print(f"SELF-CHECK {spec} {st}: right == core.chain_table: {'PASS' if ok else 'FAIL'}", flush=True)
                if not ok:
                    sys.exit(2)
            n, r0 = R.shape[0], max(0, gen - 1 - a0)
            Rm = np.where(R >= K8, -2, R)                          # below the top-8: never on a tree path
            per["chain"].append(walk(n, r0, lambda p: int(np.argmin(np.append(right[p] == 1, False)))))
            for N in KS:
                for g in GAMMAS:
                    per[f"K{N}|g{g}"].append(walk(n, r0, lambda p: proxy_check(Q[p], Rm[p], N, g)))
            if ri == 0:
                T1REF[(st, wi)] = T1
            else:
                v = (T1 >= 0) & (T1REF[(st, wi)] >= 0) & (R >= 0)
                agree[0] += ((T1 == T1REF[(st, wi)]) & v).sum(0); agree[1] += v.sum(0)
        res = {k: {"kept": float(np.sum([x[0] for x in v]) / np.sum([x[1] for x in v])), "per_window": v}
               for k, v in per.items()}
        if ri > 0:
            res["top1_agree_bands"] = {f"{lo}-{hi}": float(agree[0, lo:hi + 1].sum() / max(agree[1, lo:hi + 1].sum(), 1))
                                       for lo, hi in BANDS}
        RES[f"{key}|{st}"] = res
        print(f"{key} {st}: chain-32 {res['chain']['kept']:.3f}; tree-16 {res['K16|g1.0']['kept']:.3f} "
              f"[{res['K16|g0.5']['kept']:.3f}..{res['K16|g2.0']['kept']:.3f}]; tree-32 {res['K32|g1.0']['kept']:.3f} "
              f"[{res['K32|g0.5']['kept']:.3f}..{res['K32|g2.0']['kept']:.3f}]"
              + (f"; top-1 agreement with {KEYS[0]} " + " ".join(f"{b} {100 * v:.1f}" for b, v in res["top1_agree_bands"].items())
                 if ri > 0 else ""), flush=True)
    del D
    mx.clear_cache()
rng = np.random.default_rng(1115)
CMP = {}
for run, ctl in PAIRS:
    for st, _ in SETS:
        for k in ["chain"] + [f"K{N}|g1.0" for N in KS]:
            A = np.array(RES[f"{ctl}|{st}"][k]["per_window"], float); B = np.array(RES[f"{run}|{st}"][k]["per_window"], float)
            idx = rng.integers(0, len(A), (5000, len(A)))
            bs = (B[idx, 0].sum(1) / B[idx, 1].sum(1)) / (A[idx, 0].sum(1) / A[idx, 1].sum(1))
            r = (B[:, 0].sum() / B[:, 1].sum()) / (A[:, 0].sum() / A[:, 1].sum())
            CMP[f"{run}@{ctl}|{st}|{k}"] = {"ratio": float(r), "ci95": np.percentile(bs, [2.5, 97.5]).tolist()}
            print(f"  {run} / {ctl} {st} {k}: x{r:.4f} [{np.percentile(bs, 2.5):.4f}, {np.percentile(bs, 97.5):.4f}]", flush=True)
json.dump({"specs": SPECS, "res": RES, "cmp": CMP}, open(OUT, "w"), indent=1)
