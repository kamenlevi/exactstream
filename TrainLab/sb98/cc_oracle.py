"""WEEK98 S8 RERANK GATE (MTP Head; REPORT 'WEEK98 S8 RERANK GATE'): room for a re-ranking loss under the close-call
rule. Per run, MBPP-50 / HE-30 (evalx's first 50 / 30, sorted): the teacher-forced chain table built as core.chain_table
builds it (read-only use of core.chain / load_W / lm_logits), with the head's top-3 per cell. A token is IN when it is the
true token or in A_1 = {t in the window's stored top-32 : lp_top1 - lp_t <= 1}.
- Oracle walks (core.kept_walk's rule): a cell is accepted when any of the head's top-k (k = 1, 2, 3) is IN; k = 1 is the
  delta-1 close-call walk of cc_readout.py. Also the exact top-k walks (IN = the true token only).
- At each terminating miss of the delta-1 walk: whether #2 / #2-or-#3 is IN.
Self-check: the top-1 table equals core.chain_table on each set's first window.
Usage: cd ~/Sandbox98 && PY ~/TrainLab/sb98/cc_oracle.py OUT.json RUN..."""
import os, sys, glob, json
sys.path.insert(0, os.path.expanduser("~/Sandbox98"))
import numpy as np
import mlx.core as mx
from sb import core as C

C.limit_memory(4.5)                      # Sandbox98's caps (06:06 incident), plus clear_cache per window
OUT, RUNS = sys.argv[1], sys.argv[2:]
SETS = (("mbpp", 50), ("he", 30))
DELTA = 1.0


def files(d, n):
    return sorted(f for f in glob.glob(os.path.join(C.ROOT, "data", d, "*.npz")) if not f.endswith(".taps.npz"))[:n]


def table3(D, f, Dp=31):
    """inn [n_anchors, Dp, 3] (1 IN / 0 not / -1 no target) for the head's #1..#3 under A_1; ex [.., 3] the same for the
    true token only; rt = the exact table (#1 == true token)"""
    z = np.load(f)
    ids = np.asarray(z["ids"], np.int32); gen = int(z["gen"])
    tk = np.asarray(z["tk"], np.int64); lp = np.asarray(z["lp"], np.float32)
    W = C.load_W(z)
    a0 = max(0, gen - 2); S = len(ids); n = S - 1 - a0
    inn = np.full((n, Dp, 3), -1, np.int8); ex = np.full((n, Dp, 3), -1, np.int8)
    C.REPL.update(R=None, K=1, fn=None)
    C.CUR["f"], C.CUR["train"], C.CUR["x"] = f, False, None
    hs = []

    def on_depth(j, h, anc):
        m = min(h.shape[1], S - 1 - j - a0)
        if m > 0:
            hs.append((j, m, D.norm(h[0, :m]).astype(mx.bfloat16)))
    C.chain(D, W, mx.array(ids), Dp, on_depth, a0)
    Hs = mx.concatenate([x for _, _, x in hs], 0)
    top = []
    for c0 in range(0, Hs.shape[0], 512):                         # 512-row chunks: argpartition over the vocab
        lg = C.lm_logits(Hs[c0:c0 + 512])
        i3 = mx.argpartition(-lg, 2, axis=-1)[:, :3]
        v3 = mx.take_along_axis(lg, i3, -1)
        i3 = mx.take_along_axis(i3, mx.argsort(-v3, -1), -1)
        am = mx.argmax(lg, -1)
        i3 = mx.where((i3[:, :1] == am[:, None]), i3, mx.concatenate([am[:, None], i3[:, :2]], -1))   # ties: #1 = argmax
        top.append(i3); mx.eval(top[-1])
    top = np.array(mx.concatenate(top)).astype(np.int64)
    o = 0
    for j, m, _ in hs:
        t3 = top[o:o + m]; o += m
        true = ids[a0 + j + 1:a0 + j + 1 + m].astype(np.int64)
        rows = np.arange(a0 + j, a0 + j + m)                      # lp row r scores ids[r + 1]
        setA = np.where((lp[rows, :1] - lp[rows]) <= DELTA, tk[rows], -1)        # [m, 32], -1 = not in A
        isT = t3 == true[:, None]
        inA = (t3[:, :, None] == setA[:, None, :]).any(-1)
        inn[:m, j - 1] = (isT | inA).astype(np.int8)
        ex[:m, j - 1] = isT.astype(np.int8)
    del hs, Hs, W
    mx.clear_cache()
    return inn, ex, gen, a0


def walk(acc, r0):
    """core.kept_walk on a 0/1/-1 table: kept per check and the terminating cell (row, depth) of each check"""
    sub = acc[r0:]; kept, ends = [], []
    p = 0
    while p < sub.shape[0]:
        k = 0
        while k < min(31, sub.shape[1]) and sub[p, k] == 1:
            k += 1
        kept.append(k + 1)
        if k < sub.shape[1] and sub[p, k] == 0:
            ends.append((r0 + p, k))
        p += k + 1
    return kept, ends


R = {}
for ri, run in enumerate(RUNS):
    D = C.load_drafter(run)
    for st, nw in SETS:
        K = {f"{mode}{k}": [0, 0] for mode in ("in", "ex") for k in (1, 2, 3)}
        term = [0, 0, 0]                                           # terminating misses of the delta-1 walk, #2 IN, #2|#3 IN
        for wi, f in enumerate(files(st, nw)):
            inn, ex, gen, a0 = table3(D, f)
            if ri == 0 and wi == 0:
                ref = C.chain_table(D, f, 31)[0]
                mx.clear_cache()
                ok = ref.shape == ex.shape[:2] and bool((ref == ex[:, :, 0]).all())
                print(f"SELF-CHECK top-1 table == core.chain_table on {os.path.basename(f)}: {'PASS' if ok else 'FAIL'}",
                      flush=True)
                if not ok:
                    sys.exit(2)
            r0 = max(0, gen - 1 - a0)
            for mode, T in (("in", inn), ("ex", ex)):
                for k in (1, 2, 3):
                    acc = np.where((T[:, :, :k] == 1).any(-1), 1, np.where(T[:, :, 0] < 0, -1, 0)).astype(np.int8)
                    kp, ends = walk(acc, r0)
                    K[f"{mode}{k}"][0] += sum(kp); K[f"{mode}{k}"][1] += len(kp)
                    if mode == "in" and k == 1:
                        for r, d in ends:
                            term[0] += 1; term[1] += int(inn[r, d, 1] == 1); term[2] += int((inn[r, d, 1:3] == 1).any())
        res = {k: v[0] / v[1] for k, v in K.items()}
        res["term_n"] = term[0]; res["term_2nd_in"] = term[1] / max(term[0], 1); res["term_2or3_in"] = term[2] / max(term[0], 1)
        res["H2"] = res["in2"] / res["in1"] - 1; res["H3"] = res["in3"] / res["in1"] - 1
        R[f"{run}|{st}"] = res
        print(f"{run} {st}: kept/32 exact top-1 {res['ex1']:.3f} / top-2 {res['ex2']:.3f} / top-3 {res['ex3']:.3f}; "
              f"delta-1 IN top-1 {res['in1']:.3f} / top-2 {res['in2']:.3f} ({100 * res['H2']:+.1f} %) / top-3 {res['in3']:.3f} "
              f"({100 * res['H3']:+.1f} %); at the delta-1 walk's {term[0]} terminating misses: #2 IN "
              f"{100 * res['term_2nd_in']:.1f} %, #2 or #3 IN {100 * res['term_2or3_in']:.1f} %", flush=True)
    del D
    mx.clear_cache()
json.dump(R, open(OUT, "w"), indent=1)
