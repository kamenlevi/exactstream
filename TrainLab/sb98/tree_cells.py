"""WEEK98 S9 TREE-DYN, s30 cells (MTP Head; REPORT 'WEEK98 S9 TREE-DYN'): per window of MBPP-50 / HE-30 (evalx's first
50 / 30, sorted), the teacher-forced chain table as core.chain_table builds it (read-only use of core.chain / load_W /
lm_logits), with per cell [n_anchors, 31]:
  right  int8   1 / 0 / -1 (no target)            (= core.chain_table's rt; self-checked on each set's first window)
  rank3  int8   the true token's slot in the head's top-3 (0, 1, 2), 3 = below the top-3, -1 = no target
  hlp    f16    [.., 3] the head's log-probs of its #1..#3 (confidence: p1, log p1/p2)
  inn    int8   [.., 3] #1..#3 IN under delta 1 (the true token, or in the window's stored top-32 within 1 nat of
                its top-1), -1 = no target
  dlp    f16    the target's log-prob gap of #1 (-inf outside the stored top-32)
-> OUT.npz with keys f'{i}_{field}' per window i plus 'gen', 'a0', 'files'. Usage: cd ~/Sandbox98 && PY tree_cells.py
   RUN SET N OUT.npz"""
import os, sys, glob
sys.path.insert(0, os.path.expanduser("~/Sandbox98"))
import numpy as np
import mlx.core as mx
from sb import core as C

C.limit_memory(4.5)                      # Sandbox98's caps (06:06 incident), plus clear_cache per window
RUN, SET, N, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
DELTA = 1.0


def files(d, n):
    return sorted(f for f in glob.glob(os.path.join(C.ROOT, "data", d, "*.npz")) if not f.endswith(".taps.npz"))[:n]


def cells(D, f, Dp=31):
    z = np.load(f)
    ids = np.asarray(z["ids"], np.int32); gen = int(z["gen"])
    tk = np.asarray(z["tk"], np.int64); lp = np.asarray(z["lp"], np.float32)
    W = C.load_W(z)
    a0 = max(0, gen - 2); S = len(ids); n = S - 1 - a0
    right = np.full((n, Dp), -1, np.int8); rank3 = np.full((n, Dp), -1, np.int8)
    hlp = np.full((n, Dp, 3), np.nan, np.float16); inn = np.full((n, Dp, 3), -1, np.int8)
    dlp = np.full((n, Dp), np.nan, np.float16)
    C.REPL.update(R=None, K=1, fn=None)
    C.CUR["f"], C.CUR["train"], C.CUR["x"] = f, False, None
    hs = []

    def on_depth(j, h, anc):
        m = min(h.shape[1], S - 1 - j - a0)
        if m > 0:
            hs.append((j, m, D.norm(h[0, :m]).astype(mx.bfloat16)))
    C.chain(D, W, mx.array(ids), Dp, on_depth, a0)
    Hs = mx.concatenate([x for _, _, x in hs], 0)
    top, val = [], []
    for c0 in range(0, Hs.shape[0], 512):                         # 512-row chunks: argpartition over the vocab
        lg = C.lm_logits(Hs[c0:c0 + 512])
        lse = mx.logsumexp(lg, -1, keepdims=True)
        i3 = mx.argpartition(-lg, 2, axis=-1)[:, :3]
        v3 = mx.take_along_axis(lg, i3, -1)
        o = mx.argsort(-v3, -1)
        i3 = mx.take_along_axis(i3, o, -1); v3 = mx.take_along_axis(v3, o, -1)
        am = mx.argmax(lg, -1)
        fix = i3[:, :1] != am[:, None]                            # ties: slot 0 = core's argmax
        i3 = mx.where(fix, mx.concatenate([am[:, None], i3[:, :2]], -1), i3)
        v3 = mx.where(fix, mx.concatenate([mx.take_along_axis(lg, am[:, None], -1), v3[:, :2]], -1), v3)
        top.append(i3); val.append(v3 - lse); mx.eval(top[-1], val[-1])
    top = np.array(mx.concatenate(top)).astype(np.int64); val = np.array(mx.concatenate(val)).astype(np.float32)
    o = 0
    for j, m, _ in hs:
        t3, v3 = top[o:o + m], val[o:o + m]; o += m
        true = ids[a0 + j + 1:a0 + j + 1 + m].astype(np.int64)
        rows = np.arange(a0 + j, a0 + j + m)                      # lp row r scores ids[r + 1]
        isT = t3 == true[:, None]
        right[:m, j - 1] = isT[:, 0].astype(np.int8)
        rank3[:m, j - 1] = np.where(isT.any(1), isT.argmax(1), 3).astype(np.int8)
        hlp[:m, j - 1] = v3.astype(np.float16)
        setA = np.where((lp[rows, :1] - lp[rows]) <= DELTA, tk[rows], -1)
        inn[:m, j - 1] = (isT | (t3[:, :, None] == setA[:, None, :]).any(-1)).astype(np.int8)
        eq = tk[rows] == t3[:, :1]
        dlp[:m, j - 1] = np.where(eq.any(1), lp[rows, eq.argmax(1)] - lp[rows, 0], -np.inf).astype(np.float16)
    del hs, Hs, W
    mx.clear_cache()
    return right, rank3, hlp, inn, dlp, gen, a0


D = C.load_drafter(RUN)
out = {}; gens, a0s, names = [], [], []
for i, f in enumerate(files(SET, N)):
    right, rank3, hlp, inn, dlp, gen, a0 = cells(D, f)
    if i == 0:
        ref = C.chain_table(D, f, 31)[0]
        mx.clear_cache()
        ok = ref.shape == right.shape and bool((ref == right).all())
        print(f"SELF-CHECK right == core.chain_table on {os.path.basename(f)}: {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            sys.exit(2)
    for k, v in (("right", right), ("rank3", rank3), ("hlp", hlp), ("inn", inn), ("dlp", dlp)):
        out[f"{i}_{k}"] = v
    gens.append(gen); a0s.append(a0); names.append(os.path.basename(f))
np.savez_compressed(OUT, gen=np.array(gens), a0=np.array(a0s), files=np.array(names), **out)
print(f"{RUN} {SET}: {len(names)} windows -> {OUT}", flush=True)
