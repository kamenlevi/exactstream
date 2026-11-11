"""SR2 (NewIdeas Sun 11-08 05:2x; CHAT150): the chat drafter's output layer as a 2-bit SCREEN + 4-bit REFINE.
The MTP chain reads the full 4-bit 248K output layer per drafted row (1.49 ms of ~2.6; TrainLab data/mtp_step_bench.txt).
MTP Head's D1 measured plain 2 / 3-bit output layers (0.84 / 1.12 ms) but they change the PICK: kept -14 / -5 %.
Here the pick stays 4-bit: a 2-bit pass (0.84 ms) ranks 128-token blocks by their max logit, the top B blocks are
re-scored exactly at 4 bits, and the draft is the argmax of the refined set. Outputs never change (chat verifies every
draft); only kept per check can move, and only when the 4-bit pick falls outside the screen.
Kill lines (pre-registered here, before any run):
  KILL if the best screen keeps < 99.0 % of out4's kept per check on IFEval or on multi (kept loss > 1 %),
  or if the measured 1-row screen + refine step saves < 0.30 ms vs the 4-bit head (then it cannot pay +2 %).
  PASS (route to MTP Head / BigMoE) if kept >= 99.5 % of out4 on both sets AND it saves >= 0.40 ms per drafted row.
  Between the lines: BETWEEN (report, no route).
Method: MTP Head's own teacher-forced chain (TrainLab sticky/d1_variants.py + d24_mtptop3.top_ranks, head b5, held-out
30 IFEval + 24 multi windows, depths 1-6, the shipped cut 0.3 / K 6), imported read-only. out4 must reproduce
TrainLab data/d1_variants.txt's out4 lines exactly (the control). Shared GPU, peak ~10 GB. Writes data/chat/sr2.txt."""
import os, sys, glob, time
sys.path.insert(0, os.path.expanduser("~/TrainLab/sticky"))
import d24_mtptop3 as D24                                   # imports d22_ttt as T (MTPLX env), mlx
T = D24.T
import numpy as np
import mlx.core as mx
import mlx.nn as nn

D24.D = 6
OUT = os.path.expanduser("~/NewIdeas/data/chat/sr2.txt")
BLK = 128
L_ = []


def say(s):
    print(s, flush=True); L_.append(s)


class A:
    run = "b5"; frac = 0.03


T.M.limit_memory()
rt, lm, head, P = T.load_head(A)
q4 = lm.lm_head
wd = mx.dequantize(q4.weight, q4.scales, q4.biases, group_size=q4.group_size, bits=q4.bits)
V, H = wd.shape
NB = (V + BLK - 1) // BLK
screens = {}
for b in (2, 3):
    w, s, bb = mx.quantize(wd, group_size=64, bits=b)
    ql = nn.QuantizedLinear(H, V, bias=False, group_size=64, bits=b)
    ql.weight, ql.scales, ql.biases = w, s, bb
    mx.eval(ql.parameters())
    screens[b] = ql
del wd
say(f"SR2: output layer V {V}, input {H}, 4-bit group {q4.group_size}; blocks of {BLK} ({NB})")

# variants: (name, screen bits or None, top blocks B)
VAR = [("out4", None, 0), ("s2b4", 2, 4), ("s2b16", 2, 16), ("s3b4", 3, 4)]
PAD = NB * BLK - V


def variant_logits(x, sb, B):
    """x: the mixer output (1, n, H). Returns (rank logits, lse) where rank logits put the refined 4-bit values on the
    candidates and every other token below the refined minimum (the deployed pick is the refined argmax), and lse is
    the normalizer with refined values substituted into the screen's logits (the deployed p for the 0.3 cut)."""
    l4 = q4(x).astype(mx.float32)[0]                             # (n, V) exact values (only read at candidates)
    if sb is None:
        return l4, mx.logsumexp(l4, axis=-1)
    ls = screens[sb](x).astype(mx.float32)[0]                    # (n, V) screen
    n = ls.shape[0]
    lsp = mx.concatenate([ls, mx.full((n, PAD), -1e30, mx.float32)], axis=-1) if PAD else ls
    bmax = lsp.reshape(n, NB, BLK).max(axis=-1)                  # (n, NB)
    top = mx.argpartition(-bmax, kth=B - 1, axis=-1)[:, :B]      # (n, B) block ids
    cand = (top[:, :, None] * BLK + mx.arange(BLK)[None, None, :]).reshape(n, B * BLK)
    cand = mx.where(cand < V, cand, V)                           # the last block's padding -> a spare column V
    m = mx.zeros((n, V + 1), mx.bool_)
    m = mx.put_along_axis(m, cand, mx.ones(cand.shape, mx.bool_), axis=-1)[:, :V]   # candidate mask
    ex = mx.where(m, l4, -1e30)
    floor = mx.where(m, l4, 1e30).min(axis=-1, keepdims=True)
    rank_lg = mx.where(m, l4, mx.minimum(ls, floor - 1e-3))
    lse = mx.logsumexp(mx.where(m, l4, ls), axis=-1)
    return rank_lg, lse


def top_ranks_v(x_, a0, sb, B, rec):
    ids = x_["ids"]; S = len(ids); res = []

    def on_depth(j, h):
        n = min(h.shape[1], S - 1 - j - a0)
        xm = head.hyper_connection_mixer(h[:, :n])
        lg, lse = variant_logits(xm, sb, B)
        true = mx.array(ids[a0 + j + 1:a0 + j + 1 + n].astype(np.int32))
        tl = mx.take_along_axis(lg, true[:, None], axis=-1)
        rank = mx.minimum((lg > tl).sum(axis=-1), 3)
        p1 = mx.exp(lg.max(axis=-1) - lse)
        if sb is not None:                                        # screen recall of out4's own pick
            l4 = q4(xm).astype(mx.float32)[0]
            same = (lg.argmax(axis=-1) == l4.argmax(axis=-1))
            mx.eval(same); rec[0] += int(np.array(same).sum()); rec[1] += n
        mx.eval(rank, p1)
        res.append((np.array(p1, np.float32), np.array(rank, np.int8)))
    T.chain(lm, head, P, x_["W"], mx.array(ids)[None], D24.D, on_depth, a0)
    return res


kept_cut = None
src = open(os.path.expanduser("~/TrainLab/sticky/d1_variants.py")).read()
ns = {"np": np}
exec(src[src.index("def kept_cut"):src.index("say(\"D1 variants")], ns)   # MTP Head's own cut model, verbatim
kept_cut = ns["kept_cut"]

t0 = time.time()
for st in ("ifeval", "multi"):
    files = sorted(glob.glob(os.path.join(T.TD, "d16w", D24.PAT[st] + ".npz")))
    for nm, sb, B in VAR:
        hit = np.zeros(6); tot = np.zeros(6); TT = CC = RR = 0; rec = [0, 0]
        for f in files:
            key = os.path.basename(f)[:-4]
            x = T.window(key); gen = T.gen_of(key); a0 = max(0, gen - 2)
            r = top_ranks_v(x, a0, sb, B, rec)
            n = len(x["ids"]) - 1 - a0
            p1 = np.full((n, 6), np.nan, np.float32); rk = np.full((n, 6), -1, np.int8)
            for j, (p, k) in enumerate(r[:6]):
                p1[:len(p), j] = p; rk[:len(k), j] = k
                v = k >= 0; hit[j] += (k[v] == 0).sum(); tot[j] += v.sum()
            a, b_, c = kept_cut(p1, rk, gen, len(x["ids"]))
            TT += a; CC += b_; RR += c
        say(f"{st:6s} {nm:6s}: depth hits " + " ".join(f"{h / t:.3f}" for h, t in zip(hit, tot)) +
            f" | kept/check (cut 0.3, K6) {TT / CC:.3f}, rows {RR / CC:.2f}" +
            (f" | pick == out4 {rec[0] / max(rec[1], 1):.5f} ({rec[1] - rec[0]} of {rec[1]} differ)" if sb else "") +
            f" | {time.time() - t0:.0f} s")

# the cost side at 1 row, as the chain runs it: 4-bit head + max / lse vs screen + block top-B + gathered 4-bit refine
xr = mx.random.normal((1, 1, H)).astype(mx.bfloat16); mx.eval(xr)


def full4(x):
    lg = q4(x).astype(mx.float32)[0]
    return lg.argmax(axis=-1), mx.logsumexp(lg, axis=-1)


def sr(x, sb, B):
    ls = screens[sb](x).astype(mx.float32)[0]
    lsp = mx.concatenate([ls, mx.full((1, PAD), -1e30, mx.float32)], axis=-1) if PAD else ls
    top = mx.argpartition(-lsp.reshape(1, NB, BLK).max(axis=-1), kth=B - 1, axis=-1)[:, :B]
    cand = mx.minimum((top[:, :, None] * BLK + mx.arange(BLK)[None, None, :]).reshape(-1), V - 1)
    w = mx.dequantize(q4.weight[cand], q4.scales[cand], q4.biases[cand], group_size=q4.group_size, bits=q4.bits)
    ex = (x[0] @ w.T).astype(mx.float32)                          # (1, B*BLK)
    i = ex.argmax(axis=-1)
    return cand[i], mx.logsumexp(ls, axis=-1)


def bench(fn, reps=60):
    for _ in range(8):
        mx.eval(*fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); mx.eval(*fn()); ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[int(len(ts) * 0.9)]


m4 = bench(lambda: full4(xr))
say(f"cost at 1 row (median / p90 ms, own eval per call): out4 head + argmax + lse {m4[0]:.3f} / {m4[1]:.3f}")
for nm, sb, B in VAR[1:]:
    mt = bench(lambda: sr(xr, sb, B))
    say(f"cost at 1 row: {nm} screen + top-{B} blocks + 4-bit refine {mt[0]:.3f} / {mt[1]:.3f}  (saves {m4[0] - mt[0]:+.3f})")
say(f"peak MLX memory {mx.get_peak_memory() / 2 ** 30:.1f} GB")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(L_) + "\n")
