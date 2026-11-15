#!/usr/bin/env python3
"""t30b's own top-K (tokens + probabilities) at every drafted cell (Sun 15 Nov 2026 10:5x; the orchestrator, for MTP Head's
EAGLE-2-style tree pricing on the 125B: Kamen 15 Nov, the drafter's 98 % stays EXACT, the exact route left is TREES).

Same files, anchors, chain and cell order as `dd_ceiling.py misses` (cells_<run>_<set>.npz), so the two files align row
for row (checked at the end). The chain is TEACHER-FORCED: depth >= 2 embeds the TRUE token ids[s + j] (the hidden state is
the chain's own). So a cell's distribution is the head's at the on-true-path node of any tree drafted from that anchor, and
it equals the free-running decode's up to the check's first miss.

Per cell (anchor s = row - depth - 1; band depth j = 0..30 drafts ids[s + j + 2] from row s + j + 1; y = ids[row + 1], the
125B's greedy token):
  top_ids [N, K], top_p [N, K]: the head's top-K tokens and softmax probabilities (f32 logits over the full lm_head),
                                descending;
  p_y:    the head's probability of y;     y_rank: y's rank in the head's order (0 = the head's argmax, i.e. right);
  right:  argmax == y (identical to the cells' right: checked);
  walk_start: the cell's anchor is a check start of t30b's chain walk at 32-row checks (checkstart_walk.py's walk: from
              anchor gen - 1, keep the leading right drafts (<= 31) + the bonus, next anchor s + k + 1).
usage: dd_topk.py --set heplus24|mbppbank [--run t30b] [--k 8] -> data/deepdraft/ceiling/topk_<run>_<set>.npz
       dd_topk.py --selftest   (top-K / p_y / y_rank against numpy on random logits; tiny, no model)"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.expanduser("~/BigMoE/data/deepdraft/ceiling")


def topk_rows(mx, lg, y, K, chunk=256):
    """lg [n, V] f32 (mx), y [n] int32 (np) -> top_ids [n, K] int32, top_p [n, K] f32 (descending), p_y [n] f32,
    y_rank [n] int32, am [n] int32 (mx.argmax, as chain_hits). Rows in chunks to bound the [chunk, V] temporaries."""
    n = lg.shape[0]
    ti_, tp_, py_, yr_, am_ = [], [], [], [], []
    for c in range(0, n, chunk):
        l = lg[c:c + chunk]
        yy = mx.array(y[c:c + chunk])[:, None]
        am = mx.argmax(l, -1)
        lse = mx.logsumexp(l, axis=-1)
        ly = mx.take_along_axis(l, yy, axis=-1)[:, 0]
        yr = (l > ly[:, None]).sum(-1)
        ti = mx.argpartition(-l, kth=K - 1, axis=-1)[:, :K]
        tv = mx.take_along_axis(l, ti, axis=-1)
        mx.eval(am, lse, ly, yr, ti, tv)
        ti, tv = np.array(ti).astype(np.int32), np.array(tv, np.float32)
        o = np.argsort(-tv, axis=1, kind="stable")
        ti, tv = np.take_along_axis(ti, o, 1), np.take_along_axis(tv, o, 1)
        lse = np.array(lse, np.float32)
        ti_.append(ti); tp_.append(np.exp(tv - lse[:, None])); py_.append(np.exp(np.array(ly, np.float32) - lse))
        yr_.append(np.array(yr).astype(np.int32)); am_.append(np.array(am).astype(np.int32))
    return (np.concatenate(ti_), np.concatenate(tp_).astype(np.float32), np.concatenate(py_).astype(np.float32),
            np.concatenate(yr_), np.concatenate(am_))


def chain_topk(mx, T, lm, head, P, x, D, a0, K):
    """T.chain_hits plus each depth's top-K / p_y / y_rank; per depth j (1-based in T.chain): (right, am, ti, tp, py, yr)
    over anchors s = a0 + i, which predict ids[a0 + i + j + 1]"""
    ids = x["ids"]; S = len(ids); out = []

    def on_depth(j, h):
        n = min(h.shape[1], S - 1 - j - a0)
        lg = T.logits(lm, head, h[:, :n])[0]
        y = ids[a0 + j + 1:a0 + j + 1 + n]
        ti, tp, py, yr, am = topk_rows(mx, lg, y, K)
        out.append((am == y, am, ti, tp, py, yr))
    T.chain(lm, head, P, x["W"], mx.array(ids)[None], D, on_depth, a0)
    return out


def walk_starts(right, depth, row, gen, cap=31):
    """checkstart_walk.py's walk over one file's cells (from anchor gen - 1) -> the set of check-start anchors"""
    cell = {(int(r) - int(d) - 1, int(d)): bool(rt) for d, r, rt in zip(depth, row, right)}
    s, out = gen - 1, set()
    while (s, 0) in cell:
        out.add(s)
        k = 0
        while k < cap and cell.get((s, k), False):
            k += 1
        s = s + k + 1
    return out


def selftest():
    import mlx.core as mx
    rng = np.random.default_rng(0)
    for n, V, K in ((7, 50, 3), (300, 1000, 8), (5, 9, 8)):
        lg = rng.normal(size=(n, V)).astype(np.float32) * 3
        y = rng.integers(0, V, size=n).astype(np.int32)
        y[0] = int(lg[0].argmax())                         # one right row
        ti, tp, py, yr, am = topk_rows(mx, mx.array(lg), y, K, chunk=64)
        o = np.argsort(-lg, axis=1, kind="stable")[:, :K]
        lse = np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1)) + lg.max(1)
        assert (ti == o).all(), "top ids"
        assert np.allclose(tp, np.exp(np.take_along_axis(lg, o, 1) - lse[:, None]), rtol=1e-4, atol=1e-6), "top p"
        assert np.allclose(py, np.exp(lg[np.arange(n), y] - lse), rtol=1e-4, atol=1e-6), "p_y"
        assert (yr == (lg > lg[np.arange(n), y][:, None]).sum(1)).all(), "y_rank"
        assert (am == lg.argmax(1)).all() and yr[0] == 0, "argmax"
    # the walk: anchors 10.. (gen - 2 = 10, walk from 11); file of 3 checks
    d, r, rt = [], [], []
    for s in range(10, 60):
        for j in range(31):
            if s + j + 1 < 70:
                d.append(j); r.append(s + j + 1); rt.append(not (s == 11 and j == 2) and not (s == 14 and j == 0))
    st = walk_starts(np.array(rt), np.array(d), np.array(r), 12)
    assert min(st) == 11 and 14 in st and 15 in st and 46 not in st and 47 in st, sorted(st)
    print("SELFTEST OK: top-K ids / probs / p_y / y_rank / argmax match numpy on 3 shapes; the walk matches by hand")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["heplus24", "mbppbank"])
    ap.add_argument("--run", default="t30b"); ap.add_argument("--frac", type=float, default=0.03)
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import mlx.core as mx
    import dd_ceiling as CE
    import dd_ttt as DD
    T, M = DD.T, DD.M
    t0 = time.time()
    rt_, lm, head = M.load(a.frac)
    P = M.Trainable(head)
    P.load_weights(os.path.join(DD.CKD, f"dd_{a.run}.safetensors")); mx.eval(P.parameters())
    FS = {"heplus24": CE.HEP, "mbppbank": CE.MBB}[a.set]
    assert len(FS) == {"heplus24": 24, "mbppbank": 100}[a.set], (a.set, len(FS))
    C = {k: [] for k in ("depth", "fi", "row", "y", "right", "top_ids", "top_p", "p_y", "y_rank", "walk_start")}
    tie = 0
    for fi, f in enumerate(FS):
        while os.path.exists(T.PAUSE):
            time.sleep(1)
        z = np.load(f)
        ids = np.asarray(z["ids"], np.int32)
        x = {"W": mx.array(z["wd"]).astype(mx.bfloat16)[None], "ids": ids}
        lp = np.asarray(z["lp"], np.float32)
        gen = int(z["gen"]); a0 = max(0, gen - 2)
        F = {k: [] for k in C if k != "walk_start"}
        for j, (right, am, ti, tp, py, yr) in enumerate(chain_topk(mx, T, lm, head, P, x, 31, a0, a.k)):
            n = len(right)
            rows = a0 + np.arange(n) + j + 1                     # dd_ceiling's cells: the same rows and filters
            sel = (rows < len(lp)) & (rows >= gen - 1)
            tie += int((ti[sel, 0] != am[sel]).sum())
            for k_, v in (("depth", np.full(n, j)), ("fi", np.full(n, fi)), ("row", rows), ("y", ids[rows + 1]),
                          ("right", right), ("top_ids", ti), ("top_p", tp), ("p_y", py), ("y_rank", yr)):
                F[k_].append(np.asarray(v)[sel])
        F = {k_: np.concatenate(v) for k_, v in F.items()}
        st = walk_starts(F["right"], F["depth"], F["row"], gen)
        F["walk_start"] = np.array([int(r) - int(d) - 1 in st for d, r in zip(F["depth"], F["row"])], bool)
        for k_ in C:
            C[k_].append(F[k_])
        mx.clear_cache()
    C = {k_: np.concatenate(v) for k_, v in C.items()}
    files = np.array([os.path.basename(f) for f in FS])
    out = os.path.join(OUT, f"topk_{a.run}_{a.set}.npz")
    np.savez_compressed(out, files=files, k=a.k, **C)
    # alignment with dd_ceiling's cells (same order, same right) + the walk against checkstart_walk.py's counts
    R = {"run": a.run, "set": a.set, "cells": int(len(C["row"])), "k": a.k, "top1_neq_argmax_ties": tie,
         "wall_s": round(time.time() - t0, 1), "mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 2)}
    cf = os.path.join(OUT, f"cells_{a.run}_{a.set}.npz")
    if os.path.exists(cf):
        c = np.load(cf)
        same = len(c["row"]) == len(C["row"]) and all((c[k_] == C[k_]).all() for k_ in ("depth", "fi", "row", "right"))
        R["aligned_with_cells"] = bool(same)
    assert (C["y"][:200] == np.array([np.load(FS[int(fi)])["ids"][int(r) + 1] for fi, r in zip(C["fi"][:200], C["row"][:200])])).all(), "y"
    ws = C["walk_start"] & (C["depth"] == 0)
    R["walk_check_starts"] = int(ws.sum())
    cs = os.path.join(OUT, f"checkstart_{a.set}.json")
    if os.path.exists(cs):
        J = json.load(open(cs))
        R["checkstart_walk_starts"] = int(sum(J[k_]["n"] for k_ in ("first", "after_miss", "after_full") if k_ in J))
    yr0 = C["y_rank"][ws]
    R["at_walk_starts_depth0"] = {f"y_in_top{k_}": round(float((yr0 < k_).mean()), 4) for k_ in (1, 2, 3, 4, 8)}
    R["all_cells"] = {f"y_in_top{k_}": round(float((C["y_rank"] < k_).mean()), 4) for k_ in (1, 2, 3, 8)}
    R["p1_median_right_vs_wrong_depth0_walk"] = [round(float(np.median(C["top_p"][ws & C["right"], 0])), 4),
                                                round(float(np.median(C["top_p"][ws & ~C["right"], 0])), 4)]
    json.dump(R, open(os.path.join(OUT, f"topk_{a.run}_{a.set}.json"), "w"), indent=1)
    print("[topk]", json.dumps(R), flush=True)


if __name__ == "__main__":
    main()
