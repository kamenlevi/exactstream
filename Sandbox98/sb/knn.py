"""kNN STATE MEMORY (results/LEVERS_A.md, lever 3; MoonShots' lever 3), eval-only, s30 unchanged.
Index: every row of the Magicoder captures (data/mc + data/mc2, capture order), keyed per n in (16, 12, 8, 6, 4) by a
64-bit rolling hash of the n tokens ending at the row; value = that row's true state W (post-norm, fp16, memory-mapped).
At depth j for anchor s the n-gram ends at row p = s + j - 1; the longest n >= n_min with a hit gives the stand-in
(arm a: the most recent neighbour; b: the mean of up to 8 most recent; c: the most recent, n >= 12); else the carried one.
-> results/knn.json (bands per arm, paired per-window counts, hit rates per band, hit-cell accuracy vs the chain, cost)
usage: knn.py [--n 50]"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
from sb import core as C
from sb.compare import paired

NS = (16, 12, 8, 6, 4)
PM = np.uint64(1099511628211)
STORE = os.path.join(C.DATA, "knn")


def ngram_hash(ids, n):
    """uint64 hash of ids[p-n+1..p] for every p >= n-1 (Horner, oldest first); [S - n + 1]"""
    x = np.asarray(ids, np.uint64) + np.uint64(1)
    S = len(x)
    h = np.zeros(S - n + 1, np.uint64)
    with np.errstate(over="ignore"):
        for i in range(n):
            h = h * PM + x[i:S - n + 1 + i]
    return h


def build():
    s = json.load(open(os.path.join(C.DATA, "split.json")))
    fs = [os.path.join(C.DATA, "mc", k + ".npz") for k in s["base"] + s["pool"] + s["val"]] + \
         [os.path.join(C.DATA, "mc2", k + ".npz") for k in s.get("mc2", [])]
    os.makedirs(STORE, exist_ok=True)
    rows = sum(len(np.load(f)["ids"]) for f in fs)
    Wm = np.lib.format.open_memmap(os.path.join(STORE, "W.npy"), mode="w+", dtype=np.float16, shape=(rows, 2048))
    keys = {n: [] for n in NS}; vals = {n: [] for n in NS}
    o = 0
    for f in fs:
        z = np.load(f)
        ids = np.asarray(z["ids"], np.int32); S = len(ids)
        Wm[o:o + S] = np.array(C.load_W(z)[0].astype(mx.float16))
        for n in NS:
            if S >= n:
                keys[n].append(ngram_hash(ids, n)); vals[n].append(np.arange(o + n - 1, o + S, dtype=np.int64))
        o += S
    Wm.flush()
    idx = {}
    for n in NS:
        if not keys[n]:
            idx[n] = (np.zeros(0, np.uint64), np.zeros(0, np.int64)); continue
        k = np.concatenate(keys[n]); v = np.concatenate(vals[n])
        order = np.argsort(k, kind="stable")             # equal keys keep capture order: the last is the most recent
        idx[n] = (k[order], v[order])
    print(f"[knn] index: {len(fs)} answers, {rows} rows (store {rows * 4096 / 1e9:.2f} GB fp16)", flush=True)
    return np.load(os.path.join(STORE, "W.npy"), mmap_mode="r"), idx


def tables(ids, Wm, idx, n_min, mean8):
    """per row p: the retrieved state [S, d] (fp32), the hit mask [S], the matched n [S]"""
    S = len(ids)
    best_n = np.zeros(S, np.int32); src = [None] * S
    for n in NS:                                         # longest first
        if n < n_min or S < n:
            continue
        q = ngram_hash(ids, n)
        H, V = idx[n]
        lo = np.searchsorted(H, q, "left"); hi = np.searchsorted(H, q, "right")
        for i in np.flatnonzero(hi > lo):
            p = i + n - 1
            if best_n[p] == 0:
                best_n[p] = n
                src[p] = V[max(lo[i], hi[i] - 8):hi[i]] if mean8 else V[hi[i] - 1:hi[i]]
    R = np.zeros((S, 2048), np.float32)
    for p in np.flatnonzero(best_n):
        R[p] = np.asarray(Wm[np.sort(src[p])], np.float32).mean(0)
    return R, best_n > 0, best_n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    C.limit_memory(5); C.load_target(); C.drop_trunk()
    t0 = time.time()
    Wm, idx = build()
    print(f"[knn] built in {time.time() - t0:.0f}s", flush=True)
    D = C.load_drafter("s30")
    arms = {"a_nn1": (6, False), "b_mean8": (6, True), "c_nn1_n12": (12, False)}
    per = {m: {} for m in ["chain"] + list(arms)}
    hitc = {m: {b: [0, 0] for b in ("1-2", "3-6", "7-10", "11-14", "15-22", "23-30")} for m in arms}
    hitacc = {m: [0, 0, 0] for m in arms}                # [arm right on hit cells, chain right on the same cells, cells]
    tl, nl = 0.0, 0
    for f in C.files("mbpp")[:a.n]:
        C.wait_pause()
        z = np.load(f); ids = np.asarray(z["ids"], np.int32); gen = int(z["gen"]); a0 = max(0, gen - 2)
        key = os.path.basename(f)[:-4]
        rt0 = C.chain_table(D, f)[0]
        rts = {"chain": rt0}
        for m, (nmin, mean8) in arms.items():
            t = time.perf_counter(); R, hit, _ = tables(ids, Wm, idx, nmin, mean8); tl += time.perf_counter() - t; nl += len(ids)
            Rm, Hm = mx.array(R), mx.array(hit)

            def hook(j, h, anc, Rm=Rm, Hm=Hm):
                p = anc + j - 1
                return mx.where(Hm[p][:, None], Rm[p], h[0].astype(mx.float32))[None]
            rt = C.chain_table(D, f, fn=hook)[0]
            rts[m] = rt
            n_a, Dp = rt.shape
            P = a0 + np.arange(n_a)[:, None] + np.arange(Dp)[None, :]          # row p of cell (anchor, column j - 1)
            Hc = hit[np.minimum(P, len(ids) - 1)] & (np.arange(Dp)[None, :] >= 1) & (rt >= 0)
            for b, (lo, hi_) in zip(hitc[m], ((1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30))):
                v = rt[:, lo:hi_ + 1] >= 0
                hitc[m][b][0] += int(Hc[:, lo:hi_ + 1].sum()); hitc[m][b][1] += int(v.sum())
            hitacc[m][0] += int((rt[Hc] == 1).sum()); hitacc[m][1] += int((rt0[Hc] == 1).sum()); hitacc[m][2] += int(Hc.sum())
        for m, rt in rts.items():
            per[m][key] = C.summarize([(key, rt, gen, a0)])[1][key]
        mx.clear_cache()
    rng = np.random.default_rng(1114)
    out = {"bands": {}, "vs_chain": {}, "hit_rate": {}, "hit_cells": {}, "lookup_us_per_row": 1e6 * tl / max(nl, 1)}
    for m in per:
        tot = {b: [sum(per[m][k][b][0] for k in per[m]), sum(per[m][k][b][1] for k in per[m])] for b in per[m][next(iter(per[m]))] if b != "kept"}
        out["bands"][m] = {b: v[0] / v[1] for b, v in tot.items() if v[1]}
        out["bands"][m]["kept_per_32"] = sum(per[m][k]["kept"][0] for k in per[m]) / sum(per[m][k]["kept"][1] for k in per[m])
    for m in arms:
        out["vs_chain"][m] = {b: paired(per["chain"], per[m], b, rng) for b in ("23-30", "7-14", "0-0", "kept")}
        out["hit_rate"][m] = {b: v[0] / v[1] for b, v in hitc[m].items() if v[1]}
        r, c, n = hitacc[m]
        out["hit_cells"][m] = {"cells": n, "arm": r / max(n, 1), "chain": c / max(n, 1)}
        d = out["vs_chain"][m]["23-30"]
        print(f"[knn] {m}: 23-30 {100 * out['bands']['chain']['23-30']:.2f} -> {100 * out['bands'][m]['23-30']:.2f} "
              f"({100 * d['diff']:+.2f}, CI {100 * d['ci95'][0]:+.2f}..{100 * d['ci95'][1]:+.2f}, sign p {d['p_sign']:.3f}); "
              f"hit rate 23-30 {100 * out['hit_rate'][m]['23-30']:.1f} %; hit cells {n}: arm {100 * r / max(n, 1):.2f} vs chain "
              f"{100 * c / max(n, 1):.2f}", flush=True)
    print(f"[knn] lookup {out['lookup_us_per_row']:.1f} us per row (CPU, the table for every row of a window)", flush=True)
    json.dump(out, open(os.path.join(C.RES, "knn.json"), "w"), indent=1)
    print("[knn] DONE", flush=True)


if __name__ == "__main__":
    main()
