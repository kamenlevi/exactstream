#!/usr/bin/env python3
"""WEEK98 day-1 CEILING test (Sat 14 Nov 2026; docs/PREREG_DEEPDRAFT_CEILING_2026-11-14.md has the lines and actions).
Is the 98 %-per-guess goal at depth 30 blocked by the 125B's own near-ties, or are t30b's misses learnable?

margins: the 125B's top-1 vs top-2 margin m = lp[0] - lp[1] (nats) on every answer row, in bins, from the captured top-32
         log-probs: MBPP 100 (TrainLab dlog/d1102mbpp, the eval's set) and t30b's 113 held-out n1112 files. Also how often
         the answer token is the 125B's argmax (the capture is greedy, so it should be all rows). numpy only.
misses:  one eval pass of a run (default t30b) over MBPP 100, as dd_ttt.py's eval (chain to depth 31, anchors gen - 2),
         keeping every cell: right or not, the draft token, the 125B's margin at the target row, and the draft's rank in
         the 125B's top-32 there. Reports, at band 0 and band 23-30:
         - accuracy (a harness check against ttt30b/eval.json's MBPP bands);
         - the miss rate by margin bin, and the share of misses on near-ties (m < 0.1 / 0.25 / 0.5) and on confident rows
           (m >= 1 nat), and on the 125B's runner-up;
         - the tie-limited ceiling: 1 - (near-tie share) x (band 0's miss rate on near-ties): every confident row right,
           ties resolved as well as the freshest guess resolves them;
         - close-call acceptance: accuracy if a draft within delta nats of the 125B's top-1 also counts (0.1, 0.25, 0.5).
usage: dd_ceiling.py margins | misses [--run t30b]
"""
import argparse, glob, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
TD = os.path.expanduser("~/TrainLab/data")
CAP = os.path.expanduser("~/FableGod/deepdraft/capture/n1112")
OUT = os.path.expanduser("~/BigMoE/data/deepdraft/ceiling")
MBPP = sorted(glob.glob(os.path.join(TD, "dlog/d1102mbpp/dlog__mbpp*.npz")))[:100]
EDGES = [0, 0.1, 0.25, 0.5, 1, 2, 4, np.inf]
LAB = ["<0.1", "0.1-0.25", "0.25-0.5", "0.5-1", "1-2", "2-4", ">=4"]


def answer_rows(z):
    """rows t whose next token ids[t + 1] is an answer token and whose top-32 exists (rows >= gen - 1)"""
    ids, asst = np.asarray(z["ids"]), np.asarray(z["asst"])
    gen, n = int(z["gen"]), len(ids) - 1
    t = np.arange(max(0, gen - 1), n)
    return t[asst[t + 1] > 0]


def margins_of(fs):
    m, top_is_next, rows = [], 0, 0
    for f in fs:
        z = np.load(f)
        t = answer_rows(z)
        lp = np.asarray(z["lp"], np.float32)[t]; tk = np.asarray(z["tk"])[t]
        m.append(lp[:, 0] - lp[:, 1])
        top_is_next += int((tk[:, 0] == np.asarray(z["ids"])[t + 1]).sum()); rows += len(t)
    m = np.concatenate(m)
    h, _ = np.histogram(m, EDGES)
    return {"rows": rows, "files": len(fs), "answer_token_is_argmax": top_is_next / max(rows, 1),
            "bins": {l: float(c / max(len(m), 1)) for l, c in zip(LAB, h)},
            "share_lt": {str(d): float((m < d).mean()) for d in (0.05, 0.1, 0.25, 0.5, 1.0)},
            "median": float(np.median(m))}


def cmd_margins(a):
    s = json.load(open(os.path.expanduser("~/BigMoE/data/deepdraft/ttt30b/split.json")))
    R = {"mbpp100": margins_of(MBPP), "n1112_held_out": margins_of([os.path.join(CAP, n) for n in s["held_out"]])}
    os.makedirs(OUT, exist_ok=True)
    json.dump(R, open(os.path.join(OUT, "margins.json"), "w"), indent=1)
    for k, r in R.items():
        print(f"[ceil] {k}: {r['rows']} answer rows in {r['files']} files; answer token = the 125B's argmax on "
              f"{100 * r['answer_token_is_argmax']:.2f} %; median margin {r['median']:.2f} nats")
        print("   margin bins (nats): " + "  ".join(f"{l} {100 * v:.2f} %" for l, v in r["bins"].items()))
        print("   near-tie shares: " + "  ".join(f"m<{d} {100 * v:.2f} %" for d, v in r["share_lt"].items()))


def cmd_misses(a):
    import mlx.core as mx
    import dd_ttt as DD
    T, M = DD.T, DD.M
    t0 = time.time()
    rt_, lm, head = M.load(a.frac)
    P = M.Trainable(head)
    w = os.path.join(T.CK, "d22_b5.safetensors") if a.run == "b5" else os.path.join(DD.CKD, f"dd_{a.run}.safetensors")
    P.load_weights(w); mx.eval(P.parameters())
    cells = {k: [] for k in ("depth", "right", "m", "rank", "dlp")}   # depth = the band index (0 = the first guess)
    for f in MBPP:
        while os.path.exists(T.PAUSE):
            time.sleep(1)
        z = np.load(f)
        ids = np.asarray(z["ids"], np.int32)
        x = {"W": mx.array(z["wd"]).astype(mx.bfloat16)[None], "ids": ids}
        lp, tk = np.asarray(z["lp"], np.float32), np.asarray(z["tk"])
        gen = int(z["gen"]); a0 = max(0, gen - 2)
        hits = T.chain_hits(lm, head, P, x, 31, a0)
        for j, (right, am) in enumerate(hits):        # band index j: anchor s = a0 + i drafts ids[s + j + 2] from row s + j + 1
            n = len(right)
            rows = a0 + np.arange(n) + j + 1
            ok = rows < len(lp)
            rows, right, am = rows[ok], np.asarray(right)[ok], np.asarray(am)[ok]
            valid = rows >= gen - 1                       # the eval's valid cells: the target row has the 125B's top-32
            rows, right, am = rows[valid], right[valid], am[valid]
            m = lp[rows, 0] - lp[rows, 1]
            hit = tk[rows] == am[:, None]
            rank = np.where(hit.any(1), hit.argmax(1), 99)
            dlp = np.where(rank < 32, lp[rows, np.minimum(rank, 31)] - lp[rows, 0], -np.inf)
            for k, v in (("depth", np.full(len(rows), j)), ("right", right.astype(bool)), ("m", m), ("rank", rank), ("dlp", dlp)):
                cells[k].append(v)
        mx.clear_cache()
    C = {k: np.concatenate(v) for k, v in cells.items()}
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, f"cells_{a.run}_mbpp100.npz"), **C)
    R = {"run": a.run, "set": "mbpp100", "wall_s": time.time() - t0, "mlx_peak_gb": mx.get_peak_memory() / 1e9}
    for name, lo, hi in (("band0", 0, 0), ("band23_30", 23, 30), ("band7_10", 7, 10), ("band15_22", 15, 22)):
        sel = (C["depth"] >= lo) & (C["depth"] <= hi)
        r, m, rank, dlp = C["right"][sel], C["m"][sel], C["rank"][sel], C["dlp"][sel]
        miss = ~r
        b = np.digitize(m, EDGES[1:-1])
        R[name] = {"cells": int(sel.sum()), "acc": float(r.mean()),
                   "miss_rate_by_margin": {LAB[i]: [float(miss[b == i].mean()) if (b == i).any() else None, int((b == i).sum())] for i in range(len(LAB))},
                   "miss_share_on": {f"m<{d}": float((miss & (m < d)).sum() / max(miss.sum(), 1)) for d in (0.1, 0.25, 0.5)},
                   "miss_share_confident_m>=1": float((miss & (m >= 1)).sum() / max(miss.sum(), 1)),
                   "miss_share_runner_up": float((miss & (rank == 1)).sum() / max(miss.sum(), 1)),
                   "miss_share_outside_top32": float((miss & (rank == 99)).sum() / max(miss.sum(), 1)),
                   "row_share_m<0.25": float((m < 0.25).mean()),
                   "close_call_acc": {str(d): float((r | (dlp >= -d)).mean()) for d in (0.1, 0.25, 0.5)}}
    b0 = R["band0"]
    for name in ("band0", "band23_30"):
        x = R[name]
        mr0 = b0["miss_rate_by_margin"]
        tie_miss0 = (sum((v[0] or 0) * v[1] for k, v in mr0.items() if k in ("<0.1", "0.1-0.25"))
                     / max(sum(v[1] for k, v in mr0.items() if k in ("<0.1", "0.1-0.25")), 1))
        x["tie_limited_ceiling"] = 1 - x["row_share_m<0.25"] * tie_miss0
    json.dump(R, open(os.path.join(OUT, f"misses_{a.run}_mbpp100.json"), "w"), indent=1)
    for name in ("band0", "band7_10", "band15_22", "band23_30"):
        x = R[name]
        print(f"[ceil] {a.run} MBPP {name}: acc {100 * x['acc']:.2f} % on {x['cells']} cells; misses on m<0.1 / <0.25 / <0.5: "
              + " / ".join(f"{100 * v:.1f}" for v in x["miss_share_on"].values())
              + f" %; on confident rows (m >= 1) {100 * x['miss_share_confident_m>=1']:.1f} %; the runner-up {100 * x['miss_share_runner_up']:.1f} %"
              + f"; outside the top-32 {100 * x['miss_share_outside_top32']:.1f} %")
        print("   miss rate by margin: " + "  ".join(f"{k} {100 * v[0]:.1f}% (n {v[1]})" if v[0] is not None else f"{k} - (n 0)"
                                           for k, v in x["miss_rate_by_margin"].items()))
        print("   close-call acceptance (draft within delta nats of the top-1): "
              + "  ".join(f"delta {d}: {100 * v:.2f} %" for d, v in x["close_call_acc"].items())
              + (f"; tie-limited ceiling {100 * x['tie_limited_ceiling']:.2f} %" if "tie_limited_ceiling" in x else ""))
    print(f"[ceil] done in {R['wall_s']:.0f}s; peak {R['mlx_peak_gb']:.1f} GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["margins", "misses"])
    ap.add_argument("--run", default="t30b"); ap.add_argument("--frac", type=float, default=0.03)
    a = ap.parse_args()
    {"margins": cmd_margins, "misses": cmd_misses}[a.cmd](a)


if __name__ == "__main__":
    main()
