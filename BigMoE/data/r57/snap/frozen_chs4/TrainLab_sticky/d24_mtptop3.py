"""D-24 MTPTOP3 (MTP Head, Sun 11-01): how often is the 125B's token in the MTP head's top-2 / top-3 where its chain misses?
The exact tree lead of R-61N (the 125B's #1 is in DFlash2's top-3 at 34 % of chat first misses), for the MTP chain, which
runs chat under the switch (D-23c). Teacher-forced chain from the true prefix at every position of the held-out answers
(d22_ttt's chain: depth j's input is the true token, so up to and including the first miss it is the real chain, and past a
miss rescued by a branch holding the true token it is that branch's real continuation). Saved per anchor and depth 1-15:
the head's top-1 p and the true token's rank (0, 1, 2, 3 = 3 or worse) -> data/d24/<set>.npz; tree / cut policies are
scored on the CPU from these (sticky/d24_read.py). Shared GPU, peak ~7-11 GB (as d22_ttt eval).
Usage: [BigMoE venv] d24_mtptop3.py [--run b5] [--sets ifeval,multi,agent,mgsm,gsm,code]"""
import os, sys, glob, time, argparse
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs4/TrainLab_sticky"))
import d22_ttt as T                                          # its imports set MTPLX's env, then mlx
import numpy as np
import mlx.core as mx

PAT = {"code": "tf_code_HumanEval_*", "gsm": "tf_gsm_*", "ifeval": "tf_ifeval_*", "multi": "tf_multi_*", "agent": "agent_*",
       "mgsm": "mgsm_*"}
OUT = os.path.join(T.TD, "d24"); os.makedirs(OUT, exist_ok=True)
D = 15


def top_ranks(lm, head, P, x, a0):
    """per depth j: (top-1 p, rank of the true token capped at 3) over anchors a0..; rows past a depth's end are absent"""
    ids = x["ids"]; S = len(ids)
    res = []

    def on_depth(j, h):
        n = min(h.shape[1], S - 1 - j - a0)
        lg = T.logits(lm, head, h[:, :n])[0]
        true = mx.array(ids[a0 + j + 1:a0 + j + 1 + n].astype(np.int32))
        tl = mx.take_along_axis(lg, true[:, None], axis=-1)            # the true token's logit
        rank = mx.minimum((lg > tl).sum(axis=-1), 3)                   # how many score above it (ties: favourable)
        p1 = mx.exp(lg.max(axis=-1) - mx.logsumexp(lg, axis=-1))
        mx.eval(rank, p1)
        res.append((np.array(p1, np.float32), np.array(rank, np.int8)))
    T.chain(lm, head, P, x["W"], mx.array(ids)[None], D, on_depth, a0)
    return res


def main():
    T.M.limit_memory()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="b5"); ap.add_argument("--frac", type=float, default=0.03)
    ap.add_argument("--sets", default="ifeval,multi,mgsm,gsm,code")   # agent windows start mid-prompt (off > 0): window() refuses them
    a = ap.parse_args()
    rt, lm, head, P = T.load_head(a)
    t0 = time.time()
    for st in a.sets.split(","):
        f_out = os.path.join(OUT, f"{st}_{a.run}.npz")
        if os.path.exists(f_out):
            continue
        keys, P1, RK, GEN, IDS = [], [], [], [], []
        for f in sorted(glob.glob(os.path.join(T.TD, "d16w", PAT[st] + ".npz"))):
            while os.path.exists(T.PAUSE):
                time.sleep(1)
            key = os.path.basename(f)[:-4]
            x = T.window(key)
            gen = T.gen_of(key)
            a0 = max(0, gen - 2)
            r = top_ranks(lm, head, P, x, a0)
            n = len(x["ids"]) - 1 - a0                                   # anchors a0 .. S-2
            p1 = np.full((n, D), np.nan, np.float32); rk = np.full((n, D), -1, np.int8)
            for j, (p, k) in enumerate(r):
                p1[:len(p), j] = p; rk[:len(k), j] = k
            keys.append(key); P1.append(p1); RK.append(rk); GEN.append(gen); IDS.append(np.asarray(x["ids"], np.int32))
        np.savez(f_out, keys=np.array(keys), gen=np.array(GEN), **{f"p1_{i}": v for i, v in enumerate(P1)},
                 **{f"rk_{i}": v for i, v in enumerate(RK)}, **{f"ids_{i}": v for i, v in enumerate(IDS)})
        d1 = np.concatenate([r[:, 0] for r in RK]); d1 = d1[d1 >= 0]
        print(f"  {st} ({len(keys)} answers): depth-1 true token at rank 0 / <=1 / <=2: {np.mean(d1 == 0):.3f} / "
              f"{np.mean(d1 <= 1):.3f} / {np.mean(d1 <= 2):.3f}; {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
