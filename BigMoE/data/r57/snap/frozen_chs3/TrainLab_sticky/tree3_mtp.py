"""TREE3, the MTP half (MTP Head, Tue 11-03; the Breakthrough Push, staged verification): the trained MTP head's (D-22 b5)
teacher-forced chain on Breakthrough's 36 decision-map windows (12 HumanEval+, 12 IFEval, 12 multi; the trunk's widened
stream in data/d16w), depths 1-15, keeping at every anchor and depth the head's top-3 tokens and their probabilities (f32
softmax over the full vocab, as D-24's p1) and the true token's rank capped at 3. Row r = the MTP anchor s = a0 + r
(a0 = gen - 2), whose depth d predicts ids[s + d + 1] (d22_ttt.chain). Output: data/tree3/mtp_<set>.npz: keys, gen, a0,
t3_<i> [anchors, 15, 3] int32, p3_<i> [anchors, 15, 3] float32, rk_<i> [anchors, 15] int8 (-1 = past the answer). Prints the
MLX peak. Usage: [BigMoE venv] tree3_mtp.py [--run b5]"""
import os, sys, glob, time, argparse
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs3/TrainLab_sticky"))
import d22_ttt as T                                          # its imports set MTPLX's env, then mlx
import numpy as np
import mlx.core as mx

D = 15
OUT = os.path.join(T.TD, "tree3"); os.makedirs(OUT, exist_ok=True)
MAP = os.path.expanduser("~/FableGod/decision_map/out/routing")


def top3_chain(lm, head, P, x, a0):
    ids = x["ids"]; S = len(ids); res = []

    def on_depth(j, h):
        n = min(h.shape[1], S - 1 - j - a0)
        lg = T.logits(lm, head, h[:, :n])[0]
        true = mx.array(ids[a0 + j + 1:a0 + j + 1 + n].astype(np.int32))
        tl = mx.take_along_axis(lg, true[:, None], axis=-1)
        rank = mx.minimum((lg > tl).sum(axis=-1), 3)
        t3 = mx.argpartition(-lg, 3, axis=-1)[:, :3]
        l3 = mx.take_along_axis(lg, t3, axis=-1)
        o = mx.argsort(-l3, axis=-1)
        t3, l3 = mx.take_along_axis(t3, o, axis=-1), mx.take_along_axis(l3, o, axis=-1)
        p3 = mx.exp(l3 - mx.logsumexp(lg, axis=-1, keepdims=True))
        mx.eval(rank, t3, p3)
        res.append((np.array(t3, np.int32), np.array(p3, np.float32), np.array(rank, np.int8)))
    T.chain(lm, head, P, x["W"], mx.array(ids)[None], D, on_depth, a0)
    return res


def main():
    T.M.limit_memory()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="b5"); ap.add_argument("--frac", type=float, default=0.03)
    a = ap.parse_args()
    rt, lm, head, P = T.load_head(a)
    t0 = time.time()
    for st, pat in (("code", "tf_code_HumanEval_*"), ("ifeval", "tf_ifeval_*"), ("multi", "tf_multi_*")):
        f_out = os.path.join(OUT, f"mtp_{st}.npz")
        if os.path.exists(f_out):
            continue
        keys, GEN, A0, T3, P3, RK = [], [], [], [], [], []
        for f in sorted(glob.glob(os.path.join(MAP, pat + ".npz"))):
            key = os.path.basename(f)[:-4]
            x, gen = T.window(key), T.gen_of(key)
            a0 = max(0, gen - 2)
            r = top3_chain(lm, head, P, x, a0)
            n = len(x["ids"]) - 1 - a0
            t3 = np.full((n, D, 3), -1, np.int32); p3 = np.full((n, D, 3), np.nan, np.float32); rk = np.full((n, D), -1, np.int8)
            for j, (t, p, k) in enumerate(r):
                t3[:len(t), j] = t; p3[:len(p), j] = p; rk[:len(k), j] = k
            keys.append(key); GEN.append(gen); A0.append(a0); T3.append(t3); P3.append(p3); RK.append(rk)
            mx.clear_cache()
        np.savez(f_out, keys=np.array(keys), gen=np.array(GEN), a0=np.array(A0), **{f"t3_{i}": v for i, v in enumerate(T3)},
                 **{f"p3_{i}": v for i, v in enumerate(P3)}, **{f"rk_{i}": v for i, v in enumerate(RK)})
        d1 = np.concatenate([r[:, 0] for r in RK]); d1 = d1[d1 >= 0]
        print(f"  {st} ({len(keys)} windows): depth-1 true token at rank 0 / <= 2: {np.mean(d1 == 0):.3f} / {np.mean(d1 <= 2):.3f}; "
              f"{time.time() - t0:.0f} s; MLX peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
