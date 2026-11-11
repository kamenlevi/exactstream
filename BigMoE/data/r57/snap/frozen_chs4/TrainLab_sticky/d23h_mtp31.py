"""DEPTH32 step 1 (MTP Head, Mon 11-02; REPORT 'DEPTH32'): the trained MTP head's (D-22 b5) teacher-forced chain from every
anchor of the held-out answers to depth 31 (d22_ttt.chain_hits, as D-24 at depth 15): per anchor s = a0 .. and depth j,
right (the depth-j draft equals ids[s + j + 1]) and the argmax token. Sets: code = the 24 HumanEval+ windows (the trunk's
widened stream in data/d16w), mbpp = the same 100 MBPP answers as MATCH / EARLYTAP (data/dlog/d1102mbpp, its wd). Output
data/d23h/<set>_b5.npz: keys, gen, a0, rt_<i> [anchors, 31] int8 (1 / 0 / -1 = beyond the answer), am_<i> int32, p1_<i>
float32 (the top-1 p as D-24's, NaN beyond the answer; added Tue 11-03 before any run, for MTPPICK on MBPP). Prints the MLX peak. Usage: [BigMoE venv] d23h_mtp31.py [--run b5] [--sets code,mbpp]"""
import os, sys, glob, time, argparse
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs4/TrainLab_sticky"))
import d22_ttt as T                                          # its imports set MTPLX's env, then mlx
import numpy as np
import mlx.core as mx

D = 31
OUT = os.path.join(T.TD, "d23h"); os.makedirs(OUT, exist_ok=True)


def mbpp_window(p):
    z = np.load(p)
    ids = np.asarray(z["ids"], np.int32)
    assert z["wd"].shape[0] == len(ids) - 1, p
    return {"W": mx.array(z["wd"]).astype(mx.bfloat16)[None], "ids": ids}, int(z["gen"])


def chain_hits_p1(lm, head, P, x, D, a0):
    """T.chain_hits plus each depth's top-1 p (D-24's definition: exp(max - logsumexp) over the f32 logits)"""
    ids = x["ids"]; S = len(ids); out = []

    def on_depth(j, h):
        n = min(h.shape[1], S - 1 - j - a0)
        lg = T.logits(lm, head, h[:, :n])[0]
        am = mx.argmax(lg, -1)
        p1 = mx.exp(lg.max(axis=-1) - mx.logsumexp(lg, axis=-1))
        mx.eval(am, p1)
        a = np.array(am)
        out.append((a == ids[a0 + j + 1:a0 + j + 1 + n], a, np.array(p1, np.float32)))
    T.chain(lm, head, P, x["W"], mx.array(ids)[None], D, on_depth, a0)
    return out


def main():
    T.M.limit_memory()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="b5"); ap.add_argument("--frac", type=float, default=0.03)
    ap.add_argument("--sets", default="code,mbpp")
    a = ap.parse_args()
    rt, lm, head, P = T.load_head(a)
    t0 = time.time()
    for st in a.sets.split(","):
        f_out = os.path.join(OUT, f"{st}_{a.run}.npz")
        if os.path.exists(f_out):
            continue
        if st == "code":
            items = [(os.path.basename(f)[:-4], None) for f in sorted(glob.glob(os.path.join(T.TD, "d16w", "tf_code_HumanEval_*.npz")))]
        else:
            items = [(os.path.basename(f)[:-4], f) for f in sorted(glob.glob(os.path.join(T.TD, "dlog/d1102mbpp/dlog__mbpp*.npz")))[:100]]
        keys, GEN, A0, RT, AM, PP = [], [], [], [], [], []
        for key, f in items:
            while os.path.exists(T.PAUSE):
                time.sleep(1)
            if f is None:
                x, gen = T.window(key), T.gen_of(key)
            else:
                x, gen = mbpp_window(f)
            a0 = max(0, gen - 2)
            hits = chain_hits_p1(lm, head, P, x, D, a0)
            n = len(x["ids"]) - 1 - a0
            rt_ = np.full((n, D), -1, np.int8); am_ = np.full((n, D), -1, np.int32); pp_ = np.full((n, D), np.nan, np.float32)
            for j, (right, amx, p1) in enumerate(hits):
                rt_[:len(right), j] = right.astype(np.int8); am_[:len(amx), j] = amx; pp_[:len(p1), j] = p1
            keys.append(key); GEN.append(gen); A0.append(a0); RT.append(rt_); AM.append(am_); PP.append(pp_)
            mx.clear_cache()
        np.savez(f_out, keys=np.array(keys), gen=np.array(GEN), a0=np.array(A0), **{f"rt_{i}": v for i, v in enumerate(RT)},
                 **{f"am_{i}": v for i, v in enumerate(AM)}, **{f"p1_{i}": v for i, v in enumerate(PP)})
        d1 = np.concatenate([r[:, 0] for r in RT]); d1 = d1[d1 >= 0]
        print(f"  {st} ({len(keys)} answers): depth-1 right {d1.mean():.3f}; {time.time() - t0:.0f} s; MLX peak {mx.get_peak_memory() / 1e9:.1f} GB",
              flush=True)


if __name__ == "__main__":
    main()
