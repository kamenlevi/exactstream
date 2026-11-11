#!/usr/bin/env python3
"""g0_mtp31_set.py (drafterwall, step 2 of the pre-registered GPU test LAG2-AGENT). NOT RUN in this mission: it loads the
125B runtime through TrainLab's d22_ttt (about 10-13 GB), which is outside this session's GPU rule.
The MTP head b5, teacher-forced from every answer anchor to depth 31, on a folder of DLOG captures
(TrainLab sticky/dlog_capture.py files: ids, gen, wd). Same code path as TrainLab's sticky/d23h_mtp31.py (its
mbpp_window and chain_hits_p1 are imported, read-only); only the set and the output folder differ.
usage: [BigMoE venv] g0_mtp31_set.py <set name> <folder with dlog *.npz> [--run b5] [--max 100]
output: ~/FableGod/drafterwall/gpu_test/out/<set>_<run>.npz (keys, gen, a0, rt_<i>, am_<i>, p1_<i>), the format
gpu_test/g1_lag2_readout.py reads."""
import os, sys, glob, argparse
sys.path.insert(0, os.path.expanduser("~/TrainLab/sticky"))
import d23h_mtp31 as D                                      # imports d22_ttt as T (sets MTPLX's env, then mlx)
import numpy as np
import mlx.core as mx
T = D.T
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set"); ap.add_argument("folder")
    ap.add_argument("--run", default="b5"); ap.add_argument("--frac", type=float, default=0.03); ap.add_argument("--max", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    T.M.limit_memory()
    rt, lm, head, P = T.load_head(a)
    files = sorted(glob.glob(os.path.join(os.path.expanduser(a.folder), "*.npz")))[:a.max]
    keys, GEN, A0, RT, AM, PP = [], [], [], [], [], []
    for f in files:
        x, gen = D.mbpp_window(f)                            # any DLOG capture: W = wd, ids; gen = the prompt length
        a0 = max(0, gen - 2)
        hits = D.chain_hits_p1(lm, head, P, x, D.D, a0)
        n = len(x["ids"]) - 1 - a0
        rt_ = np.full((n, D.D), -1, np.int8); am_ = np.full((n, D.D), -1, np.int32); pp_ = np.full((n, D.D), np.nan, np.float32)
        for j, (right, amx, p1) in enumerate(hits):
            rt_[:len(right), j] = right.astype(np.int8); am_[:len(amx), j] = amx; pp_[:len(p1), j] = p1
        keys.append(os.path.basename(f)[:-4]); GEN.append(gen); A0.append(a0); RT.append(rt_); AM.append(am_); PP.append(pp_)
        mx.clear_cache()
    np.savez(os.path.join(OUT, f"{a.set}_{a.run}.npz"), keys=np.array(keys), gen=np.array(GEN), a0=np.array(A0),
             **{f"rt_{i}": v for i, v in enumerate(RT)}, **{f"am_{i}": v for i, v in enumerate(AM)}, **{f"p1_{i}": v for i, v in enumerate(PP)})
    d1 = np.concatenate([r[:, 0] for r in RT]); d1 = d1[d1 >= 0]
    print(f"{a.set} ({len(keys)} answers): depth-0 right {d1.mean():.3f}; MLX peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)

if __name__ == "__main__":
    main()
