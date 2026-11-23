"""WEEK98 (MTP Head; orchestrator 01:5x, for MoonShots' relax.py): t30b's HE+ cells WITH the drafted token per cell (am),
so rank and dlp are known on misses too. Same cells as cells_t30b_heplus.npz: t30b's own chain on the 24 HE+ windows
(d16w) with the runtime-path BANK context (as dd_taps.py evalrep --set code --ctx bank), built the way BigMoE's
dd_ceiling.py misses builds MBPP's cells (rank 99 / dlp -inf = outside the 125B's top-32). BigMoE's modules are imported
read-only; no BigMoE file is edited. Check: the bands must equal evalrep's (0 97.95 ... 23-30 89.65).
Usage: cd ~/BigMoE && ~/BigMoE/.venv/bin/python3 -I /Users/kamen/BigMoE/data/r57/snap/frozen_s98m/TrainLab_sticky/w98_heplus_cells_am.py OUT.npz"""
import os, sys, glob
sys.path.insert(0, os.path.expanduser("~/BigMoE/scripts"))
import numpy as np
import mlx.core as mx
import dd_ttt as DD

T, M = DD.T, DD.M
OUT = sys.argv[1]
BANK = os.path.expanduser("~/datasets/deepdraft/bank")
rt_, lm, head = M.load(0.03)
P = M.Trainable(head)
P.load_weights(os.path.join(DD.CKD, "dd_t30b.safetensors")); mx.eval(P.parameters())
items = sorted(glob.glob(os.path.join(T.TD, "d16w", "tf_code_HumanEval_*.npz")))
C = {k: [] for k in ("depth", "right", "m", "rank", "dlp", "am", "fi", "an", "row")}
for i, f in enumerate(items):
    while os.path.exists(T.PAUSE):
        import time; time.sleep(1)
    key_ = os.path.basename(f)[:-4]
    x = T.window(key_); gen = T.gen_of(key_)
    z = np.load(os.path.join(BANK, os.path.basename(f)))
    assert np.array_equal(np.asarray(x["ids"]), np.asarray(z["ids"])), f"{key_}: window ids != bank ids"
    x["W"] = mx.array(np.asarray(z["wd"])).astype(mx.bfloat16)[None]      # the bank's runtime-path context stream
    lp, tk = np.asarray(z["lp"], np.float32), np.asarray(z["tk"])
    a0 = max(0, gen - 2)
    for j, (right, am) in enumerate(T.chain_hits(lm, head, P, x, 31, a0)):
        rows = a0 + np.arange(len(right)) + j + 1
        ok = rows < len(lp)
        rows, right, am = rows[ok], np.asarray(right)[ok], np.asarray(am)[ok]
        valid = rows >= gen - 1
        rows, right, am = rows[valid], right[valid], am[valid]
        hit = tk[rows] == am[:, None]
        rank = np.where(hit.any(1), hit.argmax(1), 99)
        dlp = np.where(rank < 32, lp[rows, np.minimum(rank, 31)] - lp[rows, 0], -np.inf)
        for k, v in (("depth", np.full(len(rows), j)), ("right", right.astype(bool)), ("m", lp[rows, 0] - lp[rows, 1]),
                     ("rank", rank), ("dlp", dlp.astype(np.float32)), ("am", am.astype(np.int32)),
                     ("fi", np.full(len(rows), i)), ("an", rows - j - 1), ("row", rows)):
            C[k].append(v)
    mx.clear_cache()
C = {k: np.concatenate(v) for k, v in C.items()}
np.savez_compressed(OUT, files=np.array([os.path.basename(f) for f in items]), **C)
dep, ok = C["depth"], C["right"]
print(f"{len(dep)} cells from {len(items)} files -> {OUT}")
print("t30b HE+ (bank ctx) by band: " + "  ".join(
    f"{a}-{b} {100 * ok[(dep >= a) & (dep <= b)].mean():.2f}" for a, b in [(0, 0), (1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30)]))
print(f"misses outside the top-32: {100 * (C['rank'][~ok] == 99).mean():.2f} % of misses")
