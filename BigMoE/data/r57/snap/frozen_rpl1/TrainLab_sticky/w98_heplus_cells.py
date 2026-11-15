"""WEEK98 task 4 (MTP Head, for MoonShots' co-drafter join): t30b's per-cell results on the 24 HumanEval+ windows with
runtime-path context, in the schema and order of BigMoE's ceiling/cells_t30b_mbpp100.npz (dd_ceiling.py misses).

Input: the dump of BigMoE's REPLACE harness, run unmodified:
  dd_taps.py evalrep --set code --ctx bank --featdir ~/datasets/deepdraft/bank --replace wd --refresh 1000 --runs t30b
    --dump DUMP
  (--refresh 1000: the replace hook fires only at k % 1000 == 0 with k >= 1, i.e. never in 31 depths, so this is t30b's
  own chain; --ctx bank: the context stream is the bank's runtime-path wd), plus the bank files' top-32 (lp, tk).
Cells, as dd_ceiling builds them: file-major, then depth j = 0..30, then anchor s; anchor s = a0 + i drafts from target
row a0 + i + j + 1, kept if that row < len(lp) and >= gen - 1; m = lp[row, 0] - lp[row, 1] (the 125B's top-1 margin).
The dump holds right / wrong per cell but not the drafted token, so rank and dlp are known only on right cells
(rank 0, dlp 0); on wrong cells rank = -1 and dlp = NaN (unknown, not 'outside the top-32'). Extra fields: fi (file
index), an (anchor row), row (target row), files.
Usage: [BigMoE venv] python3 -I w98_heplus_cells.py DUMP OUT.npz"""
import os, sys
import numpy as np

DUMP, OUT = sys.argv[1], sys.argv[2]
BANK = os.path.expanduser("~/datasets/deepdraft/bank")
d = np.load(DUMP)
files = [str(f) for f in d["files"]]
C = {k: [] for k in ("depth", "right", "m", "rank", "dlp", "fi", "an", "row")}
for i, f in enumerate(files):
    rt, a0 = d[f"rt_{i}"], int(d[f"a0_{i}"])
    z = np.load(os.path.join(BANK, f))
    lp, ids, gen = np.asarray(z["lp"], np.float32), np.asarray(z["ids"]), int(z["gen"])
    assert rt.shape[0] == len(ids) - 1 - a0, f"{f}: dump rows {rt.shape[0]} vs bank ids {len(ids)} (a0 {a0})"
    assert a0 == max(0, gen - 2), f"{f}: a0 {a0} vs bank gen {gen}"
    for j in range(rt.shape[1]):
        col = rt[:, j]
        n = int((col >= 0).sum())
        assert (col[:n] >= 0).all() and (col[n:] < 0).all(), f"{f} depth {j}: filled cells are not a prefix"
        rows = a0 + np.arange(n) + j + 1
        keep = (rows < len(lp)) & (rows >= gen - 1)
        rows, right = rows[keep], col[:n][keep].astype(bool)
        for k, v in (("depth", np.full(len(rows), j)), ("right", right), ("m", lp[rows, 0] - lp[rows, 1]),
                     ("rank", np.where(right, 0, -1)), ("dlp", np.where(right, 0.0, np.nan).astype(np.float32)),
                     ("fi", np.full(len(rows), i)), ("an", rows - j - 1), ("row", rows)):
            C[k].append(v)
C = {k: np.concatenate(v) for k, v in C.items()}
np.savez_compressed(OUT, files=np.array(files), **C)
dep, ok = C["depth"], C["right"]
bands = [(0, 0), (1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30)]
print(f"{len(dep)} cells from {len(files)} files -> {OUT}")
print("t30b HE+ (bank ctx) by band: " + "  ".join(f"{a}-{b} {100 * ok[(dep >= a) & (dep <= b)].mean():.2f}" for a, b in bands))
print(f"near-ties (m < 0.25) {100 * (C['m'] < 0.25).mean():.2f} % of cells; median m {np.median(C['m']):.2f} nats")
