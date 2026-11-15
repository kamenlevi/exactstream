"""TREE COVERAGE for the CERTIFY98 tree line (NewIdeas, Sun 15 Nov 14:4x). On the dev dumps (BigMoE's real t30b trees;
S9c's layout and walk):
- walk-strict COVERAGE per depth: P(the 125B's word is among the verified candidates at depth d | covered at all < d),
  over each method's own walk starts. For chain-16 the candidate is the head's top-1, so this is the per-guess
  walk-strict accuracy. For tree-16 it is EAGLE-2-16's selected nodes under the true path.
- the ratio tree-16 / chain-16 kept per check with a window-clustered bootstrap (S9c's form), and the SE scaled to the
  fresh sets' window counts (HE+ 140, MBPP 300).
usage: BigMoE venv python -I -B tree_coverage.py"""
import os, sys
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools"))
import certify98 as C98
CE = os.path.expanduser("~/BigMoE/data/deepdraft/ceiling")
SETS = {"mbpp": ("mbppbank", 300), "he": ("heplus24", 140)}
NODES, DEP = 15, 31
BANDS = [(0, 0), (1, 2), (3, 6), (7, 10), (11, 14)]
rng = np.random.default_rng(1115)
OUT = os.path.expanduser("~/NewIdeas/research/week98/tree_coverage.txt"); lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
for s, (tag, nfresh) in SETS.items():
    cz, kz, tz = (np.load(f"{CE}/{p}_t30b_{tag}.npz") for p in ("cells", "topk", "tree"))
    Wc = C98.windows(f"{CE}/cells_t30b_{tag}.npz")
    dep, row, fi = cz["depth"].astype(int), cz["row"].astype(np.int64), cz["fi"].astype(np.int64)
    right, yr = kz["right"].astype(int), kz["y_rank"].astype(int)
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    wins = []
    for wi, (s0, e0) in enumerate(zip(st, en)):
        dw = dep[s0:e0]; an = row[s0:e0][dw == 0] - 1; n = Wc[wi][0].shape[0]
        R = np.full((n, DEP), -1, np.int16)
        for j in range(DEP):
            m = dw == j; R[:m.sum(), j] = np.where(right[s0:e0][m] == 1, 0, np.maximum(yr[s0:e0][m], 1))
        wins.append({"fi": int(fi[s0]), "n": n, "R": R, "row": {int(a): i for i, a in enumerate(an)},
                     "T": np.zeros((n, DEP), bool)})
    byfi = {w["fi"]: w for w in wins}
    sel = (tz["node"] < NODES) & tz["on_true"].astype(bool)
    for f_, a_, d_ in zip(tz["fi"][sel], tz["an"][sel], tz["depth"][sel]):
        w = byfi[int(f_)]; w["T"][w["row"][int(a_)], int(d_)] = True
    def walk(acc, n):
        p = C98.R0; st_ = []; ks = nc = 0
        while p < n:
            a = int(acc[p]); st_.append(p); ks += a + 1; nc += 1; p += a + 1
        return ks, nc, st_
    res = {}
    for nm in ("chain-16", "tree-16"):
        hit = np.zeros(NODES); elig = np.zeros(NODES); per = []
        for w in wins:
            cov = (w["R"] == 0) if nm == "chain-16" else w["T"]
            first = np.where((~cov[:, :NODES]).any(1), (~cov[:, :NODES]).argmax(1), NODES)
            ks, nc, sts = walk(first, w["n"]); per.append((ks, nc))
            for p in sts:
                for d in range(NODES):
                    if w["R"][p, d] < 0: break
                    elig[d] += 1; hit[d] += cov[p, d]
                    if not cov[p, d]: break
        res[nm] = (np.array(per, float), hit, elig)
        say(f"== {s} {nm}: kept/check {res[nm][0][:, 0].sum() / res[nm][0][:, 1].sum():.3f}; walk-strict coverage by band: " +
            "  ".join(f"{a}-{b} {100 * hit[a:b + 1].sum() / max(elig[a:b + 1].sum(), 1):.2f}" for a, b in BANDS))
    a, b = res["tree-16"][0], res["chain-16"][0]
    K = len(a); idx = rng.integers(0, K, (4000, K))
    r = (a[idx, 0].sum(1) / a[idx, 1].sum(1)) / (b[idx, 0].sum(1) / b[idx, 1].sum(1))
    pt = (a[:, 0].sum() / a[:, 1].sum()) / (b[:, 0].sum() / b[:, 1].sum()); se = r.std()
    sef = se * np.sqrt(K / nfresh)
    say(f"   ratio tree-16 / chain-16 x{pt:.3f}, dev windows {K}, bootstrap SE {se:.4f}, one-sided 95 % LB {pt - 1.645 * se:.3f}; "
        f"scaled to the fresh {nfresh} windows: SE {sef:.4f}, expected LB {pt - 1.645 * sef:.3f}")
    from math import erf, sqrt
    def pw(true, line): return 0.5 * (1 + erf(((true - 1.645 * sef) - line) / sef / sqrt(2)))
    for true in (pt, pt - 1.96 * se):
        say(f"   power if the true ratio is {true:.3f}: P(LB >= 1.021) {pw(true, 1.021):.2f}; "
            f"P(point >= 1.052) {0.5 * (1 + erf((true - 1.052) / sef / sqrt(2))):.2f}")
open(OUT, "w").write("\n".join(lines) + "\n")
