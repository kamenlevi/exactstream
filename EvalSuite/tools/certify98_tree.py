"""CERTIFY98's tree read-out: kept tokens per check of a best-first EAGLE-2 tree vs a chain, each under its own decode walk.
CPU only. Reads BigMoE's TOPK and TREE dumps of one run on one set (scripts/dd_topk.py 9f49e03, scripts/dd_tree.py
b198498: at every anchor, 32 best-first nodes, top-8 children, on_true = the node's token is y under a true-path parent).
  .venv/bin/python tools/certify98_tree.py <topk.npz> <tree.npz> [--label L]
  .venv/bin/python tools/certify98_tree.py reproduce      # MTP Head's S9c (TrainLab fa3ce76) on the two dev sets
Definitions (MTP Head's sb98/tree_real125.py, the same walk as certify98.kept_walk, R0 = 1):
- an N-row check is N - 1 drafted nodes; for a tree, the first N - 1 pops. accepted = the deepest on-true node among
  them + 1 (0 if none); for a chain, the run of right cells from depth 0, capped at N - 1. kept = accepted + 1.
- the walk starts at R0 and moves by kept; kept per check = sum kept / sum checks, pooled over the windows.
- tree / chain ratio: its one-sided 95 % lower bound is the delta method on log r with window-clustered (CR1) influences
  and t(K - 1), as certify98.py's bounds."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify98 as C                                    # noqa: E402

CE = os.path.expanduser("~/BigMoE/data/deepdraft/ceiling")
NS = (16, 32)
COVB = ((0, 0), (1, 2), (3, 6), (7, 10), (11, 14))                 # coverage bands: a 16-row check reaches depth 14
REPRO = (("mbppbank", {16: 7.487, 32: 9.031}, {16: 1.136, 32: 1.371}, 6.588),
         ("heplus24", {16: 11.263, 32: 16.263}, {16: 1.060, 32: 1.531}, 10.625))   # TrainLab data/sb98/tree_real125.txt


def walk(acc, r0=C.R0):
    """(sum kept, checks, starts) of the decode walk over per-row accepted counts"""
    p, ks, nc, st = r0, 0, 0, []
    while p < len(acc):
        a = int(acc[p]); st.append(p); ks += a + 1; nc += 1; p += a + 1
    return ks, nc, st


def chain_acc(rt, N):
    """the run of right cells from depth 0 (a missing cell ends it), capped at N - 1"""
    ok = rt == 1
    k = np.where((~ok).any(1), (~ok).argmax(1), rt.shape[1])
    return np.minimum(k, N - 1)


def load(topk, tree):
    """per window: rt (certify98's layout), the depth-0 anchors in row order, and the tree's accepted counts per N"""
    W = C.windows(topk)
    t, z = np.load(topk), np.load(tree)
    dep, fi, row = t["depth"], t["fi"], t["row"]
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    out, bad = [], []
    sel = z["on_true"].astype(bool)
    zfi, zan, znode, zdep = z["fi"][sel], z["an"][sel], z["node"][sel], z["depth"][sel]
    for (rt, _), a, b in zip(W, st, en):
        f = int(fi[a]); an = row[a:b][dep[a:b] == 0] - 1          # the anchor = the cell's row - 1, as in the dumps
        rowof = {int(x): i for i, x in enumerate(an)}
        acc = {N: np.zeros(rt.shape[0], np.int64) for N in NS}
        m = zfi == f
        for x, nd, d in zip(zan[m], znode[m], zdep[m]):
            r = rowof[int(x)]
            for N in NS:
                if nd < N - 1:
                    acc[N][r] = max(acc[N][r], int(d) + 1)
        ga = np.sort(z["a_an"][z["a_fi"] == f])
        if not np.array_equal(ga, np.sort(an)):
            bad.append(f"window fi {f}: the tree's anchors are not the cells' depth-0 anchors")
        flags = np.zeros(rt.shape[0], bool); ma = z["a_fi"] == f
        for x, w in zip(z["a_an"][ma], z["a_walk_start"][ma].astype(bool)):
            flags[rowof[int(x)]] = w
        s32 = walk(chain_acc(rt, 32))[2]
        if not np.array_equal(np.flatnonzero(flags), np.array([p for p in s32 if p < rt.shape[0] - 1])):
            bad.append(f"window fi {f}: a_walk_start is not the chain-32 walk")
        out.append({"rt": rt, "acc": acc})
    return out, bad


def coverage(W, accf, cap=15):
    """walk-strict COVERAGE per depth: P(the target's token is among the drafted nodes at depth d | covered at 0..d-1),
    at the walk's check starts, in certify98's walk convention (a missing cell, or the walk's last start at the answer's
    end, ends the run), so a chain's coverage is certify98.strict_counts(mode="walk") exactly"""
    num, den = np.zeros(cap), np.zeros(cap)
    for X in W:
        a = accf(X)
        for p in walk(a)[2]:
            k = np.arange(cap)
            den += a[p] >= k; num += a[p] >= k + 1
    return num, den


def ratio(T, Cc):
    """tree / chain kept per check from per-window (kept, checks) sums: (r, se of log r, one-sided 95 % lower bound)"""
    K = len(T)
    tk, tc, ck, cc = T[:, 0], T[:, 1], Cc[:, 0], Cc[:, 1]
    r = (tk.sum() / tc.sum()) / (ck.sum() / cc.sum())
    inf = tk / tk.sum() - tc / tc.sum() - ck / ck.sum() + cc / cc.sum()
    se = float(np.sqrt(K / (K - 1) * (inf ** 2).sum()))
    return float(r), se, float(r * np.exp(-C.t_one(K - 1) * se))


def score(topk, tree):
    W, bad = load(topk, tree)
    R = {"windows": len(W), "bad": bad}
    for N in NS:
        Cn = np.array([walk(chain_acc(X["rt"], N))[:2] for X in W], float)
        Tn = np.array([walk(X["acc"][N])[:2] for X in W], float)
        R[f"chain{N}"] = float(Cn[:, 0].sum() / Cn[:, 1].sum())
        R[f"tree{N}"] = float(Tn[:, 0].sum() / Tn[:, 1].sum())
        R[f"_c{N}"], R[f"_t{N}"] = Cn, Tn
    for N in NS:
        r, se, lcb = ratio(R[f"_t{N}"], R["_c16"])
        R[f"tree{N}_vs_chain16"] = {"r": r, "se_log": se, "lcb95": lcb}
    kept32 = C.summary(C.windows(topk))["kept_per_32"]["mean"]
    if abs(R["chain32"] - kept32) > 1e-9:
        bad.append(f"chain-32 {R['chain32']:.6f} != certify98's kept/32 {kept32:.6f}")
    R["cov_chain16"] = coverage(W, lambda X: chain_acc(X["rt"], 16))
    R["cov_tree16"] = coverage(W, lambda X: X["acc"][16])
    S = C.strict_counts(C.windows(topk), mode="walk", cap=15)            # the nesting: chain coverage == walk-strict
    sn, sd = sum(s[0] for s in S)[:15], sum(s[1] for s in S)[:15]
    if not (np.array_equal(sn, R["cov_chain16"][0]) and np.array_equal(sd, R["cov_chain16"][1])):
        bad.append("chain-16 coverage != certify98's walk-strict accuracy at cap 15")
    return R


def show(R, label):
    print(f"{label}: {R['windows']} windows; chain-16 {R['chain16']:.3f}, chain-32 {R['chain32']:.3f} (= certify98 kept/32)")
    for N in NS:
        x = R[f"tree{N}_vs_chain16"]
        print(f"  tree-{N}: {R[f'tree{N}']:.3f} kept per check = x{x['r']:.3f} chain-16 (one-sided 95 % lower bound "
              f"x{x['lcb95']:.3f}; se of log {x['se_log']:.4f})")
    for lab in ("chain16", "tree16"):
        n, d = R[f"cov_{lab}"]
        print(f"  coverage {lab[:-2]}-16 by band: " + "  ".join(f"{lo}-{hi} {100 * n[lo:hi + 1].sum() / d[lo:hi + 1].sum():.2f}"
                                                               for lo, hi in COVB))
    for b in R["bad"]:
        print(f"  CHECK FAILED: {b}")


def reproduce():
    ok = True
    for tag, kpc, rat, c16 in REPRO:
        R = score(f"{CE}/topk_t30b_{tag}.npz", f"{CE}/tree_t30b_{tag}.npz"); show(R, tag)
        diffs = [abs(R[f"tree{N}"] - kpc[N]) for N in NS] + [abs(R[f"tree{N}_vs_chain16"]["r"] - rat[N]) for N in NS] \
            + [abs(R["chain16"] - c16)]
        good = max(diffs) < 0.0006 and not R["bad"]                  # S9c prints 3 decimals
        print(f"  vs S9c: max |diff| {max(diffs):.4f} -> {'PASS' if good else 'FAIL'}")
        ok &= good
    print(f"TREE-REPRODUCE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv):
    if argv[:1] == ["reproduce"]:
        return reproduce()
    if len(argv) < 2:
        print(__doc__); return 2
    lab = argv[argv.index("--label") + 1] if "--label" in argv else os.path.basename(argv[1])
    R = score(argv[0], argv[1]); show(R, lab)
    return 1 if R["bad"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
