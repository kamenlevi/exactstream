> (a quote from the project owner, kept private)
From BigMoE's real t30b EAGLE-2 dumps (32 best-first pops per anchor), for K = 4 .. 32 (the first K pops; best-first
order makes every prefix a closed subtree):
  - kept per check under certify98's walk (R0 = 1), a (K + 1)-row check = K nodes + the bonus;
  - T(K) = kept(K) / kept(chain-16), the CERTIFY98 amendment 6 statistic generalized;
  - the tree's DEPTH (deepest selected node + 1) per anchor: the head steps a level-wise drafter needs (EAGLE-2 expands
    one level per head call), so it sets the DRAFT time; chain-16 needs 15 steps, chain-32 31;
  - the frontier width per level (nodes at each depth): the head call's batch.
  (the lookup-as-a-node arm is a separate step, not in this file)
Self-check: K 15 must give S9c's EAGLE-2-16 kept (11.263 / 7.487).
usage: ~/BigMoE/.venv/bin/python -I -B treek_curve.py"""
import os, sys, json
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/NewIdeas/research/week98"))
import tree_audit as TA

OUT = os.path.expanduser("~/NewIdeas/research/week98/treek_curve")
KS = [4, 8, 12, 15, 16, 20, 24, 28, 31, 32]
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)


def main():
    TA.say = lambda s="": None
    res = {}
    for s, (tag, rtag, pub32, pub16) in TA.SETS.items():
        W = TA.load_windows(tag, pub32)
        tz = np.load(f"{TA.CE}/tree_t30b_{tag}.npz", allow_pickle=True)
        fi, an, nd, dp, ot = (tz[k] for k in ("fi", "an", "node", "depth", "on_true"))
        A = {K: {} for K in KS}; D = {K: {} for K in KS}; WID = {K: [] for K in KS}
        order = np.lexsort((nd, an, fi)); fi, an, nd, dp, ot = (x[order] for x in (fi, an, nd, dp, ot))
        b = np.flatnonzero(np.append(True, (np.diff(fi) != 0) | (np.diff(an) != 0))); b = np.append(b, len(fi))
        for b0, b1 in zip(b[:-1], b[1:]):
            key = (int(fi[b0]), int(an[b0])); n_, d_, o_ = nd[b0:b1], dp[b0:b1], ot[b0:b1].astype(bool)
            for K in KS:
                sel = n_ < K; hit = o_ & sel
                A[K][key] = int(d_[hit].max()) + 1 if hit.any() else 0
                D[K][key] = int(d_[sel].max()) + 1 if sel.any() else 0
                WID[K].append(np.bincount(d_[sel], minlength=32)[:32])
        ch16 = []
        for X in W:
            p = 1
            while p < X["n"]:
                a = TA.lead_right(X["rt"][p], 0, 15); ch16.append(a + 1); p += a + 1
        c16 = float(np.mean(ch16))
        say(f"\n=== {s} ({tag}) ===  chain-16 kept {c16:.3f}")
        r = {"chain16": c16}
        for K in KS:
            ks, deps = [], []
            for X in W:
                p = 1
                while p < X["n"]:
                    key = (X["fi"], int(X["an"][p])) if p < len(X["an"]) else None
                    a = A[K].get(key, 0) if key else 0
                    ks.append(a + 1)
                    if key:
                        deps.append(D[K].get(key, 0))
                    p += a + 1
            kk = float(np.mean(ks)); w = np.mean(WID[K], 0)
            say(f"  K {K:3d}: kept {kk:6.3f}  T {kk / c16:5.3f}  depth mean {np.mean(deps):5.2f} p90 {np.percentile(deps, 90):4.0f}  "
                f"width by depth {' '.join(f'{x:.1f}' for x in w[:8])} ...")
            r[f"K{K}"] = {"kept": round(kk, 3), "T": round(kk / c16, 4), "depth_mean": round(float(np.mean(deps)), 2),
                          "depth_p90": float(np.percentile(deps, 90)), "width": [round(float(x), 2) for x in w[:16]]}
        k15 = r["K15"]["kept"]
        say(f"  SELF-CHECK K 15 = EAGLE-2-16 {k15:.3f} vs S9c {pub16} -> {'PASS' if abs(k15 - pub16) < 0.01 else 'FAIL'}")
        res[s] = r
    json.dump(res, open(OUT + ".json", "w"), indent=1)
    open(OUT + ".txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
