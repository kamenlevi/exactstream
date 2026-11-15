"""CERTIFY98: the walk-strict convention check for NewIdeas' tree-K proposal (22:1x; MoonShots' point). CPU only.
Registered (amendment 4): a walk cell with no target (past the answer's end) counts as a miss, so the walk-strict counts
reproduce kept_walk exactly. Proposed: a cell counts as reached only if it has a target. Prints, per dev set, the chain-32
walk-strict bands and the tree-32 walk's coverage bands under both conventions, and T32/32 / T32/16.
  .venv/bin/python tools/certify98_convention_check.py"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify98 as C                                    # noqa: E402
import certify98_tree as T                               # noqa: E402

BANDS = ((0, 0), (1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30))


def cov(W, accf, cap, targets_only):
    """per window (num, den) at the walk's check starts; the first uncovered depth is a miss (registered) or, with
    targets_only, a miss only if that cell has a target (TOPK's cell != -1)"""
    per = []
    for X in W:
        a, rt = accf(X), X["rt"]
        n_, d_ = np.zeros(cap), np.zeros(cap)
        for p in T.walk(a)[2]:
            k = int(a[p])
            n_[:min(k, cap)] += 1; d_[:min(k, cap)] += 1
            if k < cap and (not targets_only or rt[p, k] != -1):
                d_[k] += 1
        per.append((n_, d_))
    return per


def bands(per):
    return "  ".join(f"{lo}-{hi} {s['acc']:.2f} [{s['lcb']:.2f}]" for lo, hi in BANDS
                     for s in [C.strict_stats(per, lo, hi)])


def main():
    for tag in ("heplus24", "mbppbank"):
        tk, tr = f"{T.CE}/topk_t30b_{tag}.npz", f"{T.CE}/tree_t30b_{tag}.npz"
        R = T.score(tk, tr)
        W = T.load(tk, tr)[0]
        r = T.ratio(R["_t32"], R["_c32"]); q = R["tree32_vs_chain16"]
        print(f"== {tag}: T32/32 = tree-32 / chain-32 {r[0]:.3f} [LB {r[2]:.3f}]; T32/16 {q['r']:.3f} [LB {q['lcb95']:.3f}]")
        reg = C.strict_counts(C.windows(tk), mode="walk", cap=31)
        mine = cov(W, lambda X: T.chain_acc(X["rt"], 32), 31, False)
        same = all(np.array_equal(a[0][:31], b[0]) and np.array_equal(a[1][:31], b[1]) for a, b in zip(reg, mine))
        print(f"  the registered counting here == certify98.strict_counts(walk, cap 31): {same}")
        for conv in (False, True):
            lab = "targets only (proposed)    " if conv else "no-target = miss (registered)"
            print(f"  chain-32 walk-strict, {lab}: {bands(cov(W, lambda X: T.chain_acc(X['rt'], 32), 31, conv))}")
            print(f"  tree-32 coverage,     {lab}: {bands(cov(W, lambda X: X['acc'][32], 31, conv))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
