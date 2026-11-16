"""PLAN900 bet #1: SPAN TREE (Lunatic THINK_1116 idea 3 + ZERO), NewIdeas, Mon 16 Nov 12:3x. CPU only, read-only.
Take the deep run off the AR head:
- draft bands 0-6 with t30b's tree (<= 7 head levels per block): the first N_tree pops of depth <= 6 in BigMoE's real
  127-node dumps, in pop order (closed);
- beyond depth 6, copy the continuation of the LONGEST PRIOR MATCH (>= n_min tokens) of the context at the tree's
  main-path depth-6 node, else stop. The span rides as a chain under that node, within the 32-row budget
  (31 - N_tree span rows).
A span helps only if the true path IS the main path through depth 6 (the copy is computed for the main path). Accepted =
7 + the copied tokens' leading run that equals the 125B's tokens (bank tk0 rows). Otherwise accepted = the tree's own
true-path coverage.
Also SPAN-UB: the span computed for the true path's depth-6 node, whichever it is (an upper bound on span placement).
Head levels per block = min(7, the tree's depth).
LINE (Lunatic, written before the run): PASS >= 0.90x tree-32's kept at <= 7 levels per block on HE+; KILL < 0.85x
(HE+) or < 0.75x (MBPP).
usage: ~/BigMoE/.venv/bin/python -I -B span_tree.py"""
import os, sys, json
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/NewIdeas/research/week98"))
import tree_audit as TA

H = os.path.expanduser("~")
BANK = f"{H}/datasets/deepdraft/bank"
OUT = os.path.expanduser("~/NewIdeas/research/week98/span_tree")
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)


def longest_prior(ids, p, n_min):
    """source index j (the copy starts at j + 1) of the longest suffix of ids[..p] (length >= n_min) seen earlier; -1"""
    if p + 1 < n_min:
        return -1, 0
    tail = ids[p - n_min + 1:p + 1]
    best_j, best_n = -1, 0
    for j in range(n_min - 1, p):
        if ids[j] != ids[p]:
            continue
        if ids[j - n_min + 1:j + 1] == tail:
            n = n_min
            while j - n >= 0 and p - n >= 0 and ids[j - n] == ids[p - n] and n < 64:
                n += 1
            if n >= best_n:
                best_j, best_n = j, n
    return best_j, best_n


def main():
    TA.say = lambda s="": None
    res = {}
    for s, (tag, rtag, pub32, pub16) in TA.SETS.items():
        W = TA.load_windows(tag, pub32)
        tz = np.load(f"{TA.CE}/tree_t30b_{tag}_n127k8.npz", allow_pickle=True)
        fi, an, nd, dp, ot, pp, pa = (tz[k] for k in ("fi", "an", "node", "depth", "on_true", "path_p", "parent"))
        order = np.lexsort((nd, an, fi)); fi, an, nd, dp, ot, pp, pa = (x[order] for x in (fi, an, nd, dp, ot, pp, pa))
        b = np.flatnonzero(np.append(True, (np.diff(fi) != 0) | (np.diff(an) != 0))); b = np.append(b, len(fi))
        T = {(int(fi[b0]), int(an[b0])): (b0, b1) for b0, b1 in zip(b[:-1], b[1:])}
        bank = {}
        for X in W:
            z = np.load(os.path.join(BANK, X["file"]), allow_pickle=True)
            bank[X["fi"]] = ([int(v) for v in z["ids"]], np.asarray(z["tk"][:, 0], np.int64))
        say(f"\n=== {s} ({tag}) ===")
        r = {}

        def tree32(X, p):
            a0 = int(X["an"][p]); b0, b1 = T[(X["fi"], a0)]
            sel = nd[b0:b1] < 31; hit = ot[b0:b1].astype(bool) & sel
            return (int(dp[b0:b1][hit].max()) + 1 if hit.any() else 0), int(dp[b0:b1][sel].max()) + 1

        def span_arm(X, p, n_tree, n_min, ub):
            a0 = int(X["an"][p]); b0, b1 = T[(X["fi"], a0)]
            n_, d_, o_, p_, a_ = nd[b0:b1], dp[b0:b1], ot[b0:b1].astype(bool), pp[b0:b1], pa[b0:b1]
            cand = np.flatnonzero(d_ <= 6)
            cand = cand[np.argsort(n_[cand], kind="stable")][:n_tree]
            sel = np.zeros(len(n_), bool); sel[cand] = True
            hit = o_ & sel
            A = int(d_[hit].max()) + 1 if hit.any() else 0
            levels = int(d_[sel].max()) + 1 if sel.any() else 0
            span_rows = 31 - int(sel.sum())
            if A < 7 or span_rows <= 0 or X["rt"][p, min(A, 30)] < 0:
                return A, levels, 0
            if not ub:                                   # the main path must be the true path through depth 6
                kids = {}
                for i in np.flatnonzero(sel):
                    kids.setdefault(int(a_[i]), []).append(i)
                cur, ok = -1, True
                for d in range(7):
                    ch = kids.get(cur, [])
                    if not ch:
                        ok = False; break
                    best = max(ch, key=lambda i: p_[i])
                    if not o_[best]:
                        ok = False; break
                    cur = int(n_[best])
                if not ok:
                    return A, levels, 0
            ids, tk0 = bank[X["fi"]]
            pos6 = a0 + 8                                # the depth-6 token's position; the span starts at a0 + 9
            if pos6 >= len(ids) - 1:
                return A, levels, 0
            j, n = longest_prior(ids, pos6, n_min)
            if j < 0:
                return A, levels, 0
            m = 0
            while m < span_rows and j + 1 + m < pos6 + 1 and pos6 + m < len(tk0) and ids[j + 1 + m] == tk0[pos6 + m]:
                m += 1
            return A + m, levels, 1

        def run(fn):
            ks, lv, fired = [], [], 0
            for X in W:
                p = 1
                while p < X["n"]:
                    if p < len(X["an"]):
                        out = fn(X, p); a, l = out[0], out[1]; fired += (out[2] if len(out) > 2 else 0)
                    else:
                        a, l = 0, 0
                    ks.append(a + 1); lv.append(l); p += a + 1
            return float(np.mean(ks)), float(np.mean(lv)), fired, len(ks)

        base, blv, _, nb = run(lambda X, p: tree32(X, p))
        say(f"  tree-32 (31 pops)                 kept {base:7.3f}   head levels per block {blv:5.2f}   checks {nb}")
        r["tree-32"] = {"kept": base, "levels": blv}
        for n_tree in (7, 15, 23):
            for n_min in (8, 4):
                for ub in (False, True):
                    k, lv, fired, nck = run(lambda X, p, n_tree=n_tree, n_min=n_min, ub=ub: span_arm(X, p, n_tree, n_min, ub))
                    nm = f"span N_tree {n_tree:2d} n_min {n_min}{' UB' if ub else ''}"
                    r[nm] = {"kept": k, "levels": lv, "span_fired_share": fired / max(nck, 1), "ratio": k / base}
                    say(f"  {nm:32s}  kept {k:7.3f}  x{k / base:.3f}   head levels {lv:5.2f}   span fired on {100 * fired / max(nck, 1):4.1f} % of checks")
        for n_tree in (7, 15, 23):
            k, lv, _, _ = run(lambda X, p, n_tree=n_tree: span_arm(X, p, n_tree, 99, False))
            r[f"tree-only depth<=6 N {n_tree}"] = {"kept": k, "levels": lv, "ratio": k / base}
            say(f"  tree only, depth <= 6, N {n_tree:2d}         kept {k:7.3f}  x{k / base:.3f}   head levels {lv:5.2f}   (no span)")
        best = max((v["kept"], k) for k, v in r.items() if k.startswith("span") and not k.endswith("UB"))
        res[s] = {"arms": r, "best": best[1], "best_ratio": best[0] / base}
    he, mb = res["he"], res["mbpp"]
    mbv = res["mbpp"]["arms"].get(he["best"], {}).get("ratio", 0)
    v = "PASS" if he["best_ratio"] >= 0.90 else ("KILL" if he["best_ratio"] < 0.85 or mbv < 0.75 else "BETWEEN")
    say(f"\nLINE (HE+ >= 0.90x tree-32 kept at <= 7 levels; KILL < 0.85x HE+ or < 0.75x MBPP): {v} "
        f"(HE+ best x{he['best_ratio']:.3f} = {he['best']}; MBPP at that arm x{mbv:.3f}; MBPP best x{mb['best_ratio']:.3f})")
    json.dump(res, open(OUT + ".json", "w"), indent=1)
    open(OUT + ".txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
