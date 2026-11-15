"""LOOKUP x TREE-32 upper bound (NewIdeas, Sun 15 Nov 22:1x). CPU only. At tree-32's first uncovered depth A on the true
path (31 best-first pops), the context lookup (CTX + DS of lookup_screen.py) proposes the token at A from the true
prefix. By definition the true token there is not a selected child, so a right proposal is a NEW node.
  LEAF UB: +1 accepted token (the lookup node as a leaf), no node cost;
  CONT UB: the head's chain continues below it (cells from A + 1, capped at 31 total depth), no node cost.
LINE (written before the run): KILL for K* if LEAF UB < +3 % kept on HE+ (the budget-correct design can only be lower);
if CONT UB >= +3 % but LEAF < +3 %, the lookup needs head expansion under its node (a drafting-loop change) -> report."""
import os, sys
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/NewIdeas/research/week98"))
import tree_audit as TA
import lookup_screen as LS
TA.say = lambda s="": None; LS.say = lambda s="": None
best = LS.build_ds()
for s, (tag, rtag, pub32, pub16) in TA.SETS.items():
    W = TA.load_windows(tag, pub32)
    tz = np.load(f"{TA.CE}/tree_t30b_{tag}.npz", allow_pickle=True)
    fi, an, nd, dp, ot = (tz[k] for k in ("fi", "an", "node", "depth", "on_true"))
    K = 31; hit = ot & (nd < K); acc = {}
    for f, a_, d in zip(fi[hit], an[hit], dp[hit]):
        acc[(int(f), int(a_))] = max(acc.get((int(f), int(a_)), 0), int(d) + 1)
    res = {"tree-32": [], "LEAF UB": [], "CONT UB": []}; resc = [0, 0]
    for X in W:
        z = np.load(os.path.join(LS.BANK, X["file"]), allow_pickle=True)
        ids = np.asarray(z["ids"], np.int64); tk0 = np.asarray(z["tk"][:, 0], np.int64)
        pr = LS.ctx_props(ids, LS.NS); pr = np.where(pr >= 0, pr, LS.ds_props(ids, best))
        for arm in res:
            p = 1
            while p < X["n"]:
                a = acc.get((X["fi"], int(X["an"][p])), 0) if p < len(X["an"]) else 0
                a = min(a, K)
                if arm != "tree-32" and a < K and X["rt"][p, a] >= 0:
                    row = int(X["RW"][p, a])
                    if pr[row] >= 0 and pr[row] == tk0[row]:
                        if arm == "LEAF UB":
                            a = a + 1
                        else:
                            a = TA.lead_right(X["rt"][p], a + 1, K)
                        if arm == "LEAF UB":
                            resc[0] += 1
                    if arm == "LEAF UB":
                        resc[1] += 1
                res[arm].append(a + 1); p += a + 1
    t = np.mean(res["tree-32"])
    print(f"{s}: tree-32 kept {t:.3f}; LEAF UB {np.mean(res['LEAF UB']):.3f} (x{np.mean(res['LEAF UB']) / t:.3f}); "
          f"CONT UB {np.mean(res['CONT UB']):.3f} (x{np.mean(res['CONT UB']) / t:.3f}); lookup rescues {resc[0]} of {resc[1]} tree misses "
          f"({100 * resc[0] / max(resc[1], 1):.1f} %)")
