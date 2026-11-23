"""ANCHORCOV (MTP Head, Mon 11-02; REPORT 'ANCHORCOV'), CPU: in what share of checks is the real outcome (all m drafts
accepted, or the first miss at row k) among N pre-drafted anchors = "all" + the N - 1 lowest-cf rows (cloud task 15's
design), N = 2 / 3 / 4; beside it the rows ranked by the drafter's own first-miss probability. HPD-0's walks of DFlash2
L1-20 (data/d02b_walk_hpd0_<set>.npz: checks (k, m, A), dcf = the own picks' cf). Output: data/anchorcov.txt (appended)."""
import os
import numpy as np
TD = os.path.expanduser("~/TrainLab/data")
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
say("ANCHORCOV (DFlash2 L1-20, block 16, HPD-0's walks; outcome = all accepted or the first-miss row)")
res = {}
for st in ("code", "math", "multi", "ifeval"):
    Z = np.load(os.path.join(TD, f"d02b_walk_hpd0_{st}.npz"))
    cov = {("low", N): 0 for N in (2, 3, 4)}; cov.update({("fm", N): 0 for N in (2, 3, 4)}); n = alln = 0
    for i in range(len(Z["names"])):
        for c, (k, m, A) in enumerate(Z[f"{i}_checks"]):
            k, m = int(k), int(m)
            if m < 1:
                continue
            cf = Z[f"{i}_dcf"][c, :m].astype(np.float64)
            out = "all" if k >= m else k                      # the first miss at row k (0-based)
            n += 1; alln += out == "all"
            low = list(np.argsort(cf))                          # rows by the raw cf, lowest first
            run = np.concatenate([[1.0], np.cumprod(cf)[:-1]])  # P(rows before j all right)
            fm = list(np.argsort(-(run * (1 - cf))))            # rows by the first-miss probability
            for N in (2, 3, 4):
                cov[("low", N)] += out == "all" or out in low[:N - 1]
                cov[("fm", N)] += out == "all" or out in fm[:N - 1]
    res[st] = (n, alln, cov)
    say(f"  {st:6s}: {n} checks, all accepted {alln / n:.3f}; coverage with N anchors (all + the N-1 lowest-cf rows): "
        + ", ".join(f"N={N} {cov[('low', N)] / n:.3f}" for N in (2, 3, 4))
        + "; by first-miss probability: " + ", ".join(f"N={N} {cov[('fm', N)] / n:.3f}" for N in (2, 3, 4)))
def pooled(sets, kind, N):
    a = sum(res[s][2][(kind, N)] for s in sets); b = sum(res[s][0] for s in sets)
    return a / b
c3, ch3 = res["code"][2][("low", 3)] / res["code"][0], pooled(("multi", "ifeval"), "low", 3)
say(f"  chat (multi + IFEval pooled): N=2 {pooled(('multi', 'ifeval'), 'low', 2):.3f}, N=3 {ch3:.3f}, N=4 {pooled(('multi', 'ifeval'), 'low', 4):.3f}; "
    f"by first-miss probability N=3 {pooled(('multi', 'ifeval'), 'fm', 3):.3f}")
v = "PASS" if (c3 >= 0.50 and ch3 >= 0.50) else "KILL"
say(f"  LINE (cloud's: coverage >= 0.50 at N = 3 on code and chat): code {c3:.3f}, chat {ch3:.3f} -> {v}")
open(os.path.join(TD, "anchorcov.txt"), "a").write("\n".join(lines) + "\n\n")
