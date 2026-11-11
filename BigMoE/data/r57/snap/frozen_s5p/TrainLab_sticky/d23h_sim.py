"""DEPTH32 step 3 (MTP Head, Mon 11-02; REPORT 'DEPTH32'), CPU: the official loop at 32 rows (31 drafts) on stored tables.
Draft at anchor A: depths 1-8 = DFlash2 L1-20's own picks at A (data/d02b_walk_d23h_dfl_<set>.npz: TL_WALK_EVERY, a check at
every position), depths 9-31 = the MTP head b5's argmax from A (data/d23h/<set>_b5.npz, teacher-forced from the true prefix,
which equals the real chain up to the first miss: the only part that decides kept). Arms: MTP-32 (MTP at 1-31), HYB (DFlash2
1-8 + MTP 9-31), and both with COPYGATE's gate (~/FableGod/copygate/copygate.py, mode copygate) on the whole draft. The check
accepts the longest right prefix; the next anchor is A + k + 1. Output: data/d23h_read.txt (appended)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/FableGod/copygate"))
import copygate                                            # read-only use (the copygate session's gate; reviewed 11-02)
TD = os.path.expanduser("~/TrainLab/data")
REF = {"code": 10.26, "mbpp": 6.00}                         # DFlash2 at 32 rows, the real walk (MATCH's m_base32)
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

def simulate(ids, gen, dfl, mtp, a0, arm, gate):
    """dfl: {A: [8 own picks]}; mtp: (am [n, 31], rt [n, 31]) from anchor a0; returns the kept list"""
    am, rt = mtp
    cg = copygate.CopyGate(list(map(int, ids)), gen, "copygate") if gate else None
    A, kept, n = gen, [], len(ids)
    while A < n - 1:
        r = A - 1 - a0                                     # MTP anchor s = A - 1 predicts A + 1 at depth 1 (fixed 11-03: was A - a0)
        if r < 0 or r >= am.shape[0]:
            break
        m = min(31, n - 1 - A)
        mt = [int(x) for x in am[r, :m]]
        if arm == "hyb":
            if A not in dfl:
                break
            dr = [int(x) for x in dfl[A][:min(8, m)]] + mt[8:m]
        else:
            dr = mt
        dr = dr[:m]
        true = [int(x) for x in ids[A + 1:A + 1 + m]]
        d0 = list(dr)
        if cg is not None:
            dr, hit, info = cg.propose(list(dr), A)
        k = 0
        while k < len(dr) and dr[k] == true[k]:
            k += 1
        if cg is not None:
            cg.learn(d0, info, true, k)
        kept.append(k + 1)
        A = A + k + 1
    return kept

res = {}
for st in ("code", "mbpp"):
    fw, fm = os.path.join(TD, f"d02b_walk_d23h_dfl_{st}.npz"), os.path.join(TD, "d23h", f"{st}_b5.npz")
    if not (os.path.exists(fw) and os.path.exists(fm)):
        say(f"  {st}: tables missing"); continue
    Wz, Mz = np.load(fw), np.load(fm)
    mk = {str(k): i for i, k in enumerate(Mz["keys"])}
    acc = {a: [] for a in ("mtp", "mtp+g", "hyb", "hyb+g")}
    for w, nm in enumerate(str(x) for x in Wz["names"]):
        mi = mk.get(nm[:-4])
        if mi is None:
            continue
        ids, off, gen = Wz[f"{w}_ids"], int(Wz[f"{w}_off"]), int(Wz[f"{w}_gen"])
        assert off == 0, nm
        mids = Mz[f"am_{mi}"]
        dfl = {int(A): Wz[f"{w}_dtok"][c] for c, (k, m, A) in enumerate(Wz[f"{w}_checks"])}
        mtp = (Mz[f"am_{mi}"], Mz[f"rt_{mi}"]); a0 = int(Mz["a0"][mi])
        for arm, g in (("mtp", False), ("mtp+g", True), ("hyb", False), ("hyb+g", True)):
            acc[arm] += simulate(ids, gen, dfl, mtp, a0, arm.split("+")[0], g)
    res[st] = {a: float(np.mean(v)) if v else float("nan") for a, v in acc.items()}
    say(f"  {st}: kept per 32-row check: DFlash2-32 {REF[st]:.2f} (real walk), MTP-32 {res[st]['mtp']:.2f} (+G {res[st]['mtp+g']:.2f}), "
        f"HYB {res[st]['hyb']:.2f} (+G {res[st]['hyb+g']:.2f}); checks HYB {len(acc['hyb'])}")
if {"code", "mbpp"} <= set(res):
    best = max(("hyb", "hyb+g"), key=lambda a: res["code"][a])
    c, m = res["code"][best], res["mbpp"][best]
    v = "PASS" if (c >= 14 and m >= 7.5) else ("KILL" if (c < 13 or m < 6.5) else "BETWEEN")
    say(f"  LINE (the best realizable arm, {best}: kept(32) >= 14 HumanEval+ and >= 7.5 MBPP PASS; < 13 or < 6.5 KILL): "
        f"{c:.2f} / {m:.2f} -> {v}")
open(os.path.join(TD, "d23h_read.txt"), "a").write("DEPTH32 read-out (CPU simulation of the official loop at 32 rows)\n"
                                                    + "\n".join(lines) + "\n\n")
