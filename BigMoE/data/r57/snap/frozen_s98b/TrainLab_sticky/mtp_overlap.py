"""MTP-vs-DFlash2 overlap (MTP Head, Tue 11-03; information after DUALDRAFT / DIVERSE2, CPU): at HPD-0's anchors and depths,
both teacher-forced from the true prefix: DFlash2 L1-20's pick right (data/d02b_walk_hpd0_<set>.npz ttok) and the native
MTP head b5's draft right (data/d24/<set>_b5.npz rank 0, depths 1-15). q = P(MTP wrong | DFlash2 wrong), either-right, the
higher-confidence pick (DFlash2's tcf vs MTP's top-1 p), and kept per 16-row check at the same anchors (DFlash2 alone, MTP
alone, the pick, the oracle either). Output: data/mtp_overlap.txt (appended).
FIX (Tue 11-03, found after the 15:15 commit): D-24's row r is the MTP anchor s = a0 + r, whose depth c + 1 predicts ids[s + c + 2] (d22_ttt.chain):
for a DFlash2 check at anchor an (its depth-0 draft predicts ids[an + 1]) the live MTP chain is s = an - 1, so r = an - 1 - a0.
The first version used r = an - a0 (one position later, and the trunk's hidden at an, which the live loop does not have yet):
it compared different positions and made the two drafters look diverse (q 0.24-0.30; aligned 0.54-0.62 at depth 1)."""
import os
import numpy as np
TD = os.path.expanduser("~/TrainLab/data")
SETS = {"code": "code", "math": "gsm", "multi": "multi", "ifeval": "ifeval"}
SH = 1                                                   # r = an - SH - a0 (see FIX)
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
say("MTP vs DFlash2 overlap, ALIGNED (r = an - 1 - a0; teacher-forced at HPD-0's anchors, depths 1-15; MTP = b5 from D-24)")
kp = {}
for st, m24 in SETS.items():
    A = np.load(os.path.join(TD, f"d02b_walk_hpd0_{st}.npz")); M = np.load(os.path.join(TD, "d24", f"{m24}_b5.npz"))
    mk = {str(k): i for i, k in enumerate(M["keys"])}
    n = dw = mw = both = ei = pk = dr = 0
    K = {a: [] for a in ("dfl", "mtp", "pick", "either")}
    for w, nm in enumerate(str(x) for x in A["names"]):
        mi = mk.get(nm[:-4])
        if mi is None:
            continue
        ids, off = A[f"{w}_ids"], int(A[f"{w}_off"]); a0 = max(0, int(M["gen"][mi]) - 2)
        rk, p1 = M[f"rk_{mi}"], M[f"p1_{mi}"]
        for c, (k, m, an) in enumerate(A[f"{w}_checks"]):
            r = int(an) - SH - a0
            if r < 0 or r >= rk.shape[0]:
                continue
            first = {a: None for a in K}
            L = 0
            for j in range(min(int(m), 15)):
                if rk[r, j] < 0:
                    break
                L = j + 1
                t = int(ids[int(an) - off + 1 + j])
                d_ok = int(A[f"{w}_ttok"][c, j]) == t; m_ok = int(rk[r, j]) == 0
                p_ok = d_ok if A[f"{w}_tcf"][c, j] >= p1[r, j] else m_ok
                n += 1; dr += d_ok; dw += not d_ok; mw += not m_ok; both += (not d_ok) and (not m_ok); ei += d_ok or m_ok; pk += p_ok
                for a, ok in (("dfl", d_ok), ("mtp", m_ok), ("pick", p_ok), ("either", d_ok or m_ok)):
                    if first[a] is None and not ok:
                        first[a] = j
            for a in K:                                  # kept = right drafts before the first wrong one + the bonus
                K[a].append((L if first[a] is None else first[a]) + 1)
    if n:
        say(f"  {st:6s}: {n} positions; DFlash2 right {dr / n:.3f}, MTP right {1 - mw / n:.3f}; q = P(MTP wrong | DFlash2 wrong) "
            f"{both / max(dw, 1):.3f} (DFlash2 twins: 0.87-0.90); either right {ei / n:.3f}; higher-confidence pick {pk / n:.3f}")
        kp[st] = {a: float(np.mean(v)) for a, v in K.items()}; kp[st]["n"] = len(K["dfl"])
say("Kept per 16-row check at HPD-0's anchors (teacher-forced, exact up to the first miss): DFlash2 / MTP / pick / oracle either")
for st, d in kp.items():
    say(f"  {st:6s} ({d['n']} checks): DFlash2 {d['dfl']:.2f}, MTP {d['mtp']:.2f}, pick {d['pick']:.2f} (x{d['pick'] / d['dfl']:.3f}), "
        f"either (oracle) {d['either']:.2f}")
open(os.path.join(TD, "mtp_overlap.txt"), "a").write("\n".join(lines) + "\n\n")
