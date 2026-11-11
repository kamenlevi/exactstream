"""The adaptive certificate (Breakthrough, Mon 2026-11-02 20:2x; cloud Fable's 12 GB v3 via the orchestrator): theta lo
where the drafter is sure AND its draft equals pass 1's argmax, theta hi elsewhere. Realizable: the drafter's cf and draft
are known before the check, pass 1's argmax and p1 after it.
Per row, the draft it verifies: row A + j (j = 0 .. accepted k, j < m) verifies draft j + 1, its token dtok[j] and the
drafter's cf dcf[j] (MTP Head's HPD-0 walk of DFlash2 L1-20, ~/TrainLab/data/d02b_walk_hpd0_<dom>.npz, as dm_gate.py);
prompt rows and a full check's bonus row have none (token -1, cf nan), so only theta hi can certify them.
On an accepted row the draft is the exact token, so "agrees" means pass 1 is right there; on a check's first-miss row
the draft is wrong, so "agrees" means pass 1 and the drafter are wrong the same way: the arm's risk."""
import os
import numpy as np
import dm_gate
_D = {}

def drafts():
    """{window: (draft token per row 0 .. n - 2, the drafter's cf per row)}"""
    if not _D:
        for dom, wf in dm_gate.WALK.items():
            Z = np.load(os.path.join(dm_gate.TD, wf))
            for i, nm in enumerate(str(x) for x in Z["names"]):
                n = len(Z[f"{i}_ids"]); tok = np.full(n - 1, -1, np.int64); cf = np.full(n - 1, np.nan)
                for (k, m, A), dt, dc in zip(Z[f"{i}_checks"], Z[f"{i}_dtok"], Z[f"{i}_dcf"]):
                    for j in range(min(k + 1, m)):
                        if A + j >= n - 1: break
                        tok[A + j] = dt[j]; cf[A + j] = dc[j]
                _D[nm] = (tok, cf)
    return _D

def acert(nm, a, p1, a1, lo, hi, s):
    """certified mask over the measured rows a .. a + len(p1) - 1 (p1, a1 = pass 1's top-1 probability and argmax)"""
    tok, cf = drafts()[nm]; r = slice(a, a + len(p1))
    return (p1 >= hi) | ((p1 >= lo) & (np.nan_to_num(cf[r], nan=-1.0) >= s) & (tok[r] == a1))
