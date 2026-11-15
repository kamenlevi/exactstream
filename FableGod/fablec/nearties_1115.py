"""THINK_1115 #4, tiny CPU check on the old map data (DFlash2 HPD-0 walks, 36 HumanEval / IFEval / multi windows; the 125B's
top-8 log-probs per answer row in decision_map/out/baseline.json). Near-tie = the 125B's top-1 vs top-2 margin < 2 nats.
(a) near-tie share: all answer rows vs the walk's check-start rows vs bands 0-6 after a start; (b) P(near-tie at t+1 | at t);
(c) RECYCLE: after a miss at row A+k (wrong draft d_{k+1}), how often the drafter's already-drafted d_{k+2}, d_{k+3} equal the
true tokens at rows A+k+1, A+k+2 (the next check's band 0 / 1 targets), vs the fresh next check's own first draft."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/FableGod/decision_map")); sys.path.insert(0, os.path.expanduser("~/FableGod/fablec"))
import dm_gate
from bytes10_replay import windows
base = json.load(open(os.path.join(dm_gate.OUT, "baseline.json")))
W = windows()
for dom, ws in W.items():
    Z = np.load(os.path.join(dm_gate.TD, dm_gate.WALK[dom])); names = [str(x) for x in Z["names"]]
    nt_all = []; nt_start = []; nt_band = {b: [] for b in range(7)}; nt_miss = []; auto = [0, 0, 0, 0]
    rec = {0: [0, 0], 1: [0, 0]}; fresh = {0: [0, 0], 1: [0, 0]}; both = [0, 0]
    for w in ws:
        nm = w["nm"]; i = names.index(nm); b = base[nm]; a = int(b["a"]); tk = np.array(b["tk"]); lp = np.array(b["lp"]); N = a + len(tk)
        marg = lp[:, 0] - lp[:, 1]; nt = marg < 2.0; nt_all += nt.tolist()
        for t in range(len(nt) - 1): auto[2 * int(nt[t]) + int(nt[t + 1])] += 1
        checks = Z[f"{i}_checks"].tolist(); dtok = Z[f"{i}_dtok"]
        for ci, (k, m, A) in enumerate(checks):
            r0 = A - a                                       # the check's first row as an answer-row index (A = anchor row)
            if 0 <= r0 < len(nt): nt_start.append(bool(nt[r0]))
            for bnd in range(7):
                if 0 <= r0 + bnd < len(nt): nt_band[bnd].append(bool(nt[r0 + bnd]))
            if k < m:                                        # a miss at row A + k: the drafter's d_{k+1} was wrong
                rm = r0 + k
                if 0 <= rm < len(nt): nt_miss.append(bool(nt[rm]))
                for d in (0, 1):                             # recycled: d_{k+2+d} vs the true token at row A+k+1+d
                    r = rm + 1 + d; j = k + 1 + d
                    if 0 <= r < len(tk) and j < dtok.shape[1]:
                        rec[d][0] += int(dtok[ci, j] == tk[r, 0]); rec[d][1] += 1
                if ci + 1 < len(checks):                     # the fresh next check's own drafts at the same rows
                    k2, m2, A2 = checks[ci + 1]
                    if A2 == A + k + 1:
                        for d in (0, 1):
                            r = rm + 1 + d
                            if 0 <= r < len(tk) and d < dtok.shape[1]:
                                fresh[d][0] += int(dtok[ci + 1, d] == tk[r, 0]); fresh[d][1] += 1
                        r = rm + 1; j = k + 1
                        if 0 <= r < len(tk) and j < dtok.shape[1]:
                            both[1] += 1; both[0] += int(dtok[ci, j] == tk[r, 0] or dtok[ci + 1, 0] == tk[r, 0])
    p = lambda v: 100 * np.mean(v) if len(v) else float("nan")
    print(f"{dom}: near-tie (< 2 nats) share: all rows {p(nt_all):.1f} % ({len(nt_all)}); check-start rows {p(nt_start):.1f} % ({len(nt_start)}); "
          f"miss rows {p(nt_miss):.1f} % ({len(nt_miss)}); bands after a start 0..6: " + " ".join(f"{p(nt_band[b_]):.0f}" for b_ in range(7)))
    print(f"   P(near-tie at t+1 | at t) {100 * auto[3] / max(auto[2] + auto[3], 1):.1f} % vs P(near-tie at t+1 | not at t) {100 * auto[1] / max(auto[0] + auto[1], 1):.1f} %")
    print(f"   RECYCLE: the drafter's already-drafted d_(k+2) equals the true token after the correction {100 * rec[0][0] / max(rec[0][1], 1):.1f} % ({rec[0][1]}); d_(k+3) at band 1 {100 * rec[1][0] / max(rec[1][1], 1):.1f} %; "
          f"the fresh next check's own first draft {100 * fresh[0][0] / max(fresh[0][1], 1):.1f} % ({fresh[0][1]}), second {100 * fresh[1][0] / max(fresh[1][1], 1):.1f} %; recycled OR fresh at band 0 {100 * both[0] / max(both[1], 1):.1f} % ({both[1]})")
