"""D1 with real router probabilities (Breakthrough, 2026-10-20). 125B re-trace (data/expA/traces125/*_800_g.npz:
L<l> = router top-20 ids, G<l> = their softmax probabilities over 512) of code/chat/Bulgarian/math decode tokens.
(1) gate-mass shape: share of the top-10 renormalized mass in rank 0, ranks 0-1, ranks 0-3.
(2) frugal policies per layer (LRU, strict capacity C; only waited experts are read from the SSD; a missing unwaited
    rank uses the best-ranked resident expert among ranks 10-19, else is dropped): loads/token and the "missed mass" =
    renormalized top-10 gate mass whose expert was not computed (substituted or dropped), mean and P99 per token-layer.
    Policies: wait-top-r (r = 1, 2, 3) and wait-until-mass >= theta (theta = 0.5 .. 0.9).
CPU only (one core, ~1-2 min); run it when nobody holds a quiet lock.
"""
import glob
import numpy as np
from collections import OrderedDict

NL = 48
sets = {f.split("/")[-1].split("_")[0]: np.load(f)
        for f in sorted(glob.glob("/Users/kamen/Breakthrough/data/expA/traces125/*_800_g.npz"))}
ids = {k: np.stack([d[f"L{l}"] for l in range(NL)], 1).astype(np.int32) for k, d in sets.items()}
gp = {k: np.stack([d[f"G{l}"] for l in range(NL)], 1).astype(np.float32) for k, d in sets.items()}
w = {k: g[:, :, :10] / g[:, :, :10].sum(-1, keepdims=True) for k, g in gp.items()}


def nwait(wt, rule):
    if rule[0] == "r":
        return int(rule[1:])
    return int(np.argmax(np.cumsum(wt) >= float(rule[1:]) - 1e-6)) + 1


def run(C, rule):
    loads, missed, ntok = 0, [], 0
    for k in ids:
        tr, wt = ids[k], w[k]
        ntok += tr.shape[0]
        for l in range(NL):
            c = OrderedDict()
            for t in range(tr.shape[0]):
                row, nw = tr[t, l], nwait(wt[t, l], rule)
                alt = [int(e) for e in row[10:20] if int(e) in c]
                used, miss = [], 0.0
                for r in range(10):
                    e = int(row[r])
                    if r < nw or e in c:
                        if e not in c:
                            loads += 1
                        used.append(e)
                    else:
                        miss += wt[t, l, r]
                        a = next((x for x in alt if x not in used), None)
                        if a is not None:
                            used.append(a)
                            alt.remove(a)
                missed.append(miss)
                for e in used:
                    if e in c:
                        c.move_to_end(e)
                    else:
                        c[e] = 1
                while len(c) > C:
                    c.popitem(last=False)
    m = np.array(missed)
    return loads / ntok, m.mean(), np.percentile(m, 99)


if __name__ == "__main__":
    allw = np.concatenate([x.reshape(-1, 10) for x in w.values()])
    cum = np.cumsum(allw, -1)
    print(f"(1) {len(allw)} token-layers: mass in rank 0 {cum[:, 0].mean():.3f}, ranks 0-1 {cum[:, 1].mean():.3f}, "
          f"0-3 {cum[:, 3].mean():.3f}; ranks needed for 50/70/90 % of the mass: "
          + " / ".join(f"{(np.argmax(cum >= t, -1) + 1).mean():.2f}" for t in (0.5, 0.7, 0.9)))
    print("(2) loads/token (MB at 2.77 MB/expert) | missed mass mean / P99 per token-layer")
    for C in (51, 102):
        print(f"C={C} ({100 * C / 512:.0f} %)")
        for rule in ("r1", "r2", "r3", "m0.5", "m0.6", "m0.7", "m0.8", "m0.9"):
            L_, mm, p99 = run(C, rule)
            print(f"  {rule:5s} loads {L_:6.1f} ({L_ * 2.77:5.0f} MB)  missed mass {mm:.3f} / P99 {p99:.3f}", flush=True)
