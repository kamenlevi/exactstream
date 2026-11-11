"""RERANK-0 (MTP Head, Wed 11-04 15:09; pre-registered in REPORT 'RERANK-0'; the orchestrator's item 1 on Kamen's 300 tok/s
FAST priority): how much kept per check a better pick among DFlash2 L1-20's 16 candidates could give, at the same anchors.
Data: the TOP16 walks data/d02b_walk_kw_{set}.npz (HPD-0's anchors, block 16): per window ids / off and per check (k, m, A)
= (accepted drafts, depths available, anchor), cand [15, 16] and the teacher-forced selector softmax tsm [15, 16].
Per check: rank of the 125B's token among the candidates by tsm at each depth (1..16, 0 = not a candidate); kept = 1 + the
leading run of depths with rank in 1..r, for r = 1 (the official pick; self-check vs k + 1), 2, 3, 16.
Also the rank of the 125B's token at the official first miss. CPU only. Output: data/rerank0.txt (appended)."""
import os, time
import numpy as np
TD = os.path.expanduser("~/TrainLab/data")
RS = (1, 2, 3, 16)
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)


say(f"RERANK-0 {time.strftime('%m-%d %H:%M')}: kept per check at HPD-0's anchors if the pick were right whenever the 125B's "
    f"token is in DFlash2 L1-20's top-r candidates (teacher-forced selector order), block 16")
for s in ("code", "ifeval", "multi"):
    Z = np.load(os.path.join(TD, f"d02b_walk_kw_{s}.npz"))
    kw, ko = [], {r: [] for r in RS}
    fm = np.zeros(17, np.int64)                          # rank of the 125B's token at the official first miss (0 = absent)
    bad = tie = 0
    for w in range(len(Z["names"])):
        ids, off = Z[f"{w}_ids"], int(Z[f"{w}_off"])
        ck, cand, tsm, ttok = Z[f"{w}_checks"], Z[f"{w}_cand"], Z[f"{w}_tsm"].astype(np.float32), Z[f"{w}_ttok"]
        for c, (k, m, A) in enumerate(ck):
            k, m, A = int(k), int(m), int(A)
            D = min(15, m, len(ids) - (A - off) - 1)
            t = ids[A - off + 1:A - off + 1 + D]
            order = np.argsort(-tsm[c, :D], axis=1, kind="stable")
            ranked = np.take_along_axis(cand[c, :D], order, axis=1)
            tie += int(np.sum(ranked[:, 0] != ttok[c, :D].reshape(-1)[:D]))   # float16 ties vs the walk's own argmax
            hit = ranked == t[:, None]
            rank = np.where(hit.any(1), hit.argmax(1) + 1, 0)
            for r in RS:
                ok = (rank >= 1) & (rank <= r)
                ko[r].append((D if ok.all() else int(np.argmin(ok))) + 1)
            kw.append(k + 1)
            bad += int(ko[1][-1] != k + 1)
            if not (rank == 1).all():
                fm[rank[int(np.argmin(rank == 1))]] += 1
    kw = np.array(kw, np.float64); base = np.mean(ko[1])
    say(f"  {s}: {len(Z['names'])} windows, {len(kw)} checks; walk kept {kw.mean():.2f} per check, official pick re-simulated "
        f"{base:.2f} (differs on {bad} checks; top-1 vs the walk's own pick differs at {tie} depths: float16 ties)")
    say("    oracle kept per check: " + ", ".join(f"top-{r} {np.mean(ko[r]):.2f} (x{np.mean(ko[r]) / base:.3f})" for r in RS))
    n = fm.sum()
    say(f"    rank of the 125B's token at the official first miss ({n} misses): 2: {100 * fm[2] / n:.1f} %, 3: "
        f"{100 * fm[3] / n:.1f} %, 4-16: {100 * fm[4:].sum() / n:.1f} %, not a candidate: {100 * fm[0] / n:.1f} %")
open(os.path.join(TD, "rerank0.txt"), "a").write("\n".join(lines) + "\n\n")
