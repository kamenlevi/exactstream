"""TREE3 export (MTP Head, Tue 11-03; the Breakthrough Push, staged verification), CPU: one aligned file per decision-map
window with the top-3 tree of both drafters at every anchor and depth, and the 125B's token. Inputs: DFlash2 L1-20 at
every position (data/d02b_walk_tree3_<set>.npz: 16 candidates + the teacher-forced selector softmax, own picks + cf), the
MTP head b5 (data/tree3/mtp_<set>.npz, sticky/tree3_mtp.py), the 125B's top-8 per answer row (the map's exact run
~/FableGod/decision_map/out/baseline.json), and the map's tokens (~/FableGod/decision_map/out/routing/<window>.npz; asserted equal). Output: data/tree3_map/<set>/<window>.npz
+ data/tree3_map/README.txt; a coverage summary appended to data/tree3_read.txt."""
import os, glob
import numpy as np
TD = os.path.expanduser("~/TrainLab/data")
MAP = os.path.expanduser("~/FableGod/decision_map/out/routing")
OUT = os.path.join(TD, "tree3_map")
J = 15
README = """TREE3 (MTP Head, Tue 11-03, for the Breakthrough Push): the top-3 draft tree of DFlash2 L1-20 and of the native MTP head
b5 on the 36 decision-map windows (code = 12 HumanEval+, ifeval = 12, multi = 12), ALIGNED to the map.

Positions are the map's: token index t in `tokens` (== the map's routing/<window>.npz tokens); map row t is the 125B's row
at position t and predicts token t + 1. A check at anchor A (the last known token) drafts depth j = 0..14 = the token at
A + 1 + j = map row A + j. Every position of the answer is an anchor here (`A`), so any loop can be replayed.

Per window, arrays over anchors (n) x depth (15) [x 3]:
  A [n] anchor positions; m [n] depths with a token (<= 15); true [n, 15] the 125B's token at A + 1 + j (-1 past the end)
  dfl_t3 / dfl_p3 [n, 15, 3]  DFlash2's top-3 at (A, j) by its selector softmax over its 16 candidates, teacher-forced
                              (the predecessor is the true token at j - 1), descending; dfl_rank [n, 15] the true token's
                              rank among the 16 (16 = not a candidate); dfl_own [n, 15] / dfl_own_cf the drafter's own
                              chain (what the live engine drafts; equal to the teacher-forced pick up to the first miss)
  mtp_t3 / mtp_p3 [n, 15, 3]  the MTP head's top-3 (f32 softmax over the full vocab), teacher-forced; its chain at a check
                              anchored at A starts from the trunk's hidden at A - 1 (D-24 row A - 1 - a0, depth j + 1);
                              mtp_rank [n, 15] the true token's rank capped at 3 (-1 = no data)
  t125_t3 / t125_p3 [n, 15, 3] the 125B's own top-3 at map row A + j (the map's exact run baseline.json; -1 outside its rows)
Teacher-forced = exact along the true path up to the first miss, which is all a check's kept length depends on.
Built by sticky/tree3_mtp.py, d02b_walk_match.py (TL_WALK_EVERY + TL_WALK_TOP16), sticky/tree3_export.py (turns/tree3_job.sh).
"""
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, "README.txt"), "w").write(README)
import json
BASE = json.load(open(os.path.expanduser("~/FableGod/decision_map/out/baseline.json")))
say("TREE3 coverage (every anchor, depths 0-14, teacher-forced): the 125B's token in each drafter's top-1 / top-3")
for st in ("code", "ifeval", "multi"):
    fw, fm = os.path.join(TD, f"d02b_walk_tree3_{st}.npz"), os.path.join(TD, "tree3", f"mtp_{st}.npz")
    if not (os.path.exists(fw) and os.path.exists(fm)):
        say(f"  {st}: tables missing"); continue
    Wz, Mz = np.load(fw), np.load(fm)
    mk = {str(k): i for i, k in enumerate(Mz["keys"])}
    os.makedirs(os.path.join(OUT, st), exist_ok=True)
    acc = np.zeros((2, 8))                                   # [depth-0 / all] x [n, d1, d3, m1, m3, u1, u3, t125==true]
    for w, nm in enumerate(str(x) for x in Wz["names"]):
        key = nm[:-4]; mi = mk[key]
        ids, off, ck = Wz[f"{w}_ids"].astype(np.int64), int(Wz[f"{w}_off"]), Wz[f"{w}_checks"]
        mp = np.load(os.path.join(MAP, key + ".npz")); toks = mp["tokens"].astype(np.int64)
        assert off == 0 and np.array_equal(ids[:len(toks)], toks), (st, key, off)
        bb = BASE[key + ".npz"]; ba = int(bb["a"]); tk, lp = np.array(bb["tk"]), np.array(bb["lp"], np.float32)
        a0 = int(Mz["a0"][mi]); t3m, p3m, rkm = Mz[f"t3_{mi}"], Mz[f"p3_{mi}"], Mz[f"rk_{mi}"]
        n = len(ck)
        A = ck[:, 2].astype(np.int64); m = np.minimum(ck[:, 1], J).astype(np.int64)
        true = np.full((n, J), -1, np.int64)
        dfl_t3 = np.full((n, J, 3), -1, np.int32); dfl_p3 = np.zeros((n, J, 3), np.float16); dfl_rank = np.full((n, J), 16, np.int8)
        mtp_t3 = np.full((n, J, 3), -1, np.int32); mtp_p3 = np.zeros((n, J, 3), np.float16); mtp_rank = np.full((n, J), -1, np.int8)
        t125_t3 = np.full((n, J, 3), -1, np.int32); t125_p3 = np.zeros((n, J, 3), np.float16)
        cand, tsm = Wz[f"{w}_cand"], Wz[f"{w}_tsm"].astype(np.float32)
        for c in range(n):
            a = int(A[c])
            for j in range(int(m[c])):
                t = a + 1 + j
                if t >= len(ids):
                    break
                true[c, j] = ids[t]
                o = np.argsort(-tsm[c, j], kind="stable")
                cs = cand[c, j][o]
                dfl_t3[c, j] = cs[:3]; dfl_p3[c, j] = tsm[c, j][o[:3]]
                hit = np.nonzero(cs == ids[t])[0]
                dfl_rank[c, j] = int(hit[0]) if len(hit) else 16
                r = a - 1 - a0                               # the MTP chain at this check: the trunk's hidden at A - 1
                if 0 <= r < t3m.shape[0] and rkm[r, j] >= 0:
                    mtp_t3[c, j] = t3m[r, j]; mtp_p3[c, j] = p3m[r, j]; mtp_rank[c, j] = rkm[r, j]
                row = a + j - ba                              # map row A + j (predicting token t) in the run's rows
                if 0 <= row < len(tk):
                    ob = np.argsort(-lp[row], kind="stable")[:3]
                    t125_t3[c, j] = tk[row][ob]; t125_p3[c, j] = np.exp(lp[row][ob])
        np.savez(os.path.join(OUT, st, key + ".npz"), tokens=toks, gen=np.int64(mp["gen"]), A=A, m=m, true=true,
                 dfl_t3=dfl_t3, dfl_p3=dfl_p3, dfl_rank=dfl_rank, dfl_own=Wz[f"{w}_dtok"].astype(np.int32),
                 dfl_own_cf=Wz[f"{w}_dcf"].astype(np.float16), mtp_t3=mtp_t3, mtp_p3=mtp_p3, mtp_rank=mtp_rank,
                 t125_t3=t125_t3, t125_p3=t125_p3)
        ok = (true >= 0) & (mtp_rank >= 0) & (t125_t3[..., 0] >= 0)
        for k, sel in ((0, ok & (np.arange(J)[None] == 0)), (1, ok)):
            d1 = dfl_rank == 0; d3 = dfl_rank <= 2; m1 = mtp_rank == 0; m3 = mtp_rank <= 2
            acc[k] += [sel.sum(), (d1 & sel).sum(), (d3 & sel).sum(), (m1 & sel).sum(), (m3 & sel).sum(), ((d1 | m1) & sel).sum(),
                       ((d3 | m3) & sel).sum(), ((t125_t3[..., 0] == true) & sel).sum()]
    for k, nm_ in ((0, "depth 0"), (1, "all depths")):
        n_, d1, d3, m1, m3, u1, u3, tt = acc[k]
        say(f"  {st:6s} {nm_:10s} ({int(n_)} rows): DFlash2 top-1 {d1 / n_:.3f} / top-3 {d3 / n_:.3f}; MTP top-1 {m1 / n_:.3f} / top-3 "
            f"{m3 / n_:.3f}; either top-1 {u1 / n_:.3f} / top-3 {u3 / n_:.3f}; the 125B's top-1 == the token {tt / n_:.3f}")
open(os.path.join(TD, "tree3_read.txt"), "a").write("\n".join(lines) + "\n\n")
