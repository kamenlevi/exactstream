"""copyrun_groups.py (Fable FULL, LOSSY350): what a wide copy pass computes, from recorded routing.
Data [K]: the decision map's routing capture (decision_map/out/routing/<window>: ids [rows, 48 layers, 10 ranks], row p =
the row whose input is token p), the 12 code map windows, the HPD-0 walk for their tokens.
For each answer's verbatim prompt-copy run (as copyrun_replay.py): cut it into blocks of W rows and count, per block, the
distinct (expert, layer) groups of ranks 0-1 (a strict row's loads; its members add pairs, not groups) and of ranks 0-9
(exact rows); and the share of those experts the prompt's own rows already used (ranks 0-9), i.e. warm before the answer.
For comparison: the same counts on 16-row blocks of the non-copy answer rows.
The npz files are the project's own captures (allow_pickle only for the walk's names array). CPU, seconds.
Run: nice -n 15 ~/BigMoE/.venv/bin/python -I copyrun_groups.py > copyrun_groups.out"""
import json, os
import numpy as np
H = os.path.expanduser("~")
W = np.load(f"{H}/TrainLab/data/d02b_walk_hpd0_code.npz", allow_pickle=True)
B = json.load(open(f"{H}/FableGod/decision_map/out/baseline.json"))
TRIG = 8
def groups(E, k):
    """E [rows, 48, 10] -> sum over layers of distinct experts among ranks < k"""
    return sum(len(np.unique(E[:, l, :k])) for l in range(48))
res = {w: dict(n=0, g2=0, g10=0, warm2=0, tot2=0, rows=0) for w in (16, 32, 64, 128)}
non = dict(n=0, g2=0, g10=0, rows=0); nwin = 0
for wi, nm in enumerate(str(x) for x in W["names"]):
    if nm not in B: continue
    f = f"{H}/FableGod/decision_map/out/routing/{nm}"
    if not os.path.exists(f): continue
    R = np.load(f)["ids"].astype(np.int32)
    ids = W[f"{wi}_ids"].astype(int).tolist(); ck = W[f"{wi}_checks"]
    g0 = int(ck[0, 2]) + 1; P = ids[:g0]; A = ids[g0:]
    if len(R) < len(ids) - 1: A = A[:max(0, len(R) - g0)]
    pos = {}
    for i, t in enumerate(P): pos.setdefault(t, []).append(i)
    q = 0; runs = []
    while q < len(A):
        best = 0
        for i in pos.get(A[q], ()):
            L = 0
            while q + L < len(A) and i + L < len(P) and A[q + L] == P[i + L]: L += 1
            best = max(best, L)
        if best >= 8: runs.append((q, best)); q += best
        else: q += 1
    nwin += 1
    warm = [set(np.unique(R[:g0, l]).tolist()) for l in range(48)]
    cop = np.zeros(len(A), bool)
    for q, L in runs:
        cop[q + TRIG:q + L] = True
        for Wd in res:
            s = g0 + q + TRIG; e = min(g0 + q + L, len(R))
            for b in range(s, e, Wd):
                E = R[b:min(b + Wd, e)]
                if len(E) < min(Wd, 16) // 2: continue
                r = res[Wd]; r["n"] += 1; r["rows"] += len(E); r["g2"] += groups(E, 2); r["g10"] += groups(E, 10)
                for l in range(48):
                    u = np.unique(E[:, l, :2]).tolist(); r["tot2"] += len(u); r["warm2"] += sum(1 for x in u if x in warm[l])
    idx = [g0 + x for x in range(len(A)) if not cop[x] and g0 + x < len(R)]
    for b in range(0, len(idx) - 15, 16):
        E = R[idx[b:b + 16]]; non["n"] += 1; non["rows"] += 16; non["g2"] += groups(E, 2); non["g10"] += groups(E, 10)
print(f"{nwin} code map windows with routing")
for Wd in (16, 32, 64, 128):
    r = res[Wd]
    if not r["n"]: continue
    print(f"copy blocks of W {Wd:3d}: {r['n']:3d} blocks, {r['rows'] / r['n']:.1f} rows each; groups per block: ranks 0-1 {r['g2'] / r['n']:.0f} ({r['g2'] / r['rows']:.1f} per row), "
          f"ranks 0-9 {r['g10'] / r['n']:.0f} ({r['g10'] / r['rows']:.1f} per row); ranks 0-1 experts already used by the prompt's rows {100 * r['warm2'] / r['tot2']:.1f} %")
print(f"non-copy answer rows, 16-row blocks: {non['n']} blocks; groups per block: ranks 0-1 {non['g2'] / max(1, non['n']):.0f} ({non['g2'] / max(1, non['rows']):.1f} per row), "
      f"ranks 0-9 {non['g10'] / max(1, non['n']):.0f} ({non['g10'] / max(1, non['rows']):.1f} per row)")
