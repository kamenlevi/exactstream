"""Round 23 evaluation: how well does the MTP head predict the next verify pass's experts (data/mtp_route_pred125.*)?
Per verify row r (positions t..t+2 of a depth-3 pass), the MTP draft step r+1 gave a top-20 per trunk layer. Measures:
  recall@B   |predicted top-B ∩ actual top-10| / 10, by layer group, vs the previous position's own top-10 (row r-1, or
             the previous pass's last valid row) as a baseline;
  misses     a per-layer LRU of C slots replays the passes (all rows' experts loaded, MRU, as BigMoE today); a miss is
             "covered" if it is in the union of the pass's MTP predictions at that layer (top-B per predicted row), i.e.
             it could have been read during drafting; "wasted" = predicted, not cached, not used by the pass.
usage: route_pred_eval.py [C] [B ...]   (default C=128 = 17 GB, B = 10 20)"""
import json, os, sys
from collections import OrderedDict, defaultdict
import numpy as np
D = os.path.expanduser("~/MoonShots/data")
z = np.load(f"{D}/mtp_route_pred125.npz")
meta = json.load(open(f"{D}/mtp_route_pred125.json"))
ids, rows, seqs, prompt = z["ids"], z["rows"], z["seq"], z["prompt"]
pred, pp, ps, pj = z["pred"], z["pred_prompt"], z["pred_seq"], z["pred_step"]
P, NL, RM, K = ids.shape
C = int(sys.argv[1]) if len(sys.argv) > 1 else 128
Bs = [int(x) for x in sys.argv[2:]] or [10, 20]
# map each MTP draft step to (pass index, row): the next gated pass of the prompt after the seq it was drafted under
by_prompt = defaultdict(list)
for i in range(P):
    by_prompt[int(prompt[i])].append(i)
predmap = {}                                           # (pass, row) -> [NL, PRED]
for k in range(len(pred)):
    cand = [i for i in by_prompt[int(pp[k])] if seqs[i] > ps[k]]
    if not cand:
        continue
    i = cand[0]
    r = int(pj[k]) - 1
    if 0 <= r < rows[i]:
        predmap[(i, r)] = pred[k]
# valid rows (accepted path) from the callbacks, as wrongpath_sim.py
committed = defaultdict(int)
for (p, s, n) in meta["callbacks"]:
    committed[(p, s)] += n
valid = np.ones((P, RM), bool)
for i in range(P):
    n = committed.get((int(prompt[i]), int(seqs[i])), 0)
    if n:
        valid[i, min(n, rows[i]):] = False
groups = [(0, 12), (12, 24), (24, 36), (36, 48)]
print(f"{P} passes, {len(predmap)} (pass, row) predictions, rows/pass {np.bincount(rows).tolist()}")
# recall
for B in Bs:
    rec = np.zeros((NL, 3)); base = np.zeros((NL, 3)); cnt = np.zeros((NL, 3))
    for (i, r), pr in predmap.items():
        if r > 2:
            continue
        for L in range(NL):
            act = set(ids[i, L, r].tolist())
            if -1 in act:
                continue
            rec[L, r] += len(act & set(pr[L, :B].tolist())) / K
            if r >= 1:
                prev = set(ids[i, L, r - 1].tolist())
            else:
                j = i - 1
                if j < 0 or prompt[j] != prompt[i]:
                    continue
                last = int(valid[j, :rows[j]].sum()) - 1
                prev = set(ids[j, L, max(0, last)].tolist())
            base[L, r] += len(act & prev) / K
            cnt[L, r] += 1
    print(f"\nrecall@{B} of the actual top-10 (MTP head | previous position's top-10 as baseline), by layer group:")
    for r in range(3):
        parts = []
        for a, b in groups:
            n = cnt[a:b, r].sum()
            parts.append(f"L{a:2d}-{b - 1:2d} {rec[a:b, r].sum() / max(n, 1):.2f} | {base[a:b, r].sum() / max(n, 1):.2f}")
        print(f"  row {r}: " + "   ".join(parts))
# miss coverage under a per-layer LRU (all rows loaded, MRU)
for B in Bs:
    caches = [OrderedDict() for _ in range(NL)]
    miss = np.zeros(NL); cov = np.zeros(NL); waste = np.zeros(NL); npass = 0
    for i in range(P):
        R = int(rows[i])
        rowsp = [r for r in range(R) if (i, r) in predmap]
        for L in range(NL):
            c = caches[L]
            need = {int(e) for r in range(R) for e in ids[i, L, r] if e >= 0}
            if not need:
                continue
            ps_ = {int(e) for r in rowsp for e in predmap[(i, r)][L, :B]}
            m = [e for e in need if e not in c]
            if i >= 60:
                miss[L] += len(m)
                cov[L] += sum(1 for e in m if e in ps_)
                waste[L] += len(ps_ - need - set(c.keys()))
            for e in need:
                if e in c:
                    c.move_to_end(e)
                else:
                    c[e] = 1
            while len(c) > C:
                c.popitem(last=False)
        if i >= 60:
            npass += 1
    print(f"\nC={C} ({C * 2.77 * NL / 1e3:.0f} GB), MTP top-{B} per predicted row: misses per pass, share covered by the "
          f"prediction (loadable during drafting), wasted prefetch reads per pass:")
    for a, b in groups:
        print(f"  L{a:2d}-{b - 1:2d}: misses {miss[a:b].sum() / npass:6.1f}, covered {cov[a:b].sum() / max(1, miss[a:b].sum()):.0%}, "
              f"wasted {waste[a:b].sum() / npass:6.1f}")
    print(f"  all   : misses {miss.sum() / npass:6.1f}, covered {cov.sum() / max(1, miss.sum()):.0%}, wasted {waste.sum() / npass:6.1f}")
