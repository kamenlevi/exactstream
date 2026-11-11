"""Tree bound (Leap): coverage of the target's argmax in the drafter's top-b, and the expected accepted tokens of a
best-first static tree of n nodes under i.i.d. depth (Sequoia-style; an UPPER-ish bound: real text is bursty), next to the
same i.i.d. chain and the simulated chain. Drafters: near-copy 10 % / 5 % (tfstep), MTP head smoke (turn17)."""
import sys, os, heapq, numpy as np
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfb/Leap_scripts"))
from a1_ceiling import tau

def ranks(idp, idq, B=8):
    hit = idq[:, :B] == idp[:, :1]
    return np.where(hit.any(1), hit.argmax(1), B)          # rank of the target's top-1 in the drafter's list (B = miss)

def best_tree(r, n):
    """r[k] = P(target token = drafter rank k); expected accepted drafts of the best n-node tree (i.i.d. depths)."""
    r = sorted(r, reverse=True)
    heap, tot = [(-r[0], (0,))], 0.0
    seen = 0
    while heap and seen < n:
        negp, path = heapq.heappop(heap)
        tot += -negp; seen += 1
        heapq.heappush(heap, (negp * r[0], path + (0,)))   # first child
        k = path[-1]
        if k + 1 < len(r):                                  # next sibling
            heapq.heappush(heap, (negp / r[k] * r[k + 1], path[:-1] + (k + 1,)))
    return tot + 1                                          # + the pass's own token

def show(name, rk, cls=None, seq=None, pos=None):
    groups = [("all", np.ones(len(rk), bool))] if cls is None else [(c, cls == c) for c in ("tf_ifeval", "tf_multi", "tf_code", "tf_gsm")]
    for c, m in groups:
        r = np.array([(rk[m] == k).mean() for k in range(8)])
        cov = np.cumsum(r)
        sim = ""
        if seq is not None:
            a = (rk == 0).astype(float); tn, tp = 0, 0.0
            for s in np.unique(seq[m]):
                ms = np.where(seq == s)[0]; ms = ms[np.argsort(pos[ms])]
                tn += len(ms); tp += len(ms) / tau(a[ms], 15)
            sim = f" | chain16 simulated {tn / tp:5.2f}"
        print(f"  {name:14s} {c:9s} top-1/2/4/8 {cov[0]:.3f} {cov[1]:.3f} {cov[3]:.3f} {cov[7]:.3f} | iid chain16 "
              f"{best_tree([r[0]], 15):5.2f} | best tree n=16/32/64/128: " + " ".join(f"{best_tree(r, n):5.2f}" for n in (15, 31, 63, 127)) + sim)

E = os.path.expanduser("~/EvalSuite/")
P = np.load(E + "results/ref/tfstep.npz")
cls = np.array([str(s) for s in P["sets"]])[P["seq"]]
for nm, f in (("near-copy 10%", "w2f_r10"), ("near-copy 5%", "w2f_r05")):
    Q = np.load(E + f"results/{f}/tfstep.npz")
    show(nm, ranks(P["ids"], Q["ids"]), cls, P["seq"], P["pos"])
z = np.load(os.path.expanduser("~/TrainLab/data/budget125_tf_R4_plan4433333333_mtp_pos.npz"))
show("MTP smoke gsm", ranks(z["exact__tgt_top_ids"], z["exact__mtp_top_ids"]))
