"""BREAK98 (MoonShots, Tue 17 Nov 2026 07:4x): SESSION VALUE PREDICTION (SVP) of the 125B's state, CPU screen on the
existing true-state captures (BigMoE's bank: tf_code_HumanEval_*.npz = heplus24's 24 files, dlog__mbpp*.npz = the
bank's MBPP 100; per position: wd = the head's true input state [10240], tk = the 125B's top-32 next tokens, ids).
Mechanism (from CPU value prediction: last-value / context (FCM) predictors with a confidence gate): at a drafted
position t, depth d past the anchor a = t - d, key = the last n drafted tokens (longest of 16 / 12 / 8 / 6 / 4); if the
key ended earlier in THIS session at s <= a (prompt + verified answer: true states the checks already computed), the
head's stand-in state becomes the TRUE state wd[s] instead of its own carried one.
Measured here, per depth band, over the generated region (on the true path, drafted tokens = true tokens):
  hit       the share of positions with a match (n >= 4)
  agree1    at hits: the 125B's own top-1 at s == its top-1 at t (tk[s,0] == tk[t,0]): the retrieved true state makes
            the SAME next-token call as the true state (a CPU proxy for "the head fed wd[s] predicts as if fed wd[t]")
  copy1     at hits: the n-gram continuation ids[s+1] == ids[t+1] (token copying: the killed span / copy family)
  cos, r2   wd[s] vs wd[t] (r2 against the file's mean state)
  base      the depth-0 stand-in: the anchor's own true state wd[a] vs wd[t] (agree1 / cos), i.e. no retrieval
usage: ~/BigMoE/.venv/bin/python -I -B svp_screen.py [--maxd 30]   (CPU only, ~1-3 min)"""
import argparse
import bisect
import glob
import json
import os
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--maxd", type=int, default=30)
ap.add_argument("--pool", default="all", choices=["all", "gen"])
ap.add_argument("--out", default="")
A = ap.parse_args()
A.out = A.out or os.path.expanduser(f"~/MoonShots/week98/break98/svp_screen_{A.pool}")
NS = (16, 12, 8, 6, 4)
BANDS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 30)]
SETS = {"heplus24": sorted(glob.glob(os.path.expanduser("~/datasets/deepdraft/bank/tf_code_HumanEval_*.npz"))),
        "mbppbank": sorted(glob.glob(os.path.expanduser("~/datasets/deepdraft/bank/dlog__mbpp*.npz")))}
t0 = time.time()
RES, L_ = {}, []


def say(s=""):
    print(s, flush=True)
    L_.append(s)


for sname, files in SETS.items():
    acc = {b: dict(n=0, hit=0, agree=0, copy=0, cos=0.0, r2=0.0, base_agree=0, base_cos=0.0, nlen={n: 0 for n in NS},
                   hit_agree_n={n: [0, 0] for n in NS}) for b in BANDS}
    for f in files:
        z = np.load(f)
        ids = np.asarray(z["ids"], np.int64)
        wd = np.asarray(z["wd"], np.float32)
        tk = np.asarray(z["tk"][:, 0], np.int64)
        L = wd.shape[0]
        g = int(z["gen"])
        mu = wd[g:].mean(0)
        den = ((wd - mu) ** 2).sum(1)
        nrm = np.linalg.norm(wd, axis=1) + 1e-9
        # end positions of every n-gram (positions p: ids[p - n + 1 .. p]), most recent first per key
        idx = {n: {} for n in NS}
        for n in NS:
            for p in range(n - 1, L):
                idx[n].setdefault(tuple(ids[p - n + 1:p + 1].tolist()), []).append(p)
        for t in range(g, L - 1):
            for d in range(1, A.maxd + 1):
                a = t - d
                if a < g - 1:
                    break
                b = next(bb for bb in BANDS if bb[0] <= d <= bb[1])
                R = acc[b]
                R["n"] += 1
                R["base_agree"] += int(tk[a] == tk[t])
                R["base_cos"] += float(wd[a] @ wd[t] / (nrm[a] * nrm[t]))
                s = None
                for n in NS:
                    if t - n + 1 < 0:
                        continue
                    lst = idx[n].get(tuple(ids[t - n + 1:t + 1].tolist()))
                    if lst:
                        i_ = bisect.bisect_right(lst, a)
                        if i_ and (A.pool == "all" or lst[i_ - 1] >= g):
                            s, nn = lst[i_ - 1], n
                            break
                if s is None:
                    continue
                R["hit"] += 1
                R["nlen"][nn] += 1
                ag = int(tk[s] == tk[t])
                R["agree"] += ag
                R["hit_agree_n"][nn][0] += 1
                R["hit_agree_n"][nn][1] += ag
                R["copy"] += int(ids[s + 1] == ids[t + 1])
                R["cos"] += float(wd[s] @ wd[t] / (nrm[s] * nrm[t]))
                R["r2"] += 1.0 - float(((wd[s] - wd[t]) ** 2).sum()) / float(den[t] + 1e-9)
    RES[sname] = {}
    say(f"=== {sname} ({len(files)} files): per depth band | hit share | at hits: 125B top-1 agree (retrieved vs true), "
        f"copy (n-gram next == true next), cos, r2 | no retrieval (anchor's own state): agree, cos")
    for b in BANDS:
        R = acc[b]
        h = max(R["hit"], 1)
        row = dict(n=R["n"], hit=R["hit"] / max(R["n"], 1), agree=R["agree"] / h, copy=R["copy"] / h, cos=R["cos"] / h,
                   r2=R["r2"] / h, base_agree=R["base_agree"] / max(R["n"], 1), base_cos=R["base_cos"] / max(R["n"], 1),
                   nlen={n: R["nlen"][n] / h for n in NS},
                   agree_by_n={n: (R["hit_agree_n"][n][1] / R["hit_agree_n"][n][0] if R["hit_agree_n"][n][0] else None)
                               for n in NS})
        RES[sname][f"{b[0]}-{b[1]}"] = row
        say(f"  d {b[0]:2d}-{b[1]:2d}: n {R['n']:6d} | hit {row['hit']:.3f} | agree {row['agree']:.3f} copy {row['copy']:.3f} "
            f"cos {row['cos']:.3f} r2 {row['r2']:.3f} | base agree {row['base_agree']:.3f} cos {row['base_cos']:.3f} | "
            f"match length shares " + " ".join(f"{n}:{row['nlen'][n]:.2f}" for n in NS)
            + " | agree by length " + " ".join(f"{n}:{row['agree_by_n'][n]:.2f}" for n in NS if row['agree_by_n'][n] is not None))
say(f"done {time.time() - t0:.0f} s")
os.makedirs(os.path.dirname(A.out), exist_ok=True)
open(A.out + ".txt", "w").write("\n".join(L_) + "\n")
json.dump(RES, open(A.out + ".json", "w"), indent=1)
