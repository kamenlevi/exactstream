"""Wrong-path expert study (round 22): do experts loaded only for rejected draft rows pollute the expert cache?
Trace-driven and exact for miss counts: routing depends only on the tokens, never on the cache, so every policy below sees
the same passes (data/verify_trace125.npz/json from verify_trace125.py: 125B, MTPLX committed history, depth 3, greedy).
Each round's accepted drafts = committed tokens - 1; rows after the first rejected draft are wrong-path. All rows of a
pass are verified, so their experts are loaded in every policy; the policies differ only in what the cache keeps after
the pass (per-layer LRU of C slots, persistent across prompts):
  mru        every expert the pass used goes to the MRU end (BigMoE today, LRU-like)
  demote     as mru, then experts used only by wrong-path rows move to the LRU end (evicted first)
  validonly  recency updated only by valid rows: a wrong-path-only expert that was cached keeps its old position,
             one that was loaded for the pass is inserted at the LRU end
  nowrong    upper bound: wrong-path rows never existed (no loads for them at all; not achievable)
CPU analogue: wrong-path memory references are often useful prefetches (Mutlu et al. 2005), so demotion can also hurt;
this measures which way it goes for MoE drafts.
Cost model (round 20, fitted to BigMoE's eager benches): 0.40 ms per stalled layer + 0.083 ms per MB, 2.77 MB per expert.
usage: wrongpath_sim.py [C ...]   (default 96 128 160 192 = 13, 17, 21, 26 GB of experts)"""
import json, os, sys
from collections import OrderedDict
import numpy as np
D = os.path.expanduser("~/MoonShots/data")
z = np.load(f"{D}/verify_trace125.npz")
meta = json.load(open(f"{D}/verify_trace125.json"))
ids, rows, seqs, prompt = z["ids"], z["rows"], z["seq"], z["prompt"]
P, NL, RM, K = ids.shape
# label rows: a pass's committed tokens = the sum over token_callbacks made while it was the latest gated pass (MTPLX may
# call back more than once per round); accepted drafts = committed - 1. Checked against MTPLX's accepted_by_depth.
from collections import defaultdict
valid = np.ones((P, RM), bool)
cbs = meta["callbacks"]
committed = defaultdict(int)
for (p, s, n) in cbs:
    committed[(p, s)] += n
bad = 0
acc_lab = defaultdict(lambda: [0] * 8)
for i in range(P):
    n = committed.get((int(prompt[i]), int(seqs[i])), 0)
    if n == 0:
        bad += 1; continue
    a = n - 1
    if a > rows[i] - 1:
        bad += 1; a = rows[i] - 1
    valid[i, a + 1:] = False
    for k in range(a):
        acc_lab[int(prompt[i])][k] += 1
for m in meta["meta"]:
    print(f"  prompt {m['prompt']}: accepted by depth from labels {acc_lab[m['prompt']][:3]} vs MTPLX {m['accepted_by_depth']}")
tokens = sum(n for (_, _, n) in cbs)
wrong_rows = int(sum((~valid[i, 1:rows[i]]).sum() for i in range(P)))
print(f"trace: {P} passes, {tokens} committed tokens, rows/pass {np.bincount(rows).tolist()}, wrong-path rows {wrong_rows} "
      f"({wrong_rows / max(1, int(rows.sum())):.1%} of rows), unmatched {bad}")


def run(C, policy, warm=60):
    miss = stalled = 0
    n_tok = 0
    caches = [OrderedDict() for _ in range(NL)]
    for i in range(P):
        R = int(rows[i])
        use_rows = [r for r in range(R) if valid[i, r] or policy != "nowrong"]
        for L in range(NL):
            c = caches[L]
            a = ids[i, L]
            if a[0, 0] < 0:
                continue
            vset = {int(e) for r in range(R) if valid[i, r] for e in a[r]}
            need = {int(e) for r in use_rows for e in a[r]}
            m = [e for e in need if e not in c]
            if i >= warm:
                miss += len(m); stalled += bool(m)
            # the pass touches its experts in row order; then the policy decides what stays where
            for e in need:
                if e in c:
                    if policy in ("mru", "demote", "nowrong") or e in vset:
                        c.move_to_end(e)
                else:
                    c[e] = 1
                    if policy == "validonly" and e not in vset:
                        c.move_to_end(e, last=False)
            if policy == "demote":
                for e in need - vset:
                    c.move_to_end(e, last=False)
            while len(c) > C:
                c.popitem(last=False)
        if i >= warm:
            n_tok += int(valid[i, :R].sum())        # tokens committed by this pass = valid rows (accepted + 1)
    ms = stalled * 0.40 + miss * 2.77 * 0.083
    return miss / n_tok, stalled / n_tok, ms / n_tok


Cs = [int(x) for x in sys.argv[1:]] or [96, 128, 160, 192]
print("C (GB)        policy     misses/token  stalled layers/token  est. stall ms/token")
for C in Cs:
    for pol in ("mru", "demote", "validonly", "nowrong"):
        m, s, ms = run(C, pol)
        print(f"{C:3d} ({C * 2.77 * NL / 1e3:4.1f})  {pol:10s} {m:8.1f}      {s:8.2f}              {ms:6.2f}", flush=True)
