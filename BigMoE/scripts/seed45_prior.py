"""SEED45 prior file (BIGMOE_EVICT_PRIOR's set_prior input): per layer and expert, the share of cap45's live checks that
needed the expert (16-row union), leave-one-prompt-out per prompt ("loo_<prompt>", 48 x 512) and over all six ("all").
usage: .venv/bin/python scripts/seed45_prior.py -> data/seed45/prior45.npz"""
import os
import numpy as np
CAP = os.path.expanduser("~/BigMoE/data/cap45")
NL, E = 48, 512
ORDER = ("humaneval_HumanEval_68", "math500_test_prealgebra_1640", "humaneval_HumanEval_91", "math500_test_algebra_694",
         "math500_test_algebra_1282", "gsm8k_708")
cnt, nck = {}, {}
for name in ORDER:
    z = np.load(os.path.join(CAP, name + "_gate_0.npz"))
    seq, L, rows, roff, ids = z["seq"], z["L"], z["rows"], z["row_off"], z["ids"]
    need, kept = {}, set()
    for n, (s, l, r) in enumerate(zip(seq, L, rows)):
        s, l, r = int(s), int(l), int(r)
        d = need.setdefault(s, {})
        if l in d:
            kept.add(s)                                              # the commit replay
        else:
            d[l] = np.unique(ids[roff[n]:roff[n] + r])
    c = np.zeros((NL, E)); k = 0
    for s, d in need.items():
        if len(d) == NL and s in kept:
            for l in range(NL):
                c[l, d[l]] += 1
            k += 1
    cnt[name], nck[name] = c, k
out = {"all": (sum(cnt.values()) / sum(nck.values())).astype(np.float32)}
for name in ORDER:
    out["loo_" + name] = (sum(cnt[o] for o in ORDER if o != name) / sum(nck[o] for o in ORDER if o != name)).astype(np.float32)
os.makedirs(os.path.expanduser("~/BigMoE/data/seed45"), exist_ok=True)
np.savez(os.path.expanduser("~/BigMoE/data/seed45/prior45.npz"), **out)
print({k: (v.shape, round(float(v.mean()), 4)) for k, v in out.items()})
