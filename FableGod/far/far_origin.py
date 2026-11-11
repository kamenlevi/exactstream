"""FAR-G1 origin spacing (Fable FAR, Wed 2026-11-04 22:1x; the orchestrator's follow-up 1): if the far draft is captured only at
every S-th check (an origin), with F far tokens per origin, how much of the kept-row ceiling (PERFECT-KEPT, far_ceiling.py)
survives on M-38's frame? Per-look forecast at decision i: the last origin o <= i in the prompt; check j in (i, i + H] is
forecast (its kept rows' true experts) only if its tokens lie within the origin's F far tokens (the kept tokens of checks
o + 1 .. j sum to <= F). Staleness is modelled optimistically (the far draft stays on the true path), so this measures the
horizon coverage of an origin schedule, not the draft's drift.
usage: OMP_NUM_THREADS=1 ~/BigMoE/.venv/bin/python -I far_origin.py -> far_origin.txt (one core, ~1 min)"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("far_frame", os.path.expanduser("~/FableGod/far/far_frame.py"))
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
P, NL, E, NI, H, BIG = F.P, F.NL, F.E, F.NI, F.H, F.BIG
PRM, PCHK, PEND = F.PRM, F.PCHK, F.PEND
OUT = os.path.expanduser("~/FableGod/far/far_origin.txt")
t0 = time.time()
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


KEPTSETS = [[np.stack([np.isin(np.arange(E), c["rows"][l][:c["kept"]]) for l in range(NL)]) for c in cks] for _, cks, _ in P]
KEPTN = [np.array([c["kept"] for c in cks]) for _, cks, _ in P]


def look_fn(S, Fh):
    def f(l, i):
        d = np.full(E, BIG)
        p, t = PRM[i], PCHK[i]
        if t < 0:
            o = -1                                                   # the prefill: the first origin is the prompt start
        else:
            o = (t // S) * S                                         # the last origin at or before this check
        used = 0 if o < 0 else 0
        cum = 0                                                      # kept tokens from check o + 1 onward
        for j in range(i + 1, min(i + H, PEND[i]) + 1):
            tj = PCHK[j]
            cum = int(KEPTN[p][o + 1:tj + 1].sum())
            if cum > Fh:
                break
            m = KEPTSETS[p][tj][l]
            d[m & (d == BIG)] = j - i
        return d
    return f


say(f"FAR ORIGIN ({time.strftime('%a %Y-%m-%d %H:%M')}): PERFECT-KEPT forecast with far-draft origins every S checks, F far "
    f"tokens per origin; M-38's frame, H {H}; K {F.KS}")
for K in F.KS:
    ref = F.LRU_REF[K]
    for S in (1, 2, 4, 8):
        for Fh in (64, 96, 128, 10 ** 6):
            if S == 1 and Fh not in (64, 10 ** 6):
                continue
            St = {"n": 0, "pos": 0, "tp": 0, "fp": 0}
            miss, _ = F.replay(K, "fh", look=look_fn(S, Fh), stats=St)
            m = F.pooled(miss)
            norig = sum(len(range(0, len(c), S)) for _, c, _ in P)
            say(f"  K {K} S {S} F {Fh if Fh < 10 ** 6 else 'inf':>4}: {m:6.1f} misses / kept token, {100 * (ref[0] - m) / ref[0]:5.1f} % fewer than LRU | "
                f"recall {St['tp'] / St['pos']:.2f}, precision {St['tp'] / max(St['tp'] + St['fp'], 1):.2f} | origins {norig} ({time.time() - t0:.0f} s)")
open(OUT, "w").write("\n".join(lines) + "\n")
