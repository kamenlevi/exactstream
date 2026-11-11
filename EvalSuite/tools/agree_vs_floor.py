"""Paired final-answer agreement with the reference: variant vs floor (decode sets). usage: agree_vs_floor.py VARIANT"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evalsuite import decode_report as R
from evalsuite.report import paired_ci, mcnemar_p

if __name__ == "__main__":
    var = sys.argv[1]
    its = R.items()
    d = {n: R.load(n) for n in ("ref", "floor", var)}
    keys = sorted(set(d["ref"]) & set(d["floor"]) & set(d[var]))
    av, af, per = [], [], {}
    for s, fn in R.SCORERS.items():
        ks = [k for k in keys if its[k]["set"] == s]
        sc = {n: fn({k: d[n][k] for k in ks}, its) for n in d}
        a1 = [float(sc[var][k][0] == sc["ref"][k][0]) for k in ks]
        a2 = [float(sc["floor"][k][0] == sc["ref"][k][0]) for k in ks]
        av += a1; af += a2; per[s] = (int(sum(a1)), int(sum(a2)), len(ks))
    av, af = np.array(av), np.array(af)
    e, lo, hi = paired_ci(100 * (av - af))
    b, c = int(((av == 1) & (af == 0)).sum()), int(((av == 0) & (af == 1)).sum())
    print(f"agreement with ref: {var} {100*av.mean():.1f}% ({int(len(av)-av.sum())} disagree), floor {100*af.mean():.1f}% "
          f"({int(len(af)-af.sum())}); difference {e:+.1f} [{lo:+.1f}, {hi:+.1f}] pt, exact McNemar p={mcnemar_p(b, c):.3f} "
          f"(only {var} agrees {b}, only floor agrees {c}); per set (variant, floor, n): {per}")
