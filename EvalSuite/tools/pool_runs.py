"""Pool repeat decode runs of a small-RAM config against the reference (CPU; EvalSuite venv):
per item, the mean correctness over the runs minus the reference's; paired 90 % CI over items; core truncations / loops.
    .venv/bin/python tools/pool_runs.py b12,b12.r2 b8,b8.r2 b8fast"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from evalsuite import decode_report as R
from evalsuite.report import paired_ci

CORE = ("math", "ifeval", "humaneval")


def main(groups):
    its = R.items()
    ref = R.load("ref")

    def correct(rows, s):
        sc = R.SCORERS[s]({k: r for k, r in rows.items() if its.get(k, {}).get("set") == s}, its)
        return {k: v[1] for k, v in sc.items()}

    cref = {s: correct(ref, s) for s in CORE}
    core = [k for k in ref if its.get(k, {}).get("set") in CORE]
    for g in groups:
        runs = g.split(",")
        data = [R.load(v) for v in runs]
        print(f"== {' + '.join(runs)}")
        for s in CORE:
            cv = [correct(d, s) for d in data]
            keys = sorted(set(cref[s]).intersection(*cv))
            per = np.array([[c[k] for k in keys] for c in cv])           # runs x items
            r = np.array([cref[s][k] for k in keys])
            e, lo, hi = paired_ci(100 * (per.mean(0) - r))
            accs = " / ".join(f"{100 * p.mean():.1f}" for p in per)
            print(f"  {s:9s} n={len(keys)}: runs {accs} %, pooled {100 * per.mean():.1f} % vs ref {100 * r.mean():.1f} %: "
                  f"{e:+.1f} [{lo:+.1f}, {hi:+.1f}] pt")
        tv = [sum(d[k]["finish"] == "length" for k in core) for d in data]
        lv = [sum(R.looped(d[k]["completion_ids"]) for k in core) for d in data]
        print(f"  truncations / loops (core): runs {tv} / {lv} vs ref "
              f"{sum(ref[k]['finish'] == 'length' for k in core)} / {sum(R.looped(ref[k]['completion_ids']) for k in core)}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["b12,b12.r2", "b8,b8.r2", "b8fast"])
