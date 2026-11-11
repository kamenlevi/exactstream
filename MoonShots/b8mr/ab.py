"""Decode A/B of the multi-row phase B (run_gpu.sh gen runs): per-pass means over S = 4 passes only (matched S),
mr - wait verify-ms difference with a bootstrap 95 % CI over passes (texts differ between arms, so unpaired).
usage: ab.py  (reads data/b8mr/gen_P_ARM[_f0.1]_rR.json)"""
import json, os
import numpy as np
D = os.path.expanduser("~/MoonShots/data/b8mr")
rng = np.random.default_rng(0)


def load(p, a, ft, r):
    f = os.path.join(D, f"gen_{p}_{a}{ft}_r{r}.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    ps = [x for x in d["passes"] if x["S"] == 4]
    g = lambda k: np.array([x[k] for x in ps], float)
    return d, ps, {"ms": g("t_verify") * 1e3, "reads": g("x_misses"), "waited": g("x_waited"), "late": g("x_late"),
                   "over": g("x_over"), "stalled": g("x_stalled"), "io": g("x_io_s") * 1e3}


def boot(a, b, n=4000):
    m = [rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


for ft, rounds in (("", (1, 2)), ("_f0.1", (3,))):
    for p in ("code", "prose"):
        for r in rounds:
            w, m = load(p, "wait", ft, r), load(p, "mr", ft, r)
            if w is None or m is None:
                continue
            row = [f"{p:5s} {'10%' if ft else ' 5%'} r{r}"]
            for nm, (d, ps, v) in (("wait", w), ("mr", m)):
                row.append(f"{nm}: {d['tok_s']:.1f} tok/s {d['tok_per_pass']:.2f} tok/pass, S=4 passes {len(ps)}/{len(d['passes'])}, "
                           f"verify {v['ms'].mean():.1f} ms, reads {v['reads'].mean():.0f}, waited {v['waited'].mean():.0f}, "
                           f"late {v['late'].mean():.0f}, over {v['over'].sum():.0f}, stalled {v['stalled'].mean():.0f}, io {v['io'].mean():.1f} ms")
            dv = m[2]["ms"].mean() / w[2]["ms"].mean() - 1
            lo, hi = boot(w[2]["ms"], m[2]["ms"]) / w[2]["ms"].mean()
            row.append(f"mr vs wait per pass (S=4): {dv:+.1%} [{lo:+.1%}, {hi:+.1%}]; reads {m[2]['reads'].mean() / w[2]['reads'].mean() - 1:+.1%}, "
                       f"waited {m[2]['waited'].mean() / w[2]['waited'].mean() - 1:+.1%}; tok/s {m[0]['tok_s'] / w[0]['tok_s'] - 1:+.1%}")
            print("\n  ".join(row))
