"""WHERE THE ERRORS LIVE (NewIdeas, Sun 15 Nov 01:3x): by the 125B's own margin (top-1 minus top-2 log-prob, nats), the
share of rows, t30b's chain accuracy at depth 0 and at 23-30 (BigMoE's CEILING cells, MBPP 100), and the 2B's
teacher-forced agreement (MoonShots' rows2b). Then: what accuracy each margin bin needs for 98 % overall if every bin
moves to the same miss-rate share. CPU, read-only inputs. usage: python -I where_errors_live.py"""
import numpy as np, os
H = os.path.expanduser
BINS = [(0, 0.25), (0.25, 1), (1, 2), (2, 4), (4, 1e9)]
def lab(a, b): return f"{a:g}-{b:g}" if b < 1e8 else f">={a:g}"
c = np.load(H("~/BigMoE/data/deepdraft/ceiling/cells_t30b_mbpp100.npz"))
dep, right, m = c["depth"], c["right"], c["m"]
for s in ("mbpp", "heplus", "val"):
    d = np.load(H(f"~/MoonShots/week98/rows2b_{s}.npz"), allow_pickle=True)
    agree = d["top1"].astype(np.int64) == d["tk0"].astype(np.int64); m125 = d["margin125"]
    print(f"== {s}: 2B agreement {100 * agree.mean():.2f} % over {len(agree)} rows")
    print(f"   {'margin bin':>10} {'rows %':>7} {'2B agree':>9}" + ("  {:>9} {:>9}".format("t30b d0", "t30b 23-30") if s == "mbpp" else ""))
    for a, b in BINS:
        sel = (m125 >= a) & (m125 < b)
        line = f"   {lab(a, b):>10} {100 * sel.mean():7.1f} {100 * agree[sel].mean():9.2f}"
        if s == "mbpp":
            s0 = (dep == 0) & (m >= a) & (m < b); s2 = (dep >= 23) & (dep <= 30) & (m >= a) & (m < b)
            line += f"  {100 * right[s0].mean():9.2f} {100 * right[s2].mean():9.2f}"
        print(line)
    # miss budget: overall miss <= 2 %; where must the 2B's misses go?
    miss = 1 - agree
    tot = miss.mean()
    print(f"   2B misses {100 * tot:.2f} % of rows; share of its misses by bin: " + ", ".join(
        f"{lab(a, b)} {100 * miss[(m125 >= a) & (m125 < b)].sum() / max(miss.sum(), 1):.0f} %" for a, b in BINS))
    sure = m125 >= 4
    need_unsure = (0.98 - sure.mean() * agree[sure].mean()) / max((~sure).mean(), 1e-9)
    print(f"   for 98 % overall with sure rows as they are ({100 * agree[sure].mean():.2f} %), the rows under 4 nats need {100 * need_unsure:.1f} % (now {100 * agree[~sure].mean():.1f} %)")
