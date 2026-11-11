"""R57_RELAX_DELTA's replay check (MoonShots, Wed 2026-11-04): relax_delta.relaxed_k, the function the r57 patch calls,
decides every tested code position of RELAX_350's data (FableGod's relax_curves.py, executed read-only up to its analysis
loop: the HPD-0 walks' drafts and cf joined with the 125B's stored top-8), and kept per check comes from relax_curves' own
per-depth profile. It must reproduce RELAX_350's code table: relax_curves' profile kept (EXACT 7.78, R1 0.25 / 0.5 / 1.39 = 8.20 / 8.61 /
9.52 at 16 rows), the ratios to exact (1.054 / 1.107 / 1.224; 1.158 / 1.220 / 1.354 at 32) and the deviation (0.61 /
1.19 / 2.38 %); RELAX_350's engine kept = today's 10.54 x the ratio (10.5 -> 11.7 at 0.5 nat, 12.9 at 1.39).
usage: [BigMoE venv] python relax_delta_replay.py -> data/relax_delta_replay.txt"""
import math, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.expanduser("~/MoonShots/patches/r57_relax"))
import relax_delta as RLD

SRC = os.path.expanduser("~/FableGod/relax_curves.py")
TABLE = {0.25: (1.054, 1.158, 0.61), 0.5: (1.107, 1.220, 1.19), 1.39: (1.224, 1.354, 2.38)}   # kept ratio 16 / 32, deviated %
RAW = {None: (7.78, 8.54), 0.25: (8.20, 9.01), 0.5: (8.61, 9.49), 1.39: (9.52, 10.53)}         # relax_curves.out's profile kept
BASE16 = 10.54                                                   # RELAX_350: today's measured exact kept at 16 rows


def main():
    src = open(SRC).read()
    cut = src.index("\nfor dom in DOMS:\n    P = [q for q in POS")
    env = {"__name__": "relax_curves_prefix"}
    exec(compile(src[:cut], SRC, "exec"), env)                    # POS, kept_from_profile (no analysis printed)
    P = [q for q in env["POS"] if q["dom"] == "code"]
    lines = [f"== R57_RELAX_DELTA replay check (relax_delta.relaxed_k on RELAX_350's {len(P)} code positions; kept per check "
             "from relax_curves' profile at 16 / 32 rows; table in brackets)"]
    ok = True

    def run(delta, thr=None):
        acc, n, dev = defaultdict(float), defaultdict(int), 0
        for q in P:
            am = 0 if q["right"] else 1                              # stand-in ids: the draft 0, the argmax 0 iff right
            dlm = RLD.row_dlm(q["lpd"], q["lp0"])
            may = [False] if delta is None else (None if thr is None else [q["cf"] == q["cf"] and q["cf"] >= thr])
            k, _ = RLD.relaxed_k([0], [am], [dlm], delta or 0.0, may)   # delta None = the exact rule (no row relaxes)
            acc[q["depth"]] += k; n[q["depth"]] += 1
            dev += k and not q["right"]
        prof = {j: acc[j] / n[j] for j in n}
        return env["kept_from_profile"](prof, 16), env["kept_from_profile"](prof, 32), 100 * dev / len(P)

    e16, e32, edev = run(None)
    ok &= abs(e16 - RAW[None][0]) <= 0.01 and abs(e32 - RAW[None][1]) <= 0.01 and edev == 0
    lines.append(f"  EXACT: profile kept {e16:.2f} / {e32:.2f} [{RAW[None][0]} / {RAW[None][1]}], deviated {edev:.2f} %")
    for d, (t16, t32, tdev) in TABLE.items():
        k16, k32, dev = run(d)
        r16, r32 = k16 / e16, k32 / e16                          # RELAX_350's 32-row ratio is against today's 16-row exact
        good = (abs(k16 - RAW[d][0]) <= 0.01 and abs(k32 - RAW[d][1]) <= 0.01 and abs(r16 - t16) <= 0.002
                and abs(r32 - t32) <= 0.002 and abs(dev - tdev) <= 0.01)
        ok &= good
        lines.append(f"  R1({d}): profile kept {k16:.2f} / {k32:.2f} [{RAW[d][0]} / {RAW[d][1]}], ratio to exact {r16:.3f} / {r32:.3f} "
                     f"[{t16} / {t32}], engine kept at 16 rows {BASE16 * r16:.1f}, deviated {dev:.2f} % [{tdev}] -> "
                     + ("match" if good else "MISMATCH"))
    for thr in (0.9, 0.5):
        for d in (0.5, 1.39):
            k16, k32, dev = run(d, thr)
            lines.append(f"  R1({d}) on rows with cf >= {thr}: ratio to exact {k16 / e16:.3f} / {k32 / e16:.3f}, engine kept at 16 "
                         f"rows {BASE16 * k16 / e16:.1f}, deviated {dev:.2f} %")
    lines.append(f"  -> {'RELAX_350 code table reproduced (profile kept, ratios, deviation)' if ok else 'MISMATCH: see above'}")
    print("\n".join(lines))
    open(os.path.expanduser("~/MoonShots/data/relax_delta_replay.txt"), "a").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
