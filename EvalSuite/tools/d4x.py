"""Decision 4 on unrounded numbers, with the pre-registered IFEval stage 2 (PREREG_nax6, PREREG_f1; IFX's design).
  .venv/bin/python tools/d4x.py <arm run> <core reference run> [<stage-2 reference run>]
Core: math 50, IFEval 100, HumanEval+ 60 on the keys both runs hold. Per set: the paired delta (arm - reference) is
above -5 points AND the exact two-sided McNemar p is at least 0.05; a gain with p < 0.05 is DIFFERS (HIGHER).
Stage 2: only if IFEval alone FAILS on the core while math and HumanEval+ do not. IFEval's line is then decision 4 on
IFX's 220 stage-1 prompts (results/ifx/stage1_keys.txt): the arm's ifevalx rows vs the stage-2 reference's. The
triggering core 100 become information.
Last line `D4X: PASS | FAIL | DIFFERS (HIGHER) | STAGE2 (decode IFX's 220)`; exit 0 PASS / DIFFERS, 1 FAIL,
10 STAGE2 wanted, 2 rows missing."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from evalsuite import decode_report as dr
from evalsuite.report import mcnemar_p

SETS = ("math", "ifeval", "humaneval")
N_CORE = {"math": 50, "ifeval": 100, "humaneval": 60}
VERDICT = None       # the last main()'s verdict, for callers (tools/f1_report.py)
STAGE1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "ifx", "stage1_keys.txt")


def rule(b, c, n):
    """(verdict, delta in points, p) on unrounded numbers; b = only the arm right, c = only the reference right"""
    d, p = 100.0 * (b - c) / n, mcnemar_p(b, c)
    if p < 0.05 and b > c:
        return "DIFFERS (HIGHER)", d, p
    return ("PASS" if d > -5 and p >= 0.05 else "FAIL"), d, p


def line(name, sa, sr, keys):
    ca = sum(sa[k][1] for k in keys) / len(keys)
    cr = sum(sr[k][1] for k in keys) / len(keys)
    b = sum(1 for k in keys if sa[k][1] == 1 and sr[k][1] == 0)
    c = sum(1 for k in keys if sa[k][1] == 0 and sr[k][1] == 1)
    v, d, p = rule(b, c, len(keys))
    print(f"{name} (n {len(keys)}): {100 * ca:.1f} % vs {100 * cr:.1f} %, delta {d:+.2f} pt, McNemar p {p:.4f} "
          f"(arm only {b} / reference only {c}): {v}")
    return v


def main(arm, ref, ref2=None):
    global VERDICT
    its = dr.items()
    A, R = dr.load(arm), dr.load(ref)
    if A is None or R is None:
        print(f"no decode rows for {arm if A is None else ref}")
        return 2
    out = {}
    for s in SETS:
        keys = sorted(k for k in its if its[k].get("set") == s and k in A and k in R)
        if len(keys) < N_CORE[s]:
            print(f"{s}: {len(keys)} of {N_CORE[s]} rows in both {arm} and {ref}")
            return 2
        sa = dr.SCORERS[s]({k: A[k] for k in keys}, its)
        sr = dr.SCORERS[s]({k: R[k] for k in keys}, its)
        out[s] = line(s, sa, sr, keys)
    others_ok = all(out[s] != "FAIL" for s in ("math", "humaneval"))
    if out["ifeval"] == "FAIL" and others_ok:
        xk = open(STAGE1).read().split()
        R2 = dr.load(ref2) if ref2 else None
        if R2 is None or not all(k in A for k in xk) or not all(k in R2 for k in xk):
            have = sum(k in A for k in xk)
            print(f"IFEval alone fails on the core: stage 2 wanted ({have} of {len(xk)} IFX rows in {arm}; "
                  f"reference {ref2 or 'not given'})")
            print("D4X: STAGE2 (decode IFX's 220)")
            VERDICT = "STAGE2"
            return 10
        from evalsuite import data as D
        xits = {it["key"]: dict(it, set="ifeval") for it in D.load("dec_ifevalx")}   # score_ifeval checks the set
        sa = dr.score_ifeval({k: A[k] for k in xk}, xits)
        sr = dr.score_ifeval({k: R2[k] for k in xk}, xits)
        print("stage 2 (IFEval's line on IFX's fresh prompts; the core 100 above are information):")
        out["ifeval"] = line("ifevalx stage 1", sa, sr, xk)
    final = "FAIL" if "FAIL" in out.values() else "DIFFERS (HIGHER)" if "DIFFERS (HIGHER)" in out.values() else "PASS"
    print(f"D4X: {final}")
    VERDICT = final
    return 1 if final == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
