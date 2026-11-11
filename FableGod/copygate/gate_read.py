#!/usr/bin/env python3
"""gate_read.py (FableGod copygate): the COPYGATE walk's read-out and the pre-registered verdict (GATE.md section 5).
Kept per check (accepted + 1, mean over checks) per arm and set vs the same-rows base arm (m_base16 / m_base32)."""
import os
import numpy as np

TD = os.path.expanduser("~/TrainLab/data")
SETS = ("code", "mbpp", "agent")
out = []


def say(s):
    print(s, flush=True)
    out.append(s)


def checks(tag, s):
    p = f"{TD}/d02b_walk{tag}_{s}.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return [z[f"{i}_checks"] for i in range(len(z["names"]))]


def kept(c):
    return float(np.mean([k + 1 for x in c for k, m, a in x]))


R = {}
for arm in ("g2_16", "g2_32", "g1_16", "g1_32", "g0_16"):
    for s in SETS:
        c, b = checks(f"_g_{arm}", s), checks(f"_m_base{arm[-2:]}", s)
        if c is not None:
            R[(arm, s)] = kept(c) / kept(b)
say("COPYGATE read-out (DFlash2 L1-20, official loop; kept per check vs the same-rows base walk m_base16 / m_base32)")
say("  arm      " + "".join(s.rjust(10) for s in SETS))
for arm in ("g2_16", "g2_32", "g1_16", "g1_32", "g0_16"):
    say(f"  {arm:8s} " + "".join((f"x{R[(arm, s)]:.3f}" if (arm, s) in R else "-").rjust(10) for s in SETS))
same = [checks("_g_gall16", s) is not None and all(np.array_equal(x, y) for x, y in
        zip(checks("_g_gall16", s), checks("_m_match16", s))) for s in SETS]
say(f"  sanity: gall16 == m_match16 bit for bit on {sum(same)} of 3 sets" + ("" if all(same) else "  -> VOID (the gate code differs from MATCH)"))


def verdict(a16, a32):
    g = lambda s, a: R.get((a, s), float("nan"))
    c16, c32, mb, ag = g("code", a16), g("code", a32), g("mbpp", a16), g("agent", a16)
    if c16 >= 1.035 and c32 >= 1.18 and mb >= 1.00 and ag >= 1.04:
        return "PASS", (c16, c32, mb, ag)
    if c16 < 1.03 or c32 < 1.15 or mb < 0.99:
        return "KILL", (c16, c32, mb, ag)
    return "BETWEEN", (c16, c32, mb, ag)


v2, x2 = verdict("g2_16", "g2_32")
v1, x1 = verdict("g1_16", "g1_32")
say(f"  LINE (G2 copygate: HumanEval+ kept(16) >= x1.035 and kept(32) >= x1.18 and MBPP kept(16) >= x1.00 and agent kept(16) "
    f">= x1.04 PASS; HumanEval+ kept(16) < x1.03 or kept(32) < x1.15 or MBPP kept(16) < x0.99 KILL): "
    f"x{x2[0]:.3f} / x{x2[1]:.3f} / x{x2[2]:.3f} / x{x2[3]:.3f} -> {v2}")
if v2 == "BETWEEN" and x2[3] < 1.04 and x2[0] >= 1.035 and x2[1] >= 1.18 and x2[2] >= 1.00:
    say(f"  fallback (pre-registered: G2 misses only agent's line -> G1 pduel decides): x{x1[0]:.3f} / x{x1[1]:.3f} / "
        f"x{x1[2]:.3f} / x{x1[3]:.3f} -> {v1} for G1")
open(f"{TD}/gate_read.txt", "w").write("\n".join(out) + "\n")
