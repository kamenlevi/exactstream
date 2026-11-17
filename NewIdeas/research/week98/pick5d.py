#!/usr/bin/env python3
"""pick5d.py (NewIdeas, Tue 17 Nov; the time is the commit's): G1's branch pick, mechanically, from EvalSuite's
PREREG_lossygate_2026-11-15.md as filed: amendment 5d (refined 14:0x, the 00:5x fallback with EvalSuite's three
readings, confirmed 01:0x), 5c's RES entry rule (refined 13:4x; d = the 1c-tiers RUNAWAY read) and 5e (int8-free).
Nothing here changes a rule; it only applies them, so the pick at V2's verdict cannot slip.

Inputs (flags; --auto also reads the COMQOM map read-out and Fit12's KILL_RESULT.json when they exist):
  --tiers    PASS | FAIL_RUNAWAY | FAIL_DELTA | FAIL_DEC4 | FAIL_IFEVAL   (1c-tiers: V1 forward + V2 pooled)
  --runaway  PASS | FAIL      (the RUNAWAY guard's read at V2; 5c (d))
  --map      the non-tie ratio bt_cq / bt_cq_ctrl, or none (no map read by 09:15 Tue)
  --C / --D  KILL | SURVIVE | BETWEEN | VOID | none   (the kill test's comments-only / comments + MBPP docstring arms)
  --probe    pass | fail_engaged | not_engaged | none (BigMoE's g1c / g1cr probe re-run on '#'-line keys)
  --keytest  envs with a key-test PASS (default: the g1b key test, 7 / 9: all but g1c, g1cr)
Output: the branch, RES yes / no, the env, and every rule used.
usage: /usr/bin/python3 -I pick5d.py --tiers ... --runaway ... --map 0.42 --C SURVIVE --D BETWEEN --probe pass
       /usr/bin/python3 -I pick5d.py --selftest
"""
import argparse
import json
import os
import re

KT_DEFAULT = "g1,g1r,g1cd,g1cdr,g1l,g1lr,lx"           # g1b key test 00:38-00:48: 7 / 9 PASS (g1c / g1cr: probe gap)
RES_A = (0.98, 0.56, 1.34, 0.87)                        # bt_g1h8_res all-row / non-tie vs the line (g1g2_map.out)
RES_B = (0.06, 0.30)                                    # the paired RES increment on windows 1-12 vs <= +0.3 pt
ENV = {"G1": ("g1", "g1r"), "C": ("g1c", "g1cr"), "D": ("g1cd", "g1cdr"), "L": ("g1l", "g1lr")}
NAME = {"G1": "G1 as filed (5b, int8-free per 5e)", "C": "1c 1.6e-2 + comment-QOM, comments only",
        "D": "1c 1.6e-2 + comment-QOM WITH MBPP docstrings", "L": "1c at lambda 0.8e-2",
        "NOT RUN": "G1 NOT RUN (IFEval identity FAIL = a scope bug; 5d item 4)"}


def pick(tiers, runaway, mp, C, D, probe, kt):
    why = []
    if tiers == "PASS":
        br = "G1"; why.append("1c-tiers PASSES -> G1 as filed (5d item 1; int8-free per 5e)")
    elif tiers == "FAIL_IFEVAL":
        return "NOT RUN", False, None, ["IFEval identity FAIL -> scope bug, G1 not run in any branch (5d item 4)"]
    else:
        why.append(f"1c-tiers {tiers} -> the branch rule (5d item 4: RUNAWAY, pooled-delta and decision-4 fails alike)")
        void = "VOID" in (C, D)
        nokt = C in (None, "none") and D in (None, "none")
        if mp is None:
            br = "L"; why.append("no map read by 09:15 -> lambda 0.8e-2 (5d 14:0x)")
        elif mp <= 0.5:
            if void or nokt:
                br = "C"; why.append(f"map {mp:.2f} CONFIRMED + control VOID / no kill test -> comments-only")
            elif C == "SURVIVE":
                br = "C"; why.append(f"map {mp:.2f} CONFIRMED + C SURVIVE -> comments-only")
            elif D == "SURVIVE":
                br = "D"; why.append(f"map {mp:.2f} CONFIRMED + C {C} + D SURVIVE -> MBPP docstrings")
            elif C == "KILL":
                br = "L"; why.append(f"map {mp:.2f} CONFIRMED + C KILL + D {D} -> lambda 0.8e-2")
            else:
                br = "C"; why.append(f"map {mp:.2f} CONFIRMED + C {C} + D {D} -> comments-only (the 'otherwise' row)")
        elif mp < 0.8:
            if void or nokt:
                br = "L"; why.append(f"map {mp:.2f} BETWEEN + control VOID / no kill test -> lambda 0.8e-2")
            elif C == "SURVIVE":
                br = "C"; why.append(f"map {mp:.2f} BETWEEN + C SURVIVE -> comments-only")
            elif D == "SURVIVE":
                br = "D"; why.append(f"map {mp:.2f} BETWEEN + C {C} + D SURVIVE -> MBPP docstrings")
            else:
                br = "L"; why.append(f"map {mp:.2f} BETWEEN + C {C} + D {D} -> lambda 0.8e-2")
        else:
            br = "L"; why.append(f"map {mp:.2f} >= 0.8x -> lambda 0.8e-2")
    a = RES_A[0] <= RES_A[2] and RES_A[1] <= RES_A[3]
    b = RES_B[0] <= RES_B[1]
    d = runaway == "PASS"
    res = a and b and d
    why.append(f"RES (5c): (a) bt_g1h8_res {RES_A[0]} / {RES_A[1]} vs {RES_A[2]} / {RES_A[3]} {'holds' if a else 'FAILS'}; "
               f"(b) paired +{RES_B[0]} pt vs +{RES_B[1]} {'holds' if b else 'FAILS'}; (d) RUNAWAY {runaway} "
               f"{'holds' if d else 'FAILS'}; (c) = the env's key test, below")
    # the 00:5x fallback and EvalSuite's readings 1-2 (C or D without a usable '#'-path PASS)
    if br in ("C", "D") and probe == "fail_engaged":
        why.append("probe re-run engaged '#' and FAILED -> counts against C and D (reading 2) -> lambda 0.8e-2")
        br = "L"
    if br == "C":
        env_c = ENV["C"][1 if res else 0]
        has = env_c in kt or probe == "pass"
        if not has:
            if D == "KILL":
                why.append(f"{env_c} has no key-test PASS (probe {probe}) and D KILL -> lambda 0.8e-2 (00:5x fallback)")
                br = "L"
            else:
                why.append(f"{env_c} has no key-test PASS (probe {probe}) -> D, as D's arm is {D}, not KILL (00:5x fallback)")
                br = "D"
        else:
            why.append(f"{env_c} PASS ({'probe re-run' if probe == 'pass' else 'key test'}) -> comments-only stands")
    env = ENV[br][1 if res else 0]
    if res and env not in kt and not (br == "C" and probe == "pass"):
        why.append(f"(c): {env} has no key-test PASS -> RES drops; the RES-free env")
        res = False; env = ENV[br][0]
    if env not in kt and not (br == "C" and probe == "pass"):
        why.append(f"WARNING: {env} has no key-test PASS on record; EvalSuite's G1_PICK writer will refuse it")
    return br, res, env, why


def auto_inputs():
    m = None; C = D = None
    f = os.path.expanduser("~/Breakthrough/data/m1/comqom_map")
    for fn in sorted(os.listdir(f)) if os.path.isdir(f) else []:
        if fn.endswith((".out", ".txt")):
            for line in open(os.path.join(f, fn), errors="replace"):
                g = re.search(r"READING \(non-tie\): bt_cq / bt_cq_ctrl = [\d.]+ / [\d.]+ = ([\d.]+|inf)", line)
                if g:
                    m = float(g.group(1))
    k = os.path.expanduser("~/Fit12/data/res1c/killtest/KILL_RESULT.txt")
    if os.path.exists(k):
        t = open(k).read()
        print("KILL_RESULT.txt (read it; set --C / --D by hand from the DISCORDANT reading):\n" + t[-1500:])
    return m, C, D


def selftest():
    kt = KT_DEFAULT.split(",")
    cases = [  # (tiers, runaway, map, C, D, probe) -> (branch, res)
        (("PASS", "PASS", None, None, None, "none"), ("G1", True)),
        (("PASS", "FAIL", None, None, None, "none"), ("G1", False)),
        (("FAIL_IFEVAL", "PASS", 0.3, "SURVIVE", "SURVIVE", "pass"), ("NOT RUN", False)),
        (("FAIL_RUNAWAY", "FAIL", None, "SURVIVE", "SURVIVE", "pass"), ("L", False)),
        (("FAIL_RUNAWAY", "FAIL", 0.4, "SURVIVE", "BETWEEN", "pass"), ("C", False)),
        (("FAIL_RUNAWAY", "FAIL", 0.4, "SURVIVE", "BETWEEN", "not_engaged"), ("D", False)),
        (("FAIL_RUNAWAY", "FAIL", 0.4, "SURVIVE", "KILL", "not_engaged"), ("L", False)),
        (("FAIL_RUNAWAY", "FAIL", 0.4, "SURVIVE", "BETWEEN", "fail_engaged"), ("L", False)),
        (("FAIL_RUNAWAY", "FAIL", 0.4, "BETWEEN", "SURVIVE", "none"), ("D", False)),
        (("FAIL_RUNAWAY", "FAIL", 0.4, "KILL", "BETWEEN", "pass"), ("L", False)),
        (("FAIL_RUNAWAY", "FAIL", 0.4, "BETWEEN", "BETWEEN", "pass"), ("C", False)),
        (("FAIL_RUNAWAY", "FAIL", 0.4, "VOID", "VOID", "pass"), ("C", False)),
        (("FAIL_RUNAWAY", "FAIL", 0.6, "SURVIVE", "KILL", "pass"), ("C", False)),
        (("FAIL_RUNAWAY", "FAIL", 0.6, "BETWEEN", "SURVIVE", "pass"), ("D", False)),
        (("FAIL_RUNAWAY", "FAIL", 0.6, "BETWEEN", "BETWEEN", "pass"), ("L", False)),
        (("FAIL_RUNAWAY", "FAIL", 0.6, "VOID", "VOID", "pass"), ("L", False)),
        (("FAIL_RUNAWAY", "FAIL", 0.9, "SURVIVE", "SURVIVE", "pass"), ("L", False)),
        (("FAIL_DELTA", "PASS", 0.4, "SURVIVE", "BETWEEN", "pass"), ("C", True)),
        (("FAIL_DELTA", "PASS", 0.4, "SURVIVE", "BETWEEN", "not_engaged"), ("D", True)),
        (("FAIL_DELTA", "PASS", None, None, None, "none"), ("L", True)),
    ]
    bad = 0
    for args, want in cases:
        br, res, env, why = pick(*args, kt)
        ok = (br, res) == want
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {args} -> {br} RES {res} env {env}" + ("" if ok else f" (want {want})"))
    print(f"selftest: {len(cases) - bad} / {len(cases)} ok")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers"); ap.add_argument("--runaway"); ap.add_argument("--map", default="none")
    ap.add_argument("--C", default="none"); ap.add_argument("--D", default="none"); ap.add_argument("--probe", default="none")
    ap.add_argument("--keytest", default=KT_DEFAULT); ap.add_argument("--auto", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    mp = None if a.map == "none" else float(a.map)
    if a.auto:
        m2, _, _ = auto_inputs()
        print(f"auto: map ratio {m2} (flag: {mp})")
        mp = m2 if mp is None else mp
    br, res, env, why = pick(a.tiers, a.runaway, mp, a.C, a.D, a.probe, a.keytest.split(","))
    for w in why:
        print("- " + w)
    print(json.dumps({"branch": br, "name": NAME[br], "res": res, "env": env}))


if __name__ == "__main__":
    main()
