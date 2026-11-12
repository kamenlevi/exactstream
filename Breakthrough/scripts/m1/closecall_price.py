#!/usr/bin/env python3
"""Price of lossy acceptance at the 125B's close calls (Breakthrough, Thu 12 Nov 2026 18:3x; Kamen 18:2x via the
orchestrator; pre-registered docs/CLOSECALL_ACCEPT_PRICE.md e75badd) [P: replay estimates on teacher-forced data].
Data (read-only): DFlash2 L1-20 per-check replays on D-02a's captures, R-61N's 16-row file (ifeval, multi, agent, mgsm,
gsm) and NT-1's 32-row code file. Estimators = NewIdeas nt_analyze.py's, unchanged (LB = +1 per forgiven check; MOD =
continue with the exact conditional survival, later misses forgiven at the rule's first-miss rate; RUNS 200, rng 0 with
the same draw order), plus a count of deviating (forgiven) tokens. Output ~/Breakthrough/data/m1/closecall_price.out.
Usage: nice -n 15 ~/BigMoE/.venv/bin/python -I -B closecall_price.py
"""
import json
import math
import os

import numpy as np

R61 = os.path.expanduser("~/TrainLab/data/r61n/chatnt_b16.jsonl")
NT1 = os.path.expanduser("~/NewIdeas/data/nt/nt_accept_b32.jsonl")
WALK_CODE = os.path.expanduser("~/TrainLab/data/d02b_walk_code.npz")
CYC = {"code": 0.065, "chat": 0.044}            # s per 16-row check at 45 GB (the orchestrator's measured figures)
MTP_B5 = {"code": 9.50, "ifeval": 2.74}         # exact kept per 16-row check, MTP b5 (drafterwall a1)
RELAX = {"ifeval": (-3.0, 0.0362), "code": (-3.3, None)}   # pts at R1(0.5); IFEval's measured deviating-token share
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


RULES = [("rho 0.5", lambda r: r["pd"] >= 0.5 * r["p1"][r["k"]]),
         ("rho 0.25", lambda r: r["pd"] >= 0.25 * r["p1"][r["k"]]),
         ("rho 0.1", lambda r: r["pd"] >= 0.1 * r["p1"][r["k"]]),
         ("rho 0.75", lambda r: r["pd"] >= 0.75 * r["p1"][r["k"]]),
         ("rho 0.61 = R1(0.5)", lambda r: r["pd"] >= math.exp(-0.5) * r["p1"][r["k"]]),
         ("delta 0.05", lambda r: r["p1"][r["k"]] - r["pd"] <= 0.05),
         ("delta 0.1", lambda r: r["p1"][r["k"]] - r["pd"] <= 0.1),
         ("delta 0.2", lambda r: r["p1"][r["k"]] - r["pd"] <= 0.2)]


def analyze(recs):
    nb_max = max(r["nb"] for r in recs)
    tok_exact = float(np.mean([r["acc"] for r in recs]))
    miss = [r for r in recs if r["k"] < r["nb"]]
    reach = np.zeros(nb_max); hits = np.zeros(nb_max)
    for r in recs:
        for j in range(min(r["nb"], r["k"] + 1)):
            reach[j] += 1; hits[j] += r["hit"][j]
    surv = hits / np.maximum(1, reach)
    rng = np.random.default_rng(0)
    out = {}
    for name, rule in RULES:
        forg = [r for r in miss if rule(r)]
        f = len(forg) / max(1, len(miss))
        lbk = [r["acc"] + (1 if rule(r) and r["k"] < r["nb"] and r["k"] + 2 <= r["rem"] else 0) if r["k"] < r["nb"]
               else r["acc"] for r in recs]
        lb = float(np.mean(lbk))
        lb_dev = sum(1 for r in recs if r["k"] < r["nb"] and rule(r) and r["k"] + 2 <= r["rem"]) / sum(lbk)
        tot = dev = 0.0; RUNS = 200
        for _ in range(RUNS):
            for r in recs:
                if r["k"] >= r["nb"] or not rule(r):
                    tot += r["acc"]; continue
                j = r["k"] + 1; d = 1
                while j < r["nb"]:
                    if rng.random() < surv[j]:            # the same draw order as nt_analyze's `a or b`
                        j += 1
                    elif rng.random() < f:
                        j += 1; d += 1
                    else:
                        break
                kept = min(j + 1, r["rem"])
                tot += kept; dev += min(d, kept)
        mod = tot / (RUNS * len(recs))
        out[name] = dict(f=f, lb=lb, mod=mod, lbdev=lb_dev, moddev=dev / tot, rp=sum(1 for r in forg if r["pd"] < 0.01))
    return tok_exact, out, len(recs)


def table(dom, recs, cyc, extra=None):
    tok, out, n = analyze(recs)
    say("")
    say(f"== {dom}: {n} checks, exact kept per check {tok:.2f} ({tok / cyc:.0f} tok/s at {1000 * cyc:.0f} ms per check)"
        + (f"; {extra}" if extra else ""))
    say("| rule | forgives (first misses) | kept LB | kept MOD | tok/s LB-MOD | deviating tokens LB-MOD | forgiven with p_d < 0.01 |")
    say("|---|---|---|---|---|---|---|")
    for name, o in out.items():
        say(f"| {name} | {o['f']:.2f} | {o['lb']:.2f} ({o['lb'] / tok - 1:+.0%}) | {o['mod']:.2f} ({o['mod'] / tok - 1:+.0%}) | "
            f"{o['lb'] / cyc:.0f}-{o['mod'] / cyc:.0f} | {100 * o['lbdev']:.1f}-{100 * o['moddev']:.1f} % | {o['rp']} |")
    return tok, out


r16 = [json.loads(l) for l in open(R61)]
r32 = [json.loads(l) for l in open(NT1)]
DOM = {"tf_ifeval": "ifeval", "tf_multi": "multi", "agent": "agent", "mgsm": "mgsm", "tf_gsm": "gsm"}
say("Close-call (lossy) acceptance price: DFlash2 L1-20 replays, teacher-forced; nt_analyze's LB / MOD estimators")
res = {}
chat = [r for r in r16 if r["dom"] in ("tf_ifeval", "tf_multi")]
res["chat"] = table("chat (ifeval + multi)", chat, CYC["chat"])
for d in ("tf_ifeval", "tf_multi", "agent", "mgsm", "tf_gsm"):
    res[DOM[d]] = table(DOM[d], [r for r in r16 if r["dom"] == d], CYC["chat"])
z = np.load(WALK_CODE, allow_pickle=True)
ck = np.concatenate([z[f"{s}_checks"] for s in range(len(z["names"]))])
code16 = float(np.mean(ck[:, 0] + 1))           # columns [accepted, checkable, anchor]; d02b_walk.txt says 9.57
assert abs(code16 - 9.57) < 0.01, code16
res["code32"] = table("code, 32-row checks (NT-1, HumanEval 24)", r32, 2 * CYC["code"],
                      f"16-row exact baselines: DFlash2 walk {code16:.2f}, MTP b5 {MTP_B5['code']:.2f}")

say("")
say("SELF-CHECK vs R-61N's chat LB 2.32 / 2.41 / 2.49 and NT-1's code LB 10.34 / 10.41 / 10.47 (rho 0.5 / 0.25 / 0.1); "
    "MOD within 0.05 of 2.46 / 2.74 / 3.10 and 11.27 / 12.08 / 12.79:")
ok = True
for key, lbref, modref in (("chat", (2.32, 2.41, 2.49), (2.46, 2.74, 3.10)), ("code32", (10.34, 10.41, 10.47), (11.27, 12.08, 12.79))):
    o = res[key][1]
    for name, l, m in zip(("rho 0.5", "rho 0.25", "rho 0.1"), lbref, modref):
        g = f"{o[name]['lb']:.2f}" == f"{l:.2f}" and abs(o[name]["mod"] - m) <= 0.05
        ok = ok and g
        say(f"  {key} {name}: LB {o[name]['lb']:.2f} (ref {l}), MOD {o[name]['mod']:.2f} (ref {m}) {'PASS' if g else 'FAIL'}")
say(f"SELF-CHECK {'PASS' if ok else 'FAIL'}")

say("")
o61 = res["ifeval"][1]["rho 0.61 = R1(0.5)"]
say(f"CALIBRATION vs RELAX's live R1(0.5) on IFEval (kept +7.2 %, 3.62 % of tokens deviating): estimate LB "
    f"{o61['lb'] / res['ifeval'][0] - 1:+.1%} .. MOD {o61['mod'] / res['ifeval'][0] - 1:+.1%}; deviating "
    f"{100 * o61['lbdev']:.1f} .. {100 * o61['moddev']:.1f} %")

say("")
say("ESTIMATE TABLE (kept per check: the live R1(0.5) point sits inside LB..MOD; tok/s at 44 ms chat / 65 ms code per "
    "16-row check; expected quality: RELAX's measured R1(0.5) hit scaled by the deviating-token share, MOD) [P]")
say("| rule | IFEval kept (DFlash2 / MTP b5 est) | IFEval tok/s (MTP b5) | chat kept | code kept per 16 (MTP b5 est) | "
    "code tok/s (MTP b5) | expected IFEval hit | expected HumanEval+ hit |")
say("|---|---|---|---|---|---|---|---|")
ti, oi = res["ifeval"]; tc, oc = res["chat"]; t32, o32 = res["code32"]
dev61i = oi["rho 0.61 = R1(0.5)"]["moddev"]; dev61c = o32["rho 0.61 = R1(0.5)"]["moddev"]
say(f"| exact (no rule) | {ti:.2f} / {MTP_B5['ifeval']:.2f} | {MTP_B5['ifeval'] / CYC['chat']:.0f} | {tc:.2f} | "
    f"{MTP_B5['code']:.2f} | {MTP_B5['code'] / CYC['code']:.0f} | 0 | 0 |")
for name in ("rho 0.75", "rho 0.61 = R1(0.5)", "rho 0.5", "rho 0.25", "rho 0.1", "delta 0.05", "delta 0.1", "delta 0.2"):
    gi_lo, gi_hi = oi[name]["lb"] / ti, oi[name]["mod"] / ti
    gc_lo, gc_hi = o32[name]["lb"] / t32, o32[name]["mod"] / t32
    hi_i = RELAX["ifeval"][0] * oi[name]["moddev"] / dev61i
    hi_c = RELAX["code"][0] * o32[name]["moddev"] / dev61c
    say(f"| {name} | {oi[name]['lb']:.2f}-{oi[name]['mod']:.2f} / {MTP_B5['ifeval'] * gi_lo:.2f}-{MTP_B5['ifeval'] * gi_hi:.2f} | "
        f"{MTP_B5['ifeval'] * gi_lo / CYC['chat']:.0f}-{MTP_B5['ifeval'] * gi_hi / CYC['chat']:.0f} | "
        f"{oc[name]['lb']:.2f}-{oc[name]['mod']:.2f} | {MTP_B5['code'] * gc_lo:.2f}-{min(16, MTP_B5['code'] * gc_hi):.2f} | "
        f"{MTP_B5['code'] * gc_lo / CYC['code']:.0f}-{min(16, MTP_B5['code'] * gc_hi) / CYC['code']:.0f} | "
        f"{hi_i:+.1f} pts | {hi_c:+.1f} pts |")
say("Upper bound (drafterwall, byte floor, lossy oracle gate + lag-2 ceiling) [P]: HumanEval+ 706 tok/s (964 with the "
    "perfect pick at 64 rows), MBPP 441 (591); kept 20.1 vs 12.6 per 32-row check on HumanEval+, 12.1 vs 6.2 on MBPP.")
open(os.path.expanduser("~/Breakthrough/data/m1/closecall_price.out"), "w").write("\n".join(lines) + "\n")
