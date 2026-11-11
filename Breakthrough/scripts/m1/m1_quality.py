#!/usr/bin/env python3
"""M1 quality side (CHAT150 strike, Breakthrough Sun 2026-11-08): chat flips of the decision map's existing arms for the
45 GB chat miss levers, read-only from ~/FableGod/decision_map/out (arm JSONs, baseline.json) and ~/FableGod/floor_rows.json.
Per arm and domain: all-row top-1 flips vs the baseline argmax, non-tie flips (rows the 125B itself calls sure or mid),
VERDICT2 against the pre-registered lines (code 1.34 [0.87], IFEval 5.42 [2.22], multi 3.69 [1.63]), and paired McNemar
(only-arm / only-reference flips, z) against two references: qom_0.5_live (QOM at cf 0.5, the cleanest QOM arm) and
qom_0_live_ball (QOM-KEEP + DRAFT17, the 45 GB FAST code rule). Writes ~/Breakthrough/data/m1/m1_quality.out."""
import json, math, os, sys
OUT = os.path.expanduser("~/FableGod/decision_map/out"); DST = os.path.expanduser("~/Breakthrough/data/m1")
LINES = {"code": (1.34, 0.87), "ifeval": (5.42, 2.22), "multi": (3.69, 1.63)}
ARMS = ["qom_0.9_live", "qom_0.5_live", "qom_0_live", "qom_0.5_live_bonus5g", "qom_0_live_bonus", "qom_0_live_bq",
        "qom_0_live_ball", "qom_0_live_ball_nw1", "qom_0_live_ball_nw3", "qom_0.5_live_t3", "qom_0_live_t3",
        "qom_0_live_bq_t3", "qom_0_k1_live_t3", "top_k8", "q1u_0.9_live", "qom_0.5_live_c164", "hyb_0.9_u8_live_t3",
        "t3_top2_all"]
REFS = ["qom_0.5_live", "qom_0_live_ball"]
OUTF = "m1_quality.out"
if os.environ.get("M1Q_ARMS"):   # Sun 11:5x: other arm sets (e.g. the 24 / 12 GB chat gates), refs and output name by env
    ARMS = os.environ["M1Q_ARMS"].split(); REFS = os.environ.get("M1Q_REFS", " ".join(REFS)).split()
    OUTF = os.environ.get("M1Q_OUTF", "m1_quality_env.out")
base = json.load(open(os.path.join(OUT, "baseline.json")))
cls = {}
for r in json.load(open(os.path.expanduser("~/FableGod/floor_rows.json"))):
    cls[(r["nm"], int(r["i"]))] = r["cls"]


def flips(arm):
    p = os.path.join(OUT, arm + ".json")
    if not os.path.exists(p): return None
    A = json.load(open(p)); F = {}
    for nm, b in base.items():
        if nm not in A or not isinstance(A[nm], dict) or "tk" not in A[nm]: continue
        v = A[nm]["tk"]
        for j, tb in enumerate(b["tk"]):
            f = int(j < len(v) and v[j][0] != tb[0]) if j < len(v) else 0
            F[(nm, j)] = (b["dom"], f, cls.get((nm, int(b["a"]) + j)) in ("sure", "mid"))
    return F


def main():
    lines = []
    def say(s=""): print(s, flush=True); lines.append(s)
    FL = {a: flips(a) for a in ARMS}
    say("M1 quality: chat (and code) flips of the map's arms at the 45 GB pool unless the name says otherwise")
    say("| arm | windows (code / IFEval / multi) | code all-row [non-tie] | IFEval | multi | VERDICT2 (code / IFEval / multi) |")
    say("|---|---|---|---|---|---|")
    for a in ARMS:
        F = FL[a]
        if F is None: say(f"| {a} | missing | | | | |"); continue
        cells, vd, nw = [], [], []
        for d in ("code", "ifeval", "multi"):
            rows = [(f, nt) for (dom, f, nt) in F.values() if dom == d]
            nwin = len({nm for (nm, j), (dom, f, nt) in F.items() if dom == d})
            nw.append(str(nwin))
            if not rows: cells.append("-"); vd.append("-"); continue
            n = len(rows); fl = sum(f for f, _ in rows); nn = sum(1 for _, nt in rows if nt); fn = sum(f for f, nt in rows if nt)
            ar, ntr = 100.0 * fl / n, 100.0 * fn / max(nn, 1)
            cells.append(f"{ar:.2f} % [{ntr:.2f}] ({fl}/{n})")
            vd.append("PASS" if ar <= LINES[d][0] and ntr <= LINES[d][1] else "FAIL")
        say(f"| {a} | {' / '.join(nw)} | " + " | ".join(cells) + f" | {' / '.join(vd)} |")
    say("")
    say("Paired McNemar on all rows (only-arm / only-reference flips, z); non-tie rows in brackets")
    for ref in REFS:
        R = FL[ref]
        if R is None: continue
        say(f"vs {ref}:")
        for a in ARMS:
            F = FL[a]
            if F is None or a == ref: continue
            parts = []
            for d in ("code", "ifeval", "multi"):
                keys = [k for k, v in F.items() if v[0] == d and k in R]
                if not keys: continue
                b_ = sum(1 for k in keys if F[k][1] and not R[k][1]); c_ = sum(1 for k in keys if R[k][1] and not F[k][1])
                kn = [k for k in keys if F[k][2]]
                bn = sum(1 for k in kn if F[k][1] and not R[k][1]); cn = sum(1 for k in kn if R[k][1] and not F[k][1])
                z = (b_ - c_) / math.sqrt(b_ + c_) if b_ + c_ else 0.0; zn = (bn - cn) / math.sqrt(bn + cn) if bn + cn else 0.0
                parts.append(f"{d} {b_} / {c_}, z {z:+.2f} [{bn} / {cn}, z {zn:+.2f}]")
            say(f"  {a}: " + "; ".join(parts))
    os.makedirs(DST, exist_ok=True)
    open(os.path.join(DST, OUTF), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
