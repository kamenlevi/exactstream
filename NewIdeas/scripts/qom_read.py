"""Read-out of the Q1-ON-MISS map arms (NewIdeas, Tue 3 Nov 2026 17:2x; pre-registered in ~/NewIdeas/QUEUE.md before any
GPU run). Arms (masks ~/NewIdeas/data/dm_masks/<arm>, scripts/dm_oea_masks.py): qom_0.9_live, qom_0.5_live, qom_0_live =
on live-cut checks, rows whose drafter cf >= theta keep ranks 0-1, the check's union members and every rank 2-9 expert
resident in the base pool (CPU LRU replay at laysz45_256's C_m); Q1's SSD reads at that theta. Lines (the union arms'):
per domain all-row flips <= Q1 at 0.99's level (gate_top2_cf099) = PASS, <= 1.5x = BETWEEN, above = KILL. Paired (McNemar)
vs gate_top2_cf099 and vs q1u_0.9_live. Reads ~/FableGod/decision_map/out (read-only); writes ~/NewIdeas/data/QOM_read.txt.
Usage (BigMoE venv): python qom_read.py [arm ...]"""
import json, math, os, re, sys
import numpy as np
OUT = os.path.expanduser("~/FableGod/decision_map/out"); DST = os.path.expanduser("~/NewIdeas/data/QOM_read.txt")
base = json.load(open(os.path.join(OUT, "baseline.json"))); DOMS = ("code", "ifeval", "multi")
ARMS = sys.argv[1:] or ["qom_0.9_live", "qom_0.5_live", "qom_0_live"]
lines = []
def say(s=""): print(s, flush=True); lines.append(s)


def flips(name):
    p = os.path.join(OUT, name + ".json")
    if not os.path.exists(p): return None
    R = json.load(open(p)); v = {}
    for nm, b in base.items():
        m = min(len(b["tk"]), len(R[nm]["tk"]))
        f = np.r_[np.array(R[nm]["tk"][:m])[:, 0] != np.array(b["tk"][:m])[:, 0], np.zeros(len(b["tk"]) - m, bool)]
        v.setdefault(b["dom"], []).append(f)
    return {d: np.concatenate(x) for d, x in v.items()}


F = {a: flips(a) for a in ["gate_top2_cf099", "gate_top2_cf09", "q1u_0.9_live"] + ARMS}
def twin(a):   # Wed 03:4x: _t3 / _c164 arms are also paired with their twin (the 3-bit cost / the C 160 cost), as registered;
    if "_ball_" in a: return a.replace("_ball_", "_bq_")          # Wed 14:0x screen 6 (DRAFT17): all last rows gated vs C, same pool
    if re.search(r"_c\d+_t3(keep)?$", a):                         # Wed 07:4x screen 4 (the 24 GB pool): top-1 (B') vs top-2 (C)
        return re.sub(r"_t3keep$", "_t3", a.replace("_k1_", "_")) if "_k1_" in a else re.sub(r"_c\d+_t3$", "_t3", a)
    if a.endswith("_bq"): return a[:-3]                            # Wed 04:4x: screen 1 in QOM's form vs the ungated arm
    if a.endswith("_bq_t3"): return a[:-6] + "_t3"                 # its 3-bit form vs the ungated 3-bit arm
    if a.endswith("_bq_t2"): return a[:-3] + "_t3"                 # Wed 05:5x: 2-bit sure rows (M-34) vs the 3-bit form
    if not (re.search(r"_k1(_|$)", a) and a.endswith("_t3")):    # top-1 arms pair below, not with their 4-bit twin
        for s_ in ("_t3", "_c164"):   # Wed 04:0x: <arm>_bonus[N] (BYTES10 screen 1) with its ungated arm
            if a.endswith(s_): return a[:-len(s_)]
    m_ = re.match(r"(.+)_bonus\d*$", a)
    if m_: return m_.group(1)
    m_ = re.fullmatch(r"hyb_([0-9.]+)_k1_u10_live_t3", a)       # screen 2: top-1 vs its top-2 twin (QUEUE 04:0x)
    if m_: return f"q1u_{m_.group(1)}_live_t3"
    m_ = re.fullmatch(r"qom_([0-9.]+)_k1_live_t3", a)            # screen 2b: QOM top-1 vs QOM top-2 at 3 bits
    return f"qom_{m_.group(1)}_live_t3" if m_ else None
def extra(a):   # Wed 07:4x screen 4: (B) also vs the 45 GB-pool top-1 arms (A, if run, and 2b)
    return (re.sub(r"_c\d+_t3(keep)?$", "_t3", a), "qom_0_k1_live_t3") if re.search(r"_k1_.*_c\d+_t3(keep)?$", a) else ()
for a_ in ARMS:
    for t_ in ((twin(a_),) if twin(a_) else ()) + extra(a_):
        if t_ not in F: F[t_] = flips(t_)
lvl = {d: 100 * F["gate_top2_cf099"][d].mean() for d in DOMS}
say(f"Q1-ON-MISS arms: all-row flips per domain vs Q1 at 0.99's level (code {lvl['code']:.2f} / IFEval {lvl['ifeval']:.2f} / "
    f"multi {lvl['multi']:.2f} %); PASS <= 1x, BETWEEN <= 1.5x, KILL above")
say("| arm | code | IFEval | multi | verdict per domain |")
say("|---|---|---|---|---|")
for a in ["gate_top2_cf09", "q1u_0.9_live"] + ARMS:
    X = F[a]
    if X is None: say(f"| {a} | not run | | | |"); continue
    v = []
    for d in DOMS:
        r = 100 * X[d].mean() / lvl[d]; v.append("PASS" if r <= 1 else "BETWEEN" if r <= 1.5 else "KILL")
    say(f"| {a} | " + " | ".join(f"{100 * X[d].mean():.2f} % ({int(X[d].sum())})" for d in DOMS) + f" | {' / '.join(v)} |")
for a in ARMS:                    # Wed 04:0x: BYTES10 screen 1's second condition (Fable C's line): <= the ungated arm + 0.3 points
    if re.search(r"_(bonus\d*|bq|bq_t2)$", a) and F[a] is not None and F.get(twin(a)) is not None:
        say(f"  {a}: vs {twin(a)} + 0.3 points: " + " / ".join(
            f"{d} {100 * F[a][d].mean():.2f} vs {100 * F[twin(a)][d].mean() + 0.3:.2f} ({'ok' if 100 * F[a][d].mean() <= 100 * F[twin(a)][d].mean() + 0.3 else 'OVER'})" for d in DOMS))
say("\nPaired (McNemar: only-A, only-B, z):")
for a in ARMS:
    for b in ("gate_top2_cf099", "q1u_0.9_live") + ((twin(a),) if twin(a) else ()) + extra(a):
        if F[a] is None or F[b] is None: continue
        for d in DOMS:
            x, y = F[a][d], F[b][d]; oa, ob = int((x & ~y).sum()), int((~x & y).sum())
            say(f"  {a} vs {b} {d}: {oa} / {ob}, z {(oa - ob) / math.sqrt(max(oa + ob, 1)):+.2f}")
open(DST, "w").write("\n".join(lines) + "\n")
