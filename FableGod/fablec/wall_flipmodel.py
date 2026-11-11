"""WALL 1105: a CPU predictor of decision-map flips from keep masks (Fable C, Thu 5 Nov 2026). numpy, ~1 min, no GPU.
Idea: expert outputs on one input are orthogonal with near-equal norms (wall_probe2.out), so the relative error a mask puts
on a row's routed output at a layer follows from the kept RANKS alone: eps^2 = sum of the dropped ranks' squared weights /
all ten (the engine's mean weight by rank). A row's damage = mean eps^2 over the end layers (0-7, 36-47) and over the middle
layers (8-35); its context damage = an exponentially decayed mean of the earlier rows' damage in the window. A logistic
model on the NON-TIE rows (class and margin from floor_rows.json) is fit on rule-based strict arms measured on the map
(uniform topk2 / topk5; gate_top2_cf0999, gate_top5_cf099, gate_top4_cf099, gate_bands, gate_top0_cf0999_L08_35) and then
asked to predict arms of other FORMS it never saw (resident keep, union members, the split): today's twin, split 0.99 /
0.9, the chat route, q1u_0.9_live, oea-style top-8. If those land, it prices unmeasured arms (A / B / C, cloud3's b2d).
Output: wall_flipmodel.out."""
import ast, gzip, json, os, sys
from collections import OrderedDict
import numpy as np
sys.path.insert(0, os.path.expanduser("~/FableGod/fablec"))
from bytes10_replay import windows, live_blocks, POOLS  # noqa: E402

B = os.path.expanduser("~/Breakthrough/data"); OUTD = os.path.expanduser("~/FableGod/decision_map/out")
WP = np.array([0.227, 0.150, 0.117, 0.097, 0.084, 0.075, 0.069, 0.064, 0.060, 0.057]); W2 = WP ** 2; W2 /= W2.sum()
TAIL2 = np.concatenate([np.cumsum(W2[::-1])[::-1], [0.0]])           # eps^2 of keeping ranks < k (k = 0..10)
END = np.array([l < 8 or l >= 36 for l in range(48)]); MID = ~END
LAM = float(os.environ.get("LAM", "0.9"))
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


fr = json.load(open(os.path.expanduser("~/FableGod/floor_rows.json")))
ROW = {}
for r in fr:
    fl = r["flip"] if isinstance(r["flip"], dict) else ast.literal_eval(r["flip"])
    ROW[(r["nm"], int(r["i"]))] = (r["cls"], float(r["m0"]), float(r["p1"]), bool(fl["topk2"]), bool(fl["topk5"]))
W = windows(); WIN = {w["nm"]: (dom, w) for dom, ws in W.items() for w in ws}


def load(p): return json.load(gzip.open(p)) if p.endswith(".gz") else json.load(open(p))


def arm_flips(path, basepath):
    A = load(path); base = load(basepath); out = {}
    for nm, v in A.items():
        if nm not in base or nm not in WIN or "tk" not in v: continue
        ta, tb = v["tk"], base[nm]["tk"]; a0 = int(base[nm]["a"])
        if len(ta) != len(tb): continue
        for j in range(len(ta)): out[(nm, a0 + j)] = int(ta[j][0] != tb[j][0])
    return out


def rule_damage(arm):
    """strict rule-based arms: per window -> (D_end[N], D_mid[N]) mean eps^2 per layer group"""
    out = {}
    for nm, (dom, w) in WIN.items():
        N = len(w["ids"]); g = w["g"]; a = w["a"]; De = np.zeros(N); Dm = np.zeros(N)
        for i in range(N):
            c = g[i] if i >= a else -1.0
            if arm == "topk2": ke = km = 2
            elif arm == "topk5": ke = km = 5
            elif i < a: ke = km = 10
            elif arm == "gate_top2_cf0999": ke = km = 2 if c >= 0.999 else 10
            elif arm == "gate_top5_cf099": ke = km = 5 if c >= 0.99 else 10
            elif arm == "gate_top4_cf099": ke = km = 4 if c >= 0.99 else 10
            elif arm == "gate_bands": ke = km = 2 if c >= 0.999 else (4 if c >= 0.99 else (8 if c >= 0.9 else 10))
            elif arm == "gate_top0_cf0999_L08_35": ke = 10; km = 0 if c >= 0.999 else 10
            De[i] = TAIL2[ke]; Dm[i] = TAIL2[km]
        out[nm] = (De, Dm)
    return out


def replay_damage(rule, pool="45"):
    """resident-keep / member / split forms through the LRU replay (wall_layer_replay's rules) -> per-row damages"""
    Cm = POOLS[pool][0]; out = {}
    def k0_of(c):
        if rule == "EXACT" or c < 0: return 10
        if rule in ("K8",): return 8
        if rule == "Q1U90": return 2 if c >= 0.9 else 10
        if rule == "HYB90U8": return 2 if c >= 0.9 else 8
        if rule == "B2D": return 2 if c >= 0.99 else (5 if c >= 0.9 else 8)
        if rule == "B2C": return 2 if c >= 0.999 else (5 if c >= 0.9 else 8)
        if rule.startswith("QOM5"): return 2 if c >= 0.5 else 10
        return 2
    def strict(c, reach, l):
        if rule in ("K8", "Q1U90", "HYB90U8", "B2D", "B2C"): return True
        if rule.startswith("QOM"): return False
        parts = rule.split("+"); thr = {"SPLIT99": 0.99, "SPLIT90": 0.9}[parts[0]]
        if c >= thr: return True
        if "M90" in parts and c >= 0.9 and 12 <= l <= 35: return True
        return "R07" in parts and c >= 0 and reach < 0.7
    def zero(c, l): return "Z999" in rule and c >= 0.999 and 8 <= l <= 35
    for dom in ("code", "ifeval", "multi"):
        pools = [OrderedDict() for _ in range(48)]
        for w in W[dom]:
            ids = w["ids"]; a = w["a"]; N = len(ids); De = np.zeros(N); Dm = np.zeros(N)
            for l in range(48):
                c_ = pools[l]
                for e in ids[:a, l].ravel().tolist():
                    if e in c_: c_.move_to_end(e)
                    else:
                        c_[e] = 1
                        if len(c_) > Cm[l]: c_.popitem(last=False)
            for rows, cfs, kp in live_blocks(w, bonus="next"):
                cfs = list(cfs)
                if cfs[-1] < 0 and not rule.endswith("BQ") and rule not in ("K8", "Q1U90", "HYB90U8"): cfs[-1] = 1.0
                E = ids[rows]; R = len(rows); k0 = [k0_of(c) for c in cfs]; rch = []; acc = 1.0
                for c in cfs:
                    rch.append(acc); acc *= (c if c >= 0 else 1.0)
                for l in range(48):
                    c_ = pools[l]; zr = [zero(cc, l) for cc in cfs]; st = [strict(cc, q, l) for cc, q in zip(cfs, rch)]
                    loads = set()
                    for r in range(R):
                        if not zr[r]: loads.update(E[r, l, :k0[r]].tolist())
                    for e in loads:
                        if e in c_: c_.move_to_end(e)
                        else:
                            c_[e] = 1
                            if len(c_) > Cm[l]: c_.popitem(last=False)
                    keepset = set()
                    for r in range(R):
                        if zr[r]:
                            if r < kp: (De if END[l] else Dm)[rows[r]] += 1.0
                            continue
                        row = E[r, l].tolist(); kk = k0[r]; d2 = 0.0
                        keepset.update(row[:kk])
                        for q, e in enumerate(row[kk:], start=kk):
                            if e in loads or (not st[r] and e in c_): keepset.add(e)
                            else: d2 += W2[q]
                        if r < kp: (De if END[l] else Dm)[rows[r]] += d2
                    for e in keepset:
                        if e in c_: c_.move_to_end(e)
            out[w["nm"]] = (De / END.sum(), Dm / MID.sum())
    return out


def features(dmg, uniform=False):
    """-> dict (nm, i) -> feature vector for answer rows; context = decayed mean of earlier rows' damage"""
    F = {}
    for nm, (dom, w) in WIN.items():
        De, Dm = dmg[nm]; a = w["a"]; ce = cm = 0.0
        for i in range(len(De)):
            if i >= a and (nm, i) in ROW:
                cls, m0, p1, _, _ = ROW[(nm, i)]
                F[(nm, i)] = (dom, cls, np.array([1.0, dom == "ifeval", dom == "multi", np.log(m0 + 0.05), p1, De[i], Dm[i], ce, cm,
                                                   np.sqrt(De[i]), np.sqrt(ce)]))
            ce = LAM * ce + (1 - LAM) * De[i]; cm = LAM * cm + (1 - LAM) * Dm[i]
    return F


def fit(X, y, ridge=1e-3, it=40):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30))); Wt = p * (1 - p) + 1e-9
        H = (X * Wt[:, None]).T @ X + ridge * np.eye(X.shape[1]); b += np.linalg.solve(H, X.T @ (y - p) - ridge * b)
    return b


TRAIN = [("topk2", None), ("topk5", None), ("gate_top2_cf0999", B + "/floor1"), ("gate_top5_cf099", B + "/floor1"), ("gate_bands", B + "/floor1"),
         ("gate_top4_cf099", B + "/floor1_36"), ("gate_top0_cf0999_L08_35", B + "/floor1_36")]
DATA = {}
for arm, d in TRAIN:
    F = features(rule_damage(arm))
    if d is None: fl = {k: int(v[3] if arm == "topk2" else v[4]) for k, v in ROW.items()}
    else: fl = arm_flips(os.path.join(d, arm + ".json"), os.path.join(d, "baseline.json"))
    DATA[arm] = [(F[k][0], F[k][1], F[k][2], fl[k]) for k in F if k in fl]


def rate(samples, pred=None, sel=lambda c: c != "tie"):
    out = []
    for dom in ("code", "ifeval", "multi"):
        idx = [i for i, s in enumerate(samples) if s[0] == dom and sel(s[1])]
        if not idx: out.append("-"); continue
        out.append(f"{100 * np.mean([samples[i][3] if pred is None else pred[i] for i in idx]):.2f}")
    return " / ".join(out)


def design(samples): return np.stack([s[2] for s in samples])


nt = lambda S: [s for s in S if s[1] != "tie"]
ALL = [s for arm in DATA for s in nt(DATA[arm])]
b = fit(design(ALL), np.array([s[3] for s in ALL], float))
say(f"flip model (non-tie rows; LAM {LAM}); features [1, ifeval, multi, log margin, p1, own end, own mid, ctx end, ctx mid, sqrt own end, sqrt ctx end]; coef {np.round(b, 2).tolist()}")
say("| arm (training, rule-based strict) | measured non-tie flips code / IFEval / multi | model in sample | model leave-this-arm-out |")
say("|---|---|---|---|")
for arm in DATA:
    S = nt(DATA[arm]); X = design(S); p_in = 1 / (1 + np.exp(-X @ b))
    REST = [s for a2 in DATA if a2 != arm for s in nt(DATA[a2])]; b2 = fit(design(REST), np.array([s[3] for s in REST], float)); p_out = 1 / (1 + np.exp(-X @ b2))
    say(f"| {arm} | {rate(S)} | {rate(S, p_in)} | {rate(S, p_out)} |")

TEST = [("QOM0", "45", B + "/dm_s7/results_thu/qom_0_live_ball+code.json.gz", OUTD + "/baseline.json"),
        ("SPLIT99", "45", B + "/dm_s7/results_thu/qom_0_split0.99_live_ball+code.json.gz", OUTD + "/baseline.json"),
        ("SPLIT90", "45", B + "/dm_s7/results_thu/qom_0_split0.9_live_ball+code.json.gz", OUTD + "/baseline.json"),
        ("QOM5BQ", "45", B + "/dm_s7/results_wed/qom_0.5_live_bonus5g.json.gz", OUTD + "/baseline.json"),
        ("QOM5BQ", "24", B + "/dm_s7/results_wed/qom_0.5_live_c147_bonus5g.json.gz", OUTD + "/baseline.json"),
        ("Q1U90", "45", B + "/dm_s7/results_wed/q1u_0.9_live_bonus9g.json.gz", OUTD + "/baseline.json"),
        ("HYB90U8", "45", B + "/floor1/hyb_0.9_u8_live_t3.json", B + "/floor1/baseline.json"),
        ("K8", "45", B + "/floor1/oea_k8.json", B + "/floor1/baseline.json")]
say("")
say("| form (never seen by the fit) | pool | measured non-tie flips | model | mean own damage (end / mid) on non-tie rows |")
say("|---|---|---|---|---|")
for rule, pool, path, bp in TEST:
    dmg = replay_damage(rule, pool); F = features(dmg); fl = arm_flips(path, bp)
    S = nt([(F[k][0], F[k][1], F[k][2], fl[k]) for k in F if k in fl]); X = design(S); p = 1 / (1 + np.exp(-X @ b))
    say(f"| {rule} ({os.path.basename(path).split('.json')[0]}) | {pool} | {rate(S)} | {rate(S, p)} | {X[:, 5].mean():.3f} / {X[:, 6].mean():.3f} |")
say("")
say("| unmeasured form | pool | model: non-tie flips code / IFEval / multi | mean own damage (end / mid) |")
say("|---|---|---|---|")
REF = {k: None for k in ROW}
for rule in ("SPLIT99+R07", "SPLIT99+R07+Z999", "SPLIT99+R07+M90+Z999", "SPLIT90+R07", "B2D", "B2C"):
    dmg = replay_damage(rule, "45"); F = features(dmg); S = nt([(F[k][0], F[k][1], F[k][2], 0) for k in F]); X = design(S); p = 1 / (1 + np.exp(-X @ b))
    say(f"| {rule} | 45 | {rate(S, p)} | {X[:, 5].mean():.3f} / {X[:, 6].mean():.3f} |")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wall_flipmodel.out"), "w").write("\n".join(lines) + "\n")
