"""Sat 7 Nov 00:3x (NewIdeas ae954df / 4b2c45c '200 STACK'; GOAL's lines): the MBPP12 quality pair of the guarded INERT form,
read before the HumanEval palindrome decides whether stackt runs.

usage (EvalSuite's venv: evalplus + its MBPP items), from ~/BigMoE:
  ~/EvalSuite/.venv/bin/python -I scripts/sat_mbpp_verdicts.py <C> <out.json> [T16]
Reads data/r57/{mq,mt,mtb,mqb}_<C>_runs.json / _tokens.json (palindrome mq / mt / mtb / mqb = qall, qall + T16) and the arm's
R57_INERT_LOG files data/sat/{mt,mtb}_inert.jsonl. Test verdicts: EvalSuite's tools/mbpp_score.py on every pass, compared per
pass pair (mq-mt, mqb-mtb). Prints one line, then 'PICK <T16|none>' last.

T16's MBPP12 line (GOAL Fri 21:18, ~/FableGod/goal/build_1106/GUARD.md, written before any T16 run):
  PASS  verdicts equal to the control's on >= 11 of 12 in EACH pass pair with no net loss above 1, AND no arm run ends at the
        cap where its control ended by itself, AND median tokens per task <= 1.05
  KILL  >= 2 tasks flip to fail in either pass pair, OR any arm run at the cap where its control ended, OR tokens per task >= 1.15
  VOID  rows open = 0, or kept whole > 0 (T16 keeps by the rule only), or GOAL's audit (build_1106/inert_report.py --arm T16)
        finds an accept outside the rule ("-> the arm is void")
  BETWEEN otherwise. Speed is recorded, not the gate: K ratio (kept per check) PASS >= 1.08, KILL < 1.03.
Precedence: VOID, then KILL, then PASS. Only a PASS picks T16 (no fallback form: G16 / D16 / C16 are out)."""
import json
import os
import statistics as st
import subprocess
import sys

sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
import mbpp_score as M  # noqa: E402  (EvalSuite's scorer, read-only use)

C, OUT = sys.argv[1], sys.argv[2]
FORM = (sys.argv[3:] or ["T16"])[0]
assert FORM == "T16", "tonight's only guarded form is T16 (NewIdeas 4b2c45c)"
CTL, ARM = ("mq", "mqb"), ("mt", "mtb")
LOGS = {"mt": "data/sat/mt_inert.jsonl", "mtb": "data/sat/mtb_inert.jsonl"}
REPORT = os.path.expanduser("~/FableGod/goal/build_1106/inert_report.py")
ITS = M.items()


def runs(t):
    try:
        return {r["key"]: r for r in json.load(open(f"data/r57/{t}_{C}_runs.json"))}
    except (OSError, ValueError):
        return {}


def verdicts(t):
    try:
        tok = json.load(open(f"data/r57/{t}_{C}_tokens.json"))
    except (OSError, ValueError):
        return {}
    rows = {k: {"completion_ids": list(v.values())[0][0]} for k, v in tok.items()}
    return {k: bool(v[0]) for k, v in M.score(rows, ITS).items()}


R = {t: runs(t) for t in CTL + ARM}
V = {t: verdicts(t) for t in R}
ks = sorted(set.intersection(*[set(R[t]) for t in CTL + ARM])) if all(R.values()) else []
res = dict(form=FORM, tasks=len(ks), control_passed={t: sum(V[t].values()) for t in CTL},
           arm_passed={t: sum(V[t].values()) for t in ARM})
per = []
for c, x in zip(CTL, ARM):
    fl = [k for k in ks if V[c].get(k) and not V[x].get(k)]
    ga = [k for k in ks if not V[c].get(k) and V[x].get(k)]
    per.append(dict(pair=f"{c}-{x}", equal=sum(V[c].get(k) == V[x].get(k) for k in ks), flips=fl, gains=ga,
                    net_loss=len(fl) - len(ga), cap=[k for k in ks if R[x][k].get("cut") and not R[c][k].get("cut")]))
res["pairs"] = per
t_ = [(R[ARM[0]][k]["n"] + R[ARM[1]][k]["n"]) / (R[CTL[0]][k]["n"] + R[CTL[1]][k]["n"]) for k in ks]
kk = lambda r: r["kept"] if isinstance(r["kept"], (int, float)) else st.mean(r["kept"])
k_ = [(kk(R[ARM[0]][k]) + kk(R[ARM[1]][k])) / (kk(R[CTL[0]][k]) + kk(R[CTL[1]][k])) for k in ks]
T = st.median(t_) if t_ else float("nan")
K = st.median(k_) if k_ else float("nan")
ins = [r.get("inert") or {} for x in ARM for r in R[x].values()]
open_, whole = sum(i.get("open", 0) for i in ins), sum(i.get("whole", 0) for i in ins)
void = []
if ks and open_ == 0:
    void.append("rows open = 0")
if whole > 0:
    void.append(f"kept whole {whole} > 0")
audit = []
for c, x in zip(CTL, ARM):
    try:
        o = subprocess.run([os.path.expanduser("~/BigMoE/.venv/bin/python"), "-I", REPORT, f"data/r57/{c}_{C}_runs.json",
                            f"data/r57/{x}_{C}_runs.json", "--log", LOGS[x], "--arm", FORM], capture_output=True, text=True,
                           timeout=120).stdout
    except (OSError, subprocess.SubprocessError) as e:
        o = f"audit failed: {type(e).__name__}"
    audit += [ln.strip() for ln in o.splitlines() if "audit" in ln]
    if "-> the arm is void" in o:
        void.append(f"{x}: GOAL's audit found an accept outside the rule")
res.update(T=T, K=K, rows_open=open_, kept_whole=whole, audit=audit, void=void)
if len(ks) < 12 or any(len(V[t]) < 12 for t in CTL + ARM):
    v = "MISSING"
elif void:
    v = "VOID"
elif any(len(p["flips"]) >= 2 or p["cap"] for p in per) or T >= 1.15:
    v = "KILL"
elif all(p["equal"] >= 11 and p["net_loss"] <= 1 for p in per) and T <= 1.05:
    v = "PASS"
else:
    v = "BETWEEN"
res["verdict"] = v
res["speed"] = "PASS" if K >= 1.08 else "KILL" if K < 1.03 else "BETWEEN"
print(f"MBPP12 {FORM} vs qall ({len(ks)} tasks, C {C}): {v}: " + "; ".join(
    f"{p['pair']} equal {p['equal']} / {len(ks)}, flips {p['flips'] or 'none'}, gains {p['gains'] or 'none'}, at the cap where the "
    f"control ended {p['cap'] or 'none'}" for p in per) + f"; median tokens per task {T:.3f}; tests passed {res['arm_passed']} vs qall "
    f"{res['control_passed']}; rows open {open_}, kept whole {whole}; void: {void or 'none'}  (GOAL's line: PASS >= 11 / 12 equal in each "
    f"pair, net loss <= 1, no arm at the cap, tokens <= 1.05; KILL >= 2 flips, an arm at the cap, or tokens >= 1.15)")
print(f"  speed (recorded, not the gate): K ratio {K:.3f} -> {res['speed']} (PASS >= 1.08, KILL < 1.03)")
for a in audit:
    print("  " + a)
res["pick"] = FORM if v == "PASS" else "none"
json.dump(res, open(OUT, "w"), indent=1, default=str)
print(f"PICK {res['pick']}")
