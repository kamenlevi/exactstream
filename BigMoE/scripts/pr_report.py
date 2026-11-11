"""The partial-sum pre-route capture's report (r57_screen R57_PREROUTE=1; the orchestrator 11-02 22:2x, Fable A's
~/FableGod/NEWIDEAS_SCREEN.md 1): per variant (r57 = L+1's router on L's MoE input; full = on L+1's MLP-side read of its
pre-attention stream, only L+1's attention write missing; p3 / p6 = the same read after L's partial output with ranks >= 3 /
>= 6 dropped; fullA = L+1's ATTENTION-side read of L's full output, the control: files made before Wed 4 Nov 05:5x have no
fullA and their full / p3 / p6 ARE that attention-side read, which L+1's router never sees), recall@10 per row
(|predicted top-10 & true top-10| / 10) and of the union (the check's reads), precision@96 / recall@96 of the ranked
unique list (each expert by its best row log-prob, the look-ahead's PF_RANKED at PF_B 96) vs the true union, by domain and
layer group. Fable C's line (AUDIT C1-1, the orchestrator 23:0x): recall >= 0.85 AND precision@96 >= 0.80. Line
(pre-registered): PASS recall@10 >= 0.85 for p3 (>= 0.90 for p6); KILL p3 < 0.75; else BETWEEN. Judged on code + chat
pooled per row; the union recall and r57 (this run's own R-57 baseline) beside it.
usage: .venv/bin/python scripts/pr_report.py [TAG=pr_256] -> data/r57/<TAG>_preroute.txt"""
import json, os, sys
import numpy as np

TAG = sys.argv[1] if len(sys.argv) > 1 else "pr_256"
R = os.environ.get("PR_DIR", os.path.expanduser("~/BigMoE/data/r57"))
runs = json.load(open(os.path.join(R, f"{TAG}_preroute.json")))
L = []
def say(s=""):
    print(s); L.append(s)
V = ("r57", "full", "fullA", "p3", "p6")
dom = lambda r: "code" if r["key"].startswith("humaneval") else "agent" if r["key"].startswith("agent") else \
    "math" if r["key"].startswith(("math500", "gsm8k")) else "chat"
S = {}                                                  # (domain, variant) -> [(layer, rows, row recall, union recall)]
for r in runs:
    for v, xs in r["samples"].items():
        S.setdefault((dom(r), v), []).extend(xs); S.setdefault(("all", v), []).extend(xs)
say(f"Pre-route recall@10 ({TAG}: {len(runs)} runs, {sum(len(r['samples'].get('r57', [])) for r in runs)} layer samples)")
if not any("fullA" in r["samples"] for r in runs):
    say("NOTE: made before the read fix (Wed 4 Nov 05:5x): full / p3 / p6 here are L+1's ATTENTION-side HC read, not the "
        "MLP-side read its router sees; not a pre-route measurement")
say("| domain | variant | recall per row | recall of the union | precision@96 | recall@96 | layers 0-7 | 8-39 | 40-46 (per row) |")
say("|---|---|---|---|---|---|---|---|---|")
for d in ("code", "chat", "agent", "math", "all"):
    for v in V:
        x = np.array(S.get((d, v), []))
        if not len(x):
            continue
        g = lambda a, b: f"{x[(x[:, 0] >= a) & (x[:, 0] < b), 2].mean():.3f}" if ((x[:, 0] >= a) & (x[:, 0] < b)).any() else "-"
        pr = (f"{x[:, 4].mean():.3f} | {x[:, 5].mean():.3f}" if x.shape[1] > 5 else "- | -")
        say(f"| {d} | {v} | {x[:, 2].mean():.3f} | {x[:, 3].mean():.3f} | {pr} | {g(0, 8)} | {g(8, 40)} | {g(40, 47)} |")
a = {v: np.array(S.get(("all", v), [[0, 0, np.nan, np.nan]]))[:, 2].mean() for v in V}
verdict = "PASS" if a["p3"] >= 0.85 and a["p6"] >= 0.90 else "KILL" if a["p3"] < 0.75 else "BETWEEN"
say(f"\nVerdict (pre-registered): p3 {a['p3']:.3f}, p6 {a['p6']:.3f} per row (R-57 here {a['r57']:.3f}, full {a['full']:.3f}"
    + (f", fullA {a['fullA']:.3f}" if a["fullA"] == a["fullA"] else "") + ") "
    f"-> {verdict} (PASS p3 >= 0.85 and p6 >= 0.90; KILL p3 < 0.75)")
x3 = np.array(S.get(("all", "p3"), []))
if len(x3) and x3.shape[1] > 5:
    p96 = x3[:, 4].mean()
    say(f"Fable C's line (C1-1): p3 recall per row {a['p3']:.3f} and precision@96 {p96:.3f} -> "
        f"{'PASS' if a['p3'] >= 0.85 and p96 >= 0.80 else 'not met'} (recall >= 0.85 AND precision@96 >= 0.80)")
open(os.path.join(R, f"{TAG}_preroute.txt"), "w").write("\n".join(L) + "\n")
