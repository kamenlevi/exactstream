"""P1-OEA mixed-run read-out (NewIdeas, Tue 3 Nov 2026): the two-pass check end to end (certified rows cut, the rest
exact, through the context), as dm_cert_read.py reads Breakthrough's cert_* runs. Per run, spec 'run:src:theta' (the
certificate = each measured row's own p1 in the all-rows run src >= theta): certified share, two-pass certified-row flips
(Wilson 95 %), uncertified (exact) rows' flips through the context, all rows, windows with a flip; flips vs the baseline
argmax. Reads ~/FableGod/decision_map/out (read-only); writes ~/NewIdeas/data/P1OEA_mixed.txt when run without args.
Usage (BigMoE venv): python p1_mixed_read.py [run:src:theta ...]
Default: p1c_top_k2_p08:p1_top_k2:0.8 (control) p1c_oea_k1_p08:p1_oea_k1:0.8; self-test: cert_top2_p08:topk2:0.8
(must give CERT_read's code 0.27 % (8/2976))."""
import json, math, os, sys
import numpy as np
OUT = os.path.expanduser("~/FableGod/decision_map/out"); DST = os.path.expanduser("~/NewIdeas/data/P1OEA_mixed.txt")
base = json.load(open(os.path.join(OUT, "baseline.json")))
specs = sys.argv[1:] or ["p1c_top_k2_p08:p1_top_k2:0.8", "p1c_oea_k1_p08:p1_oea_k1:0.8"]
lines = []
def say(s=""): print(s, flush=True); lines.append(s)


def wilson(k, n, z=1.96):
    if not n: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, 100 * (c - h)), 100 * (c + h)


say("P1-OEA mixed runs (two-pass end to end); flips vs the baseline argmax")
say("| run (certificate) | domain | certified share | two-pass certified-row flips [95 %] | uncertified (exact) rows through the context | all rows | windows with a flip |")
say("|---|---|---|---|---|---|---|")
T = {}
for sp in specs:
    run, src, thr = sp.split(":"); thr = float(thr)
    if not os.path.exists(os.path.join(OUT, run + ".json")): say(f"| {run} | not run | | | | | |"); continue
    R = json.load(open(os.path.join(OUT, run + ".json"))); S = json.load(open(os.path.join(OUT, src + ".json")))
    for dom in ("code", "ifeval", "multi"):
        n = nc = fc = fu = nu = wf = nw = 0
        for nm, b in base.items():
            if b["dom"] != dom: continue
            m = min(len(b["tk"]), len(R[nm]["tk"]), len(S[nm]["lp"]))
            cert = np.exp(np.array(S[nm]["lp"])[:m, 0]) >= thr
            f = np.array(R[nm]["tk"][:m])[:, 0] != np.array(b["tk"][:m])[:, 0]
            n += m; nc += int(cert.sum()); fc += int(f[cert].sum()); fu += int(f[~cert].sum()); nu += int((~cert).sum())
            wf += int(f.any()); nw += 1
        lo, hi = wilson(fc, nc); T[(run, dom)] = (fc, nc, hi, fc + fu, n)
        say(f"| {run} ({src} p1 >= {thr}) | {dom} | {100 * nc / n:.1f} % | {100 * fc / max(nc, 1):.2f} % ({fc}/{nc}) [{lo:.2f}-{hi:.2f}] | "
            f"{100 * fu / max(nu, 1):.2f} % ({fu}/{nu}) | {100 * (fc + fu) / n:.2f} % | {wf}/{nw} |")
if not sys.argv[1:] and ("p1c_top_k2_p08", "code") in T and ("p1c_oea_k1_p08", "code") in T:
    cf, cn, chi, _, _ = T[("p1c_top_k2_p08", "code")]; af, an, _, _, _ = T[("p1c_oea_k1_p08", "code")]; f = 100 * af / max(an, 1)
    v = "KILL" if f > 5 or f > 2 * chi else "PASS" if f <= 1 and f <= chi else "BETWEEN"
    say(f"\n## Verdict (pre-registered in QUEUE before the runs; code, theta 0.8): P1-K1 two-pass certified-row flips {f:.2f} % vs the "
        f"control's upper bound {chi:.2f} % -> {v} (PASS: <= 1 % and <= the control's upper bound; KILL: > 5 % or > 2x the bound)")
if not sys.argv[1:]: open(DST, "w").write("\n".join(lines) + "\n")
