"""AUDIT 1105, check F3 on data (Fable C, Thu 5 Nov 2026, 07:1x; CPU, seconds). For each measured map arm in
~/Breakthrough/data/floor1 (per-window 'tk' = the arm's top tokens per answer row, 'off'), join the arm's argmax with the
batch's own baseline argmax, classify every answer row by the cf of the draft it verifies (bytes10_replay.windows(): g[row],
-1 = no draft) and report the flip rate per cf band, over all answer rows and over the live-realized rows (A .. first miss of
each live segment). Purpose: the cost of top-8 (+ members) on the unsure bands, which cloud3's band table prices at 0.00 %.
Output: audit_k8_bands.out."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/FableGod/fablec"))
from bytes10_replay import windows, live_segments  # noqa: E402

F1 = os.path.expanduser("~/Breakthrough/data/floor1")
ARMS = sys.argv[1:] or ["oea_k8", "hyb_0.9_u8_live_t3", "gate_bands", "gate_bands+hc8w", "gate_top5_cf099", "gate_top2_cf0999"]
EDGES = [(-1.5, 0.0, "no draft"), (0.0, 0.5, "0-0.5"), (0.5, 0.9, "0.5-0.9"), (0.9, 0.99, "0.9-0.99"), (0.99, 0.999, "0.99-0.999"), (0.999, 1.01, ">=0.999")]
base = json.load(open(os.path.join(F1, "baseline.json")))
W = windows()
lines = []
for arm in ARMS:
    p = os.path.join(F1, arm + ".json")
    if not os.path.exists(p): lines.append(f"{arm}: no file"); continue
    A = json.load(open(p))
    lines.append(f"==== {arm} ({len(A)} windows) vs floor1/baseline.json; flips = argmax differs; rows by the cf band of the draft the row verifies")
    for dom, ws in W.items():
        acc = {b[2]: [0, 0, 0, 0] for b in EDGES}; tot = [0, 0, 0, 0]   # all flips, all rows, realized flips, realized rows
        for w in ws:
            nm = w["nm"]
            if nm not in A or nm not in base: continue
            ta, tb = A[nm]["tk"], base[nm]["tk"]; N = len(w["ids"]); g = w["g"]
            if len(ta) != len(tb): lines.append(f"  {nm}: length mismatch {len(ta)} vs {len(tb)}"); continue
            off = N - len(ta)
            real = set()
            for (k0, m0, A0), dc0 in zip(w["checks"], w["dcf"]):
                for (k, m, A_, dc) in live_segments(k0, m0, A0, dc0):
                    for r in range(A_, min(A_ + min(k, m) + 1, N)): real.add(r)
            for r in range(off, N):
                fl = int(ta[r - off][0] != tb[r - off][0]); c = float(g[r])
                for lo, hi, name in EDGES:
                    if lo <= c < hi:
                        a = acc[name]; a[0] += fl; a[1] += 1
                        if r in real: a[2] += fl; a[3] += 1
                        break
                tot[0] += fl; tot[1] += 1
                if r in real: tot[2] += fl; tot[3] += 1
        s = f"  {dom}: all rows {100 * tot[0] / max(tot[1], 1):.2f} % ({tot[0]}/{tot[1]}); realized {100 * tot[2] / max(tot[3], 1):.2f} % ({tot[2]}/{tot[3]}) | by band (all / realized): "
        s += "; ".join(f"{name} {100 * a[0] / max(a[1], 1):.1f} % ({a[1]}) / {100 * a[2] / max(a[3], 1):.1f} % ({a[3]})" for name, a in acc.items() if a[1])
        lines.append(s)
out = "\n".join(lines); print(out); open(os.path.join(os.path.expanduser("~/FableGod/fablec"), "audit_k8_bands.out"), "w").write(out + "\n")
