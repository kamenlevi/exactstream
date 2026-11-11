"""Per-domain row counts, flips and paired differences for the union arms (read-only on the map's out/)."""
import json, os, math
import numpy as np
OUT = os.path.expanduser("~/FableGod/decision_map/out")
base = json.load(open(os.path.join(OUT, "baseline.json")))
def flipv(name):
    R = json.load(open(os.path.join(OUT, name + ".json"))); v = {}
    for nm, b in base.items():
        m = min(len(b["tk"]), len(R[nm]["tk"]))
        f = np.array(R[nm]["tk"][:m])[:, 0] != np.array(b["tk"][:m])[:, 0]
        v.setdefault(b["dom"], []).append(f)
    return {d: np.concatenate(x) for d, x in v.items()}
arms = ["gate_top2_cf099", "q1u_cf09", "q1u_cf09_t3", "oea_k8", "oea_k6", "hyb_0.9_u8", "hyb_0.9_u6", "hyb_0.9_u8_t3"]
F = {a: flipv(a) for a in arms}
for d in ("code", "ifeval", "multi"):
    n = len(F["q1u_cf09"][d]); print(f"{d}: rows {n}")
    for a in arms:
        k = int(F[a][d].sum()); p = k / n; se = math.sqrt(p * (1 - p) / n)
        print(f"  {a:16s} {k:4d} flips {100*p:.2f} % (+-{100*se:.2f} 1 SD)")
    for a, b in (("q1u_cf09_t3", "q1u_cf09"), ("q1u_cf09_t3", "gate_top2_cf099"), ("hyb_0.9_u8_t3", "gate_top2_cf099"), ("hyb_0.9_u6", "gate_top2_cf099")):
        x, y = F[a][d], F[b][d]; only_a, only_b = int((x & ~y).sum()), int((~x & y).sum())
        z = (only_a - only_b) / math.sqrt(max(only_a + only_b, 1))
        print(f"  paired {a} vs {b}: only-{a} {only_a}, only-{b} {only_b}, McNemar z {z:+.2f}")
