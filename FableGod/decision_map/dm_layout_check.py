"""Layout check read-out (Breakthrough; the orchestrator Tue 18:0x): the map's JSONs from the converted-store run
(out_layout/test: BIGMOE_EXT_BUILD=build_nat, DM_R3=r3_overlay) vs the control run in the same slot (out_layout/ctrl:
today's ext and records3), and vs the stored out/ runs: per name and window, top-8 ids and log-probs bitwise.
Usage (BigMoE venv): python dm_layout_check.py -> out_layout/LAYOUT_check.txt"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "out_layout")
lines = []
def cmp(a, b, label):
    if not (os.path.exists(a) and os.path.exists(b)): lines.append(f"{label}: missing"); return None
    A, B = json.load(open(a)), json.load(open(b)); ok = True; n = 0
    for nm, x in A.items():
        if not isinstance(x, dict) or "tk" not in x: continue
        y = B.get(nm); n += 1
        same = y is not None and np.array_equal(np.array(x["tk"]), np.array(y["tk"][:len(x["tk"])])) and \
            np.array_equal(np.array(x["lp"], np.float32), np.array(y["lp"][:len(x["lp"])], np.float32))
        ok &= bool(same)
        if not same and y is not None:
            m = min(len(x["tk"]), len(y["tk"])); t1 = np.mean(np.array(x["tk"])[:m, 0] == np.array(y["tk"])[:m, 0])
            lines.append(f"  {label} {nm}: DIFFERS (top-1 agreement {t1:.4f})")
    lines.append(f"{label}: {n} windows, {'all bitwise equal' if ok else 'DIFFER'}"); return ok
res = []
for name in ("baseline", "t3_top2_all"):
    res.append(cmp(os.path.join(D, "test", name + ".json"), os.path.join(D, "ctrl", name + ".json"), f"{name} test vs ctrl"))
    cmp(os.path.join(D, "test", name + ".json"), os.path.join(HERE, "out", name + ".json"), f"{name} test vs stored out/")
v = "PASS" if res and all(r is True for r in res) else "FAIL" if any(r is False for r in res) else "INCOMPLETE"
lines.append(f"LAYOUT CHECK: {v} (the converted store + build_nat vs today's store and ext, same slot)")
open(os.path.join(D, "LAYOUT_check.txt"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
