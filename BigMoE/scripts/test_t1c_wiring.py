"""Desk test of the screen's R57_TIERS=1c wiring (Mon 16 Nov; Fit12 LOSSY_1C_TIERS_SPEC.md section 6 check 1 + the counter).
The screen's own _t1c (taken from its source with ast, not re-typed) runs on the decision map's captured exact routing of
the 12 code windows (~/FableGod/decision_map/out/xcap: answer rows' top-10 ids + normalised weights), in 16-row checks
(what the gate runs) and 32-row blocks from the first answer row, MLX on the CPU. Per (row, rank) slot and layer:
  - drop mask (km = tier != 2) and no-LSB marks (t["nolsb"] = (tier[:, :2] != 0, None): MSB-3 or drop) == bigmoe.tiers1c.tiers_np's tiers;
  - the counter [e4, e3, edrop] == the distinct union experts by tiers_np's tier, and e4 + e3 + edrop == the union;
  - per check-layer means vs Fit12's reference (16 rows: union 76.6 = e4 8.7 + e3 20.5 + edrop 47.5; 32: 114.2 / 15.3 /
    30.7 / 68.2).
  .venv/bin/python scripts/test_t1c_wiring.py [screen.py]  -> data/deepdraft/t1c_wiring_test.json"""
import ast
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import mlx.core as mx
from bigmoe import tiers1c as T

mx.set_default_device(mx.cpu)
HERE = os.path.dirname(os.path.abspath(__file__))
SCREEN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "r57_screen_fast.py")
M = os.path.expanduser("~/FableGod/decision_map/out")
REF = {16: (76.6, 8.7, 20.5, 47.5), 32: (114.2, 15.3, 30.7, 68.2)}


def screen_t1c(path):
    src = open(path).read()
    fns = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "_t1c"]
    assert len(fns) == 1, len(fns)
    FG = {"t1cn": []}
    ns = {"T1C": T, "mx": mx, "FG": FG, "TIERS_S": 0.98, "TIERS_LAM": 0.016}
    exec(compile(ast.Module(body=[fns[0]], type_ignores=[]), path, "exec"), ns)
    return ns["_t1c"], FG, ast.get_source_segment(src, fns[0])


class Layer:
    def __init__(self, w):
        self._bm = {"tier": {"scores": mx.array(w[None].astype(np.float32))}}


def main():
    f1c, FG, seg = screen_t1c(SCREEN)
    files = sorted(f for f in os.listdir(os.path.join(M, "bt", "tiers_1c")) if "_code_" in f)
    stats = {R: [] for R in (16, 32)}
    bad = {"km": 0, "nolsb": 0, "counts": 0, "union": 0}
    slots = 0
    for f in files:
        z = np.load(os.path.join(M, "xcap", f))
        ids, w = z["ids"].astype(np.int64), z["w"]
        m, L = ids.shape[0], ids.shape[1]
        for R in (16, 32):
            for b0 in range(0, m - R + 1, R):                 # full checks only (the reference's form)
                r = slice(b0, b0 + R)
                for l in range(L):
                    i_, w_ = ids[r, l], w[r, l]
                    tn = T.tiers_np(i_, w_)
                    lay = Layer(w_)
                    FG["t1cn"] = []
                    km = np.array(f1c(lay, mx.array(i_[None].astype(np.int32))))[0]
                    nl = lay._bm["tier"]["nolsb"]
                    c = [int(x) for x in np.array(FG["t1cn"][0])]
                    bad["km"] += int((km != (tn != 2)).sum())
                    bad["nolsb"] += int(nl[1] is not None) + int((np.array(nl[0]) != (tn[:, :2] != 0)).sum())   # MSB-3 or drop (Leap)
                    u = np.unique(i_)
                    te = {int(e): int(tn[i_ == e].min()) for e in u}
                    want = [sum(1 for v in te.values() if v == t_) for t_ in (0, 1, 2)]
                    bad["counts"] += int(c != want)
                    bad["union"] += int(sum(c) != len(u))
                    stats[R].append([len(u)] + c)
                    slots += tn.size
    out = {"screen": os.path.relpath(SCREEN, os.path.join(HERE, "..")), "windows": len(files), "slots": slots,
           "mismatches": bad, "means": {}}
    for R in (16, 32):
        a = np.array(stats[R], float)
        mu = [round(float(x), 2) for x in a.mean(0)]
        out["means"][f"{R}_rows"] = {"check_layers": len(a), "union": mu[0], "e4": mu[1], "e3": mu[2], "edrop": mu[3],
                                     "reference (Fit12)": dict(zip(("union", "e4", "e3", "edrop"), REF[R]))}
    out["verdict"] = "PASS" if not any(bad.values()) else "FAIL"
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, "..", "data/deepdraft/t1c_wiring_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
