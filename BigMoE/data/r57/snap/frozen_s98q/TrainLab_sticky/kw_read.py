"""KNOWLEDGE WALL test (Fable KNOWLEDGE_WALL.md 0.5, mission 24; REPORT 'KWALL'): at the 125B's mid rows of the 12 code map
windows (~/FableGod/floor_rows.json, class 'mid'), is the 125B's token inside DFlash2's top-3 (the teacher-forced selector
softmax over its 16 candidates at HPD-0's anchors: data/d02b_walk_kw_code.npz, TL_WALK_TOP16), and how often is DFlash2 or
the native MTP chain right (head b5, teacher-forced from the same anchor: data/d24/code_b5.npz, rank 0)? PASS top-3 >= 85 %
or the union >= 80 %; KILL top-3 < 75 % and union < 72 %. Output: data/kw_read.txt (appended)."""
import json, os
import numpy as np
TD = os.path.expanduser("~/TrainLab/data")
R = [r for r in json.load(open(os.path.expanduser("~/FableGod/floor_rows.json"))) if r["dom"] == "code" and r["cf"] is not None]
Z = np.load(os.path.join(TD, "d02b_walk_kw_code.npz")); M = np.load(os.path.join(TD, "d24", "code_b5.npz"))
names = [str(x) for x in Z["names"]]
mk = {str(k): i for i, k in enumerate(M["keys"])}
POS = {}
for w, nm in enumerate(names):
    for c, (k, m, A) in enumerate(Z[f"{w}_checks"]):
        for j in range(min(int(k) + 1, 15, int(m))):
            POS[(nm, int(A) + j)] = (w, c, j, int(A))
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
agg = {}
for r in R:
    key = (r["nm"], r["i"])
    if key not in POS:
        continue
    w, c, j, A = POS[key]
    ids, off = Z[f"{w}_ids"], int(Z[f"{w}_off"])
    t = int(ids[r["i"] - off + 1])
    cand, sm = Z[f"{w}_cand"][c, j], Z[f"{w}_tsm"][c, j].astype(np.float32)
    o = np.argsort(-sm)
    top3 = t in set(int(x) for x in cand[o[:3]]); d1 = int(cand[o[0]]) == t
    mi = mk.get(r["nm"][:-4]); mt = None
    if mi is not None:
        a0 = max(0, int(M["gen"][mi]) - 2); rr = A - 1 - a0     # MTP anchor s = A - 1 (fixed 11-03: was A - a0, one position late)
        rk = M[f"rk_{mi}"]
        if 0 <= rr < rk.shape[0]:
            mt = int(rk[rr, j]) == 0
    g = agg.setdefault(r["cls"], {"n": 0, "top3": 0, "d1": 0, "mn": 0, "mt": 0, "un": 0})
    g["n"] += 1; g["top3"] += top3; g["d1"] += d1
    if mt is not None:
        g["mn"] += 1; g["mt"] += mt; g["un"] += (d1 or mt)
say("KWALL read-out (12 code map windows, positions the official loop reached at HPD-0's anchors; the 125B's class from the map)")
for cl in ("sure", "mid", "tie"):
    if cl in agg:
        g = agg[cl]
        say(f"  {cl:5s}: {g['n']} rows; DFlash2 top-1 right {g['d1'] / g['n']:.3f}, the 125B's token in DFlash2's top-3 {g['top3'] / g['n']:.3f}; "
            f"MTP chain right {g['mt'] / max(g['mn'], 1):.3f}, union (DFlash2 or MTP right) {g['un'] / max(g['mn'], 1):.3f} (on {g['mn']} rows with the MTP chain)")
if "mid" in agg:
    g = agg["mid"]; t3, un = g["top3"] / g["n"], g["un"] / max(g["mn"], 1)
    v = "PASS" if (t3 >= 0.85 or un >= 0.80) else ("KILL" if (t3 < 0.75 and un < 0.72) else "BETWEEN")
    say(f"  LINE (mid rows: top-3 >= 0.85 or union >= 0.80 PASS; top-3 < 0.75 and union < 0.72 KILL): top-3 {t3:.3f}, union {un:.3f} -> {v}")
open(os.path.join(TD, "kw_read.txt"), "a").write("\n".join(lines) + "\n\n")
