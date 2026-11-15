"""KWALL (b), the MTP half (MTP Head, Mon 11-02; REPORT 'KWALL'): the native MTP head's (b5) depth-1 accuracy per class at the
code map rows (~/FableGod/floor_rows.json), from D-24's teacher-forced ranks (data/d24/code_b5.npz: anchor s, depth 1 predicts
token s + 2, so map row i (predicting i + 1) reads s = i - 1; fixed 11-03, the first run read s = i). Beside it DFlash2 L1-20's own accuracy at the same rows in the official loop
(HPD-0's walk: the row's draft at its depth). CPU. Output: data/kwall_mtp.txt (appended)."""
import json, os
import numpy as np
TD = os.path.expanduser("~/TrainLab/data")
R = [r for r in json.load(open(os.path.expanduser("~/FableGod/floor_rows.json"))) if r["dom"] == "code"]
M = np.load(os.path.join(TD, "d24", "code_b5.npz")); H = np.load(os.path.join(TD, "d02b_walk_hpd0_code.npz"))
mk = {str(k): i for i, k in enumerate(M["keys"])}
hn = {str(n): i for i, n in enumerate(H["names"])}
POS = {}
for nm, w in hn.items():
    for c, (k, m, A) in enumerate(H[f"{w}_checks"]):
        for j in range(min(int(k) + 1, 15, int(m))):
            POS[(nm, int(A) + j)] = (w, c, j)
agg = {}
for r in R:
    mi = mk.get(r["nm"][:-4])
    if mi is None:
        continue
    a0 = max(0, int(M["gen"][mi]) - 2); rr = r["i"] - 1 - a0   # row i predicts i + 1: MTP anchor s = i - 1 (fixed 11-03: was i - a0)
    rk = M[f"rk_{mi}"]
    if not (0 <= rr < rk.shape[0]):
        continue
    g = agg.setdefault(r["cls"], [0, 0, 0, 0, 0])
    g[0] += 1; g[1] += int(rk[rr, 0]) == 0; g[2] += int(rk[rr, 0]) <= 2
    p = POS.get((r["nm"], r["i"]))
    if p:
        w, c, j = p; ids, off = H[f"{w}_ids"], int(H[f"{w}_off"])
        g[3] += 1; g[4] += int(H[f"{w}_dtok"][c, j]) == int(ids[r["i"] - off + 1])
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
say("KWALL (b), MTP half: the native MTP head b5 at depth 1 (teacher-forced, D-24) per the 125B's class, 12 code map windows")
for cl in ("sure", "mid", "tie"):
    if cl in agg:
        n, t1, t3, dn, d1 = agg[cl]
        say(f"  {cl:5s}: {n} rows; MTP depth-1 top-1 right {t1 / n:.3f}, in its top-3 {t3 / n:.3f}; DFlash2 (its draft at the row's depth, "
            f"official loop) right {d1 / max(dn, 1):.3f} on {dn} rows")
tot = sum(g[0] for g in agg.values())
say(f"  all {tot} rows: MTP depth-1 right {sum(g[1] for g in agg.values()) / tot:.3f}")
if "mid" in agg:
    mm = agg["mid"][1] / agg["mid"][0]
    v = "a cheap reader exists (PASS branch)" if mm >= 0.90 else ("the wall, if the near-copy's mid >= 0.90 (pending)" if mm <= 0.85 else "BETWEEN")
    say(f"  LINE (cloud's: MTP mid >= 0.90 -> a cheap reader; <= 0.85 -> the wall if the near-copy >= 0.90): MTP mid {mm:.3f} -> {v}")
open(os.path.join(TD, "kwall_mtp.txt"), "a").write("\n".join(lines) + "\n\n")
