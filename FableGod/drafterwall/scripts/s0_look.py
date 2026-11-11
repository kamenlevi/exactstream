#!/usr/bin/env python3
"""s0_look.py (exploratory): print real first-miss events of the DFlash2 own chain on the code tree3 tables:
context tail, the true token, both drafters' picks, the 125B's top-3 with probabilities, depth, class."""
import numpy as np, glob, os, json, sys
TD = os.path.expanduser("~/TrainLab/data/tree3_map")
vocab = json.load(open(os.path.expanduser("~/BigMoE/models/bare/vocab.json"))); id2 = {v: k for k, v in vocab.items()}
def s(t): return id2.get(int(t), "?").replace("Ġ", "␣").replace("Ċ", "⏎")
dom = sys.argv[1] if len(sys.argv) > 1 else "code"; nshow = int(sys.argv[2]) if len(sys.argv) > 2 else 40
rng = np.random.default_rng(0); shown = 0
files = sorted(glob.glob(f"{TD}/{dom}/*.npz"))
for f in files[:4]:
    z = np.load(f); own, true, A = z["dfl_own"], z["true"], z["A"]; toks = z["tokens"] if "tokens" in z.files else None
    n = own.shape[0]; a = 0
    print("==", os.path.basename(f), "keys:", [k for k in z.files][:20])
    while a < n and shown < nshow:
        m = int(z["m"][a])
        if m < 2: break
        eq = own[a, :m] == true[a, :m]
        k = int(np.argmin(eq)) if not eq.all() else m
        if k < m:
            p3 = z["t125_p3"][a, k]; t3 = z["t125_t3"][a, k]
            cls = "sure" if p3[0] >= 0.99 else ("tie" if p3[0] - p3[1] <= 0.1 else "mid")
            ctx = "".join(s(t) for t in true[a, max(0, k - 6):k])
            print(f"A={int(A[a]):4d} depth {k:2d} [{cls}] ...{ctx} | true {s(true[a, k])!r}  dfl {s(own[a, k])!r} (tf top3 {[s(t) for t in z['dfl_t3'][a, k]]} p {np.round(z['dfl_p3'][a, k], 2)})  mtp {[s(t) for t in z['mtp_t3'][a, k]]} p {np.round(z['mtp_p3'][a, k], 2)}  125B {[s(t) for t in t3]} p {np.round(p3, 3)}")
            shown += 1
        a += k + 1
