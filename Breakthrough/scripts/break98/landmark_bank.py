"""landmark_bank.py (Breakthrough, Tue 17 Nov 2026 08:0x; BREAK98 LANDMARK-ANS screen 2, pre-registered in
docs/IDEAS_1117_landmark.md, 07:5x + the 08:0x addendum, 88b0b41, before running): the 125B confirmation on the bank.
~/datasets/deepdraft/bank: HE+ 24 (tf_code_HumanEval_*) and MBPP 100 (dlog__mbpp*). Per row t: wd[t] = the head's
true input state (10,240 wide), tk[t, 0] = the 125B's next-token call, ids (one longer than the rows), gen.
Answer-internal landmarks (scripts/break98/landmark_source.py, answer_only, k = 8..2, q <= p - d); at covered positions:
cos(wd[q], wd[p]) vs the anchor wd[p - d] and the 1-back true state wd[p - 1]; the share within 0.40 rad (MoonShots'
line); the 125B's top-1 agreement tk[q, 0] == tk[p, 0]; copy right ids[q + 1] == ids[p + 1].
CONFIRMED: HE+, d = 8: coverage >= 25 % AND median landmark cos >= median anchor cos + 0.10. MBPP and other d: info.
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/break98/landmark_bank.py  -> data/break98/landmark_bank.out"""
import glob
import math
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from landmark_source import LandmarkIndex  # noqa: E402

BANK = os.path.expanduser("~/datasets/deepdraft/bank")
SETS = {"HE+": "tf_code_HumanEval_*.npz", "MBPP": "dlog__mbpp*.npz"}
DS = (1, 4, 8, 16)
C40 = math.cos(0.40)
OUT = os.path.join(HERE, "..", "..", "data", "break98", "landmark_bank.out")
lines = []


def say(s=""):
    lines.append(s)
    print(s, flush=True)


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


t0 = time.time()
verdict = None
say("LANDMARK-ANS screen 2 on the bank (pre-registered 07:5x + 08:0x addendum): answer-internal landmarks; wd = the "
    "head's true input state")
for sname, pat in SETS.items():
    files = sorted(glob.glob(os.path.join(BANK, pat)))
    align = []
    acc = {d: dict(n=0, cl=[], ca=[], c1=[], top=[], cp=[]) for d in DS}
    for f in files:
        z = np.load(f)
        wd = np.asarray(z["wd"], np.float32)
        wd /= np.linalg.norm(wd, axis=1, keepdims=True) + 1e-9
        ids = z["ids"].astype(np.int64)
        tk0 = z["tk"][:, 0].astype(np.int64)
        gen = int(z["gen"])
        L = wd.shape[0]
        a = np.arange(gen, L - 1)
        align.append(float(np.mean(tk0[a] == ids[a + 1])) if len(a) else float("nan"))
        idx = LandmarkIndex(ids[:L], gen, answer_only=True)
        for d in DS:
            A = acc[d]
            for p in range(gen + 1, L):
                if p - d < 0:
                    continue
                A["n"] += 1
                q, k = idx.lookup(ids, p, p - d)
                if q is None:
                    continue
                A["cl"].append(float(wd[q] @ wd[p]))
                A["ca"].append(float(wd[p - d] @ wd[p]))
                A["c1"].append(float(wd[p - 1] @ wd[p]))
                A["top"].append(tk0[q] == tk0[p])
                A["cp"].append(ids[q + 1] == ids[p + 1])
    say("")
    say(f"== {sname}: {len(files)} files; alignment check tk[t, 0] == ids[t + 1] on answer rows: median "
        f"{np.median(align):.3f} (min {np.min(align):.3f})")
    say("| d | coverage | cos landmark | cos anchor wd[p-d] | cos 1-back wd[p-1] | landmark - anchor (paired) | angle landmark "
        "(rad) | within 0.40 rad: landmark / 1-back | top-1 agree | copy right |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    for d in DS:
        A = acc[d]
        m = len(A["cl"])
        cl, ca, c1 = (np.array(A[k]) for k in ("cl", "ca", "c1"))
        say(f"| {d} | {100 * m / max(A['n'], 1):.1f} % ({m}/{A['n']}) | {med(cl):.3f} | {med(ca):.3f} | {med(c1):.3f} | "
            f"{med(cl - ca) if m else float('nan'):+.3f} | {math.acos(max(-1, min(1, med(cl)))) if m else float('nan'):.3f} | "
            f"{100 * np.mean(cl >= C40) if m else float('nan'):.1f} % / {100 * np.mean(c1 >= C40) if m else float('nan'):.1f} % | "
            f"{100 * np.mean(A['top']) if m else float('nan'):.1f} % | {100 * np.mean(A['cp']) if m else float('nan'):.1f} % |")
        if sname == "HE+" and d == 8:
            verdict = (m / max(A["n"], 1), med(cl), med(ca))
cov, ml, ma = verdict
ok = cov >= 0.25 and ml >= ma + 0.10
say("")
say(f"VERDICT (pre-registered, HE+, d = 8): coverage {100 * cov:.1f} % (>= 25), median cos landmark {ml:.3f} vs anchor "
    f"{ma:.3f} (needs +0.10) -> {'CONFIRMED' if ok else 'NOT CONFIRMED'}")
say(f"wall {time.time() - t0:.1f} s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n")
