"""D-18 MTP1 (MTP Head, Fri 10-30; REPORT 'D-18'): CPU bounds for a hybrid draft, the native MTP head's FIRST guess at
block position 1 (revealed to DFlash2, which L1-20 was trained to use: pmask 0.5) and DFlash2 L1-20 drafting positions
2-15, one 16-row check as now. Same data as D-17's router read-out (sticky/d17_route.py: both drafters at every position
of the same answers; tokens per pass = accepted + 1, walked on the hybrid's own path). Per pass at p, with mR = the MTP
head's first guess is right and kD[p] = DFlash2's accepted count:
- mR false: 0 accepted (the first draft is wrong);
- mR true and kD[p] >= 1: kD[p] (DFlash2's own chain; revealing its own right first token is taken to change nothing);
- mR true and kD[p] = 0: LOW 1 (only the MTP token) / HIGH 1 + min(kD[p + 1], 14) (DFlash2 anchored one token later;
  optimistic: that check has one more feature row than the hybrid would).
Also the depth-1 table: DFlash2 right / MTP right on the same positions.
Usage: taskpolicy -b nice -n 10 [BigMoE venv] python sticky/d18_hybrid.py -> data/d18_hybrid.txt"""
import os
import numpy as np
from tokenizers import Tokenizer

H = os.path.expanduser
TD = H("~/TrainLab/data")
tk = Tokenizer.from_file(H("~/BigMoE/models/bare/tokenizer.json"))
END = {tk.token_to_id(t) for t in ("<|im_end|>", "<|endoftext|>")} - {None}
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def texts(dom):                                     # as sticky/d17_route.py: [(out, mtp, kD by output position)]
    W = np.load(os.path.join(TD, f"d02b_walk_d17e_{dom}.npz"))
    res = []
    for s, nm in enumerate(W["names"]):
        nm = str(nm)
        z = np.load(H("~/Chat1000/data/agent125/") + nm[len("agent_"):]) if dom == "agent" else np.load(os.path.join(TD, "d17", nm))
        out, mtp = [int(t) for t in z["out"]], z["mtp"]
        gen, off, ids = int(W[f"{s}_gen"]), int(W[f"{s}_off"]), W[f"{s}_ids"]
        n = min(len(out), off + len(ids) - gen)
        assert [int(t) for t in ids[gen - off:gen - off + n]] == out[:n], nm
        kD = {int(a) - gen + 1: int(k) for k, m, a in W[f"{s}_checks"]}
        e = next((i for i, t in enumerate(out[:n]) if t in END), n - 1)
        res.append((out[:e + 1], mtp, kD))
    return res

def mright(out, mtp, p):
    return p < len(out) and mtp[p][0] >= 0 and int(mtp[p][0]) == out[p]

def walk(T, kfun):
    tok = passes = 0
    for out, mtp, kD in T:
        p, L = 1, len(out)
        while p < L:
            k = min(kfun(out, mtp, kD, p), L - p)
            tok += k + 1; passes += 1; p += k + 1
    return tok / max(passes, 1)

def dfl(out, mtp, kD, p):
    return kD.get(p, 0)

def hyb(high):
    def f(out, mtp, kD, p):
        if not mright(out, mtp, p):
            return 0
        if kD.get(p, 0) >= 1:
            return kD[p]
        return 1 + min(kD.get(p + 1, 0), 14) if high else 1
    return f

say("D-18 MTP1 bounds: MTP head's first guess + DFlash2 L1-20 for positions 2-15 (one 16-row check); tokens per pass")
for dom in ("code", "gsm", "ifeval", "multi", "agent"):
    if not os.path.exists(os.path.join(TD, f"d02b_walk_d17e_{dom}.npz")):
        say(f"  {dom}: no every-position walk"); continue
    T = texts(dom)
    c = np.zeros((2, 2), int)                       # [DFlash2 depth-1 right][MTP depth-1 right], every position
    for out, mtp, kD in T:
        for p in range(1, len(out)):
            if p in kD:
                c[int(kD[p] >= 1), int(mright(out, mtp, p))] += 1
    n = c.sum()
    d0, lo, hi = walk(T, dfl), walk(T, hyb(False)), walk(T, hyb(True))
    say(f"  {dom} ({len(T)} answers): DFlash2 {d0:.2f}; hybrid LOW {lo:.2f} ({lo - d0:+.2f}, {lo / d0 - 1:+.1%}) HIGH {hi:.2f} "
        f"({hi - d0:+.2f}, {hi / d0 - 1:+.1%})")
    say(f"    depth 1 at every position (n {n}): DFlash2 {c[1].sum() / n:.3f}, MTP {c[:, 1].sum() / n:.3f}; "
        f"MTP right where DFlash2 wrong {c[0, 1] / n:.3f}, DFlash2 right where MTP wrong {c[1, 0] / n:.3f}")
open(os.path.join(TD, "d18_hybrid.txt"), "w").write("\n".join(lines) + "\n")
