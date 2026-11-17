#!/usr/bin/env python3
"""BREAK98: LANDMARK-ANS bound with the REAL (clustered) coverage (NewIdeas, Tue 17 Nov; time = the commit's). A landmark
exists for the state after answer row q iff the k-gram of TRUE tokens ending at q occurred earlier IN THE ANSWER (rows >=
gen). True tokens = the drafted ones whenever the draft is still right, which is the only case that matters. Each
landmark refreshes the head's carried state with true-state quality (Breakthrough's ceiling statement); later depths run
on that anchor's own chain until the next landmark. Compared with random refreshes at the same coverage. Chain-32 kept.
usage: taskpolicy -b ~/BigMoE/.venv/bin/python -I -B b98_landmark_real.py"""
import os, sys
import numpy as np
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools"))
import certify98 as C98  # noqa: E402
import tree_audit as TA  # noqa: E402
import b98sim as S  # noqa: E402
BANK = os.path.expanduser("~/datasets/deepdraft/bank")
SETS = {"he": ("heplus24", 13.77), "mbpp": ("mbppbank", 6.92)}

def lm_rows(ids, gen, k):
    seen = set(); lm = np.zeros(len(ids), bool)
    for q in range(gen + k - 1, len(ids)):
        key = tuple(ids[q - k + 1:q + 1].tolist())
        lm[q] = key in seen
        seen.add(key)
    return lm

def walk(W, masks):
    ks = []
    for X, lm in zip(W, masks):
        rt = X["rt"]; n = rt.shape[0]
        g = np.full(rt.shape, -1, np.int8)
        for p in range(n):
            r = 0
            for d in range(S.DEP):
                if d > 0 and p + d < n and lm[p + d]:
                    r = d
                a = p + r
                if a >= n or d - r >= S.DEP:
                    break
                g[p, d] = rt[a, d - r]
        g[rt < 0] = -1
        ks += C98.kept_walk(g, C98.R0)
    return float(np.mean(ks))

def main():
    rng = np.random.default_rng(17)
    for name, (tag, pub) in SETS.items():
        W = TA.load_windows(tag, pub)
        base = walk(W, [np.zeros(X["rt"].shape[0], bool) for X in W])
        lab = S.kept([X["rt"] for X in W], S.refresh(1))
        for k in (2, 3, 4, 8):
            masks, cov = [], []
            for X in W:
                z = np.load(os.path.join(BANK, X["file"]), allow_pickle=True)
                lr = lm_rows(z["ids"], int(z["gen"]), k)
                an = np.asarray(X["an"])
                m = np.zeros(X["rt"].shape[0], bool)
                ok = (an >= 0) & (an < len(lr))
                m[:len(an)][ok[:len(m)]] = lr[an[ok]][:m[:len(an)][ok[:len(m)]].shape[0]]
                masks.append(m); cov.append(m[1:].mean() if len(m) > 1 else 0)
            c = float(np.mean(np.concatenate([mm[1:] for mm in masks])))
            real = walk(W, masks)
            rnd = walk(W, [rng.random(len(mm)) < c for mm in masks])
            print(f"== {name} k={k}: landmark coverage of walk anchors {100 * c:.0f} %; own {base:.2f}, LAB {lab:.2f}; "
                  f"REAL landmarks {real:.2f} ({100 * (real / base - 1):+.0f} %, {100 * (real - base) / (lab - base):.0f} % of the gap); "
                  f"random at the same coverage {rnd:.2f} ({100 * (rnd / base - 1):+.0f} %)")

if __name__ == "__main__":
    main()
