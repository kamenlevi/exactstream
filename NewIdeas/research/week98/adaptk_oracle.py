#!/usr/bin/env python3
"""ADAPTIVE K screen (NewIdeas, Tue 17 Nov 01:0x): choose the check's row count PER CHECK (16..128 rows) instead of one
fixed K. Why now: on E14's costs K* is a tie on HE+ and tree-16 by ~20 % on MBPP (KSTAR_E14.md), and under the lossy
model the sets prefer different K, so a per-check choice could beat every fixed K.

Prior art in the project: the live 45 GB loop already TRUNCATES rows by running confidence (S45c, t 0.3, ADOPTED) and
pushing the cut higher was a KILL (S45d). This screen asks the other direction: GROW past 16 rows when the tree is
confident (OPT-Tree / Sequoia-style size choice), on the t30b 127-node dumps (n127k8).

Three readings per set and cost model (drafting hidden, overhead 4 ms, as treek_final):
  fixed  : the best single K;
  ORACLE : per window, the K sequence minimizing total time for the walk's tokens (Dinkelbach DP over positions, exact
           for this choice set, with hindsight of acceptance) = the ceiling of any per-check policy;
  POLICY : no hindsight. At each check pick K maximizing E[kept_K] - lam * T_K, with E[kept_K] = 1 + sum of the
           drafter's path_p over the K - 1 best-first pops (the expected accepted length of a tree if path_p is
           calibrated); lam = the policy's own tokens per ms (fixed point).
KILL if the ORACLE gains < 3 % over the best fixed K on both sets (no policy can do better than the oracle).
Costs: R(16), R(32) from E14 (costs_e14.json, ratio of means); other K from Leap's v2 onepass growth rescaled to E14's
R32 (1 + (R - 1) x 0.4372 / 0.478) [est]; c(K) from v2; lossy = 25 ms x R_floor(K) / R_floor(32) x c(K).
usage: taskpolicy -b ~/BigMoE/.venv/bin/python -I -B adaptk_oracle.py
"""
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tree_audit as TA  # noqa: E402

ROWS = [16, 24, 32, 48, 64, 96, 128]
V2 = json.load(open(os.path.expanduser("~/Leap/data/week98/treek_costs_v2.json")))
A16, OV = 62.49, 4.0
G = (1.4372 - 1.0) / (1.478 - 1.0)
R_E14 = {r: (1.0 if r == 16 else 1.4372 if r == 32 else 1.0 + (V2["R"]["onepass"][str(r)] - 1.0) * G) for r in ROWS}
C = {r: V2["c"][str(r)] for r in ROWS}
FL = V2["R"]["floor"]
MODELS = {
    "e14": {r: R_E14[r] * C[r] * A16 + OV for r in ROWS},
    "lossy": {r: 25.0 * FL[str(r)] / FL["32"] * C[r] + OV for r in ROWS},
}
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def per_anchor(dump):
    """for each anchor key (fi, an): accepted depth a_K (0..K-1) and expected kept 1 + sum path_p over the first K-1 pops"""
    fi, an, nd, dp, ot, pp = (dump[k] for k in ("fi", "an", "node", "depth", "on_true", "path_p"))
    key = fi.astype(np.int64) * 10_000_000 + an.astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    A, E = {}, {}
    for r in ROWS:
        K = r - 1
        m = ot & (nd < K)
        acc = np.zeros(len(uk), np.int64)
        np.maximum.at(acc, inv[m], dp[m].astype(np.int64) + 1)
        A[r] = np.minimum(acc, K)
        e = np.zeros(len(uk), np.float64)
        sel = nd < K
        np.add.at(e, inv[sel], pp[sel].astype(np.float64))
        E[r] = 1.0 + e
    return {int(k): i for i, k in enumerate(uk)}, A, E


def walk_fixed(W, idx, A, r, T):
    kept = tm = 0.0
    for X in W:
        p = 1
        while p < X["n"]:
            i = idx.get(X["fi"] * 10_000_000 + int(X["an"][p])) if p < len(X["an"]) else None
            a = int(A[r][i]) if i is not None else 0
            kept += a + 1; tm += T[r]; p += a + 1
    return kept, tm


def walk_oracle(W, idx, A, T, iters=8):
    lam = None
    for _ in range(iters):
        if lam is None:
            k0, t0 = walk_fixed(W, idx, A, 16, T); lam = k0 / t0
        kept = tm = 0.0
        for X in W:
            n = X["n"]
            V = np.zeros(n + 140); choice = np.zeros(n + 1, np.int64); gain = np.zeros(n + 1, np.int64)
            for p in range(n - 1, 0, -1):
                i = idx.get(X["fi"] * 10_000_000 + int(X["an"][p])) if p < len(X["an"]) else None
                best = -1e18
                for r in ROWS:
                    a = int(A[r][i]) if i is not None else 0
                    v = (a + 1) - lam * T[r] + (V[p + a + 1] if p + a + 1 < n else 0.0)
                    if v > best:
                        best, choice[p], gain[p] = v, r, a
                V[p] = best
            p = 1
            while p < n:
                r, a = int(choice[p]), int(gain[p])
                kept += a + 1; tm += T[r]; p += a + 1
        new = kept / tm
        if abs(new - lam) < 1e-9:
            break
        lam = new
    return kept, tm


def walk_policy(W, idx, A, E, T, iters=8):
    k0, t0 = walk_fixed(W, idx, A, 16, T); lam = k0 / t0
    use = {}
    for _ in range(iters):
        kept = tm = 0.0; use = {r: 0 for r in ROWS}
        for X in W:
            p = 1
            while p < X["n"]:
                i = idx.get(X["fi"] * 10_000_000 + int(X["an"][p])) if p < len(X["an"]) else None
                if i is None:
                    r = 16
                else:
                    r = max(ROWS, key=lambda q: E[q][i] - lam * T[q])
                a = int(A[r][i]) if i is not None else 0
                use[r] += 1; kept += a + 1; tm += T[r]; p += a + 1
        new = kept / tm
        if abs(new - lam) < 1e-9:
            break
        lam = new
    return kept, tm, use


def main():
    res = {}
    say("ADAPTIVE K screen: fixed best K vs ORACLE (hindsight DP) vs POLICY (expected kept from path_p), drafting hidden")
    say("costs per check (ms): " + "; ".join(f"{m} " + " ".join(f"{r}:{T[r]:.1f}" for r in ROWS) for m, T in MODELS.items()))
    for s, (tag, _rt, pub32, _p16) in TA.SETS.items():
        W = TA.load_windows(tag, pub32)
        z = np.load(os.path.join(TA.CE, f"tree_t30b_{tag}_n127k8.npz"), allow_pickle=True)
        dump = {k: z[k] for k in ("fi", "an", "node", "depth", "on_true", "path_p")}
        idx, A, E = per_anchor(dump)
        say(f"\n=== {s} ({tag}): {len(W)} windows, {len(idx)} anchors")
        for m, T in MODELS.items():
            fx = {r: walk_fixed(W, idx, A, r, T) for r in ROWS}
            tps = {r: 1000 * k / t for r, (k, t) in fx.items()}
            rb = max(tps, key=tps.get)
            ko, to = walk_oracle(W, idx, A, T); o = 1000 * ko / to
            kp, tp, use = walk_policy(W, idx, A, E, T); pl = 1000 * kp / tp
            say(f"  [{m}] fixed: " + " ".join(f"{r}:{tps[r]:.1f}" for r in ROWS) + f" | best fixed {rb} rows {tps[rb]:.1f}")
            say(f"  [{m}] ORACLE {o:.1f} ({100 * (o / tps[rb] - 1):+.1f} % vs best fixed) | POLICY {pl:.1f} "
                f"({100 * (pl / tps[rb] - 1):+.1f} %), K use {use}")
            res.setdefault(s, {})[m] = {"fixed": tps, "best_fixed": rb, "oracle": o, "policy": pl, "policy_use": use}
        # calibration check of the expected-kept proxy at 16 and 32 rows (mean E vs mean realized, over all anchors)
        for r in (16, 32, 128):
            say(f"  calibration rows {r}: mean E[kept] {np.mean(E[r]):.3f} vs mean realized {np.mean(A[r] + 1):.3f}")
    json.dump(res, open(os.path.join(HERE, "adaptk_oracle.json"), "w"), indent=1)
    open(os.path.join(HERE, "adaptk_oracle.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
