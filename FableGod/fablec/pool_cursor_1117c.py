"""BREAK98c (Fable C, Tue 17 Nov 2026, 12:0x): price the one unknown of a POOL drafter (the head keeps a pool of drafted
nodes ahead of the cursor; each check takes the best 31 pool nodes under the cursor; a fresh re-rooted draft happens
only when the pool has nothing under the correction). The unknown: kept per check when a block's draft is CONTINUED
from the head's drifted, unrefilled pool instead of re-rooted at the true 125B state. Walk along each file (kept per
CHECK, the engine's number; the per-anchor mean is length-biased), on the real 127-pop k8 dumps, CPU, seconds:
 (A) today: every check = pops 0-30 of the pool at the current anchor (reproduces the published tree-32).
 (B) pool engine: after a block ending at the correction c' (the on-true node at the miss depth), if the pool has c'
     expanded with >= MIN nodes under it, the next check = the first 31 pool nodes (pop order) under c', no re-root
     and no refill (the dumps cannot refill: a lower bound on a real pool); else a fresh pool at the new anchor.
 (C) the own chain's hazard by depth from the cells (survivorship-biased: for reference only).
usage: ~/BigMoE/.venv/bin/python -I -B pool_cursor_1117c.py
"""
import os, json
import numpy as np
H = os.path.expanduser("~")
CE = f"{H}/BigMoE/data/deepdraft/ceiling"
OUT = f"{H}/FableGod/fablec/pool_cursor_1117c.out"
SETS = {"HE+": "heplus24", "MBPP": "mbppbank"}
B = 31
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

def load_pools(tag):
    tz = np.load(f"{CE}/tree_t30b_{tag}_n127k8.npz")
    fi, an, node, parent, dep, ontrue, yrank = (tz[k] for k in ("fi", "an", "node", "parent", "depth", "on_true", "ynext_rank"))
    key = fi.astype(np.int64) * 10**7 + an.astype(np.int64)
    order = np.lexsort((node, key))
    key, node, parent, dep, ontrue, yrank, fi, an = (x[order] for x in (key, node, parent, dep, ontrue, yrank, fi, an))
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]); ends = np.r_[starts[1:], len(key)]
    # the root's candidates: the chain cells' y_rank at depth 0 (the head's top-8 from the anchor's true state)
    tk = np.load(f"{CE}/topk_t30b_{tag}.npz")
    m0 = tk["depth"] == 0
    root_rank = dict(zip((tk["fi"][m0].astype(np.int64) * 10**7 + (tk["row"][m0] - 1).astype(np.int64)).tolist(), tk["y_rank"][m0].tolist()))
    pools = {}
    for st, en in zip(starts, ends):
        nd, pa, dp, ot, yr = node[st:en], parent[st:en], dep[st:en], ontrue[st:en], yrank[st:en]
        assert np.array_equal(nd, np.arange(len(nd)))
        # each node's deepest on-true ancestor-or-self (a path leaves the true text at most once)
        deep_ot = np.full(len(nd), -1, np.int64)
        for i in range(len(nd)):
            if ot[i]: deep_ot[i] = dp[i]
            elif pa[i] >= 0: deep_ot[i] = deep_ot[pa[i]]
        pools[(int(fi[st]), int(an[st]))] = (dp, ot, deep_ot, yr, int(root_rank.get(int(key[st]), 99)))
    return pools

def block(pool, D, budget):
    """rows = the first `budget` pool nodes (pop order) strictly below the on-true node at depth D (D = -1: root).
    Returns (a, rows_used, state): a = consecutive accepted depths D+1..; state of the correction c' (the on-true
    token at depth D + a + 1) in this pool: 'expanded' (a popped node), 'candidate' (among its parent's 8 children,
    unpopped), 'absent' (outside the parent's 8)."""
    dp, ot, deep, yr, rrank = pool
    sel = (dp > D) & (deep >= D)   # under the on-true node at depth D
    idx = np.flatnonzero(sel)[:budget]
    dok = set(dp[idx][ot[idx]].tolist())
    a = 0
    while (D + 1 + a) in dok: a += 1
    corr = D + 1 + a
    if np.any(ot & (dp == corr)): state = "expanded"
    else:
        if corr == 0: state = "candidate" if rrank < 8 else "absent"
        else:
            par = np.flatnonzero(ot & (dp == corr - 1))
            state = "candidate" if (len(par) and yr[par[0]] < 8) else "absent"
    return a, len(idx), state

def run(pools, mode, minsub, disc=1.0):
    """mode A: today (a fresh pool every check). mode B: the stale pool continues under c' when it has >= minsub
    nodes there (no refill, no re-root). mode C: refill emulation: a block is READY when c' is expanded or a candidate
    in the current pool (a CPU refill during the check expands it); a ready block's draft is emulated by the
    re-rooted pool at the new anchor and its kept is discounted by `disc` (the measured no-re-root factor); only an
    absent c' forces a fresh (serial) draft."""
    files = sorted(set(f for f, _ in pools))
    tot = {"kept": 0.0, "checks": 0, "rows": 0.0, "fresh": 0, "states": {"expanded": 0, "candidate": 0, "absent": 0}}
    for f_ in files:
        anchors = sorted(a for ff, a in pools if ff == f_)
        a0, n = anchors[0], len(anchors)
        p = 1; D = -1; A = None; ready = False
        while p < n:
            if mode in ("A", "C") or A is None:
                A = a0 + p; D = -1
                if not ready: tot["fresh"] += 1
            pool = pools[(f_, A)]
            a, rows, state = block(pool, D, B)
            tot["kept"] += (a + 1) * (disc if ready else 1.0); tot["checks"] += 1; tot["rows"] += rows + 1
            tot["states"][state] += 1
            p += a + 1
            if mode == "A": A = None; ready = False; continue
            if mode == "C": ready = state != "absent"; continue
            D = D + a + 1
            dp, ot, deep = pool[:3]
            nsub = int(np.sum((dp > D) & (deep >= D)))
            if not (state == "expanded" and nsub >= minsub): A = None
    c = tot["checks"]
    return tot["kept"] / c, tot["rows"] / c, tot["fresh"] / c, c, {k: v / c for k, v in tot["states"].items()}

def main():
    RES = {}
    for s, tag in SETS.items():
        say(f"== {s} ({tag})")
        pools = load_pools(tag)
        kA, rA, fA, nA, stA = run(pools, "A", 1)
        say(f"   (A) today, fresh tree-32 every check:        kept/check {kA:6.2f}   rows/check {rA:5.1f}   fresh drafts/check {fA:.3f}   checks {nA}")
        say(f"       state of the correction per CHECK (today's walk): " + ", ".join(f"{k} {v:.3f}" for k, v in stA.items()))
        RES[s] = {"today": kA, "states_per_check": stA}
        for minsub in (1, 8, 16, 31):
            k, r, fr, n, _ = run(pools, "B", minsub)
            say(f"   (B) pool, continue if >= {minsub:2d} nodes under c': kept/check {k:6.2f}   rows/check {r:5.1f}   fresh drafts/check {fr:.3f}   checks {n}   (x{k / kA:.3f} kept; the unrefilled pool is a lower bound)")
            RES[s][f"pool_min{minsub}"] = {"kept": k, "rows": r, "fresh": fr}
        for disc in (1.0, 0.94, 0.88, 0.80):
            k, r, fr, n, st = run(pools, "C", 1, disc)
            verdict = "PASS" if (fr <= 0.25 and k >= 0.95 * kA) else "FAIL"
            say(f"   (C) REFILL emulation, pool blocks at x{disc:.2f} of a re-rooted draft: kept/check {k:6.2f} (x{k / kA:.3f})   fresh drafts/check {fr:.3f}   checks {n}   -> {verdict} (bar: fresh <= 0.25 and >= x0.95)")
            RES[s][f"refill_disc{disc}"] = {"kept": k, "fresh": fr, "verdict": verdict}
        # (C) hazard by depth, own chain
        z = np.load(f"{CE}/cells_t30b_{tag}.npz")
        depc, right, fic, row = (z[k] for k in ("depth", "right", "fi", "row"))
        keyc = fic.astype(np.int64) * 10**7 + (row - 1 - depc).astype(np.int64)
        R = {d: dict(zip(keyc[depc == d].tolist(), right[depc == d].tolist())) for d in range(31)}
        alive = set(R[0].keys()); h = []
        for d in range(31):
            cur = [k for k in alive if k in R[d]]
            h.append(sum(1 for k in cur if not R[d][k]) / max(len(cur), 1))
            alive = {k for k in cur if R[d][k]}
        say(f"   (C) own-chain hazard by depth (miss | right so far; survivorship-biased): " + " ".join(f"{x:.3f}" for x in h))
        RES[s]["hazard"] = h
    open(OUT, "w").write("\n".join(lines) + "\n")
    json.dump(RES, open(OUT.replace(".out", ".json"), "w"), indent=1)

if __name__ == "__main__":
    main()
