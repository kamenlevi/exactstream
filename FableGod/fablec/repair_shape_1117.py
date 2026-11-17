"""BREAK98 (Fable C, Tue 17 Nov 2026, 08:0x): IN-CHECK REPAIR priced on the real SHAPE. NewIdeas' b98_repair bound
(+38 % one round, = LAB at two) is a chain-32 number; the engine runs tree-32, where siblings already cover part of what
repair fixes (yesterday's lesson: chain multipliers do not transfer to trees). Replay the same bound budget-correct on
BigMoE's real 127-node k8 dumps: perfect location of the first miss at the tap layer, the true correction, the re-drafted
tail = the head's REAL tree (or chain) from the new anchor with the rows the dead nodes release (rows-neutral), rounds
0 / 1 / 2 / unlimited. Also the rows-unbudgeted variant (a full 31-node tree per round) as the upper bound. CPU, seconds.
usage: ~/BigMoE/.venv/bin/python -I -B repair_shape_1117.py
"""
import os, json
import numpy as np
H = os.path.expanduser("~")
CE = f"{H}/BigMoE/data/deepdraft/ceiling"
OUT = f"{H}/FableGod/fablec/repair_shape_1117.out"
SETS = {"HE+": ("heplus24", 13.77, 16.26, 20.77), "MBPP": ("mbppbank", 6.92, 9.03, 12.72)}
ROWS = 32          # rows per check: the correction row + 31 drafted rows / nodes
R0 = 1
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

def walk_blind(n, acc1, acc2, rstar):
    """PROBE-FREE form: no location. Rows 0..rstar-1 are the original draft (chain); at the tap the head is refreshed
    with the true layer-L state at row rstar (true only if the prefix is right) and re-drafts the remaining 31 - rstar
    rows from there (acc2 = chain or tree at anchor p + rstar). If the first miss is before rstar the block ends there."""
    ks = nc = 0; p = R0
    while p < n:
        nc += 1
        a = acc1(p, rstar)
        if a is not None and a >= rstar and p + rstar < n:
            a2 = acc2(p + rstar, ROWS - 1 - rstar)
            a = rstar + (a2 or 0)
        a = a or 0
        ks += a + 1; p += a + 1
    return ks / max(nc, 1), nc

def walk(n, acc_at, rounds, unbudgeted=False, q=1.0, rng=None):
    """acc_at(p, budget_nodes) -> right-token prefix (<= budget) of the draft from anchor p, or None past the window.
    NewIdeas' b98_repair frame: at the first miss (after a right tokens) the head is refreshed with the TRUE state of
    the last right row (anchor p + a) and re-drafts from there: its depth 0 is the correction GUESS (the tree's depth-0
    nodes are several guesses), its deeper nodes the tail. Rows-neutral: the new draft gets the rows the dead nodes
    release (budget - a); unbudgeted: a full 31-node draft per round (upper bound, costs rows). Kept = right tokens + 1
    (the exact correction from the deepest right row at layer 48), certify98's walk."""
    ks = nc = 0; p = R0; extra_rows = 0; rounds_used = 0
    while p < n:
        B = ROWS - 1; r = 0; cur = p; nc += 1
        while True:
            a = acc_at(cur, B)
            if a is None: break
            cur += a
            if a >= B or r >= rounds or cur >= n: break
            if rng is not None and rng.random() >= q: break   # the round fails: wrong location or a poor guess
            B = ROWS - 1 if unbudgeted else B - a   # dead nodes released = B - a (the a right rows stay)
            extra_rows += ROWS if unbudgeted else 0
            r += 1; rounds_used += 1
            if B <= 0: break
        ks += cur - p + 1; p = cur + 1
    return ks / max(nc, 1), nc, extra_rows / max(nc, 1), rounds_used / max(nc, 1)

def main():
    RES = {}
    for s, (tag, pchain, ptree, plab) in SETS.items():
        say(f"== {s} ({tag}): published own chain-32 {pchain}, tree-32 {ptree}, LAB chain-32 {plab}")
        # ---- chain from the cells: R[anchor index, depth] right/0/-1
        z = np.load(f"{CE}/cells_t30b_{tag}.npz")
        dep, right, fi, row = (z[k] for k in ("depth", "right", "fi", "row"))
        chain = {}
        for f_ in sorted(set(fi.tolist())):
            w = fi == f_
            anchors = np.sort(row[w][dep[w] == 0] - 1)
            assert np.all(np.diff(anchors) == 1), "anchors not consecutive"
            a0 = int(anchors[0]); n0 = len(anchors)
            R = np.full((n0, 31), -1, np.int8)
            for d in range(31):
                md = w & (dep == d)
                idx = row[md] - 1 - d - a0
                ok = (idx >= 0) & (idx < n0)
                R[idx[ok], d] = right[md][ok]
            chain[f_] = R
        # ---- tree from the 127-node dumps: per (file, anchor) the pop index of the on-true node at each depth
        tz = np.load(f"{CE}/tree_t30b_{tag}_n127k8.npz")
        nfi, nan_, nnode, ndep, ntrue = (tz[k] for k in ("fi", "an", "node", "depth", "on_true"))
        tree = {}
        for f_ in sorted(set(nfi.tolist())):
            w = nfi == f_
            anchors = np.array(sorted(set(nan_[w].tolist())))
            assert np.all(np.diff(anchors) == 1), "tree anchors not consecutive"
            a0 = int(anchors[0]); n0 = len(anchors)
            OT = np.full((n0, 128), 10**6, np.int64)   # OT[anchor, depth] = pop index of the on-true node
            m = w & ntrue.astype(bool)
            OT[nan_[m] - a0, ndep[m]] = nnode[m]
            tree[f_] = (a0, OT)
        def chain_acc(R):
            def f(p, B):
                if p >= len(R): return None
                r = R[p, :max(B, 0)]
                bad = np.flatnonzero(r != 1)
                return int(bad[0]) if len(bad) else int(len(r))
            return f
        def tree_acc(OT):
            def f(p, B):
                if p >= len(OT): return None
                ok = OT[p] < B
                bad = np.flatnonzero(~ok)
                return int(bad[0]) if len(bad) else int(len(ok))
            return f
        RES[s] = {}
        for shape, data, mk in (("chain-32", chain, chain_acc), ("tree-32", tree, tree_acc)):
            for rounds, unb in ((0, False), (1, False), (2, False), (99, False), (1, True), (2, True)):
                ks = nc = er = ru = 0.0
                for f_, D in data.items():
                    R = D if shape == "chain-32" else D[1]
                    kp, c, e, u = walk(len(R), mk(R), rounds, unb)
                    ks += kp * c; nc += c; er += e * c; ru += u * c
                kp = ks / nc; key = f"{shape} rounds {rounds if rounds < 99 else 'inf'}{' unbudgeted' if unb else ''}"
                RES[s][key] = kp
                say(f"   {key:36s} kept/check {kp:6.2f}   x{kp / RES[s][f'{shape} rounds 0']:.3f} of plain"
                    f"   extra rows/check {er / nc:5.2f}   rounds/check {ru / nc:4.2f}")
            for rounds in (1, 2):
                for q in (0.7, 0.5, 0.3):
                    rng = np.random.default_rng(17); ks = nc = 0.0
                    for f_, D in data.items():
                        R = D if shape == "chain-32" else D[1]
                        kp, c, _, _ = walk(len(R), mk(R), rounds, False, q, rng)
                        ks += kp * c; nc += c
                    kp = ks / nc; RES[s][f"{shape} rounds {rounds} q{q}"] = kp
                    say(f"   {shape} rounds {rounds} q {q} (location x guess)  kept/check {kp:6.2f}   x{kp / RES[s][f'{shape} rounds 0']:.3f}")
        # probe-free: blind refresh at a fixed row, original chain before it, re-drafted chain or tree after it
        for rstar in (4, 8, 12, 16):
            for nm2, data2, mk2 in (("chain", chain, chain_acc), ("tree", tree, tree_acc)):
                ks = nc = 0.0
                for f_ in chain:
                    R1 = chain[f_]; R2 = data2[f_] if nm2 == "chain" else data2[f_][1]
                    kp, c = walk_blind(len(R1), chain_acc(R1), mk2(R2), rstar)
                    ks += kp * c; nc += c
                kp = ks / nc; RES[s][f"blind r*{rstar} chain->{nm2}"] = kp
                say(f"   BLIND refresh at row {rstar:2d}, chain -> {nm2:5s}   kept/check {kp:6.2f}   x{kp / RES[s]['chain-32 rounds 0']:.3f} of the chain")
    open(OUT, "w").write("\n".join(lines) + "\n")
    json.dump(RES, open(OUT.replace(".out", ".json"), "w"), indent=1)

if __name__ == "__main__":
    main()
