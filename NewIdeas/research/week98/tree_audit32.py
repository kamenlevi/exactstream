"""[tree_audit32.py: tree_audit.py at NODES = 31, i.e. 32-row checks; labels saying 16 mean 32. Mon 16 Nov 12:3x] PLAN4D ITEM 3: a tree at the shallow depths + the 2B audit (NewIdeas, Sun 15 Nov 21:4x; the orchestrator's OK 21:4x).
CPU only, one process, small. Read-only inputs:
  - BigMoE's real t30b EAGLE-2 tree dumps (ceiling/tree_t30b_{heplus24,mbppbank}.npz: 32 best-first pops per anchor,
    token / y / on_true / parent / path_p) and the cells (ceiling/cells_t30b_*.npz), in S9c's layout;
  - MoonShots' 2B / 4B teacher-forced rows (week98/rows{2b,4b}_{heplus,mbpp}.npz: top1, margin2b, tk0, margin125 per
    (file, row)), joined as join_heplus.py does: key (basename, row), every cell must find its row, and margin125 must
    equal the cells' m (atol 1e-4) as a fingerprint.
Walk: certify98.kept_walk's (R0 = 1); kept = accepted + 1 per check. Budget: a 16-row check = 15 nodes + the bonus.
The audit rule is MoonShots' frozen one: tau 4 / 2 / 1 / 0.5 nats by depth (0 / 1-2 / 3-6 / 7-30), P = 1 (one repair).

ARMS (each under its own walk over the same windows):
  chain-16                 the head's chain, 15 nodes.
  audit-chain-16           MoonShots' audited chain, capped at 15 nodes: at the first row where the auditor is confident
                           and disagrees, its token replaces the draft (a repair if right, a break if wrong).
  tree-16                  EAGLE-2-16 (the first 15 pops).
  tree+audit UB            UPPER BOUND, budget-free: where the tree stops on the true path (depth A), the auditor's
                           token at A is right and confident -> a new branch, and the head's chain continues from it
                           (the cells from A+1), total capped at 15. No false branches, no node cost.
  tree+audit BR-L          BUDGET-CORRECT branch design, P = 1, L = 3 / 5 / 7: the auditor reads the tree's MAIN path
                           (best child by path_p from the root) while it is on the true text. At the first depth j where
                           it is confident and its token is NOT among the selected children there, it adds that token as
                           a branch with L head-drafted nodes below it; the tree gives up its last L + 1 pops (never the
                           branch's ancestors). Right token: rescue (accepted A + 1 + min(cont, L)). Wrong token: a false
                           branch that only costs the nodes. If the main path leaves the true text before any branch
                           fires, the auditor's reads there are unknown: OPT assumes no branch fires, PES assumes one
                           fires and costs its nodes.
SELF-CHECKS: chain-32 13.77 / 6.92 (certify98); EAGLE-2-16 11.263 / 7.487 (S9c real); audit-chain-32 at P = 1 against
MoonShots' checkstart.json (HE+ uses the same cells; MBPP there used cells_t30b_mbpp100, here mbppbank: printed only).
LINE (PLAN4D item 3, written before the run): KILL if the composed kept per check <= max(tree-16, audit-chain-16) + 2 %
on HE+ 24 (UB first: if even the upper bound misses, no budget-correct design can pass). Also reported: MBPP bank, the 4B,
and walk-strict coverage per band (0 / 1-2 / 3-6 / 7-10 / 11-14).
usage: ~/BigMoE/.venv/bin/python -I -B tree_audit.py"""
import os, sys, json
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools"))
import certify98 as C98

H = os.path.expanduser("~")
CE = f"{H}/BigMoE/data/deepdraft/ceiling"
MS = f"{H}/MoonShots/week98"
SETS = {"he": ("heplus24", "heplus", 13.77, 11.263), "mbpp": ("mbppbank", "mbpp", 6.92, 7.487)}
DEP, NODES = 31, 31
SCHED = {0: 4.0, 1: 2.0, 2: 2.0, **{j: 1.0 for j in range(3, 7)}, **{j: 0.5 for j in range(7, 31)}}
TAU = np.array([SCHED[j] for j in range(DEP)], np.float32)
BANDS = [(0, 0), (1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30)]
OUT = os.path.expanduser("~/NewIdeas/research/week98/tree_audit32")
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)


def load_windows(tag, published):
    """per window: rt [n0 + 1, DEP] (1 right / 0 wrong / -1 none), m (125B margin), fi, an; layout checked vs certify98"""
    cz = np.load(f"{CE}/cells_t30b_{tag}.npz", allow_pickle=True)
    dep, right, row, fi, mm = (cz[k] for k in ("depth", "right", "row", "fi", "m"))
    dep = dep.astype(int); right = right.astype(int)
    files = [os.path.basename(str(f)) for f in cz["files"]]
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    ref = C98.windows(f"{CE}/cells_t30b_{tag}.npz")
    W, k32 = [], []
    for wi, (s, t) in enumerate(zip(st, en)):
        dw = dep[s:t]; n0 = int((dw == 0).sum())
        rt = np.full((n0 + 1, DEP), -1, np.int8); M = np.full((n0 + 1, DEP), np.nan, np.float32)
        RW = np.full((n0 + 1, DEP), -1, np.int64)
        for j in range(DEP):
            m = dw == j; c = int(m.sum())
            rt[:c, j] = right[s:t][m]; M[:c, j] = mm[s:t][m]; RW[:c, j] = row[s:t][m]
        assert (rt == ref[wi][0]).all(), f"{tag}: window {wi} layout != certify98"
        k32 += C98.kept_walk(rt, C98.R0, 31)
        an = row[s:t][dw == 0] - 1
        W.append({"rt": rt, "M": M, "RW": RW, "fi": int(fi[s]), "file": files[int(fi[s])], "an": an, "n": n0 + 1})
    say(f"SELF-CHECK {tag}: chain-32 kept {np.mean(k32):.3f} vs {published} -> "
        f"{'PASS' if abs(np.mean(k32) - published) < 0.01 else 'FAIL'}")
    return W


def attach_aud(W, rows_path, tag):
    """A2 [n0 + 1, DEP]: auditor right (1 / 0, -1 none); G2: its margin; T2: its token; fingerprint vs the cells' m"""
    R = np.load(rows_path, allow_pickle=True)
    key = {(os.path.basename(str(n)), int(r)): (int(t == k), float(mg), float(m1), int(t), int(k))
           for n, r, t, k, mg, m1 in zip(R["fname"], R["row"], R["top1"], R["tk0"], R["margin2b"], R["margin125"])}
    miss = bad = 0
    for X in W:
        A2 = np.full(X["rt"].shape, -1, np.int8); G2 = np.full(X["rt"].shape, np.nan, np.float32)
        T2 = np.full(X["rt"].shape, -1, np.int64); K0 = np.full(X["rt"].shape, -1, np.int64)
        for r in range(X["n"] - 1):
            for d in range(DEP):
                if X["rt"][r, d] < 0:
                    continue
                v = key.get((X["file"], int(X["RW"][r, d])))
                if v is None:
                    miss += 1; continue
                A2[r, d], G2[r, d], T2[r, d], K0[r, d] = v[0], v[1], v[3], v[4]
                bad += int(not np.isclose(v[2], X["M"][r, d], atol=1e-4))
        X[f"A2_{tag}"], X[f"G2_{tag}"], X[f"T2_{tag}"], X["K0"] = A2, G2, T2, K0
    say(f"  join {os.path.basename(rows_path)}: cells without a row {miss}, margin fingerprint mismatches {bad}")
    assert miss == 0 and bad == 0


def walk(acc_of, X, cap):
    """the decode walk over one window: acc_of(r) -> accepted at check start r; returns kept list and (A, r) per check"""
    p, kept, recs = C98.R0, [], []
    while p < X["n"]:
        a = int(acc_of(p)); a = min(a, cap)
        kept.append(a + 1); recs.append((a, p)); p += a + 1
    return kept, recs


def lead_right(rt_row, start, cap):
    k = start
    while k < cap and rt_row[k] == 1:
        k += 1
    return k


def audit_chain(X, r, cap, tag, P=1):
    """MoonShots' walk rule for one check start, capped at cap nodes"""
    h, a2, g2 = X["rt"][r], X[f"A2_{tag}"][r], X[f"G2_{tag}"][r]
    k = used = 0
    while k < cap:
        hv = h[k]
        if hv == -1:
            break
        conf = (not np.isnan(g2[k])) and g2[k] >= TAU[k]
        cv = a2[k] if conf else hv
        if used >= P:
            ok = hv == 1
        elif hv == 1 and cv == 1:
            ok = True
        elif hv == 0 and cv == 1:
            ok = True; used += 1
        else:
            ok = False
        if not ok:
            break
        k += 1
    return k


def coverage(recs, X, cap):
    """walk-strict per depth: success at d < A; a failure at d = A if A < cap and the cell exists"""
    num = np.zeros(DEP); den = np.zeros(DEP)
    for a, p in recs:
        for d in range(a):
            num[d] += 1; den[d] += 1
        if a < cap and X["rt"][p, a] >= 0:
            den[a] += 1
    return num, den


def run_set(s, tag, rtag, pub32, pub16):
    say(f"\n=== {s} ({tag}) ===")
    W = load_windows(tag, pub32)
    for aud in ("2b", "4b"):
        attach_aud(W, f"{MS}/rows{aud}_{rtag}.npz", aud)
    tz = np.load(f"{CE}/tree_t30b_{tag}.npz", allow_pickle=True)
    cf = [os.path.basename(str(f)) for f in np.load(f"{CE}/cells_t30b_{tag}.npz", allow_pickle=True)["files"]]
    assert cf == [os.path.basename(str(f)) for f in tz["files"]], "tree and cells files differ"
    nfi, nan_, nnode, ndep, ontrue, pp, par, tok, yy = (tz[k] for k in
        ("fi", "an", "node", "depth", "on_true", "path_p", "parent", "token", "y"))
    order = np.lexsort((nnode, nan_, nfi))
    nfi, nan_, nnode, ndep, ontrue, pp, par, tok, yy = (a[order] for a in (nfi, nan_, nnode, ndep, ontrue, pp, par, tok, yy))
    bounds = np.flatnonzero(np.append(True, (np.diff(nfi) != 0) | (np.diff(nan_) != 0))); bounds = np.append(bounds, len(nfi))
    T = {}
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        T[(int(nfi[b0]), int(nan_[b0]))] = (b0, b1)
    rowof = {X["fi"]: {int(a): i for i, a in enumerate(X["an"])} for X in W}

    # per anchor tree facts
    yfp = [0, 0]; yfp_d = []
    for X in W:
        n = X["n"]; X["A_t"] = np.zeros(n, np.int64); X["tree"] = [None] * n
        for r, a in enumerate(X["an"]):
            b0, b1 = T[(X["fi"], int(a))]
            nd, dp, ot, ppp, pa, tk, y = (nnode[b0:b1], ndep[b0:b1], ontrue[b0:b1].astype(bool), pp[b0:b1].astype(np.float64),
                                          par[b0:b1], tok[b0:b1], yy[b0:b1])
            sel = nd < NODES; hit = ot & sel
            A = int(dp[hit].max()) + 1 if hit.any() else 0
            X["A_t"][r] = A
            X["tree"][r] = (nd, dp, ot, ppp, pa, tk, y)
            # fingerprint: the tree's true token y at an on-true node at depth d = the auditor rows' tk0 position (2B right
            # iff its token == y there); checked on depth 0 via the 2B token where the 2B is right
            for i in np.flatnonzero(ot):
                d = int(dp[i])
                if d < DEP and X["K0"][r, d] >= 0:
                    yfp[0] += 1; ok_ = int(X["K0"][r, d] == int(y[i])); yfp[1] += ok_
                    if not ok_:
                        yfp_d.append((d, float(X["M"][r, d])))
    say(f"  y fingerprint (every on-true node's y == the 125B's top-1 tk0 at its row): {yfp[1]} / {yfp[0]}")
    if yfp_d:
        dd = np.array([x[0] for x in yfp_d]); mg = np.array([x[1] for x in yfp_d])
        say(f"  mismatches: depth 0 {int((dd == 0).sum())}, depths 1-2 {int(((dd >= 1) & (dd <= 2)).sum())}, 3+ {int((dd >= 3).sum())}; "
            f"125B margin median {np.median(mg):.3f} nats, < 0.1 nats {int((mg < 0.1).sum())}")
    assert yfp[0] > 0 and yfp[1] >= 0.995 * yfp[0], "y fingerprint under 99.5 %"

    res = {}
    def report(name, fn, cap):
        ks, num, den = [], np.zeros(DEP), np.zeros(DEP)
        for X in W:
            k, recs = walk(lambda p: fn(X, p), X, cap); ks += k
            nu, de = coverage(recs, X, cap); num += nu; den += de
        bands = [round(float(100 * num[a:b + 1].sum() / den[a:b + 1].sum()), 2) if den[a:b + 1].sum() > 0 else None for a, b in BANDS]
        nb = [int(den[a:b + 1].sum()) for a, b in BANDS]
        res[name] = {"kept": round(float(np.mean(ks)), 3), "checks": len(ks), "walk_strict_bands": bands, "band_n": nb}
        return res[name]["kept"]

    # SELF-CHECKS
    k = report("chain-32", lambda X, p: lead_right(X["rt"][p], 0, 31), 31)
    say(f"SELF-CHECK {s}: this walk's chain-32 {k:.3f} vs {pub32} -> {'PASS' if abs(k - pub32) < 0.01 else 'FAIL'}")
    k = report("tree-16 (EAGLE-2)", lambda X, p: X["A_t"][p], NODES)
    say(f"SELF-CHECK {s}: EAGLE-2-16 kept {k:.3f} vs S9c {pub16} -> {'PASS' if abs(k - pub16) < 0.01 else 'FAIL'}")
    for aud in ("2b", "4b"):
        report(f"audit-chain-32 {aud}", lambda X, p, aud=aud: audit_chain(X, p, 31, aud), 31)
        report(f"audit-chain-32 {aud} P=2", lambda X, p, aud=aud: audit_chain(X, p, 31, aud, P=2), 31)
        report(f"audit-chain-32 {aud} P=inf", lambda X, p, aud=aud: audit_chain(X, p, 31, aud, P=99), 31)
    report("chain-16", lambda X, p: lead_right(X["rt"][p], 0, NODES), NODES)

    for aud in ("2b", "4b"):
        report(f"audit-chain-16 {aud}", lambda X, p, aud=aud: audit_chain(X, p, NODES, aud), NODES)

        def ub(X, p, aud=aud):
            A = int(X["A_t"][p])
            if A >= NODES or X["rt"][p, A] < 0:
                return A
            if X[f"A2_{aud}"][p, A] == 1 and X[f"G2_{aud}"][p, A] >= TAU[A]:
                return lead_right(X["rt"][p], A + 1, NODES)       # the auditor's right token at A, then the head's chain
            return A
        report(f"tree+audit UB {aud}", ub, NODES)

        for L in (3, 5, 7):
            for mode in ("OPT", "PES"):
                def br(X, p, aud=aud, L=L, mode=mode):
                    if X["tree"][p] is None:                  # the window's end row (certify98's sentinel)
                        return 0
                    nd, dp, ot, ppp, pa, tk, y = X["tree"][p]
                    sel = nd < NODES
                    # the main path, while on the true text
                    cur, j, anc, fired = -1, 0, [], None
                    while j < NODES:
                        kids = np.flatnonzero(sel & (pa == cur) & (dp == j))
                        if X["rt"][p, j] < 0:
                            break
                        g = X[f"G2_{aud}"][p, j]; conf = (not np.isnan(g)) and g >= TAU[j]
                        t2 = int(X[f"T2_{aud}"][p, j])
                        if conf and t2 not in set(int(t) for t in tk[kids]):
                            fired = (j, X[f"A2_{aud}"][p, j] == 1); break
                        if not len(kids):
                            break
                        best = int(kids[np.argmax(ppp[kids])])
                        if not ot[best]:                       # the main path leaves the true text here
                            fired = ("off", None); break
                        anc.append(best); cur = int(nd[best]); j += 1
                    if fired is None or (fired[0] == "off" and mode == "OPT"):
                        return int(X["A_t"][p])
                    # the branch costs L + 1 of the tree's last pops (never the branch's ancestors)
                    keep = sel.copy(); drop = 0
                    for i in np.argsort(-nd, kind="stable"):
                        if drop >= L + 1:
                            break
                        if keep[i] and i not in anc:
                            keep[i] = False; drop += 1
                    # a dropped node removes its subtree (pops are best-first, so a dropped node's children were later pops)
                    hit = ot & keep
                    # closure: an on-true node counts only if its on-true ancestors are kept
                    A2 = 0
                    for d in range(NODES):
                        if (hit & (dp == d)).any():
                            A2 = d + 1
                        else:
                            break
                    if fired[0] == "off" or not fired[1]:
                        return A2                                # a false (or unknown, PES) branch: cost only
                    jj = fired[0]
                    if A2 < jj:                                  # the tree lost part of the true path above the branch
                        return A2
                    cont = lead_right(X["rt"][p], jj + 1, min(NODES, jj + 1 + L)) - (jj + 1)
                    return max(A2, jj + 1 + cont)
                report(f"tree+audit BR-{L} {mode} {aud}", br, NODES)
    return res


def main():
    out = {}
    for s, (tag, rtag, pub32, pub16) in SETS.items():
        out[s] = run_set(s, tag, rtag, pub32, pub16)
    ms = json.load(open(f"{MS}/checkstart.json"))
    say("\nMoonShots' checkstart.json audit-chain-32 (P = 1, tau0 4), for the self-check:")
    for k in ("2b heplus all tau0=4 P=1", "4b heplus all tau0=4 P=1", "2b mbpp all tau0=4 P=1", "4b mbpp all tau0=4 P=1"):
        if k in ms:
            say(f"  {k}: kept/32 {ms[k]['kept32']}")
    say("\nRESULTS (kept per check under the walk; walk-strict coverage by band 0 / 1-2 / 3-6 / 7-10 / 11-14)")
    for s, res in out.items():
        base16 = res["chain-16"]["kept"]
        say(f"\n{s}: chain-16 {base16:.3f}")
        for name, v in res.items():
            bs = " ".join(f"{b:5.1f}" if b is not None else "    -" for b in v['walk_strict_bands'])
            say(f"  {name:28s} kept {v['kept']:7.3f}  x{v['kept'] / base16:5.3f} vs chain-16  walk-strict 0|1-2|3-6|7-10|11-14|15-22|23-30: {bs}")
        for aud in ("2b", "4b"):
            ref = max(res["tree-16 (EAGLE-2)"]["kept"], res[f"audit-chain-16 {aud}"]["kept"])
            ub = res[f"tree+audit UB {aud}"]["kept"]
            bo = max(res[f"tree+audit BR-{L} OPT {aud}"]["kept"] for L in (3, 5, 7))
            bp = max(res[f"tree+audit BR-{L} PES {aud}"]["kept"] for L in (3, 5, 7))
            v = 'PASS' if bp >= 1.02 * ref else ('KILL' if bo < 1.02 * ref else 'BETWEEN (OPT passes, PES does not)')
            say(f"  LINE {aud}: max(tree-16, audit-chain-16) {ref:.3f}; UB {ub:.3f} (x{ub / ref:.3f}); best BR OPT {bo:.3f} (x{bo / ref:.3f}), "
                f"PES {bp:.3f} (x{bp / ref:.3f}) -> {v}")
    json.dump(out, open(OUT + ".json", "w"), indent=1)
    open(OUT + ".txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
