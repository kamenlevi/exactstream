"""THINK_1116b, part 2 (Fable C, Mon 16 Nov 2026, 18:3x): the SHAPE question under INERT, budget-correct on the real t30b
127-node dumps. Three ways to spend K nodes: TREE (the first K pops, today's EAGLE-2 order); CHAIN (the head's best-child
chain from the root, K deep); SPAN-TREE (the invention: inside a comment / docstring row only the best child is eligible,
siblings go to code rows; top-K eligible by path probability, a closed subtree). Walk and lossy rules as in
inert_tree127_1116b.py (exact prefix; rescue by close-call delta / INERT; then the selected best-child chain judged against
the true text; never more than K nodes). CPU, read-only, ~1-2 min.
usage: ~/BigMoE/.venv/bin/python -I -B inert_shape127_1116b.py
"""
import os, sys, json
import numpy as np
sys.dont_write_bytecode = True
H = os.path.expanduser("~")
sys.path.insert(0, f"{H}/FableGod/fablec")
from inert_tree127_1116b import CE, BANK, piece, lex_classes, KS, R0, SETS
OUT = f"{H}/FableGod/fablec/inert_shape127_1116b.out"
CD = {"in comment", "in docstring"}
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

def main():
    RES = {}
    for s, (tag, p32, p64, p128) in SETS.items():
        tz = np.load(f"{CE}/tree_t30b_{tag}_n127k8.npz")
        files = [str(f) for f in tz["files"]]
        nfi, nan_, nnode, ndep, ntok, npp, npar, ntrue = (tz[k] for k in ("fi", "an", "node", "depth", "token", "path_p", "parent", "on_true"))
        order = np.lexsort((nnode, nan_, nfi))
        nfi, nan_, nnode, ndep, ntok, npp, npar, ntrue = (a[order] for a in (nfi, nan_, nnode, ndep, ntok, npp, npar, ntrue))
        bounds = np.flatnonzero(np.append(True, (np.diff(nfi) != 0) | (np.diff(nan_) != 0))); bounds = np.append(bounds, len(nfi))
        banks = {}
        for f_ in sorted(set(nfi.tolist())):
            b = np.load(os.path.join(BANK, files[f_]))
            ids = [int(x) for x in b["ids"]]; gen = int(b["gen"]); st0 = max(gen - 1, 0)
            cls = [None] * st0 + lex_classes([piece(i) for i in ids[st0:]])
            banks[f_] = {"ids": ids, "tk": b["tk"].astype(np.int64), "lp": b["lp"].astype(np.float32), "cls": cls}
        anchors = {f_: sorted(set(nan_[nfi == f_].tolist())) for f_ in banks}
        aidx = {f_: {a: i for i, a in enumerate(anchors[f_])} for f_ in banks}
        SHAPES = ("tree", "chain", "span-tree")
        ARMS = [("exact", 0, 0, False, set()), ("CC d1", 1, 1, False, set()), ("CC d1 + INERT c+d", 1, 1, True, CD)]
        acc = {(sh, K, nm): {f_: np.zeros(len(anchors[f_]), np.int64) for f_ in banks} for sh in SHAPES for K in KS for nm, *_ in ARMS}
        nsel = {(sh, K): [] for sh in SHAPES for K in KS}; dsel = {(sh, K): [] for sh in SHAPES for K in KS}
        for b0, b1 in zip(bounds[:-1], bounds[1:]):
            f_, a_ = int(nfi[b0]), int(nan_[b0]); B = banks[f_]; ids, tk, lp, cls = B["ids"], B["tk"], B["lp"], B["cls"]
            nd, dp, tkn, pp, pa, ot = nnode[b0:b1], ndep[b0:b1], ntok[b0:b1], npp[b0:b1].astype(np.float64), npar[b0:b1], ntrue[b0:b1].astype(bool)
            idx = {int(n): i for i, n in enumerate(nd)}
            n = len(nd)
            # best child flags
            o2 = np.lexsort((-pp, pa)); best = np.zeros(n, bool); best[o2[0]] = True
            best[o2[1:]] = pa[o2[1:]] != pa[o2[:-1]]
            def span_row(d):
                pos = a_ + 2 + d
                return 0 <= pos - 1 and pos < len(cls) and cls[pos - 1] == cls[pos] and cls[pos] in CD
            def cc_ok(t, d, delta):
                row = a_ + 1 + d
                if row >= len(lp) or delta <= 0: return False
                j = np.flatnonzero(tk[row] == t)
                return len(j) > 0 and float(lp[row, j[0]] - lp[row, 0]) >= -delta
            def in_span(t, d):
                if not span_row(d): return False
                pc = piece(t); pos = a_ + 2 + d
                if cls[pos] == "in comment": return "\n" not in pc
                return '"' not in pc and "'" not in pc and "\n" not in pc
            def true_tok(d):
                pos = a_ + 2 + d
                return ids[pos] if pos < len(ids) else None
            # chain nodes: best child from the root downwards
            chain = np.zeros(n, bool); roots = np.flatnonzero(pa == -1)
            if len(roots):
                cur = int(roots[np.argmax(pp[roots])]); chain[cur] = True
                while True:
                    ch = np.flatnonzero(pa == nd[cur])
                    if not len(ch): break
                    cur = int(ch[np.argmax(pp[ch])]); chain[cur] = True
            # span-tree eligibility, top-down by depth
            elig = np.zeros(n, bool)
            for i in np.argsort(dp, kind="stable"):
                par_ok = pa[i] == -1 or (int(pa[i]) in idx and elig[idx[int(pa[i])]])
                elig[i] = bool(par_ok and (best[i] or not span_row(int(dp[i]))))
            for sh in SHAPES:
                for K in KS:
                    if sh == "tree": sel = nd < K
                    elif sh == "chain": sel = chain & (dp < K)
                    else:
                        sel = np.zeros(n, bool); e = np.flatnonzero(elig)
                        sel[e[np.argsort(-pp[e], kind="stable")[:K]]] = True
                    nsel[(sh, K)].append(int(sel.sum())); dsel[(sh, K)].append(int(dp[sel].max()) + 1 if sel.any() else 0)
                    hit = ot & sel
                    A = int(dp[hit].max()) + 1 if hit.any() else 0
                    for nm, dcode, dspan, inert, spanset in ARMS:
                        a = A
                        if true_tok(A) is not None and (dcode or inert):
                            pid = -1 if A == 0 else int(nd[np.flatnonzero(hit & (dp == A - 1))[0]])
                            kids = np.flatnonzero(sel & (pa == pid) & (dp == A))
                            if len(kids):
                                cur = int(kids[np.argmax(pp[kids])]); t = int(tkn[cur])
                                sp = in_span(t, A) if spanset else False
                                if cc_ok(t, A, dspan if sp else dcode) or (inert and sp):
                                    a = A + 1; d2 = A
                                    while True:
                                        ch = np.flatnonzero(sel & (pa == nd[cur]))
                                        if not len(ch): break
                                        cur = int(ch[np.argmax(pp[ch])]); d2 += 1; t = int(tkn[cur]); tt = true_tok(d2)
                                        if tt is None: break
                                        sp = in_span(t, d2) if spanset else False
                                        if t == tt or cc_ok(t, d2, dspan if sp else dcode) or (inert and sp): a = d2 + 1
                                        else: break
                        acc[(sh, K, nm)][f_][aidx[f_][a_]] = min(a, K)
        say(f"== {s} ({tag}): published exact tree kept 32 / 64 / 128 rows: {p32} / {p64} / {p128}")
        RES[s] = {}
        for K in KS:
            for sh in SHAPES:
                say(f"   {sh:9s} K {K + 1:3d}: nodes used {np.mean(nsel[(sh, K)]):6.1f}, depth {np.mean(dsel[(sh, K)]):5.1f} levels")
                for nm, *_ in ARMS:
                    ks = nc = 0
                    for f_ in banks:
                        ac = acc[(sh, K, nm)][f_]; p = R0; nn = len(ac)
                        while p < nn:
                            a = int(ac[p]); ks += a + 1; nc += 1; p += a + 1
                    kp = ks / nc; RES[s][f"{sh}:{K + 1}:{nm}"] = kp
                    say(f"      {nm:20s} kept/check {kp:7.3f}")
    open(OUT, "w").write("\n".join(lines) + "\n")
    json.dump(RES, open(OUT.replace(".out", ".json"), "w"), indent=1)

if __name__ == "__main__":
    main()
