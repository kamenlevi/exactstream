"""THINK_1116b audit, budget-correct version (Fable C, Mon 16 Nov 2026, 18:2x). Lossy acceptance ON the real t30b 127-node
tree dumps (BigMoE, tree_t30b_{heplus24,mbppbank}_n127k8.npz), tree-K = the first K pops (as treek_final does), K = 31 / 63 / 127
(32 / 64 / 128-row checks). The walk per anchor: exact accepted depth A = the selected true-path prefix; at depth A the
head's best SELECTED child under the true parent is tried under the lossy rule (close-call within delta nats of the 125B's
top-1, from the bank's stored top-32 log-probs; INERT = the true text is wholly inside a comment / docstring there and
the drafted token does not end the span); after a rescue the walk follows the selected best-child chain and judges each
node against the true text (equal / close-call / inert), as NewIdeas' tree_cc32.py does for close-call (the standing
approximation: after a lossy acceptance the walk goes on along the true text). Never more than K nodes. CPU, read-only.
usage: ~/BigMoE/.venv/bin/python -I -B inert_tree127_1116b.py
"""
import os, sys, json, glob, keyword
import numpy as np
from tokenizers import Tokenizer
sys.dont_write_bytecode = True
H = os.path.expanduser("~")
CE = f"{H}/BigMoE/data/deepdraft/ceiling"
BANK = f"{H}/datasets/deepdraft/bank"
TOKJ = sorted(glob.glob(f"{H}/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-MLX-4bit/snapshots/*/tokenizer.json"))[0]
OUT = f"{H}/FableGod/fablec/inert_tree127_1116b.out"
SETS = {"he": ("heplus24", 16.26, 20.77, 23.70), "mbpp": ("mbppbank", 9.03, 10.19, 11.29)}
KS = (31, 63, 127)
R0 = 1
KW = set(keyword.kwlist) | {"True", "False", "None", "self"}
PUN = set("()[]{}:,.;=+-*/<>!&|^%~@")
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
sys.path.insert(0, f"{H}/FableGod/fablec")
from inert_tree_1116b import lex_classes          # the slot_prior lexer + docstring class (importing runs nothing: guarded below)

tok = Tokenizer.from_file(TOKJ)
PIECE = {}
def piece(t):
    if t not in PIECE: PIECE[t] = tok.decode([int(t)])
    return PIECE[t]

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
        anchors = {}
        for f_ in banks: anchors[f_] = sorted(set(nan_[nfi == f_].tolist()))
        aidx = {f_: {a: i for i, a in enumerate(anchors[f_])} for f_ in banks}
        ARMS = [("exact", 0, 0, False, set()),
                ("CC d1", 1, 1, False, set()),
                ("#8 d1 code / d3 span", 1, 3, False, {"in comment", "in docstring"}),
                ("CC d1 + INERT c+d", 1, 1, True, {"in comment", "in docstring"}),
                ("CC d2 + INERT c+d", 2, 2, True, {"in comment", "in docstring"}),
                ("INERT c+d only", 0, 0, True, {"in comment", "in docstring"})]
        acc = {K: {nm: {f_: np.zeros(len(anchors[f_]), np.int64) for f_ in banks} for nm, *_ in ARMS} for K in KS}
        depthK = {K: [] for K in KS}
        for b0, b1 in zip(bounds[:-1], bounds[1:]):
            f_, a_ = int(nfi[b0]), int(nan_[b0]); B = banks[f_]; ids, tk, lp, cls = B["ids"], B["tk"], B["lp"], B["cls"]
            nd, dp, tkn, pp, pa, ot = nnode[b0:b1], ndep[b0:b1], ntok[b0:b1], npp[b0:b1].astype(np.float64), npar[b0:b1], ntrue[b0:b1].astype(bool)
            idx = {int(n): i for i, n in enumerate(nd)}
            def cc_ok(t, d, delta):                       # close-call: the 125B's log-prob of the drafted token within delta of its top-1
                row = a_ + 1 + d
                if row >= len(lp) or delta <= 0: return False
                j = np.flatnonzero(tk[row] == t)
                return len(j) > 0 and float(lp[row, j[0]] - lp[row, 0]) >= -delta
            def in_span(t, d, spanset):                   # the true text wholly inside a span there, the draft does not end it
                pos = a_ + 2 + d
                if pos >= len(cls) or pos - 1 < 0: return False
                cb, ca = cls[pos - 1], cls[pos]
                if cb != ca or ca not in spanset: return False
                pc = piece(t)
                if ca == "in comment": return "\n" not in pc
                return '"' not in pc and "'" not in pc and "\n" not in pc
            def true_tok(d):
                pos = a_ + 2 + d
                return ids[pos] if pos < len(ids) else None
            for K in KS:
                sel = nd < K
                hit = ot & sel
                A = int(dp[hit].max()) + 1 if hit.any() else 0
                depthK[K].append(int(dp[sel].max()) + 1)
                for nm, dcode, dspan, inert, spanset in ARMS:
                    a = A
                    if true_tok(A) is not None and (dcode or inert):
                        pid = -1 if A == 0 else int(nd[np.flatnonzero(hit & (dp == A - 1))[0]])
                        kids = np.flatnonzero(sel & (pa == pid) & (dp == A))
                        if len(kids):
                            cur = int(kids[np.argmax(pp[kids])]); t = int(tkn[cur])
                            sp = in_span(t, A, spanset) if spanset else False
                            ok = cc_ok(t, A, dspan if sp else dcode) or (inert and sp)
                            if ok:
                                a = A + 1; d2 = A
                                while True:                 # the selected best-child chain below the rescued node
                                    ch = np.flatnonzero(sel & (pa == nd[cur]))
                                    if not len(ch): break
                                    cur = int(ch[np.argmax(pp[ch])]); d2 += 1; t = int(tkn[cur]); tt = true_tok(d2)
                                    if tt is None: break
                                    sp = in_span(t, d2, spanset) if spanset else False
                                    if t == tt or cc_ok(t, d2, dspan if sp else dcode) or (inert and sp): a = d2 + 1
                                    else: break
                    acc[K][nm][f_][aidx[f_][a_]] = min(a, K)
        say(f"== {s} ({tag}): {len(bounds) - 1} anchors, {len(banks)} windows; published exact kept 32 / 64 / 128 rows: {p32} / {p64} / {p128}")
        RES[s] = {}
        for K in KS:
            say(f"   tree-{K + 1} ({K} nodes + bonus; mean selected depth {np.mean(depthK[K]):.1f} levels):")
            base = None
            for nm, *_ in ARMS:
                ks = nc = 0
                for f_ in banks:
                    ac = acc[K][nm][f_]; p = R0; n = len(ac)
                    while p < n:
                        a = int(ac[p]); ks += a + 1; nc += 1; p += a + 1
                kp = ks / nc; RES[s][f"{K + 1}:{nm}"] = kp
                if base is None: base = kp
                say(f"      {nm:24s} kept/check {kp:7.3f}  x{kp / base:.3f} vs exact tree-{K + 1}")
    open(OUT, "w").write("\n".join(lines) + "\n")
    json.dump(RES, open(OUT.replace(".out", ".json"), "w"), indent=1)

if __name__ == "__main__":
    main()
