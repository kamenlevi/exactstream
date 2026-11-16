"""LIM800 RERANK (Lunatic, THINK_1116b, Mon 16 Nov 18:0x). CPU only, read-only, one process, < 0.5 GB.
Same cells / TOPK / walk / lexer as lim800_slack.py (imported). Baseline rule = CC d1 + INERT (comment / docstring /
prose misses accepted; code rows strict). Questions:
 1. TREE HEADROOM on code rows: ceilings if code misses whose 125B token is the head's rank <= r (r = 1, 3, 7) were
    rescued (what any tree that spends its side rows only on code positions could buy).
 2. SCOPE RE-RANK (zero rows, zero GPU bytes): on CODE cells, if the head's top-1 is an identifier-like token never seen
    earlier in the answer text but a lower-ranked top-8 candidate (identifier-like, p >= p1 * ratio) was seen earlier,
    draft that candidate instead. Net effect on all code cells (right -> wrong and wrong -> right), then the walk.
 3. SYNTAX VETO (zero rows, zero GPU bytes): on CODE cells, if the head's top-1 closes a bracket that does not match the
    innermost open bracket (or closes with none open), draft the best top-8 candidate that is not such a close.
Re-ranked cells are judged against the text token y (right iff the new draft == y). The walk is the chain-32 walk.
usage: ~/BigMoE/.venv/bin/python3 -I lim800_rerank.py"""
import os, sys, json, re
import numpy as np
SRC = open(os.path.expanduser("~/FableGod/lunatic/lim800_slack.py")).read().split("RES = {}")[0]
exec(SRC.replace('OUT = f"{H}/FableGod/lunatic/lim800_slack.out"', 'OUT = f"{H}/FableGod/lunatic/lim800_rerank.out"'))
CE_TOPK = {}
def load_topk_cells(tag):
    tz = np.load(f"{CE}/topk_t30b_{tag}.npz")
    dep = tz["depth"].astype(int)
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    out = []
    for s, t in zip(st, en):
        dw = dep[s:t]; n0 = int((dw == 0).sum())
        T8 = np.full((n0 + 1, DEP, 8), -1, np.int64); P8 = np.zeros((n0 + 1, DEP, 8), np.float32)
        YR = np.full((n0 + 1, DEP), 99, np.int32)
        for j in range(DEP):
            msk = dw == j; c = int(msk.sum())
            T8[:c, j] = tz["top_ids"][s:t][msk]; P8[:c, j] = tz["top_p"][s:t][msk]; YR[:c, j] = tz["y_rank"][s:t][msk]
        out.append((T8, P8, YR))
    return out

IDRE = re.compile(r"^\s?[A-Za-z_][A-Za-z_0-9]*$")
def is_ident(s): return bool(IDRE.match(s))
OPEN, CLOSE = "([{", ")]}"
def scope_and_brackets(ids, states):
    """per position p (before token p): the set of identifier-like token ids seen in [0, p) and the bracket stack top"""
    seen = set(); seen_at = []; stack = []; tops = []
    for t, s in zip(ids, states):
        seen_at.append(frozenset(seen)); tops.append(stack[-1] if stack else "")
        tx = txt(t)
        if is_ident(tx): seen.add(int(t))
        if s == 0:
            for ch in tx:
                if ch in OPEN: stack.append(ch)
                elif ch in CLOSE and stack: stack.pop()
    return seen_at, tops

def bad_close(tx, top):
    for ch in tx:
        if ch in OPEN: return False
        if ch in CLOSE:
            return (not top) or OPEN[CLOSE.index(ch)] != top
    return False

RES = {}
for tag in ("heplus24", "mbppbank"):
    W = load(tag); TK = load_topk_cells(tag)
    inert = lambda A: np.isin(A["S"], (1, 2, 4))
    base_acc = lambda A: (np.nan_to_num(A["DL"], nan=-99) >= -1) | inert(A)
    say(f"== {tag}")
    b = walk(W, base_acc)[0]
    say(f"   base cc1 + inert            kept/32 {b:6.2f}")
    # 1. tree headroom on code rows
    for r in (1, 3, 7):
        k = 0
        ks = []
        for A, (T8, P8, YR) in zip(W, TK):
            rr = A["rt"].copy(); rr[(rr == 0) & base_acc(A)] = 1
            rr[(rr == 0) & (A["S"] == 0) & (YR <= r)] = 1
            ks += C98.kept_walk(rr, C98.R0)
        say(f"   + rescue code misses with 125B token at head rank <= {r}: kept/32 {np.mean(ks):6.2f}  x{np.mean(ks)/b:5.3f}")
    # 2 + 3. re-rankers
    for name in ("scope0.3", "scope0.1", "scope0.03", "syntax", "scope0.1+syntax"):
        ks = []; w2r = r2w = ncode = 0
        for A, (T8, P8, YR) in zip(W, TK):
            n0 = A["n0"]; y0 = A["Y"][:n0, 0]; S0 = A["S"][:n0, 0]
            seen_at, tops = scope_and_brackets(y0, S0)
            rr = A["rt"].copy()
            for i in range(n0 + 1):
                for j in range(DEP):
                    if rr[i, j] < 0 or A["S"][i, j] != 0: continue
                    p = i + j
                    if p >= n0: continue
                    cand = list(T8[i, j]); pr = list(P8[i, j]); pick = cand[0]
                    if name.startswith("scope"):
                        ratio = float(name.split("+")[0][5:])
                        t1 = txt(cand[0])
                        if is_ident(t1) and int(cand[0]) not in seen_at[p]:
                            for c, q in zip(cand[1:], pr[1:]):
                                if c >= 0 and is_ident(txt(c)) and int(c) in seen_at[p] and q >= pr[0] * ratio:
                                    pick = c; break
                    if name.endswith("syntax"):
                        if bad_close(txt(pick), tops[p]):
                            for c in cand:
                                if c >= 0 and c != pick and not bad_close(txt(c), tops[p]):
                                    pick = c; break
                    ncode += 1
                    new = int(pick == A["Y"][i, j]); old = int(rr[i, j])
                    if new != old:
                        if new: w2r += 1
                        else: r2w += 1
                        rr[i, j] = new
            rr[(rr == 0) & base_acc(A)] = 1
            ks += C98.kept_walk(rr, C98.R0)
        say(f"   re-rank {name:16s} kept/32 {np.mean(ks):6.2f}  x{np.mean(ks)/b:5.3f}  (code cells {ncode}: wrong->right {w2r}, right->wrong {r2w})")
    RES[tag] = b
    say()
open(OUT, "w").write("\n".join(lines) + "\n")
