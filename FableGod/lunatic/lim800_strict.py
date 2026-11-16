"""LIM800 STRICT SPAN ENDS (Lunatic, THINK_1116b, Mon 16 Nov 18:2x). The frame's INERT (span ends exact) applied to the
three measurements of IDEAS_1116b: (1) where the check-ending misses sit, (2) tree headroom (rescue code + span-end
misses whose 125B token is the head's rank <= r), (3) where EAGLE-2-31's walk-start side rows sit, now splitting INERT
positions into span-end decisions (the main or side token carries a newline in a comment / prose span or a triple
quote in a docstring) and pure wording (waste under INERT). CPU, read-only, < 0.5 GB.
usage: ~/BigMoE/.venv/bin/python3 -I lim800_strict.py"""
import os, sys
import numpy as np
SRC = open(os.path.expanduser("~/FableGod/lunatic/lim800_slack.py")).read().split("RES = {}")[0]
exec(SRC.replace('OUT = f"{H}/FableGod/lunatic/lim800_slack.out"', 'OUT = f"{H}/FableGod/lunatic/lim800_strict.out"'))
RR = open(os.path.expanduser("~/FableGod/lunatic/lim800_rerank.py")).read()
exec(RR[RR.index("CE_TOPK = {}"):RR.index("IDRE = ")])
def ends(a, b, s):
    if s in (1, 4): return "\n" in a or "\n" in b
    if s == 2: return any(q in a or q in b for q in ('"""', "'''"))
    return False
def inert_strict(A):
    m = np.isin(A["S"], (1, 2, 4))
    for i, j in zip(*np.nonzero(m & (A["rt"] == 0))):
        if ends(txt(A["DR"][i, j]), txt(A["Y"][i, j]), A["S"][i, j]): m[i, j] = False
    return m
cc1 = lambda A: np.nan_to_num(A["DL"], nan=-99) >= -1
for tag in ("heplus24", "mbppbank"):
    W = load(tag); TK = load_topk_cells(tag)
    acc = [cc1(A) | inert_strict(A) for A in W]
    ks = []; cls = {"CODE": 0, "SPAN-END": 0, "STR": 0}
    rescue = {1: [], 3: [], 7: []}
    for A, ac, (T8, P8, YR) in zip(W, acc, TK):
        r = A["rt"].copy(); r[(r == 0) & ac] = 1
        ks += C98.kept_walk(r, C98.R0)
        sub = r[C98.R0:]; p = 0
        while p < sub.shape[0]:
            k = 0
            while k < 31 and sub[p, k] == 1: k += 1
            if k < 31 and sub[p, k] == 0:
                s = A["S"][p + C98.R0, k]
                cls["CODE" if s == 0 else "STR" if s == 3 else "SPAN-END"] += 1
            p += k + 1
        for rk in rescue:
            r2 = r.copy(); r2[(r2 == 0) & (np.isin(A["S"], (0, 1, 2, 4))) & (YR <= rk)] = 1
            rescue[rk] += C98.kept_walk(r2, C98.R0)
    b = np.mean(ks); n = sum(cls.values())
    say(f"== {tag}: CC d1 + INERT (span ends strict) kept/32 {b:6.2f}; check-ending misses n {n}: " +
        ", ".join(f"{k} {100*v/n:.1f} %" for k, v in cls.items()))
    for rk, v in rescue.items():
        say(f"   rescue code + span-end misses with the 125B token at head rank <= {rk}: kept/32 {np.mean(v):6.2f}  x{np.mean(v)/b:5.3f}")
    # tree side rows at walk starts
    cz = np.load(f"{CE}/cells_t30b_{tag}.npz"); dep = cz["depth"]
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0))
    an_first = {int(cz["fi"][s]): int(cz["row"][s]) - 1 for s in st}; win_of = {int(cz["fi"][s]): w for w, s in enumerate(st)}
    tz = np.load(f"{CE}/tree_t30b_{tag}.npz")
    ws = set(zip(tz["a_fi"][tz["a_walk_start"]].tolist(), tz["a_an"][tz["a_walk_start"]].tolist()))
    fi, an, node, d, tok = tz["fi"], tz["an"], tz["node"], tz["depth"], tz["token"]
    key = fi.astype(np.int64) * 1000000 + an.astype(np.int64); order = np.argsort(key, kind="stable"); kk = key[order]
    bd = np.append(np.flatnonzero(np.append(True, kk[1:] != kk[:-1])), len(kk))
    cnt = {"CODE": 0, "SPAN-END decision": 0, "INERT wording (waste)": 0, "STR": 0}; nsel = 0
    for b0, b1 in zip(bd[:-1], bd[1:]):
        idx = order[b0:b1]; idx = idx[node[idx] < 31]
        if len(idx) == 0: continue
        f0, a0 = int(fi[idx[0]]), int(an[idx[0]])
        if (f0, a0) not in ws: continue
        A = W[win_of[f0]]; i = a0 - an_first[f0]; S0 = A["S"][:A["n0"], 0]; nsel += len(idx)
        for depth in np.unique(d[idx]):
            g = idx[d[idx] == depth]
            if len(g) < 2: continue
            p = i + int(depth); s = int(S0[p]) if 0 <= p < len(S0) else 0
            toks = [txt(t) for t in tok[g]]
            if s == 0: cnt["CODE"] += len(g) - 1
            elif s == 3: cnt["STR"] += len(g) - 1
            elif any(ends(t, "", s) for t in toks): cnt["SPAN-END decision"] += len(g) - 1
            else: cnt["INERT wording (waste)"] += len(g) - 1
    ns = sum(cnt.values())
    say(f"   walk-start side rows {ns} of {nsel} selected ({100*ns/nsel:.1f} %): " +
        ", ".join(f"{k} {100*v/ns:.1f} %" for k, v in cnt.items()))
open(OUT, "w").write("\n".join(lines) + "\n")
