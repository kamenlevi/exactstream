"""LIM800 SPAN ENDS STRICT (Lunatic, THINK_1116b, Mon 16 Nov 18:2x). The frame's INERT keeps span ends exact; the lexer
tags a span's closing token by the state at its start (INERT). Here a cell is INERT only if neither the drafted nor the
text token carries a span end: a newline in a comment / prose span, a triple quote in a docstring. CPU, read-only.
usage: ~/BigMoE/.venv/bin/python3 -I lim800_spanend.py"""
import os, sys
import numpy as np
SRC = open(os.path.expanduser("~/FableGod/lunatic/lim800_slack.py")).read().split("RES = {}")[0]
exec(SRC.replace('OUT = f"{H}/FableGod/lunatic/lim800_slack.out"', 'OUT = f"{H}/FableGod/lunatic/lim800_spanend.out"'))
def inert_strict_ends(A):
    m = np.isin(A["S"], (1, 2, 4))
    for i, j in zip(*np.nonzero(m & (A["rt"] == 0))):
        a, b = txt(A["DR"][i, j]), txt(A["Y"][i, j]); s = A["S"][i, j]
        if s in (1, 4) and ("\n" in a or "\n" in b): m[i, j] = False
        if s == 2 and any(q in a or q in b for q in ('"""', "'''")): m[i, j] = False
    return m
for tag in ("heplus24", "mbppbank"):
    W = load(tag)
    cc1 = lambda A: np.nan_to_num(A["DL"], nan=-99) >= -1
    a = walk(W, lambda A: cc1(A) | np.isin(A["S"], (1, 2, 4)))[0]
    b = walk(W, lambda A: cc1(A) | inert_strict_ends(A))[0]
    c = walk(W, lambda A: cc1(A) | inert_strict_ends(A) | ((A["S"] == 0) & (np.nan_to_num(A["M"], nan=0) >= 4)))[0]
    say(f"== {tag}: CC d1 + INERT {a:6.2f} | span ends strict {b:6.2f} (x{b/a:5.3f}) | + fix code m >= 4 {c:6.2f}")
open(OUT, "w").write("\n".join(lines) + "\n")
