"""LIM800 TREE WASTE (Lunatic, THINK_1116b, Mon 16 Nov 18:2x). CPU only, read-only, < 1 GB.
Of EAGLE-2's 31 selected nodes per anchor (the first 31 best-first pops in BigMoE's tree_t30b_{tag}.npz, 32 pops per
anchor), how many are SIDE rows (a second or later node at the same depth = an alternative for the same token
position), and in which lexer state does that position sit (lim800_slack lexer)? Side rows in COMMENT / DOC / PROSE
spans are spent where INERT already accepts any draft. Only walk-start anchors are counted (the checks real decoding
makes under the exact chain walk is not reproduced here; all anchors and walk starts are both reported).
usage: ~/BigMoE/.venv/bin/python3 -I lim800_treewaste.py"""
import os, sys
import numpy as np
SRC = open(os.path.expanduser("~/FableGod/lunatic/lim800_slack.py")).read().split("RES = {}")[0]
exec(SRC.replace('OUT = f"{H}/FableGod/lunatic/lim800_slack.out"', 'OUT = f"{H}/FableGod/lunatic/lim800_treewaste.out"'))
for tag in ("heplus24", "mbppbank"):
    W = load(tag)
    cz = np.load(f"{CE}/cells_t30b_{tag}.npz")
    dep = cz["depth"]; st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0))
    an_first = {int(cz["fi"][s]): int(cz["row"][s]) - 1 for s in st}
    win_of = {int(cz["fi"][s]): w for w, s in enumerate(st)}
    tz = np.load(f"{CE}/tree_t30b_{tag}.npz")
    fi, an, node, d = tz["fi"], tz["an"], tz["node"], tz["depth"]
    sel = node < 31
    side_state = {"all": np.zeros(5), "ws": np.zeros(5)}; n_side = {"all": 0, "ws": 0}; n_sel = {"all": 0, "ws": 0}
    ws_set = set(zip(tz["a_fi"][tz["a_walk_start"]].tolist(), tz["a_an"][tz["a_walk_start"]].tolist()))
    # group by (fi, an)
    key = fi.astype(np.int64) * 1000000 + an.astype(np.int64)
    order = np.argsort(key, kind="stable"); ks = key[order]
    bounds = np.flatnonzero(np.append(True, ks[1:] != ks[:-1])); bounds = np.append(bounds, len(ks))
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        idx = order[b0:b1]; idx = idx[sel[idx]]
        if len(idx) == 0: continue
        f0, a0 = int(fi[idx[0]]), int(an[idx[0]])
        A = W[win_of[f0]]; i = a0 - an_first[f0]
        S0 = A["S"][:A["n0"], 0]
        dd = d[idx]
        for grp in ("all", "ws"):
            if grp == "ws" and (f0, a0) not in ws_set: continue
            n_sel[grp] += len(idx)
            for depth in np.unique(dd):
                c = int((dd == depth).sum())
                if c > 1:
                    p = i + int(depth)
                    s = int(S0[p]) if 0 <= p < len(S0) else 0
                    side_state[grp][s] += c - 1; n_side[grp] += c - 1
    say(f"== {tag}")
    for grp in ("all", "ws"):
        ss = side_state[grp]
        say(f"   {grp:3s} anchors: selected nodes {n_sel[grp]}, side rows {n_side[grp]} ({100*n_side[grp]/max(1,n_sel[grp]):.1f} %)"
            f" -> in CODE {100*ss[0]/max(1,n_side[grp]):.1f} %, COMMENT {100*ss[1]/max(1,n_side[grp]):.1f} %, DOC "
            f"{100*ss[2]/max(1,n_side[grp]):.1f} %, STR {100*ss[3]/max(1,n_side[grp]):.1f} %, PROSE {100*ss[4]/max(1,n_side[grp]):.1f} %")
open(OUT, "w").write("\n".join(lines) + "\n")
