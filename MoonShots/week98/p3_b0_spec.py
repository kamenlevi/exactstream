"""WEEK98 PILLAR 3, B0 (MoonShots, Sun 15 Nov 2026; pre-registered in LEVERS_A_MOONSHOTS.md "PILLAR 3", c12c192): the
hit rate of SPECULATIVE next-block drafting for tree-32, on BigMoE's real t30b tree dumps (read-only), under tree-32's own
walk (certify98's rule from R0, as MTP Head's tree_k98.py).
Tree-32 = the first 31 pops at each anchor. Under the drafter's own model, the chance that check n stops exactly at node
v is path_p(v) x (1 - sum of v's in-tree children's p); for the root, 1 - sum of its in-tree children's p. The guess is
the most likely stop (G1), or the two most likely (G2). The guessed bonus at v is v's best child NOT in the tree (by
rank in its top-8). A hit needs the actual stop node == a guessed node AND the true next token == that node's guessed
bonus. Also: the share of checks that stop at a leaf (the block used to its end on that path), and the accept depth.
Self-check: the walk's kept/32 must reproduce S9c's real EAGLE-2-32 (HE+ 16.263, MBPP 9.031).
TRUE-STATE (lab): the dump's trees are drafted from the 125B's true state at every anchor.
usage: ~/BigMoE/.venv/bin/python -I -B p3_b0_spec.py"""
import json, os, sys
import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/EvalSuite/tools"))
import certify98 as C98                                   # noqa: E402  (R0 only)

CE = H("~/BigMoE/data/deepdraft/ceiling")
SETS = {"he": ("heplus24", 16.263), "mbpp": ("mbppbank", 9.031)}
N1 = 31                                                    # tree-32 = 31 drafted nodes


def anchors(tag):
    z = np.load(f"{CE}/tree_t30b_{tag}.npz")
    fi, an, node = z["fi"], z["an"], z["node"]
    order = np.lexsort((node, an, fi))
    cols = {k: z[k][order] for k in ("fi", "an", "node", "parent", "depth", "token", "path_p", "y", "on_true",
                                      "top_ids", "top_p", "ynext_rank")}
    starts = np.r_[0, np.flatnonzero((np.diff(cols["fi"]) != 0) | (np.diff(cols["an"]) != 0)) + 1, len(order)]
    root = {(int(f), int(a)): (ti, tp) for f, a, ti, tp in zip(z["a_fi"], z["a_an"], z["root_top_ids"], z["root_top_p"])}
    per = {}
    for s, e in zip(starts[:-1], starts[1:]):
        f, a = int(cols["fi"][s]), int(cols["an"][s])
        per.setdefault(f, {})[a] = {k: v[s:e] for k, v in cols.items()} | {"root": root[(f, a)]}
    return per


def check(A):
    """one anchor's tree-32: (accepted, stop node or -1, is_leaf, G1 hit, G2 hit, G1 node right, G2 node right)"""
    nd = A["node"] < N1
    par, dep, onp = A["parent"][nd], A["depth"][nd], A["on_true"][nd]
    node, tok, pp, ti, tp, yr = A["node"][nd], A["token"][nd], A["path_p"][nd], A["top_ids"][nd], A["top_p"][nd], A["ynext_rank"][nd]
    idx = {int(n): i for i, n in enumerate(node)}
    kids = {}
    for i, p in enumerate(par):
        kids.setdefault(int(p), []).append(i)
    # the actual stop: the deepest on_true node (on_true marks the true path), else the root
    on = np.flatnonzero(onp)
    stop = int(node[on[np.argmax(dep[on])]]) if len(on) else -1
    acc = int(dep[on].max()) + 1 if len(on) else 0
    leaf = stop >= 0 and not kids.get(stop)
    # the drafter's own stop probabilities and guessed bonus per candidate stop
    cand = []
    rti, rtp = A["root"]
    for v in [-1] + [int(n) for n in node]:
        ch = kids.get(v, [])
        chtok = {int(tok[c]) for c in ch}
        if v == -1:
            ids, ps, base = rti, rtp, 1.0
        else:
            i = idx[v]; ids, ps, base = ti[i], tp[i], float(pp[i])
        pin = sum(float(p) for t, p in zip(ids, ps) if int(t) in chtok)
        r_guess = next((r for r, t in enumerate(ids) if int(t) not in chtok), None)
        cand.append((base * max(0.0, 1.0 - pin), v, r_guess))
    cand.sort(key=lambda x: -x[0])
    # the true next token's rank at the actual stop
    if stop >= 0:
        y_rank = int(yr[idx[stop]])
    else:                                                  # stop at the root: rank of the true depth-0 token in the root's top-8
        y0 = A["y"][A["depth"] == 0]
        y_rank = next((r for r, t in enumerate(rti) if len(y0) and int(t) == int(y0[0])), 99)
    hits = []
    for g in (1, 2):
        G = cand[:g]
        node_ok = any(v == stop for _, v, _ in G)
        hit = any(v == stop and r is not None and r == y_rank for _, v, r in G)
        hits.append((hit, node_ok))
    return acc, stop, leaf, hits[0][0], hits[1][0], hits[0][1], hits[1][1]


def main():
    out = {}
    for s, (tag, pub) in SETS.items():
        per = anchors(tag)
        kept, rows = [], []
        for f in sorted(per):
            W = per[f]; ans = sorted(W)
            p = C98.R0
            while p < len(ans) + 1:                        # tree_real125's walk: n = anchors + 1 slots
                if p == len(ans):                          # the extra end slot: no tree, kept 1 (not a hit-rate check)
                    kept.append(1); p += 1
                    continue
                r = check(W[ans[p]])
                rows.append(r); kept.append(r[0] + 1); p += r[0] + 1
        R = np.array(rows, dtype=object)
        k32 = float(np.mean(kept))
        acc = np.array([r[0] for r in rows])
        res = dict(checks=len(rows), kept32=round(k32, 3), selfcheck=("PASS" if abs(k32 - pub) < 0.01 else f"FAIL vs {pub}"),
                   stop_at_leaf=round(100 * float(np.mean([r[2] for r in rows])), 1),
                   stop_at_root=round(100 * float(np.mean([r[1] == -1 for r in rows])), 1),
                   G1_hit=round(100 * float(np.mean([r[3] for r in rows])), 1),
                   G2_hit=round(100 * float(np.mean([r[4] for r in rows])), 1),
                   G1_node=round(100 * float(np.mean([r[5] for r in rows])), 1),
                   G2_node=round(100 * float(np.mean([r[6] for r in rows])), 1),
                   accepted_quantiles={q: int(np.percentile(acc, q)) for q in (10, 25, 50, 75, 90)},
                   accepted_mean=round(float(acc.mean()), 2))
        out[s] = res
        print(s, json.dumps(res))
    json.dump(out, open(H("~/MoonShots/week98/p3_b0_spec.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
