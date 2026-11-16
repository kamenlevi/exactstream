"""THINK_1115 #3 TIE-STUBS (ZERO idea 2), NewIdeas, written Mon 16 Nov 08:2x; run on the 32-pop dumps now (mechanics +
partial evidence) and on BigMoE's 127-node dumps Tuesday (the verdict).
A placed #2 branch at a predicted tie gets a RESERVED #1 continuation (a stub of s nodes: the #2 node's own best-child
chain), because on path probability alone its subtree always loses (K98). EAGLE-2's pop order fills the rest of the
K-node budget. Every selection is a closed subtree: stubs are chains from a placed sibling whose parent is a spine node
that is kept; the fill takes pops in order, skipping nodes whose parent is not kept.
Placement rules (draft-time only):
  P-THETA: along the spine (each spine node = the best child of the previous), at depths 0..D (D 6), place the #2 child
           of the spine parent where the spine edge's probability < theta (0.6 / 0.8 / 0.9), at most t ties (1 / 2 / 3);
  P-KEY:   the same, but only after a HARD start (THINK #1's key) or in a COMMENT (THINK #7's key); EAGLE-2 otherwise;
  ORACLE:  one stub at the true path's first divergence from the spine (the ceiling of any placement rule).
Stub length s = 2 / 4 (nodes after the #2 node; truncated where the dump has no deeper pops, which is common at 32).
Accepted = the deepest on-true selected node with all on-true ancestors selected, + 1. Walk: certify98's (R0 = 1).
LINE (ZERO's, written before the run): KILL if the best non-oracle rule <= EAGLE-2-K x 1.02 on HE+ (MBPP reported).
usage: ~/BigMoE/.venv/bin/python -I -B tie_stubs.py [--dump-dir DIR --dump-fmt FMT] [--K 15]"""
import os, sys, json, argparse
import numpy as np
from tokenizers import Tokenizer
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/NewIdeas/research/week98"))
import tree_audit as TA
import slot_prior as SP


def select(n_, d_, p_, a_, o_, K, rule, s, keyed_on):
    """returns a boolean selection of K nodes for one anchor, per rule"""
    N = len(n_); idx = {int(x): i for i, x in enumerate(n_)}
    kids = {}
    for i in range(N):
        kids.setdefault(int(a_[i]), []).append(i)
    for k in kids:
        kids[k].sort(key=lambda i: -p_[i])
    sel = np.zeros(N, bool); used = 0
    if rule is not None and keyed_on:
        kind, theta, t = rule
        # the spine: best child chain from the root
        spine = []; cur = -1
        while cur in kids and len(spine) < 40:
            c = kids[cur][0]; spine.append(c); cur = int(n_[c])
        stubs = []
        if kind == "oracle":
            for j, c in enumerate(spine):
                if not o_[c]:
                    par = -1 if j == 0 else int(n_[spine[j - 1]])
                    if par == -1 or o_[spine[j - 1]]:
                        tr = [x for x in kids.get(par, []) if o_[x]]
                        if tr:
                            stubs.append((j, tr[0]))
                    break
        else:
            for j, c in enumerate(spine[:7]):
                if len(stubs) >= t:
                    break
                par = -1 if j == 0 else int(n_[spine[j - 1]])
                pe = p_[c] / (p_[spine[j - 1]] if j > 0 else 1.0)
                if pe < theta and len(kids.get(par, [])) > 1:
                    stubs.append((j, kids[par][1]))
        # reserve: the spine up to each stub's parent, the #2 node, and its best-child chain of s nodes
        for j, sib in stubs:
            for c in spine[:j]:
                if not sel[c] and used < K:
                    sel[c] = True; used += 1
            chain = [sib]; cur = int(n_[sib])
            while len(chain) < s + 1 and cur in kids:
                nx = kids[cur][0]; chain.append(nx); cur = int(n_[nx])
            for c in chain:
                if not sel[c] and used < K:
                    sel[c] = True; used += 1
    # fill with EAGLE-2's pop order, parents first
    for i in np.argsort(n_, kind="stable"):
        if used >= K:
            break
        if sel[i]:
            continue
        par = int(a_[i])
        if (par == -1) or (par in idx and sel[idx[par]]):
            sel[i] = True; used += 1
    return sel


def accepted(sel, n_, d_, a_, o_):
    idx = {int(x): i for i, x in enumerate(n_)}
    best = 0
    for i in np.flatnonzero(sel & o_):
        j, ok = i, True
        while int(a_[j]) != -1:
            pj = idx.get(int(a_[j]))
            if pj is None or not sel[pj] or not o_[pj]:
                ok = False; break
            j = pj
        if ok:
            best = max(best, int(d_[i]) + 1)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default=TA.CE); ap.add_argument("--dump-fmt", default="tree_t30b_{tag}.npz")
    ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--out", default=os.path.expanduser("~/NewIdeas/research/week98/tie_stubs"))
    a = ap.parse_args(); K = a.K
    TA.say = lambda s="": None
    tok = Tokenizer.from_file(SP.TOKJ)
    lines, res = [], {}
    def say(s=""):
        print(s, flush=True); lines.append(s)
    RULES = {"EAGLE-2": None, "ORACLE s2": ("oracle", 0, 1), "ORACLE s4": ("oracle", 0, 1)}
    for th in (0.6, 0.8, 0.9):
        for t in (1, 2, 3):
            RULES[f"THETA {th} t{t}"] = ("theta", th, t)
    say(f"dumps {a.dump_dir}/{a.dump_fmt}; K {K} nodes (+ bonus)")
    for s_, (tag, rtag, pub32, pub16) in TA.SETS.items():
        W = TA.load_windows(tag, pub32)
        tz = np.load(os.path.join(a.dump_dir, a.dump_fmt.format(tag=tag)), allow_pickle=True)
        fi, an, nd, dp, ot, pp, pa = (tz[k] for k in ("fi", "an", "node", "depth", "on_true", "path_p", "parent"))
        order = np.lexsort((nd, an, fi)); fi, an, nd, dp, ot, pp, pa = (x[order] for x in (fi, an, nd, dp, ot, pp, pa))
        b = np.flatnonzero(np.append(True, (np.diff(fi) != 0) | (np.diff(an) != 0))); b = np.append(b, len(fi))
        T = {(int(fi[b0]), int(an[b0])): (b0, b1) for b0, b1 in zip(b[:-1], b[1:])}
        com = {}
        for X in W:
            z = np.load(os.path.join(SP.BANK, X["file"]), allow_pickle=True)
            ids = [int(x) for x in z["ids"]]; gen = int(z["gen"]); st = max(gen - 1, 0)
            cls = [None] * st + SP.lex_classes([tok.decode([i]) for i in ids[st:]])
            com[X["fi"]] = {int(q): (cls[int(q) + 1] == "in comment") if 0 <= int(q) + 1 < len(cls) else False for q in X["an"]}
        say(f"\n=== {s_} ({tag}) ===")
        r = {}
        for name, rule in RULES.items():
            for keyed in ((False, True) if rule is not None and rule[0] == "theta" else (False,)):
                stub_s = 4 if name.endswith("s4") else 2
                for s_len in ((stub_s,) if rule is None or rule[0] == "oracle" else (2, 4)):
                    ks = []
                    for X in W:
                        p, prev = 1, None
                        while p < X["n"]:
                            if prev is None:
                                hard = True
                            else:
                                q, qa = prev
                                mm = X["M"][q, qa] if qa < K and X["rt"][q, qa] >= 0 else np.inf
                                last = [X["M"][q, j] for j in range(max(0, qa - 3), qa)]
                                hard = (mm < 2) or any((not np.isnan(v)) and v < 2 for v in last)
                            if p < len(X["an"]):
                                a0 = int(X["an"][p]); b0, b1 = T[(X["fi"], a0)]
                                n_, d_, p_, a_, o_ = nd[b0:b1], dp[b0:b1], pp[b0:b1].astype(np.float64), pa[b0:b1], ot[b0:b1].astype(bool)
                                on = (hard or com[X["fi"]].get(a0, False)) if keyed else True
                                sel = select(n_, d_, p_, a_, o_, K, rule, s_len, on)
                                acc = accepted(sel, n_, d_, a_, o_)
                            else:
                                acc = 0
                            ks.append(acc + 1); prev = (p, acc); p += acc + 1
                    nm = name if rule is None or rule[0] == "oracle" else f"{name} s{s_len}" + (" KEYED" if keyed else "")
                    r[nm] = float(np.mean(ks))
        base = r["EAGLE-2"]
        ok = (K != 15) or abs(base - pub16) < 0.01
        say(f"  EAGLE-2-{K + 1} kept {base:.3f}" + (f" (S9c {pub16}: {'PASS' if ok else 'FAIL'})" if K == 15 else ""))
        for nm, v in sorted(r.items(), key=lambda kv: -kv[1]):
            if nm != "EAGLE-2":
                say(f"  {nm:28s} kept {v:7.3f}  x{v / base:.4f}")
        best = max((v, k) for k, v in r.items() if not k.startswith("ORACLE") and k != "EAGLE-2")
        res[s_] = {"kept": r, "best": best[1], "best_ratio": best[0] / base,
                   "oracle_ratio": max(v for k, v in r.items() if k.startswith("ORACLE")) / base}
    v = "PASS" if res["he"]["best_ratio"] > 1.02 else "KILL"
    say(f"\nLINE (best rule > EAGLE-2-{K + 1} x 1.02 on HE+): {v} (HE+ x{res['he']['best_ratio']:.4f} {res['he']['best']}, "
        f"oracle x{res['he']['oracle_ratio']:.4f}; MBPP x{res['mbpp']['best_ratio']:.4f}, oracle x{res['mbpp']['oracle_ratio']:.4f})")
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    open(a.out + ".txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
