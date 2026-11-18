"""TREE16 M4 (BigMoE; spec ~/Leap/data/week98/tree16_spec.md): the loop's exact tree check, one function per phase,
shared by the t16b screen (R57_TREE16=1) and the slice loop test (scripts/tree16_m4loop.py). Sun 15 Nov 2026 17:0x.

Per check, in the chain loop's own order (the MTP path):
  1. tr = T16D.draft(model, head, tok, n)              t30b's best-first tree, root + n pops (M1, tree16_draft)
  2. snap = CS.snapshot_untrimmable_cache_lazy(cache)  the loop's own snapshot
  3. lg, hw = verify(model, cache, tr, a, prev)        inside ref2.verify_scope: B1's tree_meta + Leap's tree_verify;
                                                       lg [T, V] node logits, hw [T, Wd] node widened streams
  4. path, bonus, k, new = accept(tr, am)              the walk; new = the path's tokens + the bonus (the loop cuts EOS)
  5. kind = commit(model, cache, snap.states, path, tr, a, text, ref2)   tree16_commit (M2): "chain" | "tree"
  6. pend = pairs(hw, tr, path, bonus, len(new))       MTPHead.append's input at the next check (the accepted pairs)
fits(a, T): the tree path only while the QSA indexer still selects every block (MTPLX picks "update_only", the dense
causal mask, while last_nb <= block_topk = 2048 / 4 = 512, i.e. a + T <= 2048; kept 4 tokens inside). Past it the check
runs as a chain (B3's per-node selection, Breakthrough Mon-Tue, lifts this).
Exactness rests on M3 (T1 every node == its path's chain row; T2 commit + probe == the chain's, bitwise on both slices)
and the slice loop test (tree16_m4loop.py: a multi-check decode with teacher-forced trees == greedy AR, token for token).
"""
import hashlib
import os
import subprocess
import sys

import numpy as np

_LP = [os.path.expanduser(p_) for p_ in ("/Users/kamen/BigMoE/data/r57/snap/frozen_s98n/Leap_scripts/week98", "/Users/kamen/BigMoE/data/r57/snap/frozen_s98n/Leap_scripts")]
_B = os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98n/Breakthrough_tree16")   # B1 tree_meta + B2 ple_tree
_SAVED = list(sys.path)                                  # tree16 inserts Leap's dirs at the FRONT on import; the loop's own
sys.path[:0] = _LP                                       # order is restored after (Leap's dirs go to the END, where the
import tree16 as TV                                      # tree pieces' lazy imports, leapk / gdnbody, still resolve)
sys.path[:] = _SAVED + [p_ for p_ in _LP + [_B] if p_ not in _SAVED]
import tree16_commit as TC                               # M2 (walk, tree_commit, mtp_pairs)
from tree_meta import tree_meta

BUDGET = 2048                                            # qwen4_exp indexer_budget (models/bare/config.json)
MARGIN = 4
PREV = 9                                                 # committed tokens handed to tree_meta (it needs >= 2)
_ST = {"installed": False, "ple": False, "qsa": False}


def install(ref2, ple=True, qsa=False):
    """after the loop's first ref2.apply(): Leap's wrappers over the installed hooks; B2's ple_tree for the PLE layer;
    with qsa, B3's qsa_tree in Leap's QSA_TREE slot (d6b5884, Sun 17:4x): per-node QSA selection above ~2K, no fallback"""
    if _ST["installed"]:
        return
    TV.install(ref2)
    if ple:
        from ple_tree import ple_tree
        TV.PLE_TREE[0] = ple_tree
        _ST["ple"] = True
    if qsa:
        assert hasattr(TV, "QSA_TREE"), "tree16 has no QSA_TREE slot (Leap d6b5884 or later)"
        import qsa_tree
        TV.QSA_TREE[0] = qsa_tree
        _ST["qsa"] = True
    _ST["installed"] = True


def fits(a, T):
    return _ST["qsa"] or a + T <= BUDGET - MARGIN


def qsa_offset(cache):
    for c in cache:
        if callable(getattr(c, "is_trimmable", None)) and c.is_trimmable():
            return int(c.offset)
    raise AssertionError("no QSA entry in the cache")


def verify(model, cache, tr, a, prev):
    """the caller holds ref2.verify_scope(model) (attention phase, target_verify, exact_verify, capture scope)"""
    toks = [int(x) for x in tr["tokens"]]
    L = qsa_offset(cache)
    assert L == a, ("cache offset vs the loop's position", L, a)
    assert fits(a, len(toks)), ("past the dense indexer budget", a, len(toks))
    meta = tree_meta(toks, [int(p) for p in tr["parents"]], a, [int(x) for x in prev][-PREV:])
    lg, hw = TV.tree_verify(model, cache, toks, meta)
    return lg, hw


def accept(tr, am):
    path, bonus = TC.walk(tr["tokens"], tr["parents"], am)
    new = [int(tr["tokens"][n]) for n in path[1:]] + [int(bonus)]
    return path, int(bonus), len(path) - 1, new


def accept_cc(tr, lg, am, delta, cc_max=0):
    """CC(1) on the tree walk (BigMoE Wed 18 Nov; the rule sent to EvalSuite + NewIdeas Tue 17 Nov 10:5x). From the root, at
    node u (the 125B's logits z_u = lg[u]):
      (1) a child whose token is argmax z_u is accepted (exact, as TC.walk);
      (2) else, with delta > 0, the child c with the largest z_u[c] (ties: the earlier node, the head's preferred sibling)
          is accepted when z_u[c] >= max z_u - delta: a close call;
      (3) else the walk stops and the check emits argmax z_u (the bonus).
    Any number of close calls per check, as relax_delta.relaxed_k (cc_max > 0 caps them). delta 0 = TC.walk's result.
    lg: [T, V] or [1, T, V] node logits; am: the per-node argmax. Returns path, bonus, k, new, close calls, sibling accepts
    (accepted nodes that are not their parent's first child, i.e. not the head's top-1 there)."""
    import mlx.core as mx
    tok = [int(x) for x in tr["tokens"]]; par = [int(p) for p in tr["parents"]]; T = len(tok)
    amv = [int(x) for x in (am.tolist() if hasattr(am, "tolist") else am)]
    kids = {}
    for n in range(1, T):
        kids.setdefault(par[n], []).append(n)
    dl = None
    if delta > 0 and T > 1:
        L = (lg[0] if lg.ndim == 3 else lg)[:T].astype(mx.float32)
        pi, ti = mx.array(par[1:], dtype=mx.int32), mx.array(tok[1:], dtype=mx.int32)
        rows = L[pi]
        z = mx.take_along_axis(rows, ti[:, None], axis=1)[:, 0] - rows.max(axis=-1)
        dl = [None] + z.tolist()
    path, u, ncc = [0], 0, 0
    while True:
        ch = kids.get(u, [])
        nxt = next((c for c in ch if tok[c] == amv[u]), None)
        if nxt is None and dl is not None and ch and (cc_max <= 0 or ncc < cc_max):
            c = max(ch, key=lambda c_: (dl[c_], -c_))
            if dl[c] >= -delta:
                nxt, ncc = c, ncc + 1
        if nxt is None:
            break
        path.append(nxt)
        u = nxt
    bonus = amv[u]
    nsib = sum(1 for n in path[1:] if kids[par[n]][0] != n)
    return path, bonus, len(path) - 1, [tok[n] for n in path[1:]] + [bonus], ncc, nsib


def tree_block(S, nodes, k):
    """runs.json "tree" (EvalSuite's engagement term tree:sib_accepted@run)"""
    return dict(nodes=nodes, k=k, checks=S["checks"], sib_accepted=S.get("sib", 0), close_calls=S.get("cc", 0))


def commit(model, cache, states, path, tr, a, text, ref2):
    return TC.tree_commit(model, cache, states, path, int(tr["T"]), a, text=text, ref2=ref2)


def pairs(hw, tr, path, bonus, n):
    h, nxt = TC.mtp_pairs(hw, tr["tokens"], path, bonus)
    return h[:, :n], nxt[:n]


def main_k(path):
    """how many accepted drafts lie on the main chain (nodes 1..D-1)"""
    km = 0
    while km + 1 < len(path) and int(path[km + 1]) == km + 1:
        km += 1
    return km


def new_stats():
    return dict(checks=0, fallback=0, T=0, D=0, k=0, k_main=0, side=0, main_end=0, chain=0, tree=0,
                draft_ms=[], commit_ms=[], kd=[], sib=0, cc=0)


def record(S, tr, path, kind, draft_ms, commit_ms):
    T, D, k = int(tr["T"]), int(tr["main_chain_len"]), len(path) - 1
    km = main_k(path)
    S["checks"] += 1; S["T"] += T; S["D"] += D; S["k"] += k; S["k_main"] += km
    S["side"] += int(k > km)                             # the walk left the main chain (a side branch kept more)
    S["main_end"] += int(k == km == D - 1)               # the walk used the whole main chain (a chain might go on)
    S[kind] += 1
    S["draft_ms"].append(round(draft_ms, 3)); S["commit_ms"].append(round(commit_ms, 3)); S["kd"].append([k, km, D])


def summary(S):
    n = max(S["checks"], 1)
    out = {k_: S[k_] for k_ in ("checks", "fallback", "side", "main_end", "chain", "tree")}
    out.update(T_mean=round(S["T"] / n, 3), D_mean=round(S["D"] / n, 3), kept_mean=round(1 + S["k"] / n, 3),
               kept_main_mean=round(1 + S["k_main"] / n, 3),
               draft_ms=float(np.median(S["draft_ms"])) if S["draft_ms"] else None,
               commit_ms=float(np.median(S["commit_ms"])) if S["commit_ms"] else None, kd=S["kd"],
               sib_accepted=S.get("sib", 0), close_calls=S.get("cc", 0))
    return out


def provenance():
    """sha256 (12) of every TREE16 piece this process imported + the owners' HEADs"""
    def sh(p):
        try:
            return hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]
        except OSError:
            return None

    def head(d):
        try:
            return subprocess.run(["git", "-C", os.path.expanduser(d), "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return None
    import tree16_draft as T16D
    import tree_meta as TM
    mods = {"tree16": TV.__file__, "tree_att": TV.TA.__file__, "tree_gdn": TV.TG.__file__, "tree_meta": TM.__file__,
            "tree16_draft": T16D.__file__, "tree16_commit": TC.__file__, "tree16_loop": __file__}
    if _ST["ple"]:
        import ple_tree as PT
        mods["ple_tree"] = PT.__file__
    if _ST["qsa"]:
        mods["qsa_tree"] = TV.QSA_TREE[0].__file__
    return dict(files={k_: [os.path.abspath(v_), sh(v_)] for k_, v_ in mods.items()},
                heads={"Leap": head("~/Leap"), "Breakthrough": head("~/Breakthrough"), "BigMoE": head("~/BigMoE")})
