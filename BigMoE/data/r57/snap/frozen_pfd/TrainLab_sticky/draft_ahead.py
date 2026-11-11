"""DRAFT-AHEAD (MTP Head, Wed 11-04; the orchestrator's item 2 on Kamen's 300 tok/s FAST priority; BREAK_THE_WALL 4.2;
DA-0's replay: a usable pre-drafted block on 34 % of code checks): pre-draft the next block during check i for a few guessed
outcomes, each on a cloned drafter cache, and adopt the one matching the check's real outcome (k accepted, next anchor
am[k]); otherwise draft serially as today. An adopted pre-draft is the serial draft: the same propose() on the same inputs
(the check's taps for rows 0..k, the same anchor token, a copy of the same drafter cache), so the drafts are the same.

The engine side (~/BigMoE/scripts/r57_screen.py, BigMoE's file; the PATCH notes at the end) needs propose() to also return
its candidates, unary logits and selector projection (propose_x below), for the 2nd-ranked guesses.
"""
import copy
import numpy as np
import mlx.core as mx


def clone_cache(dcache):
    """a drafter cache copy whose updates never reach the original: new array objects for keys / values (an in-place
    update on the copy, e.g. RotatingKVCache's 1-row path, rebinds only the copy's arrays)"""
    out = []
    for c in dcache:
        c2 = copy.copy(c)
        if getattr(c, "keys", None) is not None:
            c2.keys, c2.values = c.keys[...], c.values[...]
        out.append(c2)
    return out


def second(cand, unary, hp, sel, pred, j):
    """the selector's 2nd-ranked candidate at depth index j given predecessor id pred (the score of r57_screen.py's chain)"""
    sc = unary[:, j] + mx.sum(sel.predecessor_codebook(mx.array([pred]))[:, None] * hp[:, j, None]
                              * sel.successor_codebook(cand[:, j]), axis=-1)
    o = mx.argsort(-sc.astype(mx.float32), axis=-1)
    return int(cand[0, j, o[0, 1]].item())


def plan(path, cf, c, cand, unary, hp, sel, anchor, seg2=False):
    """the prefixes to pre-draft for a check that verifies the first c of the block's own picks path (cf per depth):
    'end'  k = c: every checked draft right; the next anchor is the 125B's token past the cut, guessed by the own pick there;
    'low1' / 'low2' k = j: the first miss at the j-th lowest-cf depth of the segment; the next anchor is the 125B's
           correction there, guessed by the selector's 2nd-ranked candidate;
    'end2' (seg2) k = c with the 2nd-ranked candidate past the cut.
    Returns [(tag, k, guess)]; no guess past a full block (a 17-row block would supply one)."""
    pre = []
    if c < len(path):
        pre.append(("end", c, int(path[c])))
        if seg2:
            pre.append(("end2", c, second(cand, unary, hp, sel, anchor if c == 0 else int(path[c - 1]), c)))
    if c >= 1:
        for tag, j in zip(("low1", "low2"), np.argsort(np.asarray(cf[:c]), kind="stable")[:2]):
            j = int(j)
            pre.append((tag, j, second(cand, unary, hp, sel, anchor if j == 0 else int(path[j - 1]), j)))
    return pre


def launch(propose_x, ctx_rows, pre, R, MASK, dcache):
    """build the pre-drafts (lazy) and submit them: ctx_rows(k) = the check's taps for rows 0..k as [1, k + 1, W] bf16.
    Returns [(tag, k, guess, path, cf, cache)]."""
    out = []
    for tag, k, g in pre:
        dc = clone_cache(dcache)
        r = propose_x(mx.array([[g] + [MASK] * (R - 1)]), ctx_rows(k), dc)
        out.append((tag, k, g, r[0], r[1], dc))
    if out:
        mx.async_eval(*[x for o in out for x in o[3:5]])
    return out


def resolve(handles, k, anchor, tags=None):
    """the pre-draft for the real outcome (k accepted, next anchor), or None. tags: only these kinds count"""
    for tag, kp, g, pth, cf, dc in handles:
        if kp == k and g == anchor and (tags is None or tag in tags):
            return tag, pth, cf, dc
    return None


PATCH = """r57_screen.py (BigMoE's file; DRAFT-AHEAD behind R57_DA=1, DFlash2 path only, no copy override in this form):
1. propose() -> propose_x(): also return cand, unary, hp (already computed; path and cf unchanged, so the drafts stay bitwise).
2. The draft step: if da_hit is not None (from the previous check): pth, cf, dcache = da_hit's path, cf, cache and skip the
   propose; else the serial propose_x as today. Keep cand / unary / hp of the block actually used.
3. After the cut (dr known) and the verify graph is built and tp = taps() exists: pre = plan(path, cf, len(dr), cand, unary,
   hp, draft.candidate_selector, tok); handles = launch(propose_x, lambda k: tp[:k + 1][None].astype(mx.bfloat16), pre,
   R_, MASK, dcache). Safe form: launch after mx.eval(am, tp) (overlaps the CPU-side commit only). Overlap form: launch
   before it under `with mx.stream(S2)` (S2 = mx.new_stream(mx.gpu)) so the pre-drafts start at layer 45's taps and run
   beside layers 46-48: needs BigMoE's check against the ext's gate / OOO scheduling.
4. After the commit: da_hit = resolve(handles, k, am[k]) (with R57_FAST relaxed rules: the committed k and token).
5. Report drf (the drafter's critical-path ms) with hits at ~0, and the hit rate.
"""
