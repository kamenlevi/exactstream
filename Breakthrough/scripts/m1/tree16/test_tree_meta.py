"""test_tree_meta.py (Breakthrough, TREE16 B1 unit test; spec section 2): for every path of every tree, the chain layout
rebuilt from meta equals the explicit chain (positions, ancestor lists, GDN windows, PLE taps, n-gram contexts, route-(b)
columns); the fork schedule replays every node from its parent's state; Leap's tree_gdn.fork_tables (L1) agrees.
Trees: a pure chain, a bushy tree, a deep narrow tree, a single node, and 2,000 random preorder trees (T 1..16).
Usage: python3 -I -B test_tree_meta.py   (numpy only; Leap's fork_tables imported read-only when present)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tree_meta import tree_meta, validate

try:
    sys.path.insert(1, os.path.expanduser("~/Leap/scripts/week98"))
    from tree_gdn import fork_tables                     # pure Python; imports mlx (no GPU work)
except Exception as e:                                   # pragma: no cover
    fork_tables = None
    print(f"(Leap's fork_tables not importable: {e!r}; the cross-check is skipped)")

rng = np.random.default_rng(20261115)


def random_preorder(T):
    """a random tree in DFS preorder: each new node attaches to some node on the current ancestor stack"""
    parents, stack = [-1], [0]
    for n in range(1, T):
        k = int(rng.integers(0, len(stack)))
        stack = stack[:len(stack) - k]                    # pop k levels (stay at least at the root)
        if not stack:
            stack = [0]
        parents.append(stack[-1])
        stack.append(n)
    return parents


SHAPES = {
    "chain16": [-1] + list(range(15)),
    "bushy": [-1, 0, 1, 2, 2, 1, 5, 0, 7, 7, 0, 10, 11, 0, 13, 13],
    "deep_narrow": [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 2],
    "single": [-1],
}


def check(parents, L, tag):
    T = len(parents)
    tokens = rng.integers(0, 248320, T)
    prev = rng.integers(0, 248320, 9)
    m = tree_meta(tokens, parents, L, prev)
    nchk = 0
    for pi, path in enumerate(m["paths"]):
        assert path[0] == 0 and all(parents[path[j]] == path[j - 1] for j in range(1, len(path))), (tag, path)
        ct = tokens[path]
        for j, n in enumerate(path):
            assert m["depth"][n] == j and m["pos"][n] == L + j, (tag, n)
            assert list(m["anc"][n, :j + 1]) == path[:j + 1] and np.all(m["anc"][n, j + 1:] == -1), (tag, n)
            assert list(m["tail_cols"][n, :j + 1]) == [L + x for x in path[:j + 1]] and np.all(m["tail_cols"][n, j + 1:] == -1)
            ref = lambda idx: path[idx] if idx >= 0 else idx       # chain row index -> node index or state code
            assert list(m["gdn_win"][n]) == [ref(j - 3), ref(j - 2), ref(j - 1)], (tag, n, m["gdn_win"][n])
            assert list(m["ple_taps"][n]) == [ref(j - 3), ref(j - 6), ref(j - 9)], (tag, n, m["ple_taps"][n])
            tok = lambda idx: int(ct[idx]) if idx >= 0 else int(prev[idx])
            assert list(m["ngram_prev"][n]) == [tok(j - 2), tok(j - 1)], (tag, n)
            assert m["chain_rows"][n] == (path[:j + 1] == list(range(j + 1))), (tag, n)
            nchk += 1
    for n in range(T):
        assert n in m["paths"][m["owner"][n]], (tag, n)
    leaves = [n for n in range(T) if not m["children"][n]]
    assert sorted(p[-1] for p in m["paths"]) == leaves, tag
    # the fork schedule: every node starts from its parent's state (the root from the committed state)
    slots, state, live = {}, (), set()
    for n, restore, save in m["fork_ops"]:
        if restore >= 0:
            assert restore in live, (tag, n, restore)
            state = slots[restore]
            if m["children"][parents[n]][-1] == n:
                live.discard(restore)
        want_before = tuple(int(x) for x in m["anc"][n, :m["depth"][n]])
        assert state == want_before, (tag, n, state, want_before)
        state = state + (n,)
        if save >= 0:
            assert save not in live, (tag, n, save)
            slots[save] = state
            live.add(save)
        assert save < m["n_slots"] and restore < m["n_slots"], tag
    if fork_tables is not None:
        r, s, ns = fork_tables([int(p) for p in parents])
        assert [x[1] for x in m["fork_ops"]] == r and [x[2] for x in m["fork_ops"]] == s and m["n_slots"] == ns, (tag, r, s, ns)
    return nchk


tot = 0
for nm, par in SHAPES.items():
    validate(par)
    tot += check(par, 1000, nm)
for L in (0, 1, 2, 3, 9, 1024, 3000):
    tot += check(SHAPES["bushy"], L, f"bushy L {L}")
for i in range(2000):
    T = int(rng.integers(1, 17))
    tot += check(random_preorder(T), int(rng.integers(0, 4000)), f"random {i}")
bad = 0
for par in ([-1, 0, 0, 1], [0], [-1, 1], [-1, 0, 1, 0, 2]):     # not preorder / bad root / forward parent / 4's parent left the stack
    try:
        validate(par)
    except ValueError:
        bad += 1
assert bad == 4, bad
print(f"tree_meta unit test PASS: {tot} node-in-path checks over {len(SHAPES) + 7 + 2000} trees; fork schedule replays; "
      f"Leap's fork_tables {'agrees' if fork_tables is not None else 'skipped'}; 4 / 4 malformed trees refused")
