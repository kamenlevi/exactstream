"""tree_meta.py (Breakthrough, TREE16 piece B1; spec ~/Leap/data/week98/tree16_spec.md 8770fe2, draft until Mon 10:00).
Numpy, CPU. meta = tree_meta(tokens, parents, L, prev_tokens) for an EAGLE-2-style tree in DFS preorder (children in
draft-probability order, so node 0 is the root and nodes 0..D-1 are the main chain).

Model constants (the 125B, ~/BigMoE/models/bare/config.json; MTPLX qwen4_exp):
  GDN conv kernel 4            -> gdn_win holds the 3 rows before a node on its path (oldest first);
  PLE conv kernel 4, dilation ngram_size 3 -> ple_taps holds the taps 3 / 6 / 9 back (nearest first, as the spec lists);
  n-gram context ngram_size - 1 = 2        -> ngram_prev holds the 2 previous path tokens (oldest first).
Negative entries name cache state slots:
  gdn_win: -k = the row k before the root = GDN conv-state slot 3 - k (slot 2 = the newest committed row);
  ple_taps: -k = the row k before the root = PLE conv-state slot 9 - k (slot 8 = the newest);
  ngram_prev never needs a code: it reads prev_tokens (the last committed tokens, newest last).
The n-gram EOS rule (MTPLX's _shift_ignore_eos: no context across an EOS) is applied where the ids are hashed (B2), on
each node's raw 3-token history [ngram_prev, token]; ngram_prev itself is raw.

fork_ops (the GDN depth-first schedule), one entry per node in node order: (node, restore_slot, save_slot).
  - restore_slot >= 0: before the node's recurrence step, load its parent's state from that slot (the node is a later
    child; its parent's first child came right after the parent and continued the live state). The parent's LAST child
    frees the slot after loading it.
  - save_slot >= 0: after the node's step, store its state there (the node has >= 2 children).
  n_slots = the distinct slots used (= the most saved states live at once, as freed slots are reused first); at least 1,
  as L1's scratch buffer wants (Leap's fork_tables returns max(used, 1)).
tail_cols (attention route (b)): tail_cols[n, j] = L + anc[n, j] for j <= depth[n] (chain column L + j of node n reads
tree column L + anc[n, j]); -1 otherwise."""
import numpy as np

GDN_K = 4
PLE_K, PLE_DIL = 4, 3
NGRAM_CTX = 2
TMAX = 16


def validate(parents):
    """parents in DFS preorder: parents[0] = -1, and each node's parent is on the ancestor stack of the node before it"""
    P = np.asarray(parents)
    T = len(P)
    if T < 1 or P[0] != -1:
        raise ValueError("node 0 must be the root (parents[0] = -1)")
    stack = [0]
    for n in range(1, T):
        p = int(P[n])
        if not (0 <= p < n):
            raise ValueError(f"parents[{n}] = {p}: not an earlier node")
        while stack and stack[-1] != p:
            stack.pop()
        if not stack:
            raise ValueError(f"node {n}: parent {p} is not on node {n - 1}'s ancestor stack (not DFS preorder)")
        stack.append(n)


def tree_meta(tokens, parents, L, prev_tokens, tmax=TMAX):
    tokens = np.asarray(tokens, np.int64)
    parents = np.asarray(parents, np.int64)
    prev_tokens = np.asarray(prev_tokens, np.int64)
    T = len(tokens)
    if len(parents) != T:
        raise ValueError("tokens and parents differ in length")
    if T > tmax:
        raise ValueError(f"{T} nodes > {tmax}")
    if len(prev_tokens) < NGRAM_CTX:
        raise ValueError(f"prev_tokens needs the last {NGRAM_CTX} committed tokens")
    validate(parents)
    children = [[] for _ in range(T)]
    for n in range(1, T):
        children[int(parents[n])].append(n)
    depth = np.zeros(T, np.int64)
    anc = np.full((T, tmax), -1, np.int64)
    anc[0, 0] = 0
    for n in range(1, T):
        p = int(parents[n])
        depth[n] = depth[p] + 1
        anc[n, :depth[n]] = anc[p, :depth[n]]
        anc[n, depth[n]] = n
    pos = L + depth
    chain_rows = np.array([bool(np.all(anc[n, :depth[n] + 1] == np.arange(depth[n] + 1))) for n in range(T)])

    def back(n, k):
        """the row k back from node n along its path: a node index, or -(k - depth) before the root"""
        d = int(depth[n])
        return int(anc[n, d - k]) if k <= d else -(k - d)

    gdn_win = np.array([[back(n, k) for k in range(GDN_K - 1, 0, -1)] for n in range(T)], np.int64)       # 3, 2, 1 back
    ple_taps = np.array([[back(n, k) for k in (PLE_DIL, 2 * PLE_DIL, 3 * PLE_DIL)] for n in range(T)], np.int64)
    ngram_prev = np.zeros((T, NGRAM_CTX), np.int64)
    for n in range(T):
        for i, k in enumerate(range(NGRAM_CTX, 0, -1)):                                               # 2, 1 back
            b = back(n, k)
            ngram_prev[n, i] = tokens[b] if b >= 0 else prev_tokens[b]                                 # prev_tokens[-j]
    fork_ops, slot_of, free, used = [], {}, [], 0           # the same numbering as Leap's tree_gdn.fork_tables (L1)
    for n in range(T):
        restore = -1
        if n > 0:
            p = int(parents[n])
            if p != n - 1:                                   # a later child: its parent's state comes back from a slot
                restore = slot_of[p]
                if children[p][-1] == n:                     # the parent's last child frees the slot after loading it
                    free.append(slot_of.pop(p))
        save = -1
        if len(children[n]) > 1:
            if free:
                save = free.pop()
            else:
                save = used
                used += 1
            slot_of[n] = save
        fork_ops.append((n, restore, save))
    n_slots = max(used, 1)
    leaves = [n for n in range(T) if not children[n]]
    paths = [[int(x) for x in anc[n, :depth[n] + 1]] for n in leaves]
    owner = np.array([next(i for i, pth in enumerate(paths) if n in pth) for n in range(T)], np.int64)
    tail_cols = np.where(anc >= 0, L + anc, -1)
    return dict(depth=depth, pos=pos, anc=anc, chain_rows=chain_rows, gdn_win=gdn_win, ple_taps=ple_taps,
                ngram_prev=ngram_prev, fork_ops=fork_ops, n_slots=int(n_slots), paths=paths, owner=owner,
                tail_cols=tail_cols, children=children)
