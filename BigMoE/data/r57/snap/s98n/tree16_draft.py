"""TREE16 M1 (BigMoE; spec ~/Leap/data/week98/tree16_spec.md, ce2047a): t30b drafts an EAGLE-2 tree in the loop's MTP path.
Sun 15 Nov 2026 14:5x.

The policy is S9c's priced one (TrainLab fa3ce76, on b198498's dump): the root (the last sampled token, already a committed
pair in the MTP cache) plus the first N = 15 best-first pops by path probability, top-8 children per node. That is N - 1 = 14
sequential head forwards, the same count as draft_cut's 15-draft chain.

Route A (the live MTPLX head, QSA included):
- Each popped node runs model.mtp_forward(h_parent, [[token]]) on an MTP cache holding the committed pairs plus exactly
  its ancestors' own entries in depth order, the same cache content a chain along that path would hold. The positions
  come from the cache offset, as the chain's do.
- Before each pop, the cache is trimmed to the longest prefix the new node's path shares with the cached path. The
  missing ancestors' stored (raw indexer key, K, V) rows are written back with the cache's own primitives (write_raw,
  then kv.update_and_fetch); pooled blocks past the trim are rebuilt from raw by the next forward (QSACache.trim drops
  them).
- A main-chain expansion appends with no re-insertion. After the draft the cache is trimmed back to the committed
  length, as MTPHead.draft_cut does.

Output, in the spec's form: tokens / parents int32 [T] in DFS preorder with children best-first (parents[n] < n; nodes
0..D-1 = the main chain), plus per node: depth, path_p, draft p, the pop order. The main chain equals the greedy chain
(draft_cut's tokens with t = 0), which is the M1 self-check against the model.
usage: python tree16_draft.py --selftest   (no model: the cache juggling on a real QSACache vs sequential writes; the
       preorder and best-first logic on a toy head)"""
import heapq
import numpy as np


def _topk_np(logits_row, k):
    """f32 softmax top-k of one row (an mx array [V]) -> ids [k] int32, p [k] float64 (descending)"""
    import mlx.core as mx
    l = logits_row.astype(mx.float32)
    lse = mx.logsumexp(l)
    idx = mx.argpartition(-l, kth=k - 1)[:k]
    val = mx.take(l, idx)
    mx.eval(idx, val, lse)
    idx, val = np.array(idx).astype(np.int32), np.array(val, np.float64)
    o = np.lexsort((idx, -val))                          # ties (bf16 logits) -> the lower id first, as mx.argmax picks
    return idx[o], np.exp(val[o] - float(lse))


class PathCache:
    """the MTP cache's speculative tail: committed length Lc, then the entries of one root-to-node path"""

    def __init__(self, qc):
        self.qc = qc                                     # QSACache (the MTP head's single layer)
        self.Lc = int(qc.offset)
        self.path = []                                   # node ids whose entries sit at Lc, Lc + 1, ...
        self.rows = {}                                   # node id -> (raw [1,1,Di], k [B,H,1,D], v [B,H,1,Dv])

    def to(self, anc):
        """make the cached tail exactly the entries of `anc` (node ids in depth order)"""
        m = 0
        while m < len(self.path) and m < len(anc) and self.path[m] == anc[m]:
            m += 1
        drop = len(self.path) - m
        if drop:
            self.qc.trim(drop)
        for a in anc[m:]:
            raw, k, v = self.rows[a]
            if raw is not None:
                self.qc.write_raw(raw)
            self.qc.kv.update_and_fetch(k, v)
        self.path = list(anc)
        assert int(self.qc.offset) == self.Lc + len(anc), (int(self.qc.offset), self.Lc, len(anc))

    def capture(self, node):
        """after the node's own forward: its row (the last one) for its descendants"""
        import mlx.core as mx
        pos = int(self.qc.offset) - 1
        raw = None if self.qc.raw_keys is None else self.qc.raw_keys[:, pos:pos + 1, :]
        k, v = self.qc.kv.keys[..., pos:pos + 1, :], self.qc.kv.values[..., pos:pos + 1, :]
        mx.eval(*(x for x in (raw, k, v) if x is not None))
        self.rows[node] = (raw, k, v)
        self.path.append(node)

    def release(self):
        if self.path:
            self.qc.trim(len(self.path))
        self.path = []
        assert int(self.qc.offset) == self.Lc


def draft(model, mtp, root_tok, n=15, k=8, fwd=None):
    """mtp: MTPHead-like (cache, logits [1,1,V] for the root's children, h the root's hidden). Returns a dict with the
    spec's tokens / parents (DFS preorder) and per-node diagnostics. fwd(h, tok) -> (logits [1,1,V], h) overrides the
    head forward (the selftest's toy head)."""
    import mlx.core as mx
    if fwd is None:
        def fwd(h, tok):
            return model.mtp_forward(h, mx.array([[int(tok)]]), mtp_cache=mtp.cache, return_hidden=True)
    pc = PathCache(mtp.cache[0])
    ids, ps = _topk_np(mtp.logits[0, -1], k)
    heap, cnt = [], 0
    for t_, p_ in zip(ids, ps):                         # candidates: (-path_p, count, parent (-1 = root), token, depth, p)
        heapq.heappush(heap, (-float(p_), cnt, -1, int(t_), 1, float(p_))); cnt += 1
    H = {-1: mtp.h}
    par, tok, dep, pth, pdr, anc = [], [], [], [], [], {-1: []}
    try:
        for t in range(n):
            if not heap:
                break
            mp, _, parent, tk, d, p_ = heapq.heappop(heap)
            par.append(parent); tok.append(tk); dep.append(d); pth.append(-mp); pdr.append(p_)
            if t == n - 1:                               # the last pop's children are never drafted: no forward (n pops
                break                                    # = n - 1 forwards, as draft_cut's n drafts)
            pc.to(anc[parent])
            lg, h = fwd(H[parent], tk)
            pc.capture(t)
            H[t] = h; anc[t] = anc[parent] + [t]
            if True:
                ci, cp = _topk_np(lg[0, -1], k)
                for t_, q_ in zip(ci, cp):
                    heapq.heappush(heap, (mp * float(q_), cnt, t, int(t_), d + 1, float(q_))); cnt += 1
    finally:
        pc.release()
    return preorder(root_tok, par, tok, dep, pth, pdr)


def preorder(root_tok, par, tok, dep, pth, pdr):
    """pop-order nodes (parent -1 = root) -> the spec's DFS preorder, children best-first by draft p"""
    kids = {}
    for i, p in enumerate(par):
        kids.setdefault(p, []).append(i)
    for p in kids:
        kids[p].sort(key=lambda i: (-pdr[i], i))
    order, st = [], list(reversed(kids.get(-1, [])))
    while st:
        i = st.pop()
        order.append(i)
        st.extend(reversed(kids.get(i, [])))
    new = {i: j + 1 for j, i in enumerate(order)}        # node 0 = the root
    T = 1 + len(order)
    tokens = np.array([root_tok] + [tok[i] for i in order], np.int32)
    parents = np.array([-1] + [0 if par[i] < 0 else new[par[i]] for i in order], np.int32)
    depth = np.array([0] + [dep[i] for i in order], np.int32)
    out = dict(tokens=tokens, parents=parents, depth=depth, T=T,
               path_p=np.array([1.0] + [pth[i] for i in order]), p=np.array([1.0] + [pdr[i] for i in order]),
               pop=np.array([-1] + order, np.int32))
    D = 1
    while D < T and parents[D] == D - 1:
        D += 1
    out["main_chain_len"] = D
    assert all(parents[i] < i for i in range(1, T)), parents
    return out


def selftest():
    import mlx.core as mx
    from mtplx.models.qwen4_exp import QSACache
    rng = np.random.default_rng(3)
    H, D, Di = 2, 8, 4
    row = {i: (mx.array(rng.normal(size=(1, 1, Di)).astype(np.float32)), mx.array(rng.normal(size=(1, H, 1, D)).astype(np.float32)),
               mx.array(rng.normal(size=(1, H, 1, D)).astype(np.float32))) for i in range(-6, 12)}

    def put(qc, i):
        raw, k, v = row[i]
        qc.write_raw(raw); qc.kv.update_and_fetch(k, v)

    def fresh(seq):
        qc = QSACache(4)
        for i in seq:
            put(qc, i)
        return qc

    def same(a, b, n):
        ka, kb = np.array(a.kv.keys[..., :n, :]), np.array(b.kv.keys[..., :n, :])
        va, vb = np.array(a.kv.values[..., :n, :]), np.array(b.kv.values[..., :n, :])
        ra, rb = np.array(a.raw_keys[:, :n, :]), np.array(b.raw_keys[:, :n, :])
        return a.offset == b.offset == n and (ka == kb).all() and (va == vb).all() and (ra == rb).all()
    committed = list(range(-6, 0))                       # 6 committed pairs
    qc = fresh(committed)
    pc = PathCache(qc)
    nodes = {0: [], 1: [0], 2: [0, 1], 3: [], 4: [3], 5: [0, 1, 2], 6: [3, 4]}    # parents' paths (anc of the node's parent)
    for nd in range(7):                                  # pop order jumping between branches
        pc.to(nodes[nd])
        put(qc, nd); pc.capture(nd)                      # the "forward" writes the node's own row
        ref = fresh(committed + nodes[nd] + [nd])
        assert same(qc, ref, len(committed) + len(nodes[nd]) + 1), ("cache != the path's sequential writes", nd)
    pc.release()
    assert qc.offset == len(committed)

    # the toy head: logits from a hash of the path; best-first + preorder
    V = 50

    class Toy:
        def __init__(self):
            self.cache = [fresh(committed)]
            self.h = mx.array([[[-1.0]]])                # the toy hidden carries the node id (-1 = the root)
            self.logits = self._lg(())

        @staticmethod
        def _lg(path):
            g = np.random.default_rng(abs(hash(path)) % (2 ** 32))
            return mx.array(g.normal(size=(1, 1, V)).astype(np.float32) * 3)

    toy = Toy()
    paths = {-1: ()}

    def fwd(h, tk):
        key = int(round(float(np.array(h).reshape(-1)[0])))
        p_ = paths[key] + (int(tk),)
        nid = len(paths) - 1
        paths[nid] = p_
        put(toy.cache[0], 100 + nid if 100 + nid in row else -1)      # any row: the cache bookkeeping is under test
        return Toy._lg(p_), mx.array([[[float(nid)]]])
    row.update({100 + i: row[i % 6] for i in range(20)})
    out = draft(None, toy, 7, n=15, k=4, fwd=fwd)
    assert out["T"] == 16 and out["tokens"][0] == 7 and out["parents"][0] == -1
    assert toy.cache[0].offset == len(committed), "the draft left the cache longer than the committed history"
    # main chain = greedy: each main-chain node is the argmax child of its parent
    D = out["main_chain_len"]
    for i in range(1, D):
        pth = tuple(int(x) for x in out["tokens"][1:i])
        assert int(out["tokens"][i]) == int(np.argmax(np.array(Toy._lg(pth))[0, 0])), ("main chain not greedy", i)
    # best-first: path_p of the pops is non-increasing; every node's path_p = parent's x its p
    pp = out["path_p"][1:][np.argsort(out["pop"][1:])]
    assert all(pp[i] >= pp[i + 1] - 1e-15 for i in range(len(pp) - 1))
    for i in range(1, out["T"]):
        par_ = out["parents"][i]
        assert abs(out["path_p"][i] - (out["path_p"][par_] if par_ > 0 else 1.0) * out["p"][i]) < 1e-12
    print(f"SELFTEST OK: PathCache == sequential path writes on a real QSACache (7 pops, 3 branches); toy draft T 16, "
          f"main chain {D} greedy, best-first, preorder parents[n] < n, cache back to the committed length")


if __name__ == "__main__":
    selftest()
