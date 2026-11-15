#!/usr/bin/env python3
"""TREE (Sun 15 Nov 2026 11:0x; the orchestrator, for MTP Head's EAGLE-2 pricing on the 125B. S9b's open unknown is the
head's confidence OFF the true path: the children of a wrong sibling).

At every anchor of the bank's HE+ 24 / MBPP 100 windows (TOPK's depth-0 cells; walk_start marks the chain-32 walk's check
starts), t30b drafts a REAL best-first tree by path probability, the N highest path-probability nodes with top-8 children
per node:
  - Pop the best candidate. Run the head on it: fed its token with its parent's carried state at position s + d + 1. It
    attends to the true pairs u <= s plus its own ancestors' chain entries (dd_d22_ttt.chain's depth-j step, restricted to
    one path).
  - Push its top-8 children. 32 nodes per anchor.
  - Path probability falls along every path, so the first N popped nodes are the best N-node tree for every N <= 32.

Per node: fi, an (anchor s), node (pop order 0..31), parent (-1 = the root), depth d (0 = the first draft, position
s + d + 2), token, path_p, y (the 125B's token at its position, -1 past the window), on_true (token == y under a true-path
parent), top_ids / top_p (its own top-8 + probabilities), p_ynext / ynext_rank (only meaningful on true-path nodes).
Per anchor: a_fi, a_an, a_walk_start, root_top_ids / root_top_p (the depth-0 distribution).

Exactness check: true-path nodes are the teacher-forced chain, so their distributions must equal TOPK's cells (fi,
row = s + d + 2, band d + 1); the roots must equal TOPK's band 0. Both are reported.
usage: dd_tree.py --set heplus24|mbppbank [--run t30b] [--nodes 32] [--k 8] -> data/deepdraft/ceiling/tree_<run>_<set>.npz
       dd_tree.py --selftest   (tree_attn against dd_d22_ttt.ttt_attn on random data; best-first against brute force)"""
import argparse, heapq, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.expanduser("~/BigMoE/data/deepdraft/ceiling")
NEG = -1e9


def tree_attn(mx, scale, q, K1, V1, AK, AV, amask, anc):
    """dd_d22_ttt.ttt_attn for tree rows, the same ops in the same order: row i attends to the true pairs u <= anc[i]
    (K1 / V1 [1,KV,S1,hd]) and to its path's chain entries AK / AV [1,KV,n,J,hd] where amask [n,J] (its ancestors in depth
    order, then itself; padding masked to exp(NEG - m) = 0). q [1,H,n,hd] -> [1,H,n,hd]"""
    _, Hh, n, hd = q.shape
    KV, S1 = K1.shape[1], K1.shape[2]
    r = Hh // KV
    qg = q.reshape(1, KV, r, n, hd)
    st = (qg @ K1[:, :, None].swapaxes(-1, -2)) * scale                         # [1,KV,r,n,S1]
    st = mx.where(mx.arange(S1)[None, :] <= anc[:, None], st, NEG)
    J = AK.shape[3]
    sc = mx.stack([(qg * AK[:, :, None, :, jj]).sum(-1) for jj in range(J)], -1) * scale   # [1,KV,r,n,J]
    sc = mx.where(amask, sc, NEG)
    m = mx.maximum(st.max(-1, keepdims=True), sc.max(-1, keepdims=True))
    et, ec = mx.exp(st - m), mx.exp(sc - m)
    o = et @ V1[:, :, None]
    for jj in range(J):
        o = o + ec[..., jj:jj + 1] * AV[:, :, None, :, jj]
    o = o / (et.sum(-1, keepdims=True) + ec.sum(-1, keepdims=True))
    return o.reshape(1, Hh, n, hd)


class Forest:
    """best-first trees, one per row (anchor): candidates (-path_p, count, parent, token, depth); node t = the t-th pop"""

    def __init__(self, root_ti, root_tp):
        self.B = len(root_ti)
        self.cnt = 0
        self.heaps = [[] for _ in range(self.B)]
        self.par, self.dep, self.pp, self.anc = [[] for _ in range(self.B)], [[] for _ in range(self.B)], \
            [[] for _ in range(self.B)], [[] for _ in range(self.B)]
        for b in range(self.B):
            for c in range(len(root_ti[b])):
                self._push(b, float(root_tp[b][c]), -1, int(root_ti[b][c]), 0)

    def _push(self, b, p, parent, tok, d):
        heapq.heappush(self.heaps[b], (-p, self.cnt, parent, tok, d)); self.cnt += 1

    def pop(self):
        """one node per row -> parent [B], tok [B], depth [B], path_p [B], entries per row (ancestor node ids in depth
        order, then this node's id)"""
        t = len(self.par[0])
        par, tok, dep, pp, ent = [], [], [], [], []
        for b in range(self.B):
            mp, _, parent, tk, d = heapq.heappop(self.heaps[b])
            anc = (self.anc[b][parent] + [parent]) if parent >= 0 else []
            self.par[b].append(parent); self.dep[b].append(d); self.pp[b].append(-mp); self.anc[b].append(anc)
            par.append(parent); tok.append(tk); dep.append(d); pp.append(-mp); ent.append(anc + [t])
        return np.array(par), np.array(tok, np.int32), np.array(dep), np.array(pp, np.float64), ent

    def push(self, t, ti, tp):
        for b in range(self.B):
            for c in range(ti.shape[1]):
                self._push(b, self.pp[b][t] * float(tp[b, c]), t, int(ti[b, c]), self.dep[b][t] + 1)


def selftest():
    os.environ["MLX_ENABLE_TF32"] = "0"                   # true fp32: a one-row matmul vs a 5-row one differ ~1e-3 in TF32
    import mlx.core as mx
    import dd_d22_ttt as T
    rng = np.random.default_rng(1)
    KV, r, hd, S1, n, a0 = 2, 3, 16, 40, 5, 20
    q = mx.array(rng.normal(size=(1, KV * r, n, hd)).astype(np.float32))
    K1 = mx.array(rng.normal(size=(1, KV, S1, hd)).astype(np.float32)); V1 = mx.array(rng.normal(size=(1, KV, S1, hd)).astype(np.float32))
    J = 4
    ck = [mx.array(rng.normal(size=(1, KV, n, hd)).astype(np.float32)) for _ in range(J)]
    cv = [mx.array(rng.normal(size=(1, KV, n, hd)).astype(np.float32)) for _ in range(J)]
    ref = T.ttt_attn(0.25, q, K1, V1, ck, cv, a0)
    AK, AV = mx.stack(ck, 3), mx.stack(cv, 3)
    out = tree_attn(mx, 0.25, q, K1, V1, AK, AV, mx.ones((n, J), mx.bool_), a0 + mx.arange(n))
    assert (np.array(out) == np.array(ref)).all(), "tree_attn != ttt_attn (full entries)"
    # padded rows: row i keeps its first J_i entries; it must equal ttt_attn run on that row alone with J_i entries
    Ji = [1, 2, 4, 3, 1]
    am = mx.array(np.array([[jj < Ji[i] for jj in range(J)] for i in range(n)]))
    out = np.array(tree_attn(mx, 0.25, q, K1, V1, AK, AV, am, a0 + mx.arange(n)))
    for i in range(n):
        ri = np.array(T.ttt_attn(0.25, q[:, :, i:i + 1], K1, V1, [c[:, :, i:i + 1] for c in ck[:Ji[i]]],
                                 [c[:, :, i:i + 1] for c in cv[:Ji[i]]], a0 + i))[:, :, 0]
        assert np.allclose(out[:, :, i], ri, rtol=1e-6, atol=1e-6), ("padded row", i, np.abs(out[:, :, i] - ri).max())
    # the gather the run relies on: store [1,KV,M,hd][:, :, idx [B,J]] -> [1,KV,B,J,hd] = numpy's
    store = rng.normal(size=(1, KV, 12, hd)).astype(np.float32)
    ai = np.array([[0, 5], [4, 7], [11, 2]], np.int32)
    g = np.array(mx.array(store)[:, :, mx.array(ai)])
    assert g.shape == (1, KV, 3, 2, hd) and (g == store[:, :, ai]).all(), "2-D gather"
    hs = [mx.array(rng.normal(size=(1, 4, 6)).astype(np.float32)) for _ in range(3)]    # the grouped parent gather
    par = np.array([1, -1, 1, 0]); grp = [np.flatnonzero(par == q_) for q_ in np.unique(par)]
    cat = mx.concatenate([hs[int(par[g_[0]]) + 1][:, mx.array(g_)] for g_ in grp], axis=1)
    inv = np.empty(4, np.int64); inv[np.concatenate(grp)] = np.arange(4)
    hp = np.array(cat[:, mx.array(inv)])
    assert all((hp[0, b_] == np.array(hs[par[b_] + 1])[0, b_]).all() for b_ in range(4)), "parent gather"
    # best-first against brute force: toy distributions from a hash of the path, 3 rows, 32 pops
    def dist(path):
        g = np.random.default_rng(abs(hash(tuple(path))) % (2 ** 32))
        p = g.dirichlet(np.full(8, 0.3)); o = np.argsort(-p)
        return (g.choice(1000, 8, replace=False)[o]).astype(np.int32), p[o]
    rows = [[7], [8], [9]]
    R = [dist(rw) for rw in rows]
    F = Forest(np.stack([x[0] for x in R]), np.stack([x[1] for x in R]))
    paths = [[[] for _ in range(32)] for _ in rows]
    for t in range(32):
        par, tok, dep, pp, ent = F.pop()
        ti, tp = [], []
        for b in range(3):
            paths[b][t] = (paths[b][par[b]] if par[b] >= 0 else []) + [int(tok[b])]
            assert len(ent[b]) == dep[b] + 1 and ent[b][-1] == t and (par[b] < t)
            x = dist(rows[b] + paths[b][t]); ti.append(x[0]); tp.append(x[1])
        F.push(t, np.stack(ti), np.stack(tp))
    for b in range(3):                                    # brute force: every level to depth 32, the top-32 path probabilities
        best, fr = [], [((), 1.0)]                        # (a level's best 40 hold every global top-32 node of that level)
        for _ in range(33):
            nf = []
            for pth, p in fr:
                ids, pr = dist(rows[b] + list(pth))
                for i_, p_ in zip(ids, pr):
                    nf.append((pth + (int(i_),), p * float(p_)))
            nf.sort(key=lambda z: -z[1]); nf = nf[:40]; best += nf; fr = nf
        best.sort(key=lambda z: -z[1])
        top = sorted(z[1] for z in best[:32])
        got = sorted(F.pp[b])
        assert np.allclose(top, got, rtol=1e-12), ("best-first", b)
        assert all(F.pp[b][i] >= F.pp[b][i + 1] - 1e-15 for i in range(31)), "pop order not best-first"
    print("SELFTEST OK: tree_attn == ttt_attn bitwise (full rows) and per padded row; best-first pops == the brute-force "
          "top-32 path probabilities, non-increasing, parents first")


NKS = ("fi", "an", "node", "parent", "depth", "token", "path_p", "y", "on_true", "top_ids", "top_p", "p_ynext", "ynext_rank")
AKS = ("a_fi", "a_an", "a_walk_start", "root_top_ids", "root_top_p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["heplus24", "mbppbank"])
    ap.add_argument("--run", default="t30b"); ap.add_argument("--frac", type=float, default=0.03)
    ap.add_argument("--nodes", type=int, default=32); ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--part", default="", help="i/n: only windows [len*i/n, len*(i+1)/n) -> tree_<run>_<set>.part<i>of<n>.npz")
    ap.add_argument("--merge", type=int, default=0, help="n: concatenate the n parts in order -> tree_<run>_<set>.npz + report")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.check_only:                                     # the checks + preview from a saved tree npz (no model)
        TK = np.load(os.path.join(OUT, f"topk_{a.run}_{a.set}.npz"))
        z = dict(np.load(os.path.join(OUT, f"tree_{a.run}_{a.set}.npz")))
        a.nodes, a.k = int(z["nodes"]), int(z["k"])
        return report(a, {k: z[k] for k in ("fi", "an", "node", "parent", "depth", "token", "path_p", "y", "on_true", "top_ids",
                                              "top_p", "p_ynext", "ynext_rank")},
                      {k: z[k] for k in ("a_fi", "a_an", "a_walk_start", "root_top_ids", "root_top_p")}, TK, None, None)
    if a.merge:                                          # parts are contiguous window ranges: concatenation keeps the order
        TK = np.load(os.path.join(OUT, f"topk_{a.run}_{a.set}.npz"))
        Z = [dict(np.load(os.path.join(OUT, f"tree_{a.run}_{a.set}.part{i}of{a.merge}.npz"))) for i in range(a.merge)]
        a.nodes, a.k = int(Z[0]["nodes"]), int(Z[0]["k"])
        N_ = {k: np.concatenate([z[k] for z in Z]) for k in NKS}
        A_ = {k: np.concatenate([z[k] for z in Z]) for k in AKS}
        assert (np.diff(A_["a_fi"]) >= 0).all(), "parts out of order"
        np.savez_compressed(os.path.join(OUT, f"tree_{a.run}_{a.set}.npz"), files=TK["files"], nodes=a.nodes, k=a.k, **N_, **A_)
        return report(a, N_, A_, TK, round(float(sum(z["wall_s"] for z in Z)), 1), round(float(max(z["peak_gb"] for z in Z)), 2))
    import mlx.core as mx
    import dd_ceiling as CE
    import dd_ttt as DD
    from dd_topk import topk_rows
    T, M = DD.T, DD.M
    t0 = time.time()
    rt_, lm, head = M.load(a.frac)
    P = M.Trainable(head)
    P.load_weights(os.path.join(DD.CKD, f"dd_{a.run}.safetensors")); mx.eval(P.parameters())
    emb, layer = lm.model.embed_tokens, head.layers[0]
    att = layer.self_attn
    FS = {"heplus24": CE.HEP, "mbppbank": CE.MBB}[a.set]
    TK = np.load(os.path.join(OUT, f"topk_{a.run}_{a.set}.npz"))
    assert [str(x) for x in TK["files"]] == [os.path.basename(f) for f in FS], "TOPK's files differ"
    d0 = TK["depth"] == 0
    NK = ("fi", "an", "node", "parent", "depth", "token", "path_p", "y", "on_true", "top_ids", "top_p", "p_ynext", "ynext_rank")
    AKs = ("a_fi", "a_an", "a_walk_start", "root_top_ids", "root_top_p")
    N_, A_ = {k: [] for k in NK}, {k: [] for k in AKs}
    lo, hi = 0, len(FS)
    if a.part:
        i_, n_ = (int(x) for x in a.part.split("/")); lo, hi = len(FS) * i_ // n_, len(FS) * (i_ + 1) // n_
    for fi, f in enumerate(FS):
        if not lo <= fi < hi:
            continue
        while os.path.exists(T.PAUSE):
            time.sleep(1)
        z = np.load(f)
        ids = np.asarray(z["ids"], np.int32)
        S = len(ids); gen = int(z["gen"]); a0 = max(0, gen - 2)
        sel = d0 & (TK["fi"] == fi)
        anc = (TK["row"][sel] - 1).astype(np.int64)                  # the anchors of TOPK's depth-0 cells
        ws = TK["walk_start"][sel]
        B = len(anc)
        if B == 0:
            continue
        W = mx.array(z["wd"]).astype(mx.bfloat16)[None]
        idm = mx.array(ids)[None]
        n1 = S - 1                                                     # depth 1 = the true pairs, as dd_d22_ttt.chain
        mixed, hyper, inject = layer.attn_hyper_connection(T.fuse(head, P, W[:, :n1], emb(idm[:, 1:n1 + 1])).astype(mx.bfloat16))
        q, gate, K1, V1 = T.qkv(att, P, mixed, mx.arange(n1))
        o = mx.fast.scaled_dot_product_attention(q[:, :, a0:], K1, V1, scale=att.scale, mask="causal")
        n = n1 - a0
        blk = P.o_proj(o.transpose(0, 2, 1, 3).reshape(1, n, -1) * mx.sigmoid(gate[:, a0:]))
        h1 = T.finish(head, P, hyper[:, a0:], inject[:, a0:], blk)
        R = h1[:, mx.array(anc - a0)]                                  # [1,B,Wd] the roots' carried states
        lg = T.logits(lm, head, R)[0]
        rti, rtp, _, _, _ = topk_rows(mx, lg, ids[anc + 2], a.k)
        mx.eval(K1, V1, R); del mixed, hyper, inject, q, gate, o, blk, h1, lg
        A_["a_fi"].append(np.full(B, fi)); A_["a_an"].append(anc); A_["a_walk_start"].append(ws)
        A_["root_top_ids"].append(rti); A_["root_top_p"].append(rtp)
        F = Forest(rti, rtp)
        HS, KS, VS = [R], [], []
        ontrue = np.zeros((B, a.nodes), bool)
        ar = np.arange(B)
        for t in range(a.nodes):
            par, tok, dep, pp, ent = F.pop()
            grp = [np.flatnonzero(par == q_) for q_ in np.unique(par)]          # the parents' carried states, by group
            cat = mx.concatenate([HS[int(par[g_[0]]) + 1][:, mx.array(g_)] for g_ in grp], axis=1)
            inv = np.empty(B, np.int64); inv[np.concatenate(grp)] = np.arange(B)
            HP = cat[:, mx.array(inv)]
            pos = anc + dep + 1                                                 # chain depth j = d + 2 -> position s + j - 1
            x = T.fuse(head, P, HP, emb(mx.array(tok)[None]))
            mixed, hyper, inject = layer.attn_hyper_connection(x.astype(mx.bfloat16))
            q, gate, k, v = T.qkv(att, P, mixed, mx.array(pos.astype(np.int32)))
            KS.append(k); VS.append(v)
            J = int(dep.max()) + 1
            aidx = np.empty((B, J), np.int32); am = np.zeros((B, J), bool)
            for b in range(B):
                e = ent[b]
                aidx[b, :len(e)] = [qq * B + b for qq in e]; aidx[b, len(e):] = t * B + b; am[b, :len(e)] = True
            ai = mx.array(aidx)
            kst, vst = mx.concatenate(KS, axis=2), mx.concatenate(VS, axis=2)
            o = tree_attn(mx, att.scale, q, K1, V1, kst[:, :, ai], vst[:, :, ai], mx.array(am), mx.array(anc.astype(np.int32)))
            blk = P.o_proj(o.transpose(0, 2, 1, 3).reshape(1, B, -1) * mx.sigmoid(gate))
            h = T.finish(head, P, hyper, inject, blk)
            lg = T.logits(lm, head, h)[0]
            yp = anc + dep + 2                                                  # the node's position
            y = np.where(yp <= S - 1, ids[np.minimum(yp, S - 1)], -1)
            yn = np.where(yp + 1 <= S - 1, ids[np.minimum(yp + 1, S - 1)], 0)
            ti, tp, py, yr, _ = topk_rows(mx, lg, yn, a.k)
            mx.eval(h)
            HS.append(h)
            ptrue = np.where(par >= 0, ontrue[ar, np.maximum(par, 0)], True)
            ontrue[:, t] = ptrue & (tok == y)
            F.push(t, ti, tp)
            for k_, v_ in (("fi", np.full(B, fi)), ("an", anc), ("node", np.full(B, t)), ("parent", par), ("depth", dep),
                           ("token", tok), ("path_p", pp.astype(np.float32)), ("y", y), ("on_true", ontrue[:, t].copy()),
                           ("top_ids", ti), ("top_p", tp), ("p_ynext", py), ("ynext_rank", yr)):
                N_[k_].append(v_)
        del HS, KS, VS, K1, V1, R
        mx.clear_cache()
    N_ = {k: np.concatenate(v) for k, v in N_.items()}
    A_ = {k: np.concatenate(v) for k, v in A_.items()}
    if a.part:
        i_, n_ = a.part.split("/")
        np.savez_compressed(os.path.join(OUT, f"tree_{a.run}_{a.set}.part{i_}of{n_}.npz"), nodes=a.nodes, k=a.k,
                            wall_s=time.time() - t0, peak_gb=mx.get_peak_memory() / 1e9, **N_, **A_)
        print(f"[tree] part {a.part}: windows {lo}..{hi - 1}, {len(A_['a_an'])} anchors, {time.time() - t0:.0f} s", flush=True)
        return
    np.savez_compressed(os.path.join(OUT, f"tree_{a.run}_{a.set}.npz"), files=TK["files"], nodes=a.nodes, k=a.k, **N_, **A_)
    report(a, N_, A_, TK, round(time.time() - t0, 1), round(mx.get_peak_memory() / 1e9, 2))


def report(a, N_, A_, TK, wall, peak):
    # the exactness check against TOPK (teacher-forced cells): roots = band 0; true-path nodes at depth d = band d + 1
    key = lambda fi_, row, band: (fi_.astype(np.int64) * 100000 + row.astype(np.int64)) * 64 + band.astype(np.int64)
    kt = key(TK["fi"], TK["row"], TK["depth"]); o_ = np.argsort(kt); kt_s = kt[o_]

    def look(k_):
        i = np.searchsorted(kt_s, k_); ok = (i < len(kt_s)) & (kt_s[np.minimum(i, len(kt_s) - 1)] == k_)
        return o_[np.minimum(i, len(kt_s) - 1)], ok

    def cmp(ti, tp, idx):
        same1 = ti[:, 0] == TK["top_ids"][idx, 0]
        set8 = np.array([set(x) == set(y_) for x, y_ in zip(ti, TK["top_ids"][idx])])
        dp = np.abs(tp[:, 0] - TK["top_p"][idx, 0])
        return {"n": int(len(idx)), "top1_same": round(float(same1.mean()), 5), "top8_set_same": round(float(set8.mean()), 5),
                "p1_absdiff_max": round(float(dp.max()), 6), "p1_absdiff_median": float(np.median(dp))}
    ir, okr = look(key(A_["a_fi"], A_["a_an"] + 1, np.zeros(len(A_["a_an"]))))
    nt = N_["on_true"] & (N_["depth"] + 1 <= 30)
    it, okt = look(key(N_["fi"][nt], N_["an"][nt] + N_["depth"][nt] + 2, N_["depth"][nt] + 1))
    Rj = {"run": a.run, "set": a.set, "anchors": int(len(A_["a_an"])), "nodes_per_anchor": a.nodes, "k": a.k,
          "walk_starts": int(A_["a_walk_start"].sum()), "wall_s": wall, "mlx_peak_gb": peak,
          "check_roots_vs_topk_band0": cmp(A_["root_top_ids"][okr], A_["root_top_p"][okr], ir[okr]) | {"found": round(float(okr.mean()), 5)},
          "check_truepath_nodes_vs_topk": cmp(N_["top_ids"][nt][okt], N_["top_p"][nt][okt], it[okt]) | {"found": round(float(okt.mean()), 5)}}
    # preview (paired, same anchors): kept per check = accepted + 1; chain-K = TOPK's leading rights capped at K drafts,
    # tree-N = the deepest true-path node among the first N pops + 1
    rt_ = {}
    for fi_, row, band, r_ in zip(TK["fi"], TK["row"], TK["depth"], TK["right"]):
        rt_[(int(fi_), int(row) - int(band) - 1, int(band))] = bool(r_)
    A = len(A_["a_an"])
    chain = np.zeros(A, int)
    for i_, (fi_, s) in enumerate(zip(A_["a_fi"], A_["a_an"])):
        kk = 0
        while kk < 31 and rt_.get((int(fi_), int(s), kk), False):
            kk += 1
        chain[i_] = kk
    acc = np.zeros((A, a.nodes + 1), int)                               # acc[:, N] = accepted with the first N nodes
    assert len(N_["node"]) == A * a.nodes, (len(N_["node"]), A, a.nodes)
    if True:
        # N_ is ordered (window, node t, row b): reshape per window -> per anchor
        offs = np.cumsum([0] + [int((A_["a_fi"] == fi_).sum()) for fi_ in np.unique(A_["a_fi"])])
        ot = np.zeros((A, a.nodes), bool); dd = np.zeros((A, a.nodes), int)
        p0 = 0
        for wi, fi_ in enumerate(np.unique(A_["a_fi"])):
            Bw = offs[wi + 1] - offs[wi]
            blk_ot = N_["on_true"][p0:p0 + Bw * a.nodes].reshape(a.nodes, Bw).T
            blk_d = N_["depth"][p0:p0 + Bw * a.nodes].reshape(a.nodes, Bw).T
            ot[offs[wi]:offs[wi + 1]] = blk_ot; dd[offs[wi]:offs[wi + 1]] = blk_d
            p0 += Bw * a.nodes
        for N in range(1, a.nodes + 1):
            acc[:, N] = np.where(ot[:, :N], dd[:, :N] + 1, 0).max(1)
    wsel = A_["a_walk_start"].astype(bool)
    Rj["preview_kept_per_check"] = {
        sc: {f"chain_{K}": round(float(np.minimum(chain[m_], K).mean() + 1), 4) for K in (15, 31)}
        | {f"tree_{N}": round(float(acc[m_, N].mean() + 1), 4) for N in (15, 16, 31, 32) if N <= a.nodes}
        for sc, m_ in (("walk_starts", wsel), ("all_anchors", np.ones(A, bool)))}
    json.dump(Rj, open(os.path.join(OUT, f"tree_{a.run}_{a.set}.json"), "w"), indent=1)
    print("[tree]", json.dumps(Rj), flush=True)


if __name__ == "__main__":
    main()
