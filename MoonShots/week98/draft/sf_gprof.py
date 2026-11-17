"""DRAFT LINE (GPU-side, mtlprof): where a pop's fixed cost goes at the REAL r57 format (MoonShots, Tue 17 Nov 2026; after sf_tree.py --real:
1.11 ms per pop sync-free, ~0.56 ms of it bytes). Information for the fused tree level, not a verdict.

Setup as sf_tree.py --real (t30b's 10 Linears + the HC mixes as QuantizedLinear 8-bit g64, Leap's ScreenedHead 32K +
session extras, B = 1). One anchor's sync-free tree is drafted up to pop t; then pop t is re-run cut into its stages,
each evaluated alone with its inputs already on the GPU (mx.synchronize around it), --reps times:
  step (the GPU frontier) | gather (parent state, ancestor rows) | fuse (norms + fc_embedding + fc_hidden) |
  hcA (the attention HC read) | qkv | attn (KV concat + gathers + tree_attn) | oproj | finish (MLP HC read + MoE +
  combine) | head (the mixer HC + the screened logits) | topk (logsumexp + argpartition + sort + exp + id map)
Reported per stage: median wall (us) and its MLX primitive count (graph nodes, via mx.export_to_dot); their sum vs the
same pop chained (one eval) and vs the pipelined tree's per-pop median.
usage: ~/BigMoE/.venv/bin/python -I -B sf_prof.py [--set heplus24] [--window 0] [--t 8] [--reps 30] [--out <json>]
       (MEM ~9 GB: share lock with MEM, after `peer ls`)"""
import argparse
import json
import os
import re
import sys
import tempfile
import time

import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/Leap/third_party/mtlprof"))
import mtlprof as MP                                       # noqa: E402  (before mlx: GPU command-buffer times)
MP.load(2)
sys.path.insert(0, H("~/BigMoE/scripts"))
sys.path.insert(0, H("~/MoonShots/week98"))
sys.path.insert(0, H("~/MoonShots/week98/draft"))
import mlx.core as mx                                      # noqa: E402
import sf_tree as SF                                       # noqa: E402


def nodes_of(*arrs):
    """number of primitive nodes in the lazy graph of arrs (mx.export_to_dot), or -1"""
    try:
        fd, p = tempfile.mkstemp(suffix=".dot")
        os.close(fd)
        mx.export_to_dot(p, *arrs)
        s = open(p).read()
        os.unlink(p)
        return len(re.findall(r"shape=rectangle", s)) or len(re.findall(r"label=", s))
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["heplus24", "mbppbank"], default="heplus24")
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--t", type=int, default=8)
    ap.add_argument("--nodes", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--anchors", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import mlx.nn as nn
    import dd_tree as DT
    import dd_ceiling as CE
    import dd_ttt as DD
    import p3_dd_tree as P3
    T, M = DD.T, DD.M
    K, NODES = a.k, a.nodes
    rt_, lm, head = M.load(0.03)
    P = M.Trainable(head)
    P.load_weights(os.path.join(DD.CKD, "dd_t30b.safetensors")); mx.eval(P.parameters())
    emb, layer = lm.model.embed_tokens, head.layers[0]
    att = layer.self_attn

    def qlin(lin):
        w = lin.weight.astype(mx.bfloat16)
        ql = nn.QuantizedLinear(w.shape[1], w.shape[0], bias=False, group_size=64, bits=8)
        ql.weight, ql.scales, ql.biases = mx.quantize(w, group_size=64, bits=8)
        return ql
    for name in ("q_proj", "o_proj", "k_proj", "v_proj", "se_gate", "se_up", "se_down", "fc_hidden", "fc_embedding",
                 "router"):
        setattr(P, name, qlin(getattr(P, name)))
    for g in (layer.attn_hyper_connection, layer.mlp_hyper_connection, head.hyper_connection_mixer):
        for sub in ("input_mix_weight_down", "input_mix_weight_up"):
            setattr(g, sub, qlin(getattr(g, sub)))
    mx.eval(P.parameters(), head.parameters())
    sys.path.insert(0, H("~/Leap/scripts"))
    from screened_head import ScreenedHead
    RANK = np.load(P3.RANK_F)
    FS = {"heplus24": CE.HEP, "mbppbank": CE.MBB}[a.set]
    TK = np.load(os.path.join(DT.OUT, f"topk_t30b_{a.set}.npz"))
    fi = a.window
    z = np.load(FS[fi])
    ids = np.asarray(z["ids"], np.int32)
    S = len(ids); gen = int(z["gen"]); a0 = max(0, gen - 2)
    sel = (TK["depth"] == 0) & (TK["fi"] == fi)
    ancs = (TK["row"][sel] - 1).astype(np.int64)
    W = mx.array(z["wd"]).astype(mx.bfloat16)[None]
    idm = mx.array(ids)[None]
    n1 = S - 1
    mixed, hyper, inject = layer.attn_hyper_connection(T.fuse(head, P, W[:, :n1], emb(idm[:, 1:n1 + 1])).astype(mx.bfloat16))
    q, gate, K1, V1 = T.qkv(att, P, mixed, mx.arange(n1))
    o = mx.fast.scaled_dot_product_attention(q[:, :, a0:], K1, V1, scale=att.scale, mask="causal")
    n = n1 - a0
    blk = P.o_proj(o.transpose(0, 2, 1, 3).reshape(1, n, -1) * mx.sigmoid(gate[:, a0:]))
    h1 = T.finish(head, P, hyper[:, a0:], inject[:, a0:], blk)
    mx.eval(K1, V1, h1)
    sh = ScreenedHead(lm.lm_head, RANK[:32768], extra_cap=4096)
    sh.add(ids[:int(ancs.min()) + 2])
    idmap = mx.concatenate([sh.ids, sh.xids]) if sh.n_extra else sh.ids
    mx.eval(idmap)
    logits_of = lambda h: sh.logits(head.hyper_connection_mixer(h))

    stages = ["step", "gather", "fuse", "hcA", "qkv", "attn", "oproj", "finish", "head", "topk"]
    acc = {s: [] for s in stages}
    cnt = {s: [] for s in stages}
    chained, piped, chained_disp, tree_gpu, tree_disp = [], [], [], [], []
    picks = ancs[len(ancs) // 4::max(1, len(ancs) // (a.anchors + 1))][:a.anchors]
    for anc in picks:
        anc = int(anc)
        R = h1[:, anc - a0:anc - a0 + 1]
        lg0 = logits_of(R)
        mx.eval(R, lg0)
        # the pipelined tree, for the per-pop reference
        G = SF.GPUFrontier(K, NODES)
        tis, _, tp = SF.topk_gpu(lg0, K, idmap)
        HS, KS, VS = [R], [], []
        ancarr = mx.array(np.array([anc], np.int32))
        anc1 = mx.array(np.array([anc + 1], np.int32))
        snap = None
        mx.synchronize(); t0 = time.perf_counter(); g0 = MP.counts()
        for t in range(NODES):
            if t == a.t:
                mx.eval(tis, tp, HS, KS, VS, *G.arrays())
                snap = (tis, tp, list(HS), list(KS), list(VS), G.arrays())
            pop = G.step(tis, tp, t)
            par, tok, d = pop[0], pop[1], pop[2]
            HP = mx.take(mx.stack(HS, axis=0), (par + 1).reshape(1), axis=0).reshape(1, 1, -1)
            jj = mx.arange(t + 1, dtype=mx.int32)
            arow = G.nanc[t * NODES:t * NODES + t + 1]
            aidx = mx.where(jj < d, arow, t).reshape(1, t + 1).astype(mx.int32)
            am = (jj <= d).reshape(1, t + 1)
            x = T.fuse(head, P, HP, emb(tok.reshape(1, 1)))
            mixed, hyper, inject = layer.attn_hyper_connection(x.astype(mx.bfloat16))
            qq, gate, k, v = T.qkv(att, P, mixed, anc1 + d.reshape(1))
            KS.append(k); VS.append(v)
            kst, vst = mx.concatenate(KS, axis=2), mx.concatenate(VS, axis=2)
            oo = DT.tree_attn(mx, att.scale, qq, K1, V1, kst[:, :, aidx], vst[:, :, aidx], am, ancarr)
            bl = P.o_proj(oo.transpose(0, 2, 1, 3).reshape(1, 1, -1) * mx.sigmoid(gate))
            h = T.finish(head, P, hyper, inject, bl)
            lg = logits_of(h)
            tis, _, tp = SF.topk_gpu(lg, K, idmap)
            HS.append(h)
            mx.async_eval(h, tis, tp, pop, *G.arrays())
        mx.eval(HS, tis, tp)
        mx.synchronize()
        piped.append((time.perf_counter() - t0) * 1e3 / NODES)
        g1 = MP.counts()
        cbt = MP.cbs(g0["ncb"], g1["ncb"])
        busy = sorted((s, e) for s, e, rd in zip(cbt["start"], cbt["end"], cbt["ready"]) if rd)
        tot, cur_s, cur_e = 0.0, None, None
        for s_, e_ in busy:
            if cur_e is None or s_ > cur_e:
                if cur_e is not None:
                    tot += cur_e - cur_s
                cur_s, cur_e = s_, e_
            else:
                cur_e = max(cur_e, e_)
        if cur_e is not None:
            tot += cur_e - cur_s
        tree_gpu.append(tot * 1e6 / NODES)
        tree_disp.append((g1["disp"] - g0["disp"]) / NODES)
        # pop a.t, staged, from the snapshot
        t = a.t
        tis0, tp0, HS0, KS0, VS0, Garr = snap
        for rep in range(a.reps + 3):
            G2 = SF.GPUFrontier(K, NODES)
            (G2.fk, G2.fpar, G2.ftok, G2.fdep, G2.nf, G2.npp, G2.ndep, G2.nanc) = Garr
            KSl, VSl = list(KS0), list(VS0)
            tm = {}

            def run(name, fn):
                mx.synchronize(); c0 = MP.counts()
                out = fn()
                mx.eval(*[o_ for o_ in out if isinstance(o_, mx.array)])
                mx.synchronize(); c1 = MP.counts()
                cb = MP.cbs(c0["ncb"], c1["ncb"])
                tm[name] = float(sum(e - s for s, e, rd in zip(cb["start"], cb["end"], cb["ready"]) if rd)) * 1e6
                if rep == 0:
                    cnt[name].append(int(c1["disp"] - c0["disp"]))
                return out
            pop, = run("step", lambda: (G2.step(tis0, tp0, t),))
            par, tok, d = pop[0], pop[1], pop[2]
            HP, aidx, am, pos = run("gather", lambda: (
                mx.take(mx.stack(HS0, axis=0), (par + 1).reshape(1), axis=0).reshape(1, 1, -1),
                mx.where(mx.arange(t + 1, dtype=mx.int32) < d, G2.nanc[t * NODES:t * NODES + t + 1], t).reshape(1, t + 1).astype(mx.int32),
                (mx.arange(t + 1, dtype=mx.int32) <= d).reshape(1, t + 1),
                anc1 + d.reshape(1)))
            x, = run("fuse", lambda: (T.fuse(head, P, HP, emb(tok.reshape(1, 1))),))
            mixed, hyper, inject = run("hcA", lambda: layer.attn_hyper_connection(x.astype(mx.bfloat16)))
            qq, gate, k, v = run("qkv", lambda: T.qkv(att, P, mixed, pos))
            KSl.append(k); VSl.append(v)

            def attn():
                kst, vst = mx.concatenate(KSl, axis=2), mx.concatenate(VSl, axis=2)
                return (DT.tree_attn(mx, att.scale, qq, K1, V1, kst[:, :, aidx], vst[:, :, aidx], am, ancarr),)
            oo, = run("attn", attn)
            bl, = run("oproj", lambda: (P.o_proj(oo.transpose(0, 2, 1, 3).reshape(1, 1, -1) * mx.sigmoid(gate)),))
            h, = run("finish", lambda: (T.finish(head, P, hyper, inject, bl),))
            lg, = run("head", lambda: (logits_of(h),))
            run("topk", lambda: SF.topk_gpu(lg, K, idmap))
            if rep >= 3:
                for s_ in stages:
                    acc[s_].append(tm[s_])
            # the same pop chained (one eval)
            G3 = SF.GPUFrontier(K, NODES)
            (G3.fk, G3.fpar, G3.ftok, G3.fdep, G3.nf, G3.npp, G3.ndep, G3.nanc) = Garr
            KSc, VSc = list(KS0), list(VS0)
            mx.synchronize(); ts = time.perf_counter()
            pop = G3.step(tis0, tp0, t)
            par, tok, d = pop[0], pop[1], pop[2]
            HP = mx.take(mx.stack(HS0, axis=0), (par + 1).reshape(1), axis=0).reshape(1, 1, -1)
            jj = mx.arange(t + 1, dtype=mx.int32)
            aidx = mx.where(jj < d, G3.nanc[t * NODES:t * NODES + t + 1], t).reshape(1, t + 1).astype(mx.int32)
            am = (jj <= d).reshape(1, t + 1)
            x = T.fuse(head, P, HP, emb(tok.reshape(1, 1)))
            mixed, hyper, inject = layer.attn_hyper_connection(x.astype(mx.bfloat16))
            qq, gate, k, v = T.qkv(att, P, mixed, anc1 + d.reshape(1))
            KSc.append(k); VSc.append(v)
            kst, vst = mx.concatenate(KSc, axis=2), mx.concatenate(VSc, axis=2)
            oo = DT.tree_attn(mx, att.scale, qq, K1, V1, kst[:, :, aidx], vst[:, :, aidx], am, ancarr)
            bl = P.o_proj(oo.transpose(0, 2, 1, 3).reshape(1, 1, -1) * mx.sigmoid(gate))
            h = T.finish(head, P, hyper, inject, bl)
            out = SF.topk_gpu(logits_of(h), K, idmap)
            c0 = MP.counts()
            mx.eval(h, *out)
            mx.synchronize()
            c1 = MP.counts()
            cb = MP.cbs(c0["ncb"], c1["ncb"])
            if rep >= 3:
                chained.append(float(sum(e - s for s, e, rd in zip(cb["start"], cb["end"], cb["ready"]) if rd)) * 1e6)
                chained_disp.append(int(c1["disp"] - c0["disp"]))
    res = dict(set=a.set, window=a.window, t=a.t, nodes=NODES, anchors=[int(x) for x in picks], reps=a.reps,
               stage_us={s: dict(median=float(np.median(acc[s])), p10=float(np.percentile(acc[s], 10)),
                                 p90=float(np.percentile(acc[s], 90)), nodes=int(np.median(cnt[s])) if cnt[s] else -1)
                         for s in stages},
               stage_sum_us=float(sum(np.median(acc[s]) for s in stages)),
               chained_pop_gpu_us=float(np.median(chained)), chained_pop_dispatches=int(np.median(chained_disp)),
               piped_ms_per_pop=float(np.median(piped)), tree_gpu_busy_us_per_pop=float(np.median(tree_gpu)),
               tree_dispatches_per_pop=float(np.median(tree_disp)))
    for s in stages:
        r = res["stage_us"][s]
        print(f"  {s:7s} GPU {r['median']:8.1f} us [{r['p10']:.1f}-{r['p90']:.1f}]  dispatches {r['nodes']}", flush=True)
    print(f"  sum of stages' GPU {res['stage_sum_us']:.1f} us | the same pop chained: GPU {res['chained_pop_gpu_us']:.1f} us, "
          f"{res['chained_pop_dispatches']} dispatches | pipelined tree: {res['piped_ms_per_pop'] * 1e3:.1f} us wall per pop, "
          f"GPU busy {res['tree_gpu_busy_us_per_pop']:.1f} us per pop, {res['tree_dispatches_per_pop']:.0f} dispatches per pop", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
