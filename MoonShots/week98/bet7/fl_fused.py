"""WEEK98 BET 7: the fused level (deliverable 2), driven over a fl_stock.Stock object's own modules and cache.

Dependent kernels per level (F = 1): S1 prep | S2 S3 attention HC | S4 projections | S5 attention (+1 kernel at
>= 1,024 keys) | S6 o_proj | S7 S8 MLP HC (+ the attention's residual write) | S9 S9b S10 S11 MoE | S12 S13 mixer (+
the MoE's residual write, which is the layer's output hh) | S14 head | S15 tail  = 17 (18 with the 2-pass attention).
Off the critical path, with the stock's own code on the fused outputs: the indexer's raw-key row and pooled block
(cache.write_raw / _extend_pooled) and the KV row write (cache.kv.update_and_fetch), so the cache state is the stock's
by construction once the fused outputs are bit-equal.
Scope: F = 1; dense attention (the indexer's selection inactive: <= 2,048 keys); no session extras in the screen."""
import mlx.core as mx

import fl_attn
import fl_hc
import fl_level
import fl_moe


class Fused:
    def __init__(self, st):
        self.st = st
        mtp = st.mtp
        self.L0 = mtp.layers[0]
        self.attn = self.L0.self_attn

    def level(self, h, tok, sc, F=1, K=8):
        assert F == 1, "the prototype covers F = 1"
        st, mtp, L0, attn = self.st, self.st.mtp, self.L0, self.attn
        cache = st.cache[0]
        T = int(cache.offset)
        idx = attn.indexer
        assert (T + 1) // idx.ratio <= idx.block_topk, "beyond the indexer's budget: the stock level's job"
        hpre = fl_level.s1_prep(mtp, st.emb, h, tok)                                       # S1
        mixed1, inj1, _, _ = fl_hc.hc_read(L0.attn_hyper_connection, hpre)                 # S2, S3
        qf, kf, vf, iqk = fl_level.s4_proj(attn, mixed1)                                   # S4
        kb, vb = cache.kv.keys, cache.kv.values
        if kb is None or T > kb.shape[2]:
            raise RuntimeError("the cache backing must hold the T rows S5 reads")
        gated, krot = fl_attn.fused_attn(attn, qf, kf, vf, kb[0], vb[0], T)                # S5
        bo1 = fl_level.s6_oproj(attn, gated)                                               # S6
        mixed2, inj2, hw2, _ = fl_hc.hc_read(L0.mlp_hyper_connection, hpre, pending=(bo1, inj1))   # S7, S8
        bo2, _ = fl_moe.moe(L0.mlp, mixed2)                                                # S9, S9b, S10, S11
        x, _, hh, _ = fl_hc.hc_read(mtp.hyper_connection_mixer, hw2, pending=(bo2, inj2))  # S12, S13
        lg = fl_level.s14_head(st.sh, x)                                                   # S14
        kidx, klp, cand, sel, nsc, ntok = fl_level.s15_tail(lg, sc, st.sh.ids, K)          # S15
        # ---- cache maintenance, the stock's code on the fused outputs (QSAIndexer._call_rows, Attention.__call__)
        cache.indexer_budget = idx.block_topk * idx.ratio
        _, kidxr = mx.split(iqk.reshape(1, 1, 640), [idx.n_heads * idx.head_dim], axis=-1)
        cache.write_raw(kidxr.reshape(1, 1, idx.head_dim))
        idx._extend_pooled(cache, T + 1)
        k4 = krot.reshape(1, 2, 1, 256)
        v4 = vf.reshape(1, 1, 2, 256).transpose(0, 2, 1, 3)
        from mtplx.models.qwen4_exp import _after_index_block_writes
        k4 = _after_index_block_writes(k4, cache)
        cache.kv.update_and_fetch(k4, v4)
        hh3 = hh.reshape(1, 1, -1)
        return dict(h=hh3, tok=ntok.reshape(1, 1), sc=nsc, hh=hh3, kidx=kidx.reshape(1, K),
                    klp=klp.reshape(1, K), cand=cand, sel=sel, par=mx.zeros((1,), mx.uint32))

    def block(self, L, K=8, tok0=1000, mode="async", keep=False):
        st = self.st
        h, tok, sc = st.h0, mx.array([[tok0]], dtype=mx.int32), mx.zeros((1,), mx.float32)
        outs = []
        for _ in range(L):
            o = self.level(h, tok, sc, 1, K)
            h, tok, sc = o["h"], o["tok"], o["sc"]
            outs.append(o)
            if mode == "async":
                mx.async_eval(h, tok, sc, o["kidx"], o["klp"])
        mx.eval([list(o.values()) for o in outs])
        if not keep:
            st.cache[0].trim(L)
        return outs


class Fused2(Fused):
    """the fused level with the next level's input norms folded into S15 (S15P) and S1 as pure GEMVs (S1G);
    a block's first level runs S0 for that prep. Same kernels otherwise; outputs and cache writes as Fused."""

    def level(self, h, tok, sc, F=1, K=8, prep=None):
        assert F == 1, "the prototype covers F = 1"
        st, mtp, L0, attn = self.st, self.st.mtp, self.L0, self.attn
        cache = st.cache[0]
        T = int(cache.offset)
        idx = attn.indexer
        assert (T + 1) // idx.ratio <= idx.block_topk, "beyond the indexer's budget: the stock level's job"
        hn, en_in = prep if prep is not None else fl_level.s0_prep(mtp, st.emb, h, tok)    # S0 (first level only)
        hpre = fl_level.s1g(mtp, hn, en_in)                                                # S1
        mixed1, inj1, _, _ = fl_hc.hc_read(L0.attn_hyper_connection, hpre)                 # S2, S3
        qf, kf, vf, iqk = fl_level.s4_proj(attn, mixed1)                                   # S4
        kb, vb = cache.kv.keys, cache.kv.values
        if kb is None or T > kb.shape[2]:
            raise RuntimeError("the cache backing must hold the T rows S5 reads")
        gated, krot = fl_attn.fused_attn(attn, qf, kf, vf, kb[0], vb[0], T)                # S5
        bo1 = fl_level.s6_oproj(attn, gated)                                               # S6
        mixed2, inj2, hw2, _ = fl_hc.hc_read(L0.mlp_hyper_connection, hpre, pending=(bo1, inj1))   # S7, S8
        bo2, _ = fl_moe.moe(L0.mlp, mixed2)                                                # S9-S11
        x, _, hh, _ = fl_hc.hc_read(mtp.hyper_connection_mixer, hw2, pending=(bo2, inj2))  # S12, S13
        lg = fl_level.s14_head(st.sh, x)                                                   # S14
        kidx, klp, cand, sel, nsc, ntok, hn2, en2 = fl_level.s15p_tail(lg, sc, st.sh.ids, hh, mtp, st.emb, K)  # S15
        cache.indexer_budget = idx.block_topk * idx.ratio
        _, kidxr = mx.split(iqk.reshape(1, 1, 640), [idx.n_heads * idx.head_dim], axis=-1)
        cache.write_raw(kidxr.reshape(1, 1, idx.head_dim))
        idx._extend_pooled(cache, T + 1)
        k4 = krot.reshape(1, 2, 1, 256)
        v4 = vf.reshape(1, 1, 2, 256).transpose(0, 2, 1, 3)
        from mtplx.models.qwen4_exp import _after_index_block_writes
        k4 = _after_index_block_writes(k4, cache)
        cache.kv.update_and_fetch(k4, v4)
        hh3 = hh.reshape(1, 1, -1)
        return dict(h=hh3, tok=ntok.reshape(1, 1), sc=nsc, hh=hh3, kidx=kidx.reshape(1, K),
                    klp=klp.reshape(1, K), cand=cand, sel=sel, par=mx.zeros((1,), mx.uint32), prep=(hn2, en2))

    def block(self, L, K=8, tok0=1000, mode="async", keep=False):
        st = self.st
        h, tok, sc = st.h0, mx.array([[tok0]], dtype=mx.int32), mx.zeros((1,), mx.float32)
        prep = None
        outs = []
        for _ in range(L):
            o = self.level(h, tok, sc, 1, K, prep=prep)
            h, tok, sc, prep = o["h"], o["tok"], o["sc"], o["prep"]
            outs.append(o)
            if mode == "async":
                mx.async_eval(h, tok, sc, o["kidx"], o["klp"], prep[0], prep[1])
        mx.eval([[v for k, v in o.items() if k != "prep"] for o in outs])
        if not keep:
            st.cache[0].trim(L)
        return outs
