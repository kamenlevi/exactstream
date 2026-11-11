"""R57_PFLM=1|2 (Sun 8 Nov; Kamen's MUST prefill >= 500 tok/s; NewIdeas' PF-A, Leap's advice, MoonShots'
patches/layer_major_prefill/DESIGN.md): layer-major prefill with the screen's chunk boundaries, plus a per-layer expert scratch
so each layer's experts are read from records3 ONCE per prompt instead of once per 1,024-row chunk.

- Today (45 GB r57, R57_TIER=1): every prefill MoE call goes stream._sync_call -> tier.prefill_call, which builds a scratch of
  exactly the call's unique experts, fills it from records3 with the ext's CPU rebuild (an SSD read of every one of them) and
  frees it. A 1,024-row chunk routes to ~all 512 experts, so each chunk re-reads ~1.5 GB per layer (~75 GB per chunk): a 7.9K
  prompt reads ~600 GB (N45: TTFT 64 s, 124 tok/s).
- Order: `for layer: for chunk:` instead of `for chunk: for layer:`. Each (layer, chunk) step gets exactly today's inputs (its own
  hyper stream, the layer's KV / GDN cache after chunks 0..k-1), the same chunk sizes and the same kernels (MoonShots DESIGN.md:
  bit-identical when the per-call computation is unchanged; Leap: a one-call prefill is NOT bitwise, 0 / 9 hashes).
- Reads: install() wraps tier.prefill_call. While LMC["on"], a layer's first call allocates nothing new beyond one full-E scratch
  (E rows x every 4-bit piece, rows = expert ids, ~1.57 GB: about today's per-call scratch, which is freed meanwhile) and reads
  the call's experts into their rows; later chunks of the same layer read only experts not yet there. The scratch is reused
  across layers (a new layer only resets which rows are valid) and dropped after the prefill (the decode peak is unchanged).
  - R57_PFLM=1: the call runs on the full-E scratch with the ORIGINAL ids (no copy). The kernel sees E = 512 weight rows instead
    of len(unique): tier.prefill_call's docstring warns that this changed MLX's gather kernel choice on SHORT prompts (B7 prose
    sha); layer-major only runs prompts > 1 chunk. Exactness is an empirical question: R57_LOGHASH vs the chunk-major arm.
  - R57_PFLM=2: bitwise by construction: each call gathers its unique rows into a compact copy (mx.take, sorted ids) and runs on
    compact ids, exactly today's call shape. Costs one extra ~1.5 GB copy per call (the 45.0 GB line decides).
- Logits: only the last chunk goes through the mixer + head, its last 256 rows (as R57_PFNOLOG).
"""
import numpy as np
import mlx.core as mx

LMC = {"on": False, "mode": 1, "t": None, "have": set(), "sc": None, "reads": 0, "hits": 0, "calls": 0}


def install(tier, mode=1):
    """Wrap bigmoe.tier.prefill_call (stream._sync_call looks it up at call time). Inactive unless LMC['on']."""
    LMC["mode"] = int(mode)
    if getattr(tier, "_pflm_installed", False):
        return
    orig = tier.prefill_call
    from bigmoe.stream import PIECES

    def prefill_call(cls, self, x, indices, bm, *a, **kw):
        if not LMC["on"]:
            return orig(cls, self, x, indices, bm, *a, **kw)
        t = bm["tier"]
        tier.STATS["prefill_calls"] += 1
        ids = np.array(indices)
        uniq = np.unique(ids)
        E = int(t["E"])
        shp = [((E,) + tuple(sh), dt) for sh, dt in t["shapes4"]]
        if LMC["sc"] is None or [(tuple(s.shape), s.dtype) for s in LMC["sc"]] != shp:
            LMC["sc"] = None
            mx.clear_cache()
            LMC["sc"] = [mx.zeros(sh, dt) for sh, dt in shp]
            mx.eval(*LMC["sc"])
            LMC["t"] = None
        if LMC["t"] is not t:                            # a new layer: no row of the scratch is valid for it yet
            LMC["t"], LMC["have"] = t, set()
        need = [int(e) for e in uniq if int(e) not in LMC["have"]]
        if need:                                         # the same ext rebuild as today, into those experts' own rows
            tier._read_rebuilt(t, need, need, LMC["sc"])
            LMC["have"].update(need)
            LMC["reads"] += len(need)
        LMC["hits"] += len(uniq) - len(need)
        LMC["calls"] += 1
        if LMC["mode"] == 2:                             # today's call shape: a compact sorted scratch + order-preserving ids
            gi = mx.array(uniq.astype(np.int32))
            scr = [mx.take(s, gi, axis=0) for s in LMC["sc"]]
            mx.eval(*scr)
            idx = mx.array(np.searchsorted(uniq, ids).astype(np.int32)).astype(indices.dtype)
        else:
            scr, idx = LMC["sc"], indices
        saved = [getattr(getattr(self, proj), field) for proj, field in PIECES]
        for (proj, field), arr in zip(PIECES, scr):
            setattr(getattr(self, proj), field, arr)
        try:
            y = cls.__call__(self, x, idx, *a, **kw)
            mx.eval(y)
        finally:
            for (proj, field), arr in zip(PIECES, saved):
                setattr(getattr(self, proj), field, arr)
        if LMC["mode"] == 2:
            del scr
        return y

    tier.prefill_call = prefill_call
    tier._pflm_installed = True


def prefill_layer_major(model, ids, chunk, states, on_chunk=None):
    """The screen's chunk-major prefill, layer-major. Returns (cache, logits of the last chunk's last 256 rows).
    on_chunk(k, h_k) runs after the LAST layer for each chunk k (the widened stream, for the MTP head)."""
    import mtplx.models.qwen4_exp as Q
    import bigmoe.tier as BT
    lm = model.language_model
    tm = lm.model
    cache = model.make_cache()
    xs = [mx.array([ids[a:a + chunk]]) for a in range(0, len(ids), chunk)]

    def run():
        hs = []
        for x in xs:
            h = tm.embed_tokens(x)
            hs.append(mx.tile(h, (1, 1, tm.args.hc_count)))
        mx.eval(*hs)
        stage = tm._ple_stage_idx is not None and Q._COMPILED_VERIFY_PLE.get() is None
        BT._reuse_drop()
        LMC.update(on=True, t=None, have=set())
        try:
            for L, (layer, c) in enumerate(zip(tm.layers, cache)):
                for k, x in enumerate(xs):
                    if L == 0:                           # as _forward: the ssm mask of a B=1 unpadded chunk is None
                        assert Q.create_ssm_mask(hs[k], cache[tm.ssm_idx]) is None, "PFLM: an ssm mask (padding?)"
                    if stage and L == tm._ple_stage_idx:   # as _forward stages chunk k's n-grams before its layers: the
                        ple = tm.layers[tm._ple_stage_idx].ple   # stage only touches the PLE layer's cache, which holds
                        ple.ple_embedding.stage(x, cache[tm._ple_stage_idx], ple.NGRAM_IDX)   # chunks 0..k-1 here too
                    hs[k] = layer(hs[k], input_ids=x, ssm_mask=None, cache=c)
                    mx.eval(hs[k], *states([c]))
        finally:
            LMC.update(on=False, t=None, have=set(), sc=None)
            mx.clear_cache()
        if on_chunk is not None:
            for k, h in enumerate(hs):
                on_chunk(k, h)
        tm._last_widened = hs[-1]
        out = tm.hyper_connection_mixer(hs[-1])[:, -256:]   # TextModel(logits_keep=256): the mixer over the whole chunk
        return lm._head_logits(out)

    if Q.qwen4_opdiet_enabled("rope"):
        with Q._rope_table_scope():
            lg = run()
    else:
        lg = run()
    return cache, lg
