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
- R57_PFLM=3 (PF-B, Sun 08:4x, Leap's plan; mode 1's scratch on original ids): each layer is split at its MoE. Every chunk's
  attention part (PLE, attention hyper read, attention / GDN, MLP hyper read) runs in chunk order and goes to the GPU with
  mx.async_eval; meanwhile the CPU reads ALL the layer's experts into the scratch (R57_PFLM_AHEAD=1, the default: PFX routed
  93-99 % of them, so ~1-7 % more bytes; the ext holds the GIL, so the overlap is CPU read || GPU attention). Then ONE MoE
  call per group of chunks (<= R57_PFLM_TOK tokens, default 4096) on Leap's padded SwitchGLU (MLX 0.32.2's sorted gather
  leaves rows unwritten past 32,767 sorted rows unless M % 64 == 0), then each chunk's residual write. The router and the
  shared expert run once per group (per-token, batch-independent per Leap; R57_LOGHASH vs mode 1 decides). The screen passes
  on_layer (its TAP for the drafter, which layer(...) did in modes 1 / 2). R57_PFLM_CC=1: mx.clear_cache() after each layer.
"""
import os
import numpy as np
import mlx.core as mx

LMC = {"on": False, "mode": 1, "t": None, "have": set(), "sc": None, "reads": 0, "hits": 0, "calls": 0, "tier": None}


def switch_glu_padded(m, x, indices, align=64):
    """Leap's /Users/kamen/BigMoE/data/r57/snap/frozen_pfb/Leap_scripts/pfb_fix.py (Sun 08:3x): SwitchGLU's forward (sort, up / gate / down sorted gather, SwiGLU,
    unsort) with the sorted rows padded to a multiple of 64 (copies of the last row, same expert), so MLX 0.32.2's
    affine_gather_qmm_rhs_nax takes its aligned path. Bitwise == the 1,024-token calls (Leap, 1.6K / 4.7K / 7.9K)."""
    from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort
    xe = mx.expand_dims(x, (-2, -3))
    xs, idx, inv = _gather_sort(xe, indices)
    R = idx.shape[0]
    pad = (-R) % align
    if pad:
        xs = mx.concatenate([xs, mx.repeat(xs[-1:], pad, axis=0)], axis=0)
        idx = mx.concatenate([idx, mx.repeat(idx[-1:], pad, axis=0)], axis=0)
    h = m.activation(m.up_proj(xs, idx, sorted_indices=True), m.gate_proj(xs, idx, sorted_indices=True))
    y = m.down_proj(h, idx, sorted_indices=True)
    if pad:
        y = y[:R]
    return _scatter_unsort(y, inv, indices.shape).squeeze(-2)


def _scratch(t):
    """The full-E scratch for layer t's piece shapes (reallocated only when the shapes change) and, on a new layer, no valid
    rows yet. Returns E."""
    E = int(t["E"])
    shp = [((E,) + tuple(sh), dt) for sh, dt in t["shapes4"]]
    if LMC["sc"] is None or [(tuple(s.shape), s.dtype) for s in LMC["sc"]] != shp:
        LMC["sc"] = None
        mx.clear_cache()
        LMC["sc"] = [mx.zeros(sh, dt) for sh, dt in shp]
        mx.eval(*LMC["sc"])
        LMC["t"] = None
    if LMC["t"] is not t:                                # a new layer: no row of the scratch is valid for it yet
        LMC["t"], LMC["have"] = t, set()
    return E


def _read_all(t):
    """PF-B: every expert of layer t not yet in the scratch, read now (CPU) while the GPU runs the queued attention parts."""
    E = _scratch(t)
    need = [e for e in range(E) if e not in LMC["have"]]
    if need:
        LMC["tier"]._read_rebuilt(t, need, need, LMC["sc"])
        LMC["have"].update(need)
        LMC["reads"] += len(need)


def install(tier, mode=1):
    """Wrap bigmoe.tier.prefill_call (stream._sync_call looks it up at call time). Inactive unless LMC['on']."""
    LMC["mode"] = int(mode)
    LMC["tier"] = tier
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
        _scratch(t)
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
            y = switch_glu_padded(self, x, idx) if LMC["mode"] == 3 else cls.__call__(self, x, idx, *a, **kw)
            mx.eval(y)
        finally:
            for (proj, field), arr in zip(PIECES, saved):
                setattr(getattr(self, proj), field, arr)
        if LMC["mode"] == 2:
            del scr
        return y

    tier.prefill_call = prefill_call
    tier._pflm_installed = True


def _split_layers(tm, xs, hs, cache, states, stage, on_layer):
    """R57_PFLM=3 (PF-B): each layer split at its MoE (qwen4_exp DecoderLayer.__call__'s steps verbatim; REF2's patched
    hyper read / attention / write are looked up at call time as there; its _dec_call only acts on 5-16-row passes)."""
    import mtplx.models.qwen4_exp as Q
    tok = int(os.environ.get("R57_PFLM_TOK", "4096") or 4096)
    ahead = os.environ.get("R57_PFLM_AHEAD", "1") != "0"
    cc = os.environ.get("R57_PFLM_CC") == "1"
    groups, cur, n = [], [], 0                           # contiguous chunk groups of <= tok tokens (at least one chunk)
    for k, x in enumerate(xs):
        if cur and n + x.shape[1] > tok:
            groups.append(cur)
            cur, n = [], 0
        cur.append(k)
        n += x.shape[1]
    groups.append(cur)
    for layer in tm.layers:                              # a fully resident layer would run the unpadded stock call
        bm_ = layer.mlp.switch_mlp._bm
        assert "tier" in bm_ and not bm_["full"], "R57_PFLM=3: every layer must take the tier path (none fully resident)"
    for L, (layer, c) in enumerate(zip(tm.layers, cache)):
        t = layer.mlp.switch_mlp._bm["tier"]
        pend = []
        for k, x in enumerate(xs):
            if L == 0:                                   # as _forward: the ssm mask of a B=1 unpadded chunk is None
                assert Q.create_ssm_mask(hs[k], cache[tm.ssm_idx]) is None, "PFLM: an ssm mask (padding?)"
            sync = stage and L == tm._ple_stage_idx      # the PLE stage feeds chunk k's PLE read: consume it before k+1's
            if sync:
                ple = tm.layers[tm._ple_stage_idx].ple
                ple.ple_embedding.stage(x, cache[tm._ple_stage_idx], ple.NGRAM_IDX)
            h = hs[k]
            if "ple" in layer:
                h = h + layer.ple(h, x, c)
            mixed, hyper, inject = layer.attn_hyper_connection(h)
            bo = layer.linear_attn(mixed, None, c) if layer.is_linear else layer.self_attn(mixed, c)
            mixed, hyper, inject = layer.mlp_hyper_connection(hyper, pending=(bo, inject))
            pend.append((mixed, hyper, inject))
            outs = [a for a in (mixed, hyper, inject) if isinstance(a, mx.array)] + list(states([c]))
            (mx.eval if sync else mx.async_eval)(*outs)
        if ahead:
            _read_all(t)                                 # CPU read || the GPU's queued attention parts
        for g in groups:
            M = pend[g[0]][0] if len(g) == 1 else mx.concatenate([pend[k][0] for k in g], axis=1)
            Y = layer.mlp(M)                             # router + shared expert + the padded SwitchGLU (tier wrapper)
            off = 0
            for k in g:
                n_ = pend[k][0].shape[1]
                hs[k] = Q._hyper_residual_write(pend[k][1], Y if len(g) == 1 else Y[:, off:off + n_], pend[k][2])
                off += n_
                if on_layer is not None:
                    on_layer(layer, hs[k])
            mx.eval(*[hs[k] for k in g])
            del M, Y
        del pend
        if cc:
            mx.clear_cache()


def prefill_layer_major(model, ids, chunk, states, on_chunk=None, on_layer=None):
    """The screen's chunk-major prefill, layer-major. Returns (cache, logits of the last chunk's last 256 rows).
    on_chunk(k, h_k) runs after the LAST layer for each chunk k (the widened stream, for the MTP head). on_layer(layer, h_k):
    R57_PFLM=3 only, after each (layer, chunk) step (modes 1 / 2 call layer(...), whose screen wrapper taps itself)."""
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
            if LMC["mode"] == 3:
                _split_layers(tm, xs, hs, cache, states, stage, on_layer)
            for L, (layer, c) in enumerate(zip(tm.layers, cache) if LMC["mode"] != 3 else ()):
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
