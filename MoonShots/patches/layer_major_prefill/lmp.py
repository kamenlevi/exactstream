"""Layer-major prefill for BigMoE's SSD-streamed MoE (PF2), built on BigMoE's PF1 staging (BIGMOE_PF_PREFILL=1).

Chunk-major prefill (mlx-lm generate_step / bigmoe.decode.greedy: 2,048-token chunks, each through all layers) makes every chunk
stage ~all of a layer's non-resident experts again. Layer-major runs `for layer: for chunk:`, so each layer's experts are staged
once per prompt. Every (layer, chunk) step gets the same inputs as in chunk-major order: the chunk's own hidden state, and the
layer's KV/GDN cache after chunks 0..c-1. With the same chunk boundaries, the same kernels and the same masks, caches and
logits are bit-identical (tested by bench_lmp35.py).

Pieces:
  sticky_staging()  context manager: PF1 normally consumes layer L's staging buffer after one chunk. Here it is kept for all
                    chunks of L, and `ext.adopt` (moving staged experts into the LRU pool) is held off during prefill, so the
                    resident set the staged set was computed against cannot change mid-layer. Runtime monkeypatch of
                    bigmoe.stream; nothing in ~/BigMoE is edited.
  prefill_mlxlm()   driver for mlx-lm Qwen3.5/3.6 MoE text models (the 35B test bed), mirroring generate_step's prefill loop.
  prefill_qwen4()   driver for MTPLX qwen4_exp (Qwen3.8-Flash-Next 125B), mirroring bigmoe.decode.greedy's prefill.
  greedy_lmp()      bigmoe.decode.greedy with the layer-major prefill (one-command A/B on the 125B)."""
from __future__ import annotations
import os
from contextlib import contextmanager
from types import SimpleNamespace
import mlx.core as mx


@contextmanager
def sticky_staging():
    import bigmoe.stream as S
    if not S._PF_PREFILL:
        raise RuntimeError("layer-major prefill needs BigMoE's PF1 staging: set BIGMOE_PF_PREFILL=1 before importing bigmoe")
    orig_call, orig_adopt = S._sync_call_staged, S.ext.adopt
    saved = {}

    def call(cls, mod, x, indices, bm, *a, **kw):
        L = bm["L"]
        if L in saved:                                   # later chunk of the same layer: reuse its staged buffer
            S._STAGE["pending"][L] = saved[L]
        else:                                            # first chunk of layer L
            S._stage_kick(L, bm["pools"])                # normally already staged during L-1
            saved[L] = S._STAGE["pending"][L]
            saved.pop(L - 2, None)                       # its buffer is the one L reuses
        return orig_call(cls, mod, x, indices, bm, *a, **kw)

    S._sync_call_staged = call
    S.ext.adopt = lambda *a, **k: None
    try:
        yield
    finally:
        S._sync_call_staged, S.ext.adopt = orig_call, orig_adopt


def _spans(n, step):
    """chunk boundaries of mlx-lm's generate_step / greedy: all but the last token, `step` at a time."""
    out, s = [], 0
    while n - s > 1:
        k = min(step, n - s - 1)
        out.append((s, s + k))
        s += k
    return out


def prefill_mlxlm(model, prompt, cache, step=2048):
    """mlx-lm Qwen3.5/3.6 MoE: fill `cache` with all but the last prompt token, layer-major. Returns the remaining tokens
    (pass them to generate_step with prompt_cache=cache)."""
    tm = model.language_model.model
    spans = _spans(prompt.size, step)
    hs = [tm.embed_tokens(prompt[a:b][None]) for a, b in spans]
    fa_c, ssm_c = cache[tm.fa_idx], cache[tm.ssm_idx]
    # masks exactly as the reference builds them at each chunk's start (KVCache.make_mask uses only its offset)
    fa_masks = [type(fa_c).make_mask(SimpleNamespace(offset=fa_c.offset + a), b - a, return_array=False, window_size=None)
                for a, b in spans]
    ssm_masks = [ssm_c.make_mask(b - a) if hasattr(ssm_c, "make_mask") else None for a, b in spans]
    with sticky_staging():
        for L, layer in enumerate(tm.layers):
            c = cache[L]
            for i in range(len(spans)):
                hs[i] = layer(hs[i], mask=ssm_masks[i] if layer.is_linear else fa_masks[i], cache=c)
                mx.eval(hs[i], c.state)
            mx.clear_cache()
    return prompt[spans[-1][1]:] if spans else prompt


def prefill_qwen4(model, prompt_ids, cache, step=2048):
    """MTPLX qwen4_exp: fill `cache` with all but the last prompt token, layer-major, mirroring bigmoe.decode.greedy's
    `model(ids[:, i:i + step], cache=cache)` loop (whose vocab-head output is discarded)."""
    import mtplx.models.qwen4_exp as Q
    tm = model.language_model.model                     # Qwen4ExpTextModel
    ids = mx.array([list(prompt_ids)[:-1]])
    spans = [(i, min(i + step, ids.shape[1])) for i in range(0, ids.shape[1], step)]
    xs = [ids[:, a:b] for a, b in spans]
    hs = [mx.tile(tm.embed_tokens(x), (1, 1, tm.args.hc_count)) for x in xs]
    ssm_masks = [Q.create_ssm_mask(h, cache[tm.ssm_idx]) for h in hs]
    assert all(m is None for m in ssm_masks), "padded/batched caches are not supported by this driver"
    scope = Q._rope_table_scope() if Q.qwen4_opdiet_enabled("rope") else None
    if scope is not None:
        scope.__enter__()
    try:
        with sticky_staging():
            for L, (layer, c) in enumerate(zip(tm.layers, cache)):
                for i, x in enumerate(xs):
                    if L == tm._ple_stage_idx and Q._COMPILED_VERIFY_PLE.get() is None:
                        ple = layer.ple                     # the n-gram stage belongs to this chunk's step at the PLE layer
                        ple.ple_embedding.stage(x, c, ple.NGRAM_IDX)
                    hs[i] = layer(hs[i], input_ids=x, ssm_mask=ssm_masks[i], cache=c)
                    mx.eval(hs[i], *[a for a in (c.state if hasattr(c, "state") else []) if isinstance(a, mx.array)])
                mx.clear_cache()
    finally:
        if scope is not None:
            scope.__exit__(None, None, None)
    tm._last_widened = hs[-1]


def greedy_lmp(model, prompt_ids, max_tokens, on_token=None, prefill_chunk=2048):
    """bigmoe.decode.greedy with layer-major prefill; the decode loop is unchanged."""
    cache = model.make_cache()
    prefill_qwen4(model, prompt_ids, cache, step=prefill_chunk)
    y = mx.array(int(prompt_ids[-1]))
    first, out = True, []
    for _ in range(max_tokens):
        nxt = model(y.reshape(1, 1), cache=cache)
        nxt = nxt[0] if isinstance(nxt, tuple) else nxt
        y2 = mx.argmax(nxt[0, -1].astype(mx.float32))
        mx.async_eval(y2)
        if not first:
            tok = int(y.item()); out.append(tok)
            if on_token: on_token(tok)
        first = False
        y = y2
    out.append(int(y.item()))
    if on_token: on_token(out[-1])
    return out[:max_tokens]


# ------------------------------------------------------------------------------------------------------------------ PF3
# PF3: inside layer-major prefill, attention still runs chunk by chunk (causal state), but each layer's MoE runs once over
# groups of consecutive chunks up to MOE_TOKENS tokens (bigger expert GEMMs: 1.6-1.75x faster MoE per token at 8-16K in
# layered_prefill.py). Token-wise MoE, so the math is the same; whether it is bit-identical to PF2 depends on MLX's kernels
# being row-invariant at M >= 2048 (checked by bench_lmp35.py pf3 against lmp).
import os as _os
MOE_TOKENS = int(_os.environ.get("PF3_MOE_TOKENS", "8192"))
PF3_ASYNC = _os.environ.get("PF3_ASYNC", "1") == "1"   # attention phase: async_eval per chunk (host builds the next chunk meanwhile)


def _groups(spans, budget):
    out, cur, n = [], [], 0
    for i, (a, b) in enumerate(spans):
        if cur and n + (b - a) > budget:
            out.append(cur); cur, n = [], 0
        cur.append(i); n += b - a
    if cur: out.append(cur)
    return out


def prefill_mlxlm_pf3(model, prompt, cache, step=2048, moe_tokens=None):
    """prefill_mlxlm with the MoE of each layer batched over chunk groups (mlx-lm Qwen3.5/3.6 decoder layer:
    h = x + attn(norm1(x)); out = h + mlp(norm2(h)))."""
    tm = model.language_model.model
    spans = _spans(prompt.size, step)
    hs = [tm.embed_tokens(prompt[a:b][None]) for a, b in spans]
    fa_c, ssm_c = cache[tm.fa_idx], cache[tm.ssm_idx]
    fa_masks = [type(fa_c).make_mask(SimpleNamespace(offset=fa_c.offset + a), b - a, return_array=False, window_size=None)
                for a, b in spans]
    ssm_masks = [ssm_c.make_mask(b - a) if hasattr(ssm_c, "make_mask") else None for a, b in spans]
    groups = _groups(spans, moe_tokens or MOE_TOKENS)
    with sticky_staging():
        for L, layer in enumerate(tm.layers):
            c = cache[L]
            for i in range(len(spans)):                  # attention, chunk by chunk (the cache advances in order)
                f = layer.linear_attn if layer.is_linear else layer.self_attn
                hs[i] = hs[i] + f(layer.input_layernorm(hs[i]), ssm_masks[i] if layer.is_linear else fa_masks[i], c)
                (mx.async_eval if PF3_ASYNC else mx.eval)(hs[i], c.state)
            for g in groups:                             # MoE over a group of chunks at once
                cat = mx.concatenate([hs[i] for i in g], axis=1) if len(g) > 1 else hs[g[0]]
                out = cat + layer.mlp(layer.post_attention_layernorm(cat))
                o = 0
                for i in g:
                    n = spans[i][1] - spans[i][0]
                    hs[i] = out[:, o:o + n]; o += n
                mx.eval([hs[i] for i in g])
            mx.clear_cache()
    return prompt[spans[-1][1]:] if spans else prompt


def prefill_qwen4_pf3(model, prompt_ids, cache, step=2048, moe_tokens=None):
    """prefill_qwen4 with PF3: per layer, PLE + attention hyper-connection + attention + MLP hyper-connection read per chunk,
    then the MoE block once per chunk group, then each chunk's hyper residual write (DecoderLayer.__call__, split)."""
    import mtplx.models.qwen4_exp as Q
    tm = model.language_model.model
    ids = mx.array([list(prompt_ids)[:-1]])
    spans = [(i, min(i + step, ids.shape[1])) for i in range(0, ids.shape[1], step)]
    xs = [ids[:, a:b] for a, b in spans]
    hs = [mx.tile(tm.embed_tokens(x), (1, 1, tm.args.hc_count)) for x in xs]
    ssm_masks = [Q.create_ssm_mask(h, cache[tm.ssm_idx]) for h in hs]
    assert all(m is None for m in ssm_masks), "padded/batched caches are not supported by this driver"
    groups = _groups(spans, moe_tokens or MOE_TOKENS)
    scope = Q._rope_table_scope() if Q.qwen4_opdiet_enabled("rope") else None
    if scope is not None:
        scope.__enter__()
    try:
        with sticky_staging():
            for L, (layer, c) in enumerate(zip(tm.layers, cache)):
                pend = [None] * len(spans)
                for i, x in enumerate(xs):
                    if L == tm._ple_stage_idx and Q._COMPILED_VERIFY_PLE.get() is None:
                        ple = layer.ple
                        ple.ple_embedding.stage(x, c, ple.NGRAM_IDX)
                    h = hs[i]
                    if "ple" in layer:
                        h = h + layer.ple(h, x, c)
                    mixed, hyper, inject = layer.attn_hyper_connection(h)
                    blk = layer.linear_attn(mixed, ssm_masks[i], c) if layer.is_linear else layer.self_attn(mixed, c)
                    mixed, hyper, inject = layer.mlp_hyper_connection(hyper, pending=(blk, inject))
                    pend[i] = (mixed, hyper, inject)
                    (mx.async_eval if PF3_ASYNC else mx.eval)(
                        mixed, hyper, *[a for a in (inject if isinstance(inject, (list, tuple)) else [inject]) if isinstance(a, mx.array)],
                        *[a for a in (c.state if hasattr(c, "state") else []) if isinstance(a, mx.array)])
                for g in groups:
                    cat = mx.concatenate([pend[i][0] for i in g], axis=1) if len(g) > 1 else pend[g[0]][0]
                    out = layer.mlp(cat)
                    o = 0
                    for i in g:
                        n = spans[i][1] - spans[i][0]
                        hs[i] = Q._hyper_residual_write(pend[i][1], out[:, o:o + n], pend[i][2]); o += n
                    mx.eval([hs[i] for i in g])
                pend = None
                mx.clear_cache()
    finally:
        if scope is not None:
            scope.__exit__(None, None, None)
    tm._last_widened = hs[-1]


def greedy_pf3(model, prompt_ids, max_tokens, on_token=None, prefill_chunk=2048):
    """greedy_lmp with PF3 prefill."""
    global prefill_qwen4
    keep = prefill_qwen4
    prefill_qwen4 = prefill_qwen4_pf3
    try:
        return greedy_lmp(model, prompt_ids, max_tokens, on_token=on_token, prefill_chunk=prefill_chunk)
    finally:
        prefill_qwen4 = keep
