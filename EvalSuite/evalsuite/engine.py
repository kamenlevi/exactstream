"""Layer-major forward of many independent sequences on BigMoE's SSD-streamed 125B, with per-position scoring.

Prefill-only tests need logits for ~1M positions spread over ~2,000 short sequences. Chunk-major (one sequence at a
time through all 48 layers) re-streams nearly every expert per sequence (~34 GB of SSD reads each). Layer-major
(`for layer: for sequence chunk:`) stages each layer's experts once per batch: MoonShots' lmp driver (bit-identical to
chunk-major prefill, ~/MoonShots/patches/layer_major_prefill/lmp.py) generalized from the chunks of one prompt to
the chunks of many prompts, each with its own cache. Every (layer, chunk) step sees exactly the inputs it would see in
MTPLX's chunk-major prefill (attention_phase "prefill", MTPLX's prefill cache, PLE staging per chunk).

MLX 0.32.2 gather bug (PrefillKernels N3): gather_qmm_rhs_nax is wrong at bm 64 with routed rows not a multiple of
64; bm 64 engages at rows / E >= 64. Chunks are <= 2048 tokens against the full 512-expert scratch, so rows / E <= 40
(bm 32, correct). assert_safe_chunk() guards it."""
from __future__ import annotations
from contextlib import contextmanager, nullcontext
import numpy as np
from . import env  # noqa: F401
import mlx.core as mx
from . import model as M

STEP = 2048
TOPK = 10


@contextmanager
def sticky_staging():
    """Vendored from MoonShots lmp.sticky_staging: PF1 normally consumes layer L's staging buffer after one call;
    keep it for all chunks of L, and hold off ext.adopt so the resident set the staging was computed against cannot
    change mid-layer. Runtime monkeypatch of bigmoe.stream (nothing in ~/BigMoE is edited)."""
    import bigmoe.stream as S
    if not S._PF_PREFILL:
        raise RuntimeError("layer-major forward needs BIGMOE_PF_PREFILL=1 (evalsuite.env sets it)")
    orig_call, orig_adopt = S._sync_call_staged, S.ext.adopt
    saved = {}

    def call(cls, mod, x, indices, bm, *a, **kw):
        L = bm["L"]
        if L in saved:
            S._STAGE["pending"][L] = saved[L]
        else:
            S._stage_kick(L, bm["pools"])
            saved[L] = S._STAGE["pending"][L]
            saved.pop(L - 2, None)
        return orig_call(cls, mod, x, indices, bm, *a, **kw)

    S._sync_call_staged = call
    S.ext.adopt = lambda *a, **k: None
    try:
        yield
    finally:
        S._sync_call_staged, S.ext.adopt = orig_call, orig_adopt


def assert_safe_chunk(tokens, experts=512, k=TOPK):
    if tokens * k / experts >= 64 and (tokens * k) % 64:
        raise RuntimeError(f"chunk of {tokens} tokens would hit MLX 0.32.2's bm-64 gather bug")


def batches(items, max_tokens=24576):
    """group item indices into forward batches: short items packed up to max_tokens, long ones alone"""
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    out, cur, n = [], [], 0
    for i in order:
        t = len(items[i]["ids"])
        if cur and (n + t > max_tokens or t > STEP):
            out.append(cur); cur, n = [], 0
        cur.append(i); n += t
    if cur:
        out.append(cur)
    return out


def forward(rt, items, score, capture_routes=False, head_block=512):
    """items: dicts with 'ids' (token list) and 'pos' (sorted int array of positions whose next-token distribution is
    scored, each < len(ids)). score(i, pos_block, logits_block[n, V]) -> dict of lazy mx arrays (per position).
    Returns {i: {name: np.ndarray}} concatenated over the item's scored positions (+ 'routes' [n, 48, 10] uint16 when
    capture_routes)."""
    import mtplx.models.qwen4_exp as Q
    from mtplx.attention_context import attention_phase
    from mtplx import generation as G
    model = rt.model if hasattr(rt, "model") else rt
    text = model.language_model if hasattr(model, "language_model") else model
    tm = text.model
    nl = len(tm.layers)
    M.STATE.pass_id += 1
    jobs, caches = [], []
    for i, it in enumerate(items):
        ids = list(it["ids"])
        caches.append(G._make_target_prefill_cache(rt))
        arr = mx.array([ids])
        for a in range(0, len(ids), STEP):
            assert_safe_chunk(min(STEP, len(ids) - a))
            jobs.append((i, a, arr[:, a:a + STEP]))
    last = {}
    for i, a, _ in jobs:
        last[i] = max(last.get(i, 0), a)
    hs = [mx.tile(tm.embed_tokens(x), (1, 1, tm.args.hc_count)) for _, _, x in jobs]
    for (i, _, _), h in zip(jobs, hs):
        assert Q.create_ssm_mask(h, caches[i][tm.ssm_idx]) is None, "padded/batched caches are not supported"
    local = []                                            # scored positions inside each chunk (chunk-relative)
    for i, a, x in jobs:
        p = np.asarray(items[i]["pos"])
        local.append(p[(p >= a) & (p < a + x.shape[1])] - a)
    routes = [np.zeros((len(lp), nl, TOPK), np.uint16) if capture_routes and len(lp) else None for lp in local]
    rope = Q._rope_table_scope() if Q.qwen4_opdiet_enabled("rope") else nullcontext()
    with attention_phase("prefill"), rope, sticky_staging():
        for L, layer in enumerate(tm.layers):
            for k, (i, a, x) in enumerate(jobs):
                c = caches[i][L]
                if L == tm._ple_stage_idx and Q._COMPILED_VERIFY_PLE.get() is None:
                    layer.ple.ple_embedding.stage(x, c, layer.ple.NGRAM_IDX)
                M.STATE.capture = {} if routes[k] is not None else None
                hs[k] = layer(hs[k], input_ids=x, ssm_mask=None, cache=c)
                state = [s for s in (c.state if hasattr(c, "state") else []) if isinstance(s, mx.array)]
                got = None
                if routes[k] is not None:
                    got = M.STATE.capture.get(L)
                    got = got[0][0, mx.array(local[k])] if got else None
                mx.eval(hs[k], *state, *([got] if got is not None else []))
                if got is not None:
                    routes[k][:, L] = np.array(got).astype(np.uint16)
                if a == last[i]:
                    caches[i][L] = None                   # no later chunk of this sequence needs layer L's cache
            mx.clear_cache()
    M.STATE.capture = None
    res = {}
    for k, (i, a, x) in enumerate(jobs):
        lp = local[k]
        h, hs[k] = hs[k], None
        if not len(lp):
            continue
        out = tm.hyper_connection_mixer(h)
        for b0 in range(0, len(lp), head_block):
            sel = lp[b0:b0 + head_block]
            logits = text.lm_head(out[:, mx.array(sel)])[0]
            r = score(i, sel + a, logits)
            mx.eval(r)
            for name, v in r.items():
                res.setdefault(i, {}).setdefault(name, []).append(np.array(v))
        if routes[k] is not None:
            res.setdefault(i, {}).setdefault("routes", []).append(routes[k])
        del out, h
        mx.clear_cache()
    return {i: {n: np.concatenate(v) for n, v in d.items()} for i, d in res.items()}


# ---------------------------------------------------------------------------------------------- scoring heads
def logsoftmax(logits):
    x = logits.astype(mx.float32)
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def ref_head(K):
    """reference: sorted top-K log-probs (ids, fp16), entropy, target log-prob"""
    def f(targets):
        def score(i, pos, logits):
            lp = logsoftmax(logits)
            ids = mx.argpartition(-lp, kth=K - 1, axis=-1)[:, :K]
            v = mx.take_along_axis(lp, ids, axis=-1)
            o = mx.argsort(-v, axis=-1)
            ids, v = mx.take_along_axis(ids, o, axis=-1), mx.take_along_axis(v, o, axis=-1)
            ent = -(mx.exp(lp) * lp).sum(-1)
            tgt = mx.take_along_axis(lp, mx.array(targets(i, pos))[:, None], axis=-1)[:, 0]
            return {"ids": ids.astype(mx.int32), "lp": v.astype(mx.float16), "ent": ent, "tgt": tgt}
        return score
    return f


def kld_vs_ref(lq, ref_ids, ref_lp):
    """KL(P_ref || Q) from the reference's top-K and Q's full distribution; the reference's tail (outside its top-K) is
    one bucket (exact when the tail mass is 0, a lower bound otherwise; the same for every variant)."""
    lpr = ref_lp.astype(mx.float32)
    pr = mx.exp(lpr)
    lqr = mx.take_along_axis(lq, ref_ids, axis=-1)
    head = (pr * (lpr - lqr)).sum(-1)
    p_rest = mx.maximum(1.0 - pr.sum(-1), 0.0)
    q_rest = mx.maximum(1.0 - mx.exp(lqr).sum(-1), 1e-12)
    tail = mx.where(p_rest > 1e-9, p_rest * (mx.log(mx.maximum(p_rest, 1e-30)) - mx.log(q_rest)), 0.0)
    return head + tail, p_rest


def var_head(ref, targets, K=32):
    """variant: KLD vs the cached reference, top-1 agreement, target log-prob, own top-K"""
    def score(i, pos, logits):
        lq = logsoftmax(logits)
        r = ref[i]
        sel = np.searchsorted(r["pos"], pos)
        rid, rlp = mx.array(r["ids"][sel]), mx.array(r["lp"][sel])
        kld, p_rest = kld_vs_ref(lq, rid, rlp)
        top1 = mx.argmax(lq, axis=-1)
        ids = mx.argpartition(-lq, kth=K - 1, axis=-1)[:, :K]
        v = mx.take_along_axis(lq, ids, axis=-1)
        o = mx.argsort(-v, axis=-1)
        tgt = mx.take_along_axis(lq, mx.array(targets(i, pos))[:, None], axis=-1)[:, 0]
        return {"kld": kld, "p_rest": p_rest, "agree": (top1 == rid[:, 0]).astype(mx.uint8),
                "top1": top1.astype(mx.int32), "tgt": tgt, "ent": -(mx.exp(lq) * lq).sum(-1),
                "ids": mx.take_along_axis(ids, o, axis=-1).astype(mx.int32),
                "lp": mx.take_along_axis(v, o, axis=-1).astype(mx.float16)}
    return score


def letter_head(letter_ids):
    """multiple choice: full-vocab log-probs of the option letters at the answer position"""
    li = mx.array(letter_ids)

    def score(i, pos, logits):
        lp = logsoftmax(logits)
        return {"letters": lp[:, li], "top1": mx.argmax(lp, axis=-1).astype(mx.int32)}
    return score
