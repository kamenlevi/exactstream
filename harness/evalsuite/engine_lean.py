"""Lean layer-major forward (EXPERIMENTAL, not used until `verify_lean` passes on the GPU).

engine.forward spends ~10 ms of host time per (layer, sequence) call on short MC items: bigmoe.stream's
_sync_call_staged pulls the router ids to the host (np.array(indices)) for bookkeeping, and the driver evaluates every
call. With BIGMOE_PF_FULLSCR the call only needs the full-E scratch (every expert of the layer, in id order) and the
expert ids themselves, so here:
  - the first call of layer L waits for L's staged experts, kicks L+1's reads, fills the full scratch once;
  - every sequence's call of layer L is built lazily on that scratch (no host sync);
  - the driver evaluates all sequences of layer L together, before the next layer refills the scratch;
  - the PLE layer stays one sequence at a time (its n-gram staging is shared state).
Expected bit-identical to engine.forward (same scratch contents, same kernels, same expert ids); verify_lean() checks it
on real batches before anything relies on it."""
from __future__ import annotations
from contextlib import contextmanager, nullcontext
import numpy as np
from . import env  # noqa: F401
import mlx.core as mx
from . import model as M
from .engine import STEP, TOPK, assert_safe_chunk


@contextmanager
def lean_staging():
    import bigmoe.stream as S
    if not (S._PF_PREFILL and S._FULLSCR):
        raise RuntimeError("lean staging needs BIGMOE_PF_PREFILL=1 and BIGMOE_PF_FULLSCR=1")
    orig_call, orig_adopt = S._sync_call_staged, S.ext.adopt
    cur = {"L": None}

    def call(cls, mod, x, indices, bm, *a, **kw):
        L = bm["L"]
        if cur["L"] != L:
            S._stage_kick(L, bm["pools"])                    # no-op when L+1's kick during L-1 already queued it
            b, sids = S._STAGE["pending"].pop(L)
            S.ext.stage_wait(b)
            if L + 1 < S._STAGE["nl"]:
                S._stage_kick(L + 1, S._STAGE["layers"][L + 1])   # overlaps the SSD reads with layer L's compute
            if S._STAGE.get("scrbuf") is None:
                E = S._STAGE["E"]
                S._STAGE["scrbuf"] = [mx.zeros((E,) + p.shape[1:], dtype=p.dtype) for p in bm["pools"]]
                mx.eval(*S._STAGE["scrbuf"])
            S.ext.scratch_full(L, b, S._STAGE["scrbuf"], [int(e) for e in sids])
            cur["L"] = L
        saved = [getattr(getattr(mod, p), f) for p, f in S.PIECES]
        for (p, f), arr in zip(S.PIECES, S._STAGE["scrbuf"]):
            setattr(getattr(mod, p), f, arr)
        try:
            return cls.__call__(mod, x, indices, *a, **kw)
        finally:
            for (p, f), arr in zip(S.PIECES, saved):
                setattr(getattr(mod, p), f, arr)

    S._sync_call_staged = call
    S.ext.adopt = lambda *a, **k: None
    try:
        yield
    finally:
        S._sync_call_staged, S.ext.adopt = orig_call, orig_adopt
        S._STAGE["scr"] = None


def forward_lean(rt, items, score, capture_routes=False, head_block=512):
    """same contract as engine.forward"""
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
        assert Q.create_ssm_mask(h, caches[i][tm.ssm_idx]) is None
    local = []
    for i, a, x in jobs:
        p = np.asarray(items[i]["pos"])
        local.append(p[(p >= a) & (p < a + x.shape[1])] - a)
    routes = [np.zeros((len(lp), nl, TOPK), np.uint16) if capture_routes and len(lp) else None for lp in local]
    rope = Q._rope_table_scope() if Q.qwen4_opdiet_enabled("rope") else nullcontext()
    with attention_phase("prefill"), rope, lean_staging():
        for L, layer in enumerate(tm.layers):
            ple = L == tm._ple_stage_idx and Q._COMPILED_VERIFY_PLE.get() is None
            outs, got = [], {}
            for k, (i, a, x) in enumerate(jobs):
                c = caches[i][L]
                if ple:
                    layer.ple.ple_embedding.stage(x, c, layer.ple.NGRAM_IDX)
                M.STATE.capture = {} if routes[k] is not None else None
                hs[k] = layer(hs[k], input_ids=x, ssm_mask=None, cache=c)
                state = [s for s in (c.state if hasattr(c, "state") else []) if isinstance(s, mx.array)]
                if routes[k] is not None and M.STATE.capture.get(L):
                    got[k] = M.STATE.capture[L][0][0, mx.array(local[k])]
                if ple:                                   # shared n-gram staging: one sequence at a time here
                    mx.eval(hs[k], *state, *([got[k]] if k in got else []))
                else:
                    outs += [hs[k], *state] + ([got[k]] if k in got else [])
            if outs:
                mx.eval(*outs)                            # before the next layer refills the scratch
            for k, g in got.items():
                routes[k][:, L] = np.array(g).astype(np.uint16)
            for i, a, _ in jobs:
                if a == last[i]:
                    caches[i][L] = None
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


def verify_lean(rt, items, score, log=print):
    """engine.forward vs forward_lean on the same items: every returned array must be bit-identical"""
    import time
    from . import engine as E
    t = time.time()
    a = E.forward(rt, items, score, capture_routes=True)
    ta = time.time() - t
    t = time.time()
    b = forward_lean(rt, items, score, capture_routes=True)
    tb = time.time() - t
    bad = [(i, n) for i in a for n in a[i] if not np.array_equal(a[i][n], b[i][n])]
    log(f"verify_lean: {len(items)} sequences, standard {ta:.1f}s, lean {tb:.1f}s, mismatches {bad[:5]} ({len(bad)})")
    return not bad and set(a) == set(b)
