"""Fit12 T0 (Mon 2 Nov 2026; Breakthrough ~/Breakthrough/PLAN_12GB_TWOPASS.md section 4, engine side handed to Fit12): a
free-running two-pass greedy decode on B12, the streamed 12 GB engine. NON-EXACT by design: EvalSuite's same-scores test
(decision 4) judges it, and adopting it is Kamen's call.

Per decoded token (one row):
  pass 1: the row with experts at ranks 0-1 only, at the model's own routed weights (B12_NORM=mass: probability / the
          true top-10 mass), ranks 2-9 weighted 0 and never read, no renormalisation; no late ranks (no STASH writes)
  keep:   if max softmax(pass 1's logits, fp32) >= theta, its argmax is the token and pass 1's KV / recurrent state stay
  pass 2: else the row is rolled back (MTPLX rollback_after_verify on an eager snapshot taken before pass 1) and re-run
          exactly (B12's one-row path under ALLWAIT: every rank 0-9 waited); its token and state replace pass 1's
theta > 1: every token from pass 2 = B12's exact one-row AR decode (bitwise gate G1); theta 0: every token from pass 1.

apply(B) patches bt_b12._install in this process (Breakthrough's file is read, never edited); call it BEFORE B.load_b12.
generate(B, model, prompt_ids, max_tokens, theta, stop_ids) -> (tokens, finish, stats). The prompt minus its last token
takes the exact AR prefill (B.b12_prefill, as bt_b12's b12_generate); the last prompt token is the first two-pass row.
"""
import hashlib, inspect, os, time

_P1 = [False]                                          # True while pass 1's graph is built (read in bt_b12's block)
_REPL = [
    ('    FL_ALL = mx.where(R < max(2, B3, CFG["w"]), W20, 0).astype(mx.int32)[None]   # last layer: nothing late, all forced waited\n',
     '    FL_ALL = mx.where(R < max(2, B3, CFG["w"]), W20, 0).astype(mx.int32)[None]   # last layer: nothing late, all forced waited\n'
     '    FL_P1 = mx.where(R < 2, W20, 0).astype(mx.int32)[None]          # [Fit12 T0] pass 1: ranks 0-1 waited, the rest never read\n'),
    ('        if CFG["pre"] and S == 1:\n',
     '        if CFG["pre"] and S == 1 and not _F12_P1[0]:                 # [Fit12 T0] pass 1 never takes the pre-routed pair\n'),
    ('        one = S == 1\n',
     '        one = S == 1\n'
     '        _p1 = one and _F12_P1[0]                                      # [Fit12 T0] pass 1: ranks 0-1 only\n'),
    ('        if one and not last and AW[0]:                                # B12_ALLWAIT (Fit12 CMP3 / AW, 10-26): a single\n',
     '        if _p1:                                                       # [Fit12 T0]\n'
     '            fl = FL_P1\n'
     '        elif one and not last and AW[0]:                              # B12_ALLWAIT (Fit12 CMP3 / AW, 10-26): a single\n'),
    ('        dep_m, n_m = gate.post(L, rows, wk=255 | ((K * S) << 8), **pk)   # flag mode; LRU-touch every row\'s ranks < 10\n',
     '        dep_m, n_m = gate.post(L, rows, wk=(255 | ((2 * S) << 8)) if _p1 else (255 | ((K * S) << 8)), **pk)   # [Fit12 T0] LRU-touch < 2 / < 10\n'),
    ('        forced = (R < NF)[None]                                       # ranks 0-1 (+ waited / late ranks) always counted\n',
     '        forced = (R < 2)[None] if _p1 else (R < NF)[None]             # [Fit12 T0] pass 1 counts ranks 0-1 only\n'),
    ('        sel = (pool & (mx.cumsum(pool.astype(mx.int32), axis=-1) <= K - NF)) | forced   # + first resident of the rest\n',
     '        sel = forced if _p1 else ((pool & (mx.cumsum(pool.astype(mx.int32), axis=-1) <= K - NF)) | forced)   # [Fit12 T0]\n'),
    ('        if CFG["late1"] and not last and one:\n',
     '        if CFG["late1"] and not last and one and not _p1:\n'),
    ('        elif CFG["l3"] and not last and one:                          # B8: cold ranks A3..B3-1 land half a layer late\n',
     '        elif CFG["l3"] and not last and one and not _p1:              # B8: cold ranks A3..B3-1 land half a layer late (not in pass 1)\n'),
]


def apply(B):
    """patch bt_b12._install (before B.load_b12); idempotent per process"""
    if B.__dict__.get("_F12_P1") is _P1:
        return True
    src = inspect.getsource(B._install)
    for old, new in _REPL:
        assert src.count(old) == 1, ("t0_twopass: bt_b12 changed", old[:80])
        src = src.replace(old, new)
    B.__dict__["_F12_P1"] = _P1
    exec(compile(src, B.__file__ + " [t0_twopass]", "exec"), B.__dict__)
    return True


def _logits(out):
    return out[0] if isinstance(out, tuple) else out


def _decode(B, model, prompt_ids, max_tokens, theta, stop_ids, trace):
    import numpy as np
    import mlx.core as mx
    from mtplx.cache_state import rollback_after_verify, snapshot_untrimmable_cache
    from bigmoe_v3f.fuse import reset_generation_state
    reset_generation_state(model)
    B.STASH.clear(); B.PENDH.clear()
    ids = [int(t) for t in prompt_ids]
    cache = model.make_cache()
    t0 = time.perf_counter()
    if len(ids) > 1:
        B.b12_prefill(model, mx.array([ids[:-1]]), cache, 512, 4096)   # exact (decoupled) AR prefill, PFR by its defaults
    t1 = time.perf_counter()
    aw0, B.AW[0] = B.AW[0], True                         # pass 2 = the exact one-row path: every rank waited (ALLWAIT)
    h, y, out, n2, fin = hashlib.sha1(), ids[-1], [], 0, "length"
    try:
        while len(out) < max_tokens:
            x = mx.array([[y]])
            p1, path = -1.0, 2
            if theta is not None:                         # two-pass (theta None: the exact reference, no pass 1)
                snap = snapshot_untrimmable_cache(cache)  # eager copy of the recurrent / non-trimmable state
                _P1[0] = True
                try:
                    lg = _logits(model(x, cache=cache))[0, -1].astype(mx.float32)
                finally:
                    _P1[0] = False
                pm, tok = mx.max(mx.softmax(lg, axis=-1)), mx.argmax(lg)
                mx.eval(pm, tok, lg)
                p1 = float(pm.item())
                if p1 >= theta:
                    path = 1
                else:
                    rollback_after_verify(cache, snap, verified_tokens=1)
                del snap
            if path == 2:
                lg = _logits(model(x, cache=cache))[0, -1].astype(mx.float32)
                tok = mx.argmax(lg)
                mx.eval(tok, lg)
                n2 += 1
            h.update(np.array(lg.view(mx.uint32)).tobytes())   # the chosen pass's logits, bit for bit
            y = int(tok.item())
            out.append(y)
            if trace is not None:
                trace.append((round(p1, 4), path))
            if y in stop_ids:
                fin = "eos"
                break
    finally:
        B.AW[0] = aw0
        _P1[0] = False
    t2 = time.perf_counter()
    return out, fin, dict(steps=len(out), pass2=n2, rate=round(n2 / max(1, len(out)), 4), lsha=h.hexdigest()[:10],
                          pf_s=round(t1 - t0, 3), dec_s=round(t2 - t1, 3))


def generate(B, model, prompt_ids, max_tokens, theta, stop_ids=(248044, 248046), trace=None):
    """two-pass greedy AR decode (theta > 1: every row rolled back and re-run exactly, gate G1). trace (a list) gets
    (p1, path) per token. stats: steps, pass2, rate, lsha (sha1 of the chosen logits, fp32, every step), seconds"""
    return _decode(B, model, prompt_ids, max_tokens, float(theta), stop_ids, trace)


def generate_exact(B, model, prompt_ids, max_tokens, stop_ids=(248044, 248046)):
    """the reference: the same prefill and loop with no pass 1 (B12's exact one-row AR decode under ALLWAIT)"""
    return _decode(B, model, prompt_ids, max_tokens, None, stop_ids, None)
