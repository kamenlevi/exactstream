"""Fit12 T1 + T2 (Mon 2 Nov 2026; Breakthrough ~/Breakthrough/PLAN_12GB_TWOPASS.md sections 3-4, engine side handed to Fit12):
the top-2 two-pass check on B12 at 12 GB, drafter = the MTP head B12 ships (spec125, k 3, 4-row checks). FAST track (non-exact
by design; EvalSuite's same-scores test judges it, Kamen adopts). T0's file (t0_twopass.py, wired into EvalSuite) is untouched.

T1: pass 1 = the check's rows (1 to 4, any width) with experts at ranks 0-1 only, at the model's own routed weights
    (B12_NORM=mass, ranks 2-9 weight 0 and never read, no renormalisation, no late ranks). apply(B) patches bt_b12._install
    in-process (Breakthrough's file is read, never edited); call it before B.load_b12.
T2: per check (spec125.generate's loop: MTP drafts d, rows [y] + d):
    1. pass 1 on all rows (verify capture scope, lazy snapshot as spec125)
    2. certify row i: max softmax(pass 1's logits) >= theta; walk on the certified rows' argmax: n = the first row that is
       uncertified or misses its draft
    3. row n certified (a miss, or the last row): commit pass 1's rows 0..n (keep n + 1), tokens d[:n] + [argmax_n]
    4. row n uncertified: keep pass 1's rows 0..n-1 (commit keep n; n = 0: roll the window back), then pass 2 = the exact
       check of rows n.. (spec125's own verify on toks[n:], all 10 ranks), walked and committed as spec125 does
    KV: certified rows keep pass 1's entries, pass-2 rows their own; the MTP head gets the matching hidden rows.
theta > 1: every check rolls back at row 0 and runs spec125's exact check (gate G1: bitwise = the exact loop);
theta 0: no pass 2 (every row certified). generate(..., theta=None) is the exact reference loop (no pass 1).
The process env is the caller's (the suite's MTP env); the repetition guard must be off (BIGMOE_LOOP_GUARD=0 in the suite).
"""
import contextlib, hashlib, inspect, os, time

_P1 = [False]
_REPL = [
    ('    FL_ALL = mx.where(R < max(2, B3, CFG["w"]), W20, 0).astype(mx.int32)[None]   # last layer: nothing late, all forced waited\n',
     '    FL_ALL = mx.where(R < max(2, B3, CFG["w"]), W20, 0).astype(mx.int32)[None]   # last layer: nothing late, all forced waited\n'
     '    FL_P1 = mx.where(R < 2, W20, 0).astype(mx.int32)[None]          # [Fit12 T1] pass 1: ranks 0-1 waited, the rest never read\n'),
    ('        if CFG["pre"] and S == 1:\n',
     '        if CFG["pre"] and S == 1 and not _F12_T1[0]:                 # [Fit12 T1] pass 1 never takes the pre-routed pair\n'),
    ('        one = S == 1\n',
     '        one = S == 1\n'
     '        _p1 = bool(_F12_T1[0])                                        # [Fit12 T1] pass 1 (any check width): ranks 0-1 only\n'),
    ('        if one and not last and AW[0]:                                # B12_ALLWAIT (Fit12 CMP3 / AW, 10-26): a single\n',
     '        if _p1:                                                       # [Fit12 T1]\n'
     '            fl = FL_P1\n'
     '        elif one and not last and AW[0]:                              # B12_ALLWAIT (Fit12 CMP3 / AW, 10-26): a single\n'),
    ('        dep_m, n_m = gate.post(L, rows, wk=255 | ((K * S) << 8), **pk)   # flag mode; LRU-touch every row\'s ranks < 10\n',
     '        dep_m, n_m = gate.post(L, rows, wk=(255 | ((2 * S) << 8)) if _p1 else (255 | ((K * S) << 8)), **pk)   # [Fit12 T1] LRU-touch < 2 / < 10\n'),
    ('        forced = (R < NF)[None]                                       # ranks 0-1 (+ waited / late ranks) always counted\n',
     '        forced = (R < 2)[None] if _p1 else (R < NF)[None]             # [Fit12 T1] pass 1 counts ranks 0-1 only\n'),
    ('        sel = (pool & (mx.cumsum(pool.astype(mx.int32), axis=-1) <= K - NF)) | forced   # + first resident of the rest\n',
     '        sel = forced if _p1 else ((pool & (mx.cumsum(pool.astype(mx.int32), axis=-1) <= K - NF)) | forced)   # [Fit12 T1]\n'),
    ('        if CFG["late1"] and not last and one:\n',
     '        if CFG["late1"] and not last and one and not _p1:\n'),
    ('        elif CFG["l3"] and not last and one:                          # B8: cold ranks A3..B3-1 land half a layer late\n',
     '        elif CFG["l3"] and not last and one and not _p1:              # B8: cold ranks A3..B3-1 land half a layer late (not in pass 1)\n'),
]


def apply(B):
    """patch bt_b12._install (before B.load_b12); idempotent per process; not together with t0_twopass.apply"""
    if B.__dict__.get("_F12_T1") is _P1:
        return True
    assert "_F12_P1" not in B.__dict__, "t0_twopass is applied in this process"
    src = inspect.getsource(B._install)
    for old, new in _REPL:
        assert src.count(old) == 1, ("t2_twopass: bt_b12 changed", old[:80])
        src = src.replace(old, new)
    B.__dict__["_F12_T1"] = _P1
    exec(compile(src, B.__file__ + " [t2_twopass]", "exec"), B.__dict__)
    return True


def _split(out):
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, None


def generate(B, model, prompt_ids, max_tokens, theta, prefilled, stop_ids=(248044, 248046), k_mtp=3, max_draft=3,
             stats=None, log=None, skip=None):
    """spec125.generate (lookup off, early off, no width / adapt, guard off) with the two-pass check (theta: float) or
    without it (theta None: the exact reference loop). log (a list) gets one record per check: (S, path, n, m, new).
    stats: passes, pass2 (checks that ran pass 2), rows1 / rows2 (rows run in pass 1 / 2), lsha (sha1 of the logits that
    decided every committed token, fp32), tok / pass, seconds, skipped (checks run exactly by the skip rule).
    skip=(k, j) (Mon 22:5x): after k checks in a row that needed pass 2, the next j checks skip pass 1 (exact checks), then
    pass 1 is tried again; None = off (the gates' semantics)"""
    import numpy as np
    import mlx.core as mx
    import spec125 as SP
    from mtplx.cache_state import rollback_after_verify, snapshot_untrimmable_cache, snapshot_untrimmable_cache_lazy
    from mtplx.graphbank import ensure_eager_window_capacity
    assert os.environ.get("BIGMOE_LOOP_GUARD", "1") != "1", "T2 mirrors spec125 with the repetition guard off"
    if os.environ.get("BIGMOE_RESET_GEN", "1") == "1":
        try:
            from bigmoe.fuse import reset_generation_state
            reset_generation_state(model)
        except ImportError:
            pass
    if os.environ.get("BIGMOE_FRSPEC") == "1":       # = spec125.generate: the FR-Spec draft head rebuilt with the prompt's ids
        import sys
        if os.path.expanduser("~/Scout") not in sys.path:
            sys.path.append(os.path.expanduser("~/Scout"))
        from patches.frspec4 import FRSpec
        FRSpec(model, extra_ids=[int(t) for t in prompt_ids]).on()
    st = stats if stats is not None else {}
    st.update(passes=0, pass2=0, rows1=0, rows2=0, refused=0, skipped=0)
    streak, skip_left = 0, 0
    prompt_ids = [int(t) for t in prompt_ids]
    cache, head = SP.restore_state(model, prefilled, with_head=True)   # the fold's restore MOVES a marked state
    aw0 = B.AW[0]
    if hasattr(B, "_fs") and B._fs("AW"):
        B.AW[0] = True                                 # = the fold's generate wrapper: ALLWAIT for single-row passes
    h, y, out = hashlib.sha1(), prompt_ids[-1], []
    t0 = time.perf_counter()

    def forward(toks_, p1):
        """one verify forward of toks_ (pass 1 if p1); returns (logits [S, V] fp32, hidden, snap)"""
        S_ = len(toks_)
        snap_ = None
        if S_ > 1:
            snap_ = snapshot_untrimmable_cache_lazy(cache)
            ensure_eager_window_capacity(cache, S_)
        elif p1:
            snap_ = snapshot_untrimmable_cache(cache)  # a one-row pass 1 may be undone (T0's way)
        scope = model.verify_capture_scope() if S_ > 1 else contextlib.nullcontext()
        _P1[0] = p1
        try:
            with scope:
                lg_, hid_ = _split(model(mx.array([toks_]), cache=cache, return_hidden=True))
        finally:
            _P1[0] = False
        return lg_[0].astype(mx.float32), hid_, snap_

    def commit(toks_, snap_, keep, p1):
        """keep the first `keep` rows of the window just run (spec125's commit, its repair on a refusal)"""
        S_ = len(toks_)
        if S_ == 1:
            return None
        hid_ = None
        ok = model.commit_verified_window(cache, snap_.states, keep_tokens=keep, verified_tokens=S_)
        if not ok:
            st["refused"] += 1
            rollback_after_verify(cache, snap_, verified_tokens=S_)
            _P1[0] = p1
            try:
                _, hid_ = _split(model(mx.array([toks_[:keep]]), cache=cache, return_hidden=True))
            finally:
                _P1[0] = False
            mx.eval(hid_)
        model.clear_verify_capture(cache)
        if SP._COMMIT_ASYNC:
            mx.async_eval(*SP._COMMIT_ASYNC(cache))
        return hid_

    try:
        while len(out) < max_tokens:
            d = head.draft(k_mtp, 0.0, None) if head is not None else []
            d = d[:max_draft][: max(0, max_tokens - len(out) - 1)]
            toks = [y] + d
            S = len(toks)
            use_p1 = theta is not None and skip_left == 0
            if theta is not None and not use_p1:
                skip_left -= 1
                st["skipped"] += 1
            if use_p1:                                 # ---- pass 1 (top-2) on every row of the check
                lg1, hid1, snap1 = forward(toks, True)
                p1 = mx.max(mx.softmax(lg1, axis=-1), axis=-1)
                g1 = mx.argmax(lg1, axis=-1)
                mx.eval(p1, g1, hid1)
                p1, g1 = p1.tolist(), g1.tolist()
                st["rows1"] += S
                n = 0
                while n < len(d) and p1[n] >= theta and d[n] == g1[n]:
                    n += 1
                if p1[n] >= theta:                     # rows 0..n certified: pass 1 decides the whole check
                    new = d[:n] + [g1[n]]
                    rh = commit(toks, snap1, n + 1, True)
                    hid = rh if rh is not None else hid1
                    h.update(np.array(lg1[: n + 1].view(mx.uint32)).tobytes())
                    if log is not None:
                        log.append((S, 1, n, -1, len(new)))
                    path_lg = None
                else:                                  # row n uncertified: keep rows 0..n-1, pass 2 from row n
                    if n == 0:
                        rollback_after_verify(cache, snap1, verified_tokens=S)
                        if S > 1:
                            model.clear_verify_capture(cache)
                        hid_keep = None
                    else:
                        rh = commit(toks, snap1, n, True)
                        hid_keep = (rh if rh is not None else hid1)[:, :n]
                        h.update(np.array(lg1[:n].view(mx.uint32)).tobytes())
                    del snap1
                    path_lg = (n, hid_keep)
            else:
                path_lg = (0, None)
            if path_lg is not None:                    # ---- pass 2 / the exact check (spec125's verify) on toks[n:]
                n, hid_keep = path_lg
                toks2, d2 = toks[n:], d[n:]
                S2 = len(toks2)
                lg2, hid2, snap2 = forward(toks2, False)
                g2 = mx.argmax(lg2, axis=-1)
                mx.eval(g2, hid2)
                g2 = g2.tolist()
                m = 0
                while m < len(d2) and d2[m] == g2[m]:
                    m += 1
                new = d[:n] + d2[:m] + [g2[m]]
                rh = commit(toks2, snap2, m + 1, False)
                hid2 = rh if rh is not None else hid2
                h.update(np.array(lg2[: m + 1].view(mx.uint32)).tobytes())
                hid = hid2[:, : m + 1] if hid_keep is None else mx.concatenate([hid_keep, hid2[:, : m + 1]], axis=1)
                st["pass2"] += use_p1
                st["rows2"] += S2
                if log is not None:
                    log.append((S, 2 if use_p1 else 0, n, m, len(new)))
                if use_p1 and skip is not None:
                    streak += 1
                    if streak >= skip[0]:
                        streak, skip_left = 0, skip[1]
            elif skip is not None:
                streak = 0
            if head is not None:                       # = spec125 (early off): the head appends after every check
                head.append(hid[:, : len(new)], new)
            st["passes"] += 1
            c0 = len(out)
            for t in new:
                if len(out) < max_tokens:
                    out.append(t)
            if stop_ids and any(t in stop_ids for t in out[c0:]):
                j = next(i for i in range(c0, len(out)) if out[i] in stop_ids)
                del out[j + 1:]
                break
            y = new[-1]
    finally:
        B.AW[0] = aw0
        _P1[0] = False
        tcm = prefilled.pop("_b12_qsa", None) if isinstance(prefilled, dict) else None
        if tcm is not None:                            # = the fold's generate wrapper: close the state's QSA row target
            tcm.__exit__(None, None, None)
    st["time"] = time.perf_counter() - t0
    st["tok_s"] = len(out) / max(1e-9, st["time"])
    st["tok_per_pass"] = len(out) / max(1, st["passes"])
    st["lsha"] = h.hexdigest()[:10]
    return out, st
