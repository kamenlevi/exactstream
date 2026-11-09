"""R-56 diagnostic (disclosed; run after runner 29's verdict rows, docs/RESULT_r56_2026-10-30.md): where does the REF2
live loop first leave REF2's own AR, in bits? For each prompt whose ref2 row differs (results/r56/ref2.jsonl; or --keys):
  1. REF2's AR as the reference generator runs it (ref2.prefill + ar_step, PF1 on as evalsuite.env sets it), diff.at
     steps, every logits row kept; its tokens vs the stored row (results/ref2): did the reference move?
  2. The live loop exactly as live16 ran it (same drafter, C, PF1 for prefills only), every kept check row compared
     bitwise with the AR row at its position, up to the token difference: the first row that differs in bits, its check
     (start, row, rows kept, REF2 path counts), max |delta| and the top-3 of both rows there and at the token difference.
One process, `evalsuite gpu live16diag ref --C 192` -> results/r56/diag.jsonl. live16.py's loop is copied unchanged."""
from __future__ import annotations
import hashlib, json, os, sys, time
import numpy as np
import mlx.core as mx
from . import env, data as D
from .decode import footprint_peak_gb

LEAP = os.path.expanduser("~/Leap")
OUT = os.path.join(env.ROOT, "results", "r56")
R, K = 16, 16
EOS = {248044, 248046}
TAPS = (5, 15, 25, 35, 45)


def main(rt, st, keys_file, C, log):
    for k in ("MTPLX_FUSED_GATE_UP", "BIGMOE_TIER", "BIGMOE_VERIFY_MIDLOOP"):
        assert k not in os.environ, f"{k} is set"
    sys.path.insert(0, os.path.join(LEAP, "scripts"))
    sys.path.insert(1, os.path.join(LEAP, "third_party"))
    import ref2
    assert os.path.realpath(ref2.__file__) == os.path.realpath(os.path.join(LEAP, "scripts", "ref2.py")), ref2.__file__
    import bigmoe.stream as BS
    assert BS._SCRATCH_KEEP, "BIGMOE_SCRATCH_KEEP is off"
    import dflash_model_mlx as DF
    from mtplx import cache_state as CS
    Q, model = ref2.Q, rt.model
    REF = {d["key"]: d for d in map(json.loads, open(os.path.join(env.ROOT, "results", "ref2", "decode.jsonl")))}
    items = {it["key"]: it for s in ("math", "ifeval", "humaneval") for it in D.load("dec_" + s)}
    rows56 = {r["key"]: r for r in map(json.loads, open(os.path.join(OUT, "ref2.jsonl")))}
    keys = open(keys_file).read().split() if keys_file else [k for k, r in rows56.items() if not r["equal"]]
    ref2_sha = hashlib.sha256(open(ref2.__file__, "rb").read()).hexdigest()[:12]
    ref2.apply()
    selftest = ref2.self_test()
    ref2.remove(); ref2.reset_counts()
    assert all(selftest.values()), f"ref2.self_test failed: {selftest}"
    log(f"R-56 diag: keys {keys}; ref2.py {ref2_sha}; C {C}")

    def pf1(on):
        if not on and BS._STAGE["bufs"] is not None:
            BS._stage_drop()
        BS._PF_PREFILL = on

    # ------------------------------------------------ taps, drafter, proposals: live16's (= STEP 3k's), unchanged
    TAP = {"on": False, "x": [[] for _ in TAPS]}
    lay = model.language_model.model.layers
    TAP_OF = {id(lay[L]): i for i, L in enumerate(TAPS)}
    _layer_call = Q.DecoderLayer.__call__
    def _layer(self, *a, **kw):
        out = _layer_call(self, *a, **kw)
        i = TAP_OF.get(id(self))
        if TAP["on"] and i is not None:
            o = out if isinstance(out, mx.array) else out[0]
            B, L, W = o.shape
            TAP["x"][i].append(o.reshape(B, L, 4, W // 4).astype(mx.float32).mean(axis=2)[0].astype(mx.float16))
        return out
    Q.DecoderLayer.__call__ = _layer
    def tap_reset():
        TAP["x"] = [[] for _ in TAPS]
    def taps():
        f = [mx.concatenate(x) if len(x) > 1 else x[0] for x in TAP["x"]]
        return mx.stack(f, axis=1).reshape(f[0].shape[0], -1)

    draft = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
    draft.load_weights(os.path.join(LEAP, "data", "l120_ft.safetensors"), strict=False)
    draft.bind(model)
    mx.eval(draft.parameters())
    MASK, SW = int(draft.config.mask_token_id), int(draft.config.sliding_window)

    def head2s(hid):
        lg = draft.compute_logits(hid)
        B, s = 128, lg.shape
        blk = lg.reshape(*s[:-1], s[-1] // B, B)
        tb = mx.argpartition(blk.max(axis=-1), -K, axis=-1)[..., -K:]
        sel = mx.take_along_axis(blk, mx.broadcast_to(tb[..., None], (*s[:-1], K, B)), axis=-2).reshape(*s[:-1], K * B)
        j = mx.argpartition(sel, -K, axis=-1)[..., -K:]
        c = mx.take_along_axis(tb, j // B, axis=-1) * B + j % B
        return c, mx.take_along_axis(lg, c, axis=-1)

    def chain(blk, unary, cand, hp):
        sel = draft.candidate_selector
        pred, path_ = blk[:, 0], []
        for p in range(hp.shape[1]):
            sc = unary[:, p] + mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None]
                                      * sel.successor_codebook(cand[:, p]), axis=-1)
            pred = mx.take_along_axis(cand[:, p], mx.argmax(sc, axis=-1)[:, None], axis=-1)[:, 0]
            path_.append(pred)
        return mx.stack(path_, axis=1)

    def propose(blk, ctx, dcache):
        hid = draft.hidden_states(blk, ctx, dcache, 1)
        cand, unary = head2s(hid)
        return chain(blk, unary, cand, draft.candidate_selector.hidden_projection(hid))

    _midloop = Q._prefill_midloop_eval_layers
    def arm_on(ref):
        if ref:
            ref2.apply()
            Q._prefill_midloop_eval_layers = lambda rows: _midloop(rows) or (4 if 2 <= int(rows) <= 16 else 0)
        else:
            ref2.remove()
            Q._prefill_midloop_eval_layers = _midloop

    def states(cache):
        return [c.state for c in cache if c is not None and hasattr(c, "state")]

    def prefill(ids, chunk=1024):
        pf1(True)
        cache, lg = model.make_cache(), None
        tap_reset(); TAP["on"] = True
        for a in range(0, len(ids), chunk):
            lg = model(mx.array([ids[a:a + chunk]]), cache=cache)
            mx.eval(lg, states(cache), [x[-1] for x in TAP["x"]])
        TAP["on"] = False
        pf1(False)
        return cache, lg[0, -1], taps()

    def top3(v):
        v = v.astype(mx.float32)
        i = mx.argsort(-v)[:3].tolist()
        return [(int(t), float(v[t].item())) for t in i]

    def same(x, y):
        return x.shape == y.shape and x.dtype == y.dtype and \
            bool(mx.all(mx.view(x.reshape(-1), mx.uint8) == mx.view(y.reshape(-1), mx.uint8)).item())

    # ------------------------------------------------ per prompt
    with open(os.path.join(OUT, "diag.jsonl"), "a") as f:
        t_proc = time.time()
        for key in keys:
            if time.time() - t_proc > 720:
                log(f"R-56 diag: time budget (12 min): {key} and later keys not run")
                break
            it, want, r56 = items[key], REF[key]["completion_ids"], rows56[key]
            ids, at = list(it["prompt_ids"]), r56["diff"]["at"]
            P, t0 = len(ids), time.time()
            # 1. REF2's AR as the reference generator runs it (ref2.generate's calls, PF1 on throughout)
            pf1(True)
            ref2.apply()
            cache, last = ref2.prefill(model, ids)
            toks, ar = [int(mx.argmax(last).item())], []
            for _ in range(at):
                lg = ref2.ar_step(model, cache, toks[-1])
                mx.eval(lg)
                ar.append(lg)
                toks.append(int(mx.argmax(lg).item()))
            ref2.remove()
            del cache
            mx.clear_cache()
            ar_vs_stored = next((i for i, (x, y) in enumerate(zip(toks, want)) if x != y), None)
            t_ar = time.time() - t0
            # 2. the live loop as live16 ran it, every kept row vs the AR row at its position
            arm_on(True)
            ref2.reset_counts()
            cache, last, T = prefill(ids)
            dcache = draft.make_cache()
            start = max(0, P - (SW - 1))
            for c in dcache:
                c.offset = start
            ctx, a = T[start:P], P
            tok = int(mx.argmax(last).item())
            out, checks, first_bits, n_cmp, n_eq, at_rows = [tok], [], None, 0, 0, None
            while len(out) <= at and tok not in EOS:
                ref2.remove()
                dr = np.array(propose(mx.array([[tok] + [MASK] * (R - 1)]), ctx[None].astype(mx.bfloat16), dcache)[0]).tolist()
                if (trim := dcache[0].offset - a) > 0:
                    DF._trim_recent_cache(dcache, trim)
                ref2.apply()
                c0 = ref2.counts()
                snap = CS.snapshot_untrimmable_cache_lazy(cache)
                tap_reset(); TAP["on"] = True
                with ref2.verify_scope(model):
                    lg = model(mx.array([[tok] + dr]), cache=cache)
                TAP["on"] = False
                am, tp = mx.argmax(lg[0], axis=-1), taps()
                mx.eval(am, tp, lg)
                am = am.tolist()
                k = 0
                while k < R - 1 and dr[k] == am[k]:
                    k += 1
                cnt = {k_: v - c0.get(k_, 0) for k_, v in ref2.counts().items() if v - c0.get(k_, 0)}
                ck = dict(start=a - P, kept=k + 1, counts=cnt)
                checks.append(ck)
                for j in range(k + 1):                   # rows 0..k: inputs on the AR path
                    i = a - P + j
                    if i >= len(ar):
                        break
                    n_cmp += 1
                    eq = same(lg[0, j], ar[i])
                    n_eq += eq
                    if not eq and first_bits is None:
                        dlt = mx.abs(lg[0, j].astype(mx.float32) - ar[i].astype(mx.float32))
                        first_bits = dict(pos=i, check_start=a - P, row=j, kept=k + 1, counts=cnt,
                                          max_delta=float(mx.max(dlt).item()), n_diff=int(mx.sum(dlt > 0).item()),
                                          loop_top3=top3(lg[0, j]), ar_top3=top3(ar[i]),
                                          prev_checks=checks[-4:-1])
                    if i == at - 1:
                        at_rows = dict(loop_top3=top3(lg[0, j]), ar_top3=top3(ar[i]), bitwise=eq, row=j, check_start=a - P,
                                       kept=k + 1)
                model.commit_verified_window(cache, snap.states, keep_tokens=k + 1, verified_tokens=R)
                mx.async_eval(states(cache))
                new = dr[:k] + [am[k]]
                for j, x in enumerate(new):
                    if x in EOS:
                        new = new[:j + 1]
                        break
                out += new
                tok, ctx, a = out[-1], tp[:k + 1], a + k + 1
            mx.eval(states(cache))
            arm_on(False)
            del cache, dcache, ar
            mx.clear_cache()
            loop_vs_stored = next((i for i, (x, y) in enumerate(zip(out, want)) if x != y), None)
            rec = dict(key=key, n_prompt=P, at=at, stored_at=want[at], loop_at=out[at] if len(out) > at else None,
                       ar_at=toks[at], ar_vs_stored=ar_vs_stored, loop_vs_stored=loop_vs_stored,
                       loop_vs_ar_tokens=next((i for i, (x, y) in enumerate(zip(out, toks)) if x != y), None),
                       rows_compared=n_cmp, rows_bitwise=n_eq, first_bits=first_bits, at_rows=at_rows, n_checks=len(checks),
                       ar_s=round(t_ar, 1), s=round(time.time() - t0, 1), ref2_sha=ref2_sha, C=C,
                       life_gb=round(footprint_peak_gb(), 2), t=time.strftime("%H:%M:%S"))
            f.write(json.dumps(rec) + "\n"); f.flush()
            log(f"R-56 diag {key}: AR today vs stored row: first diff {ar_vs_stored}; loop vs stored {loop_vs_stored}; "
                f"kept rows bitwise = AR {n_eq}/{n_cmp}; first bit difference {first_bits and {k_: first_bits[k_] for k_ in ('pos', 'check_start', 'row', 'kept', 'max_delta', 'n_diff')}}; "
                f"at token {at}: stored {want[at]}, AR {toks[at]}, loop {rec['loop_at']}; rows there {at_rows}")
