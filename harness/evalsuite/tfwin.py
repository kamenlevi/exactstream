"""TFWIN (docs/PREREG_agentfast_2026-11-06.md): a run's committed tokens scored by a mode's exact reference along that
run's own prefix. Per row of results/<RUN>/decode.jsonl: the prompt is prefilled as replay.py does, then the committed
tokens go through REF2's exact verify path in 16-row windows, every row kept (the reference's own one-token AR bit for
bit: R-56c for REF2-T, NAX6 Part 2 for NAX6-N). A tail window shorter than 16 is padded with its last token; the padded
rows are never read or kept, as a check's rejected drafts. Per committed token: the reference's log-prob of it, its
argmax and the argmax's log-prob. Self-check first (--self RUN): that exact run's rows of SELF_KEYS must be the
reference's argmax at every token, else VOID (exit 5).
  `evalsuite gpu tfwin ref --run RUN [--keys FILE] [--self RUN]` (EVALSUITE_RELAX_VARIANT: the reference, as relax-replay;
  the same env as relax-replay)
Out: results/<RUN>/tfwin/<key>.json.gz (key, n, lp, am, lp1), results/<RUN>/tfwin_chunks.jsonl."""
from __future__ import annotations
import glob, gzip, json, math, os, threading, time
import numpy as np
import mlx.core as mx
from . import env, data as D, model as EM
from .decode import footprint_peak_gb
from .relax import OOO_ENV, el, pause_wait, swap_gb, _tick
from .relax_core import R

SELF_KEYS = ("humaneval/HumanEval/27", "ifeval/3409", "gsm8k/707")
LN_P01 = math.log(0.01)                                 # a blunder: the reference's p of the committed token < 0.01


def main(rt, st, a, log):
    chunk_s = (a.chunk_minutes * 60) or 540
    ext_ok = (os.environ.get("BIGMOE_EXT_BUILD") == "build_next"
              or (os.path.basename(env.BIGMOE_CODE.rstrip("/")) != "bigmoe-6c64148"
                  and os.environ.get("BIGMOE_EXT_BUILD") in (None, "build")))
    assert os.environ.get("BIGMOE_MAXK") == "160" and ext_ok
    got = {k_: os.environ.get(k_) for k_ in OOO_ENV}
    assert got == OOO_ENV, f"tier env {got}"
    assert a.run, "--run RUN"
    rdir = os.path.join(env.ROOT, "results", a.run)
    out_dir = os.path.join(rdir, "tfwin")
    os.makedirs(out_dir, exist_ok=True)
    threading.Thread(target=_tick, daemon=True).start()
    t_start, HARD = time.time(), chunk_s + 300
    vname = os.environ.get("EVALSUITE_RELAX_VARIANT", "ref2t")
    rv = EM.variant(vname)
    ref2 = EM.ref2_import(rv)
    assert ref2.CFG.get("key_buckets") is True, ref2.CFG
    import bigmoe.stream as BS
    import bigmoe.tier as BT
    assert BT._COMBINE and BT._OOO and BT._GLU == "3c" and int(BS.GATE_N[0]) == 160
    from mtplx import cache_state as CS
    Q, model = ref2.Q, rt.model
    ITEMS = {}

    def item(row):
        s = row["set"]
        if s not in ITEMS:
            ITEMS[s] = {it["key"]: it for it in D.load("dec_" + s)}
        return ITEMS[s][row["key"]]

    ref2.apply()
    selftest = ref2.self_test()
    ref2.remove(); ref2.reset_counts()
    assert all(selftest.values()), f"ref2.self_test failed: {selftest}"

    def pf1(on):
        if not on and BS._STAGE["bufs"] is not None:
            BS._stage_drop()
        BS._PF_PREFILL = on

    _midloop = Q._prefill_midloop_eval_layers
    def arm_on(ref):
        if ref:
            ref2.apply()
            Q._prefill_midloop_eval_layers = lambda rows_: _midloop(rows_) or (4 if 2 <= int(rows_) <= 16 else 0)
        else:
            ref2.remove()
            Q._prefill_midloop_eval_layers = _midloop

    def states(cache):
        return [c.state for c in cache if c is not None and hasattr(c, "state")]

    def prefill(ids, chunk=1024):
        """replay.py's prefill (relax.py's without the taps)"""
        pf1(True)
        cache, lg = model.make_cache(), None
        for a_ in range(0, len(ids), chunk):
            lg = model(mx.array([ids[a_:a_ + chunk]]), cache=cache)
            mx.eval(lg, states(cache))
        pf1(False)
        return cache, lg[0, -1]

    def score(ids, out):
        """the reference along `out`: per token (log p of out[i], argmax, log p of the argmax); None if cut by HARD"""
        pause_wait()
        arm_on(True)
        cache, last = prefill(ids)
        l0 = last.astype(mx.float32)
        z = mx.logsumexp(l0)
        mx.eval(l0, z)
        lp, am, lp1 = [round((l0[out[0]] - z).item(), 5)], [int(mx.argmax(l0).item())], [round((l0.max() - z).item(), 5)]
        pos = 1
        while pos < len(out):
            if el() > HARD:
                arm_on(False)
                return None
            n = min(R, len(out) - pos)
            inp = out[pos - 1:pos - 1 + n]
            inp = inp + [inp[-1]] * (R - n)               # the tail: pad to the 16-row width; padded rows are not kept
            snap = CS.snapshot_untrimmable_cache_lazy(cache)
            with ref2.verify_scope(model):
                lg = model(mx.array([inp]), cache=cache)
            l_ = lg[0, :n].astype(mx.float32)
            lse = mx.logsumexp(l_, axis=-1)
            lpc = mx.take_along_axis(l_, mx.array(out[pos:pos + n], dtype=mx.int32)[:, None], axis=-1)[:, 0] - lse
            eam = mx.argmax(l_, axis=-1)
            lpm = l_.max(axis=-1) - lse
            mx.eval(lpc, eam, lpm)
            lp += np.round(np.array(lpc), 5).tolist()
            am += eam.tolist()
            lp1 += np.round(np.array(lpm), 5).tolist()
            model.commit_verified_window(cache, snap.states, keep_tokens=n, verified_tokens=R)
            mx.async_eval(states(cache))
            pos += n
        mx.eval(states(cache))
        arm_on(False)
        del cache
        mx.clear_cache()
        return dict(lp=lp, am=am, lp1=lp1)

    # ------------------------------------------------ self-check: an exact run's own rows are the reference's argmax
    srows = {r["key"]: r for r in map(json.loads, open(os.path.join(env.ROOT, "results", a.self, "decode.jsonl")))}
    bad = []
    for k in SELF_KEYS:
        row = srows[k]
        out = row["completion_ids"]
        s = score(list(item(row)["prompt_ids"]), out)
        assert s is not None, "the self-check was cut by the hard limit"
        dev = sum(x != y for x, y in zip(s["am"], out))
        log(f"TFWIN self-check ({vname} on results/{a.self}): {k}: {len(out)} tokens, {dev} not the reference's argmax")
        if dev:
            bad.append(k)
    if bad:
        log(f"TFWIN: VOID: the reference does not reproduce results/{a.self}'s own tokens on {bad}")
        raise SystemExit(5)

    rows = [json.loads(l) for l in open(os.path.join(rdir, "decode.jsonl"))]
    want = set(open(a.keys).read().split()) if a.keys else None
    done = {os.path.basename(p)[:-len(".json.gz")].replace("__", "/") for p in glob.glob(os.path.join(out_dir, "*.json.gz"))}
    new = 0
    for row in rows:
        key = row["key"]
        if key in done or (want is not None and key not in want):
            continue
        if new and el() > chunk_s:
            log(f"TFWIN {a.run}: chunk budget ({chunk_s:.0f} s running): {new} new row(s)")
            break
        out = row["completion_ids"]
        t0 = time.time()
        s = score(list(item(row)["prompt_ids"]), out)
        if s is None:
            log(f"TFWIN {a.run} {key}: cut by the hard limit: not saved")
            break
        p = os.path.join(out_dir, key.replace("/", "__") + ".json.gz")
        with gzip.open(p + ".tmp", "wt") as g:
            json.dump(dict(key=key, n=len(out), ref=vname, **s), g)
        os.replace(p + ".tmp", p)
        new += 1
        dev = sum(x != y for x, y in zip(s["am"], out))
        blund = sum(x < LN_P01 for x in s["lp"])
        log(f"TFWIN {a.run} {key}: {len(out)} tokens, {dev} not the reference's argmax, {blund} blunder(s) "
            f"(p < 0.01), {time.time() - t0:.0f} s")
    rec = dict(run=a.run, ref=vname, self=a.self, pid=os.getpid(), start=time.strftime("%H:%M:%S", time.localtime(t_start)),
               end=time.strftime("%H:%M:%S"), C=a.C, rows=new, running_s=round(el()), life_gb=round(footprint_peak_gb(), 2),
               swap1=round(swap_gb(), 2), provenance=env.provenance())
    with open(os.path.join(rdir, "tfwin_chunks.jsonl"), "a") as g:
        g.write(json.dumps(rec) + "\n")
    log(f"TFWIN {a.run}: process done: {new} row(s), lifetime peak {rec['life_gb']} GB")
