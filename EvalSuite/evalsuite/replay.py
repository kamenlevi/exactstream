"""RELAX's exact reference (docs/PREREG_relax_2026-11-02.md): REF2-T replays a relax arm's logged checks. Per prompt it
prefills the prompt, then feeds each logged check's 16 input rows to an exact cache (top-10 on every row, no drafter, no
gate, no timing) and commits the same k+1 rows, so E sees exactly the prefix G committed. Per committed token: E's
argmax, E's log-prob of the committed token and of its own argmax, and KLD(E || G) against G's logged top-64. The
step-0 records (gate and rule off) must replay bitwise (row hashes): else VOID (exit 5). An arm whose gate-off rows are
not REF2-T's numerics (HC8: B, C) or that has none (T3ALL) checks the replay on the control's.
  `evalsuite gpu relax-replay ref --arm A [--keys FILE]` (same env as the arm; any C)
Out: results/relax_<arm>[_tag]/replay/<key>.jsonl.gz (one line per check), replay_chunks.jsonl."""
from __future__ import annotations
import glob, gzip, hashlib, json, os, sys, threading, time
import numpy as np
import mlx.core as mx
from . import env, data as D, model as EM
from .decode import footprint_peak_gb
from .relax import ARMS, EOS, OOO_ENV, REF_TIER, SETS, el, out_dir, pause_wait, rowhash, swap_gb, _tick
from .relax_core import R, kld_top


def main(rt, st, a, log):
    arm, chunk_s, C = a.arm, (a.chunk_minutes * 60) or 540, a.C
    ext_ok = (os.environ.get("BIGMOE_EXT_BUILD") == "build_next"     # the Oct 30 gate build; since BigMoE's Nov 4 build
              or (os.path.basename(env.BIGMOE_CODE.rstrip("/")) != "bigmoe-6c64148"   # ext/build serves MAXK 160 too
                  and os.environ.get("BIGMOE_EXT_BUILD") in (None, "build")))           # (MAXK is runtime)
    assert os.environ.get("BIGMOE_MAXK") == "160" and ext_ok
    got = {k_: os.environ.get(k_) for k_ in OOO_ENV}
    assert got == OOO_ENV, f"tier env {got}"
    OUT = out_dir(arm)
    os.makedirs(os.path.join(OUT, "replay"), exist_ok=True)
    threading.Thread(target=_tick, daemon=True).start()
    t_start, HARD = time.time(), chunk_s + 300
    rv = EM.variant(os.environ.get("EVALSUITE_RELAX_VARIANT", "ref2t"))   # ref2tf0 on the converted store (Fri)
    ref2 = EM.ref2_import(rv)
    assert ref2.CFG.get("key_buckets") is True, ref2.CFG
    import bigmoe.stream as BS
    import bigmoe.tier as BT
    assert BT._COMBINE and BT._OOO and BT._GLU == "3c" and int(BS.GATE_N[0]) == 160
    from mtplx import cache_state as CS
    Q, model = ref2.Q, rt.model
    items = {it["key"]: it for s in SETS for it in D.load("dec_" + s)}
    rows = {d["key"]: d for d in map(json.loads, open(os.path.join(OUT, "decode.jsonl")))}
    keys = open(a.keys).read().split() if a.keys else list(rows)
    rdir = os.path.join(OUT, "replay")
    done = {os.path.basename(p)[:-len(".jsonl.gz")].replace("__", "/") for p in glob.glob(os.path.join(rdir, "*.jsonl.gz"))}
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
        """relax.py's prefill without the taps (they only feed the drafter; the model's rows do not read them)"""
        pf1(True)
        cache, lg = model.make_cache(), None
        for a_ in range(0, len(ids), chunk):
            lg = model(mx.array([ids[a_:a_ + chunk]]), cache=cache)
            mx.eval(lg, states(cache))
        pf1(False)
        return cache, lg[0, -1]

    def replay(ids, out, recs):
        """E along G's committed prefix: one record per check"""
        pause_wait()
        arm_on(True)
        cache, last = prefill(ids)
        first = int(mx.argmax(last).item())
        res, bits = [], [0, 0]
        for rec in recs:
            if el() > HARD:
                return None, None, None
            inp, n, pos = rec["inp"], rec["n"], rec["pos"]
            assert inp[0] == out[pos - 1], (pos, inp[0], out[pos - 1])
            toks = out[pos:pos + n]
            snap = CS.snapshot_untrimmable_cache_lazy(cache)
            with ref2.verify_scope(model):
                lg = model(mx.array([inp]), cache=cache)
            l0 = lg[0, :n].astype(mx.float32)
            lse = mx.logsumexp(l0, axis=-1)
            lpc = mx.take_along_axis(l0, mx.array(toks, dtype=mx.int32)[:, None], axis=-1)[:, 0] - lse
            eam = mx.argmax(lg[0, :n], axis=-1)
            lp1 = l0.max(axis=-1) - lse
            lpg = mx.take_along_axis(l0, mx.array(rec["ti"], dtype=mx.int32), axis=-1) - lse[:, None]
            mx.eval(lpc, eam, lp1, lpg, lg)
            lpg = np.array(lpg)
            kld = [round(kld_top(lpg[j], rec["tl"][j]), 6) for j in range(n)]
            r_ = dict(c=rec["c"], pos=pos, eam=eam.tolist(), lpc=np.round(np.array(lpc), 4).tolist(),
                      lp1=np.round(np.array(lp1), 4).tolist(), kld=kld)
            if "h" in rec:
                hs = [rowhash(lg[0, j]) for j in range(n)]
                eq = sum(x == y for x, y in zip(hs, rec["h"]))
                bits[0] += eq; bits[1] += n
                r_["h_eq"] = eq
            res.append(r_)
            model.commit_verified_window(cache, snap.states, keep_tokens=rec["k"] + 1, verified_tokens=R)
            mx.async_eval(states(cache))
        mx.eval(states(cache))
        arm_on(False)
        del cache
        mx.clear_cache()
        return res, first, bits

    # ------------------------------------------------ step 0: G's gate-off rows replay bitwise
    own_exact = not ARMS[arm]["hc8"] and ARMS[arm].get("tier", REF_TIER) == REF_TIER
    s0 = glob.glob(os.path.join(OUT, "step0", "*.json")) if own_exact else []
    if not s0:                                    # an arm whose gate-off rows are not REF2-T's (HC8: B, C) or that has none
        ctl = out_dir("control")                  # (T3ALL): the replay checks itself on the control's gate-off rows
        s0 = glob.glob(os.path.join(ctl, "step0", "*.json"))
        log(f"RELAX replay {arm}: step 0 on {ctl}'s gate-off rows (the arm's own are not REF2-T's numerics)")
    assert s0, f"no usable step-0 record in {OUT}/step0 or the control's"
    z = json.load(open(max(s0, key=os.path.getmtime)))       # the newest G process's gate-off rows
    it0 = items[z["key"]]
    res0, first0, bits0 = replay(list(it0["prompt_ids"]), z["out"], z["recs"])
    ok0 = first0 == z["out"][0] and bits0[1] > 0 and bits0[0] == bits0[1]
    log(f"RELAX replay {arm} step 0: {z['key']} rows bitwise {bits0[0]}/{bits0[1]}, first token equal {first0 == z['out'][0]}")
    if not ok0:
        log(f"RELAX replay {arm}: VOID: the replay does not reproduce G's gate-off rows bitwise")
        raise SystemExit(5)

    new = 0
    for key in keys:
        if key in done or key not in rows:
            continue
        if new and el() > chunk_s:
            log(f"RELAX replay {arm}: chunk budget ({chunk_s:.0f} s running): {new} new prompt(s)")
            break
        cp = os.path.join(OUT, "checks", key.replace("/", "__") + ".jsonl.gz")
        recs = [json.loads(l) for l in gzip.open(cp, "rt")]
        out = rows[key]["completion_ids"]
        t0 = time.time()
        res, first, _ = replay(list(items[key]["prompt_ids"]), out, recs)
        if res is None:
            log(f"RELAX replay {arm} {key}: cut by the hard limit: not saved")
            break
        rp = os.path.join(rdir, key.replace("/", "__") + ".jsonl.gz")
        with gzip.open(rp + ".tmp", "wt") as g:
            g.write(json.dumps(dict(key=key, first_equal=first == out[0], n=len(out), s=round(time.time() - t0, 1))) + "\n")
            for r_ in res:
                g.write(json.dumps(r_) + "\n")
        os.replace(rp + ".tmp", rp)
        new += 1
        dev = sum(x != y for r_ in res for x, y in zip(r_["eam"], out[r_["pos"]:r_["pos"] + len(r_["eam"])]))
        log(f"RELAX replay {arm} {key}: {len(out)} tokens, {dev} deviated from E's argmax, {time.time() - t0:.0f} s")
    rec = dict(arm=arm, pid=os.getpid(), start=time.strftime("%H:%M:%S", time.localtime(t_start)), end=time.strftime("%H:%M:%S"),
               C=C, step0_bits=bits0, prompts=new, running_s=round(el()), life_gb=round(footprint_peak_gb(), 2),
               swap1=round(swap_gb(), 2), provenance=env.provenance())
    with open(os.path.join(OUT, "replay_chunks.jsonl"), "a") as g:
        g.write(json.dumps(rec) + "\n")
    log(f"RELAX replay {arm}: process done: {new} prompt(s), lifetime peak {rec['life_gb']} GB")
