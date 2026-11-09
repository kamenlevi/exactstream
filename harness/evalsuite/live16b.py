"""R-56 on REF2b (docs/PREREG_r56b_2026-10-30.md): is the 16-row live drafting loop bit-exact against REF2b's own AR on
the full completions of R-56's 36 prompts? One arm per process: `evalsuite gpu live16b ref --arm A --keys FILE`.
  ref2b:        REF2b (snap/leap-faff8e5: key buckets, slot call, commit phase, SDPA classes) on the default ext. Per
                prompt it first runs REF2b's AR the way the reference generator does (ref2.prefill + ar_step, PF1 on) to
                the item's max_tokens: its tokens, and a sha1 of every logits row's bytes -> results/r56b/ar/<key>.json.
  ref2b_gate16: the same REF2b with BigMoE's GATE16 (BIGMOE_MAXK=160, BIGMOE_EXT_BUILD=build_next: 16-row checks take
                the sync-free gate). It reads the AR files the ref2b arm wrote; it does not run its own AR.
  ref2t_gate16: R-56c, REF2-T (BigMoE's rank tiers 4433333333 / 3c / C_l 112) on the gate, with its own AR.
  ref2to_gate16: R-56d, ref2t_gate16 + BigMoE's BIGMOE_TIER_COMBINE=1 and BIGMOE_OOO=1 (R-69: the scores reach the tier
                call through model.block), its own AR; per prompt the tier calls and the OOO calls of the AR and of the
                loop are counted (the switches must engage).
  ref2cto_gate16: R-56e, REF2c-T (snap/leap-91cd316, Leap's REF2c freeze: ref2.CFG att_single + gdn_inproj on REF2-T;
                AR rows = results/ref2ct) on the gate with OOO + the fused combine, its own AR; each REF2c switch must
                run (asserted at AR step 1 and at check 1; counted per prompt).
The loop is live16's (Leap's STEP 3k harness), run to max_tokens: every kept check row is hashed and compared with the
AR row at its position, and every token with the AR's. It records the first row that differs in bits (check, row,
REF2 path counts, top-3 both sides) and stops the prompt at the first token that differs.
Rows -> results/r56b/<arm>.jsonl (resumable), one line per process -> results/r56b/chunks.jsonl."""
from __future__ import annotations
import hashlib, json, os, sys, threading, time
import numpy as np
import mlx.core as mx
from . import env, data as D, model as EM
from .decode import footprint_now_gb, footprint_peak_gb

LEAP = os.path.expanduser("~/Leap")
OUT = os.path.join(env.ROOT, "results", "r56b")   # set per arm in main: r56b (R-56b) or r56c (R-56c)
ARMS = {"ref2b": ("r56b", "ref2b", False, False),                 # (results dir, reference rows, gate, tiers)
        "ref2b_gate16": ("r56b", "ref2b", True, False),
        "ref2t_gate16": ("r56c", "ref2t", True, True),            # R-56c: REF2-T on the gate, its own AR
        "ref2to_gate16": ("r56d", "ref2t", True, "ooo"),          # R-56d: + OOO and the fused combine, its own AR
        "ref2cto_gate16": ("r56e", "ref2ct", True, "ooo"),        # R-56e: REF2c-T (Leap 91cd316) + OOO / combine, own AR
        "ref2tnaxno_gate16": ("r56n", "ref2tnaxn", True, "ooo", "ref2tnax")}   # PREREG_nax6 Part 2: NAX6-N (converted
#   store) + OOO / combine, own AR; the stored rows it is compared with are NAX6's Part 1 rows (row-major, ref2tnax)
COUNT_OF = {"tier_nax_ooo": ("nax6", "nax6_ooo")}   # REF2 switches whose calls ref2 counts under other names (NAX6)
TIER_ENV = {"BIGMOE_TIER": "4433333333", "BIGMOE_TIER_GLU": "3c", "BIGMOE_TIER_CL": "112", "BIGMOE_TIER_COMBINE": None,
            "BIGMOE_OOO": None}
OOO_ENV = dict(TIER_ENV, BIGMOE_TIER_COMBINE="1", BIGMOE_OOO="1")
R, K = 16, 16
EOS = {248044, 248046}
TAPS = (5, 15, 25, 35, 45)
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
FAN_CAP = os.path.expanduser("~/.local/share/quiet/FAN_CAP")
sha = lambda toks: hashlib.sha256(json.dumps([int(x) for x in toks]).encode()).hexdigest()[:16]
med = lambda v: float(np.median(v)) if len(v) else float("nan")

ACT = [0.0]                                              # running time: a quiet window's SIGSTOP freezes this ticker too
def _tick():
    last = time.time()
    while True:
        time.sleep(1)
        now = time.time()
        ACT[0] += min(now - last, 3.0)
        last = now
el = lambda: ACT[0]


def rowhash(x):
    """sha1 of a logits row's bytes (dtype and shape included): equal hashes = bitwise equal rows"""
    return hashlib.sha1(f"{x.dtype}{tuple(x.shape)}".encode()
                        + np.asarray(mx.view(mx.contiguous(x).reshape(-1), mx.uint8)).tobytes()).hexdigest()[:20]


def top3(x):
    v = x.astype(mx.float32)
    return [(int(t), float(v[t].item())) for t in mx.argsort(-v)[:3].tolist()]


def arfile(key):
    return os.path.join(OUT, "ar", key.replace("/", "__") + ".json")


def swap_gb():
    import re, subprocess
    m = re.search(r"used = ([\d.]+)M", subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout)
    return float(m.group(1)) / 1024 if m else 0.0


def main(rt, st, a, log):
    arm, chunk_s, C = a.arm, (a.chunk_minutes * 60) or 540, a.C
    assert arm in ARMS, f"--arm must be one of {sorted(ARMS)}"
    for k in ("MTPLX_FUSED_GATE_UP", "BIGMOE_VERIFY_MIDLOOP"):
        assert k not in os.environ, f"{k} is set"
    global OUT
    out_dir, ref_v, gate, tier = ARMS[arm][:4]
    rows_v = ARMS[arm][4] if len(ARMS[arm]) > 4 else ref_v   # the stored rows (Part 2: NAX6's row-major Part 1 rows)
    OUT = os.path.join(env.ROOT, "results", out_dir)
    ext_ok = (os.environ.get("BIGMOE_EXT_BUILD") == "build_next"     # BigMoE's Oct 30 gate build; since its Nov 4 build
              or (os.path.basename(env.BIGMOE_CODE.rstrip("/")) != "bigmoe-6c64148"   # ext/build serves MAXK 160 too
                  and os.environ.get("BIGMOE_EXT_BUILD") in (None, "build")))           # (MAXK is runtime; BigMoE 11-04 07:55)
    assert (os.environ.get("BIGMOE_MAXK") == "160" and ext_ok) == gate, \
        "the gate16 arms need BIGMOE_MAXK=160 and the gate's ext (build_next; or ext/build on a snapshot after Nov 4)"
    got_tier = {k: os.environ.get(k) for k in TIER_ENV}
    want = OOO_ENV if tier == "ooo" else TIER_ENV if tier else dict.fromkeys(TIER_ENV)
    assert got_tier == want, f"tier env {got_tier} for arm {arm}"
    threading.Thread(target=_tick, daemon=True).start()
    t_start, swap0, HARD = time.time(), swap_gb(), chunk_s + 300
    sys.path.insert(0, os.path.join(LEAP, "third_party"))   # the drafter (dflash_model_mlx)
    rv = EM.variant(ref_v)                        # the REF2 snapshot (+ CFG switches) the reference rows are made on
    ref2 = EM.ref2_import(rv)
    assert ref2.CFG.get("key_buckets") is True, ref2.CFG
    assert not ref2.CFG.get("gdn_inproj") or "MTPLX_FUSED_GDN_INPROJ" not in os.environ, "MTPLX_FUSED_GDN_INPROJ is set"
    on = [k for k, x in (rv.get("ref2_cfg") or {}).items() if x is True]   # REF2c switches: each must run (Leap 91cd316)

    def engaged(where, c0):
        off = [k for k in on if not any(ref2.counts().get(c, 0) - c0.get(c, 0) for c in COUNT_OF.get(k, (k,)))]
        assert not off, f"{where}: REF2 switches {off} never ran ({ref2.counts()})"
    import bigmoe.stream as BS
    assert BS._SCRATCH_KEEP, "BIGMOE_SCRATCH_KEEP is off"
    gate_n = int(BS.GATE_N[0]) if hasattr(BS, "GATE_N") else None
    assert (gate_n == 160) == gate, f"GATE_N {gate_n} for arm {arm}"
    BT = None
    if tier:
        import bigmoe.tier as BT
        assert (BT._COMBINE, BT._OOO) == ((tier == "ooo"),) * 2, f"bigmoe.tier COMBINE {BT._COMBINE}, OOO {BT._OOO}"
    calls = lambda: (BT.STATS["calls"], BT.OOO_STATS["calls"]) if BT else (0, 0)   # (tier calls, OOO calls)
    import dflash_model_mlx as DF
    from mtplx import cache_state as CS
    Q, model = ref2.Q, rt.model
    REF = {d["key"]: d for d in map(json.loads, open(os.path.join(env.ROOT, "results", rows_v, "decode.jsonl")))}
    items = {it["key"]: it for s in ("math", "ifeval", "humaneval") for it in D.load("dec_" + s)}
    keys = open(a.keys).read().split()
    os.makedirs(os.path.join(OUT, "ar"), exist_ok=True)
    path = os.path.join(OUT, f"{arm}.jsonl")
    done = {json.loads(l)["key"] for l in open(path)} if os.path.exists(path) else set()
    ref2_sha = hashlib.sha256(open(ref2.__file__, "rb").read()).hexdigest()[:12]
    ref2.apply()
    selftest = ref2.self_test()
    ref2.remove(); ref2.reset_counts()
    assert all(selftest.values()), f"ref2.self_test failed: {selftest}"
    log(f"R-56b {arm}: ref2.py {ref2_sha} (key buckets on" + (f"; CFG {rv['ref2_cfg']}" if rv.get("ref2_cfg") else "")
        + f"), GATE_N {gate_n}, MAXK {os.environ.get('BIGMOE_MAXK')}, "
        f"ext {os.environ.get('BIGMOE_EXT_BUILD', 'build')}, C {C}, BigMoE code {env.BIGMOE_CODE}")

    def pf1(on):
        if not on and BS._STAGE["bufs"] is not None:
            BS._stage_drop()
        BS._PF_PREFILL = on

    def stats():
        try:
            s = st.stats()
            return np.array([s["misses"], s["io_s"]], dtype=np.float64)
        except Exception:
            return np.zeros(2)

    # ------------------------------------------------ taps, drafter, proposals: live16's (= STEP 3k's), unchanged
    TAP = {"on": False, "x": [[] for _ in TAPS]}
    lay = model.language_model.model.layers
    TAP_OF = {id(lay[L]): i for i, L in enumerate(TAPS)}
    _layer_call = Q.DecoderLayer.__call__
    def _layer(self, *a_, **kw):
        out = _layer_call(self, *a_, **kw)
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
        for a_ in range(0, len(ids), chunk):
            lg = model(mx.array([ids[a_:a_ + chunk]]), cache=cache)
            mx.eval(lg, states(cache), [x[-1] for x in TAP["x"]])
        TAP["on"] = False
        pf1(False)
        return cache, lg[0, -1], taps()

    def ar_run(key, ids, cap):
        """REF2b's AR as the reference generator runs it (ref2.generate's calls, PF1 on throughout): tokens, and per
        step the logits row's hash and top-3; written to the AR file once, read back by later processes and arms"""
        p = arfile(key)
        if os.path.exists(p):
            return json.load(open(p))
        assert not (gate and not tier), f"no AR file for {key}: the ref2b arm writes it first"   # REF2-T: its own AR
        pf1(True)
        ref2.apply()
        t0, c0, e0 = time.time(), calls(), ref2.counts()
        cache, last = ref2.prefill(model, ids)
        toks, hs, t3 = [int(mx.argmax(last).item())], [], []
        while len(toks) < cap and toks[-1] not in EOS:
            lg = ref2.ar_step(model, cache, toks[-1])
            mx.eval(lg)
            hs.append(rowhash(lg)); t3.append(top3(lg))
            toks.append(int(mx.argmax(lg).item()))
            if len(hs) == 1:
                engaged(f"{key}: AR step 1", e0)
        ref2.remove()
        del cache
        mx.clear_cache()
        rec = dict(key=key, toks=toks, hash=hs, top3=t3, s=round(time.time() - t0, 1), ref2_sha=ref2_sha, C=C,
                   calls=[x - y for x, y in zip(calls(), c0)], pid=os.getpid(), t=time.strftime("%H:%M:%S"))
        if on:                                    # REF2c: each switch's calls in this AR
            rec["ref2c"] = {c: ref2.counts().get(c, 0) - e0.get(c, 0) for k in on for c in COUNT_OF.get(k, (k,))}
        tmp = p + ".tmp"
        json.dump(rec, open(tmp, "w")); os.replace(tmp, p)
        return rec

    def run(ids, ar, cap):
        while os.path.exists(PAUSE):
            time.sleep(5)
        paused, cut, tdiff, bdiff, n_cmp, n_eq = False, False, None, None, 0, 0
        arm_on(True)
        ref2.reset_counts()
        mx.synchronize()
        c1 = calls()
        t1 = time.perf_counter()
        cache, last, T = prefill(ids)
        pf = time.perf_counter() - t1
        P, want, hs = len(ids), ar["toks"], ar["hash"]
        dcache = draft.make_cache()
        start = max(0, P - (SW - 1))
        for c in dcache:
            c.offset = start
        ctx, a_ = T[start:P], P
        tok = int(mx.argmax(last).item())
        out, cyc, kept, checks = [tok], [], [], 0
        if want[0] != tok:
            tdiff = dict(at=0, got=tok, want=want[0], row="prefill")
        s0 = stats()
        td = c_prev = time.perf_counter()
        while tdiff is None and len(out) < cap and tok not in EOS:
            if el() > HARD:
                cut = True
                break
            c0 = time.perf_counter()
            if out[1:]:
                cyc.append(c0 - c_prev)
            c_prev = c0
            ref2.remove()
            dr = np.array(propose(mx.array([[tok] + [MASK] * (R - 1)]), ctx[None].astype(mx.bfloat16), dcache)[0]).tolist()
            if (trim := dcache[0].offset - a_) > 0:
                DF._trim_recent_cache(dcache, trim)
            ref2.apply()
            cnt0 = ref2.counts()
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
            for j in range(k + 1):                       # rows 0..k: inputs on the AR path; row j predicts out[len+j]
                i = a_ - P + j
                if i >= len(hs):
                    break
                n_cmp += 1
                eq = rowhash(lg[0, j]) == hs[i]
                n_eq += eq
                if not eq and bdiff is None:
                    bdiff = dict(pos=i, abs=P + i, check_start=a_ - P, row=j, kept=k + 1,
                                 counts={k_: v - cnt0.get(k_, 0) for k_, v in ref2.counts().items() if v - cnt0.get(k_, 0)},
                                 loop_top3=top3(lg[0, j]), ar_top3=ar["top3"][i])
            model.commit_verified_window(cache, snap.states, keep_tokens=k + 1, verified_tokens=R)
            mx.async_eval(states(cache))                 # SB: the commit replay submitted alone
            kept.append(k + 1); checks += 1
            if checks == 1:
                engaged("check 1", {})                   # counts were reset at this run's start
            new = dr[:k] + [am[k]]
            for j, x in enumerate(new):
                if x in EOS:
                    new = new[:j + 1]
                    break
            for j, x in enumerate(new):                  # the first token that leaves the AR ends the prompt
                i = len(out) + j
                w = want[i] if i < len(want) else None
                if x != w:
                    tdiff = dict(at=i, got=x, want=w, row=j, loop_top3=top3(lg[0, j]),
                                 ar_top3=ar["top3"][i - 1] if 0 < i <= len(ar["top3"]) else None)
                    new = new[:j + 1]
                    break
            out += new
            tok, ctx, a_ = out[-1], tp[:k + 1], a_ + k + 1
            paused = paused or os.path.exists(PAUSE)
        mx.eval(states(cache))
        dec = time.perf_counter() - td
        ms, io = (stats() - s0) / max(checks, 1)
        cnt, lc = ref2.counts(), [x - y for x, y in zip(calls(), c1)]
        arm_on(False)
        del cache, dcache
        mx.clear_cache()
        return dict(out=out[:cap], tdiff=tdiff, bdiff=bdiff, n_cmp=n_cmp, n_eq=n_eq, pf=pf, dec=dec, checks=checks, calls=lc,
                    kept=float(np.mean(kept)) if kept else 0.0, cyc=1e3 * med(cyc), paused=paused, cut=cut,
                    ms=float(ms), io=float(1e3 * io), cnt={k_: round(v / max(checks, 1), 3) for k_, v in cnt.items() if v})

    # ------------------------------------------------ the prompts
    new_rows, gate_, yielded = 0, getattr(st, "gate", None), False
    with open(path, "a") as f:
        for key in keys:
            if key in done:
                continue
            if new_rows and el() > chunk_s:
                log(f"R-56b {arm}: chunk budget ({chunk_s:.0f} s running): {new_rows} new row(s)")
                break
            yflag = os.environ.get("EVALSUITE_YIELD_FLAG")
            if yflag and os.path.exists(yflag):          # a quiet ticket waits (cli --share): stop at this row boundary
                log(f"R-56b {arm}: yielding to a quiet ticket after {new_rows} new row(s)")
                yielded = True
                break
            it = items[key]
            ids, cap = list(it["prompt_ids"]), it["max_tokens"]
            t0 = time.time()
            ar = ar_run(key, ids, cap)
            r = run(ids, ar, cap)
            if r["cut"]:
                log(f"R-56b {arm} {key}: cut at {len(r['out'])} tokens by the hard limit ({HARD:.0f} s running): not saved")
                break
            if gate_ is not None:
                gate_.check()
            n, stored = len(r["out"]), REF[key]["completion_ids"]
            row = dict(key=key, set=it["set"], n_prompt=len(ids), cap=cap, n=n, ar_n=len(ar["toks"]),
                       equal=r["out"] == ar["toks"], tdiff=r["tdiff"], rows_cmp=r["n_cmp"], rows_eq=r["n_eq"],
                       bits_equal=r["n_eq"] == r["n_cmp"] and r["n_cmp"] == len(ar["hash"]) and r["out"] == ar["toks"],
                       bdiff=r["bdiff"], ar_vs_ref2b=next((i for i, (x, y) in enumerate(zip(ar["toks"], stored)) if x != y),
                                                         None if len(ar["toks"]) == len(stored) else min(len(ar["toks"]), len(stored))),
                       sha=sha(r["out"]), ar_sha=sha(ar["toks"]), ref2b_sha=sha(stored), checks=r["checks"],
                       kept=round(r["kept"], 3), tok_s=round(n / r["dec"], 2) if r["dec"] else None, cyc_ms=round(r["cyc"], 1),
                       prefill_s=round(r["pf"], 2), per_check=r["cnt"], misses=round(r["ms"], 1), io_ms=round(r["io"], 1),
                       paused=r["paused"], ar_s=ar.get("s"), s=round(time.time() - t0, 1),
                       calls=dict(ar=ar.get("calls"), loop=r["calls"]),   # [tier calls, OOO calls] per phase
                       ar_ref2c=ar.get("ref2c"),             # REF2c: each switch's calls in the AR (the loop's: per_check)
                       mem=dict(now=round(footprint_now_gb(), 2), life=round(footprint_peak_gb(), 2)), swap_gb=round(swap_gb(), 2),
                       fan_cap=open(FAN_CAP).read().strip() if os.path.exists(FAN_CAP) else None, C=C, pid=os.getpid(),
                       t=time.strftime("%H:%M:%S"))
            f.write(json.dumps(row) + "\n"); f.flush()
            new_rows += 1
            log(f"R-56b {arm} {key}: P {len(ids)}, {n}/{len(ar['toks'])} tokens, equal {row['equal']}, rows bitwise "
                f"{r['n_eq']}/{r['n_cmp']} (AR rows {len(ar['hash'])})"
                + (f", first bit difference {r['bdiff']}" if r["bdiff"] else "") + (f", first token difference {r['tdiff']}" if r["tdiff"] else "")
                + f", AR vs results/{rows_v}: first difference {row['ar_vs_ref2b']}, {row['tok_s']} tok/s (quiet mode), kept {r['kept']:.2f}, "
                f"mem {row['mem']}" + (f", [tier, OOO] calls {row['calls']}" if BT else ""))
    rec = dict(arm=arm, pid=os.getpid(), start=time.strftime("%H:%M:%S", time.localtime(t_start)), end=time.strftime("%H:%M:%S"),
               C=C, ref2_sha=ref2_sha, ref2_code=rv.get("ref2_code"), ref2_cfg=rv.get("ref2_cfg"), selftest=selftest,
               gate_n=gate_n, maxk=os.environ.get("BIGMOE_MAXK"),
               ext=os.environ.get("BIGMOE_EXT_BUILD", "build"), bigmoe_code=env.BIGMOE_CODE, scratch_keep=BS._SCRATCH_KEEP,
               combine=BT._COMBINE if BT else None, ooo=BT._OOO if BT else None,
               rows=new_rows, running_s=round(el()), wall_s=round(time.time() - t_start), life_gb=round(footprint_peak_gb(), 2),
               swap0=round(swap0, 2), swap1=round(swap_gb(), 2), provenance=env.provenance())
    with open(os.path.join(OUT, "chunks.jsonl"), "a") as g:
        g.write(json.dumps(rec) + "\n")
    log(f"R-56b {arm}: process done: {new_rows} new row(s), lifetime peak {rec['life_gb']} GB, {rec['running_s']} s running")
    return "yield" if yielded else None
