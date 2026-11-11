"""R-56 LIVE16 (docs/PREREG_r56_2026-10-30.md): is the 16-row live drafting loop token-exact against its own reference's
AR on suite prompts? One arm per process: `evalsuite gpu live16 ref --arm ref2|stock --keys results/r56/keys.txt`.
  ref2:  Leap's ref2.py (~/Leap/scripts, pinned by the runner) with its defaults (R-50 slot call, commit phase, SDPA
         classes), vs EvalSuite's REF2 AR rows (results/ref2); plus REF2's in-process AR logits for the first rows of
         each prompt, compared bitwise with the kept check rows (as STEP 3k).
  stock: no REF2 (stock MLX kernels; BigMoE's _sync_call with BIGMOE_SCRATCH_KEEP=1), vs the stock AR rows
         (results/ref.r2).
The loop is Leap's R-40 STEP 3k harness (~/Leap 23abf98 scripts/r40s3k_screen.py: DFlash2 L1-20 drafting, 16-row checks
in the engine's verify scopes, commit keep k+1, PF1 for prefills only), run to the item's max_tokens; a prompt stops at
its first token that differs from the reference row. Each process first re-decodes one reference row (HumanEval/27,
which crosses 128 keys) with the reference's own generator (decode._gen / decode._gen_ref2): if it no longer
reproduces, the process's rows are VOID (the reference moved, not the loop).
Rows -> results/r56/<arm>.jsonl (resumable), one line per process -> results/r56/chunks.jsonl."""
from __future__ import annotations
import hashlib, json, os, sys, threading, time
import numpy as np
import mlx.core as mx
from . import env, data as D
from .decode import footprint_now_gb, footprint_peak_gb, _gen, _gen_ref2

LEAP = os.path.expanduser("~/Leap")
OUT = os.path.join(env.ROOT, "results", "r56")
REFS = {"ref2": "ref2", "stock": "ref.r2"}
CHECK_KEY = "humaneval/HumanEval/27"
R, K, NAR = 16, 16, 48
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


def swap_gb():
    import re, subprocess
    m = re.search(r"used = ([\d.]+)M", subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout)
    return float(m.group(1)) / 1024 if m else 0.0


def fan_cap():
    try:
        return open(FAN_CAP).read().strip()
    except OSError:
        return None


def main(rt, st, arm, keys_file, chunk_s, C, log):
    assert arm in REFS, f"--arm must be one of {sorted(REFS)}"
    for k in ("MTPLX_FUSED_GATE_UP", "BIGMOE_TIER", "BIGMOE_VERIFY_MIDLOOP"):   # as STEP 3k's harness pops them
        assert k not in os.environ, f"{k} is set"
    threading.Thread(target=_tick, daemon=True).start()
    t_start, chunk_s, swap0 = time.time(), chunk_s or 540, swap_gb()
    HARD = chunk_s + 240                                 # a prompt still running then is cut and not saved
    sys.path.insert(0, os.path.join(LEAP, "scripts"))
    sys.path.insert(1, os.path.join(LEAP, "third_party"))
    import ref2
    assert os.path.realpath(ref2.__file__) == os.path.realpath(os.path.join(LEAP, "scripts", "ref2.py")), ref2.__file__
    import bigmoe.stream as BS
    assert BS._SCRATCH_KEEP, "BIGMOE_SCRATCH_KEEP is off: R-56 audits the default (on)"
    import dflash_model_mlx as DF
    from mtplx import cache_state as CS
    from mtplx.sampling import SamplerConfig
    Q, model = ref2.Q, rt.model
    REF = {d["key"]: d for d in map(json.loads, open(os.path.join(env.ROOT, "results", REFS[arm], "decode.jsonl")))}
    items = {it["key"]: it for s in ("math", "ifeval", "humaneval") for it in D.load("dec_" + s)}
    keys = open(keys_file).read().split()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{arm}.jsonl")
    done = {json.loads(l)["key"] for l in open(path)} if os.path.exists(path) else set()
    ref2_sha = hashlib.sha256(open(ref2.__file__, "rb").read()).hexdigest()[:12]
    selftest = None
    if arm == "ref2":
        ref2.apply()
        selftest = ref2.self_test()
        ref2.remove(); ref2.reset_counts()
        assert all(selftest.values()), f"ref2.self_test failed: {selftest}"

    def pf1(on):
        """on: EvalSuite's PF1 prefill path; off: BigMoE's default _sync_call for calls with > MAXK ids (STEP 3k's)"""
        if not on and BS._STAGE["bufs"] is not None:
            BS._stage_drop()
        BS._PF_PREFILL = on

    # ------------------------------------------------ the reference check: one reference row, its own generator
    it = items[CHECK_KEY]
    greedy = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
    seed = int(hashlib.sha1(CHECK_KEY.encode()).hexdigest()[:8], 16)
    pf1(True)                                            # the reference rows ran with PF1 on throughout (evalsuite.env)
    t = time.time()
    if arm == "ref2":
        ref2.apply()
        toks, _ = _gen_ref2(rt, it["prompt_ids"], it["max_tokens"], greedy, seed)
        ref2.remove()
    else:
        toks, _ = _gen(rt, it["prompt_ids"], it["max_tokens"], greedy, seed)
    mx.clear_cache()
    refcheck = dict(key=CHECK_KEY, equal=toks == REF[CHECK_KEY]["completion_ids"], n=len(toks), s=round(time.time() - t, 1))
    log(f"R-56 {arm}: reference check {refcheck} (ref2.py {ref2_sha}, self-test {selftest}, SCRATCH_KEEP "
        f"{BS._SCRATCH_KEEP}, C {C})")

    def finish(new_rows):
        rec = dict(arm=arm, pid=os.getpid(), start=time.strftime("%H:%M:%S", time.localtime(t_start)),
                   end=time.strftime("%H:%M:%S"), C=C, refcheck=refcheck, ref2_sha=ref2_sha, selftest=selftest,
                   scratch_keep=BS._SCRATCH_KEEP, rows=new_rows, running_s=round(el()), wall_s=round(time.time() - t_start),
                   life_gb=round(footprint_peak_gb(), 2), swap0=round(swap0, 2), swap1=round(swap_gb(), 2),
                   fan_cap=fan_cap(), provenance=env.provenance())
        with open(os.path.join(OUT, "chunks.jsonl"), "a") as g:
            g.write(json.dumps(rec) + "\n")
        log(f"R-56 {arm}: process done: {new_rows} new row(s), lifetime peak {rec['life_gb']} GB, "
            f"{rec['running_s']} s running")
    if not refcheck["equal"]:
        log(f"R-56 {arm}: the reference row did not reproduce: no live-loop row in this process (VOID, see the PREREG)")
        finish(0)
        return

    def stats():
        try:
            s = st.stats()
            return np.array([s["misses"], s["io_s"]], dtype=np.float64)
        except Exception:
            return np.zeros(2)

    # ------------------------------------------------ taps for the drafter (dfl2c_extract's, per forward)
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
    def taps():                                          # [rows, 5 * 2560] fp16, rows of the forwards since the reset
        f = [mx.concatenate(x) if len(x) > 1 else x[0] for x in TAP["x"]]
        return mx.stack(f, axis=1).reshape(f[0].shape[0], -1)

    # ------------------------------------------------ the drafter (d02b_screen's eager loop)
    draft = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
    draft.load_weights(os.path.join(LEAP, "data", "l120_ft.safetensors"), strict=False)
    draft.bind(model)
    mx.eval(draft.parameters())
    MASK, SW = int(draft.config.mask_token_id), int(draft.config.sliding_window)

    def head2s(hid):
        """the exact two-stage top-16 (R-06): candidates [1, bs - 1, K] as vocabulary ids, their logits"""
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

    # ------------------------------------------------ the arms
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
        """ref2.prefill's calls (chunks of 1024, eval per chunk) on EvalSuite's PF1 path, plus the taps; then PF1 off"""
        pf1(True)
        cache, lg = model.make_cache(), None
        tap_reset(); TAP["on"] = True
        for a in range(0, len(ids), chunk):
            lg = model(mx.array([ids[a:a + chunk]]), cache=cache)
            mx.eval(lg, states(cache), [x[-1] for x in TAP["x"]])
        TAP["on"] = False
        pf1(False)
        return cache, lg[0, -1], taps()

    def ar_logits(ids, n):
        """REF2's own AR (ref2.generate's prefill + ar_step), the logits of the first n steps"""
        arm_on(True)
        cache, last, _ = prefill(ids)
        tok, out = int(mx.argmax(last).item()), []
        for _ in range(n):
            lg = ref2.ar_step(model, cache, tok)
            mx.eval(lg)
            out.append(lg)
            tok = int(mx.argmax(lg).item())
        arm_on(False)
        del cache
        mx.clear_cache()
        return out

    def where(lg_row, got, want):
        """at a differing token: the check row's logit gap from its argmax to the reference's token, and its top-2 gap"""
        v = lg_row.astype(mx.float32)
        t2 = sorted(mx.topk(v, 2).tolist(), reverse=True)
        return dict(gap=None if want is None else round(float((v[got] - v[want]).item()), 5), top2=round(t2[0] - t2[1], 5))

    def run(ids, ref, ar, want, cap):
        while os.path.exists(PAUSE):
            time.sleep(5)
        paused, cut, diff = False, False, None
        arm_on(ref)
        ref2.reset_counts()
        mx.synchronize()
        t1 = time.perf_counter()
        cache, last, T = prefill(ids)
        pf = time.perf_counter() - t1
        P = len(ids)
        dcache = draft.make_cache()
        start = max(0, P - (SW - 1))
        for c in dcache:
            c.offset = start
        ctx, a = T[start:P], P
        tok = int(mx.argmax(last).item())
        out, cyc, dr_t, ck_t, kept, pend = [tok], [], [], [], [], []
        if not want or want[0] != tok:
            diff = dict(at=0, got=tok, want=want[0] if want else None, row="prefill")
        s0 = stats()
        td = c_prev = time.perf_counter()
        while diff is None and len(out) < cap and tok not in EOS:
            if el() > HARD:
                cut = True
                break
            c0 = time.perf_counter()
            if out[1:]:
                cyc.append(c0 - c_prev)
            c_prev = c0
            if ref:
                ref2.remove()
            dr = np.array(propose(mx.array([[tok] + [MASK] * (R - 1)]), ctx[None].astype(mx.bfloat16), dcache)[0]).tolist()
            if (trim := dcache[0].offset - a) > 0:
                DF._trim_recent_cache(dcache, trim)
            if ref:
                ref2.apply()
            c1 = time.perf_counter()
            snap = CS.snapshot_untrimmable_cache_lazy(cache)
            tap_reset(); TAP["on"] = True
            with ref2.verify_scope(model):
                lg = model(mx.array([[tok] + dr]), cache=cache)
            TAP["on"] = False
            am, tp = mx.argmax(lg[0], axis=-1), taps()
            mx.eval(am, tp)
            am = am.tolist()
            k = 0
            while k < R - 1 and dr[k] == am[k]:
                k += 1
            model.commit_verified_window(cache, snap.states, keep_tokens=k + 1, verified_tokens=R)
            if ref:
                mx.async_eval(states(cache))             # SB: the commit replay submitted alone
            c2 = time.perf_counter()
            dr_t.append(c1 - c0); ck_t.append(c2 - c1); kept.append(k + 1)
            if ar is not None:                           # rows 0..k: inputs at positions a..a+k, all on the AR path
                pend += [(a - P + j, lg[0, j]) for j in range(k + 1) if a - P + j < len(ar)]
            new = dr[:k] + [am[k]]                       # new[j] is check row j's argmax
            for j, x in enumerate(new):
                if x in EOS:
                    new = new[:j + 1]
                    break
            for j, x in enumerate(new):                  # the first token that leaves the reference row ends the prompt
                i = len(out) + j
                w = want[i] if i < len(want) else None
                if x != w:
                    diff = dict(at=i, got=x, want=w, row=j, **where(lg[0, j], x, w))
                    new = new[:j + 1]
                    break
            out += new
            tok, ctx, a = out[-1], tp[:k + 1], a + k + 1
            paused = paused or os.path.exists(PAUSE)
        mx.eval(states(cache))
        dec = time.perf_counter() - td
        ms, io = (stats() - s0) / max(len(kept), 1)
        cnt = ref2.counts()
        bad = sum(not bool(mx.all(mx.view(x.reshape(-1), mx.uint8) == mx.view(ar[i].reshape(-1), mx.uint8)).item())
                  for i, x in pend)
        cmp = len(pend)
        del pend
        arm_on(False)
        del cache, dcache
        mx.clear_cache()
        return dict(out=out[:cap], diff=diff, pf=pf, dec=dec, checks=len(kept), kept=float(np.mean(kept)) if kept else 0.0,
                    cyc=1e3 * med(cyc), dr=1e3 * med(dr_t), ck=1e3 * med(ck_t), bad=bad, cmp=cmp, paused=paused, cut=cut,
                    ms=float(ms), io=float(1e3 * io),
                    cnt={k_: v / max(len(kept), 1) for k_, v in cnt.items() if v} if ref else None)

    # ------------------------------------------------ the prompts
    new_rows, gate = 0, getattr(st, "gate", None)
    with open(path, "a") as f:
        for key in keys:
            if key in done:
                continue
            if new_rows and el() > chunk_s:
                log(f"R-56 {arm}: chunk budget ({chunk_s:.0f} s running): {new_rows} new row(s)")
                break
            it, want = items[key], REF[key]["completion_ids"]
            ids, cap = list(it["prompt_ids"]), it["max_tokens"]
            ar = ar_logits(ids, max(NAR, 128 - len(ids) + 24)) if arm == "ref2" else None   # rows reach past absolute 112
            r = run(ids, arm == "ref2", ar, want, cap)
            del ar
            if r["cut"]:
                log(f"R-56 {arm} {key}: cut at {len(r['out'])} tokens by the hard limit ({HARD:.0f} s running): not saved")
                break
            if gate is not None:
                gate.check()                             # a timed-out gate (a frozen pass) stops the stage before the save
            n = len(r["out"])
            row = dict(key=key, set=it["set"], n_prompt=len(ids), cap=cap, n=n, ref_n=len(want),
                       ref_finish=REF[key].get("finish"), equal=r["out"] == want, diff=r["diff"],
                       sha=sha(r["out"]), ref_sha=sha(want), under128=len(ids) < 128, checks=r["checks"],
                       kept=round(r["kept"], 3), tok_s=round(n / r["dec"], 2) if r["dec"] else None,
                       cyc_ms=round(r["cyc"], 1), dr_ms=round(r["dr"], 1), ck_ms=round(r["ck"], 1),
                       prefill_s=round(r["pf"], 2), ar_cmp=r["cmp"], ar_bad=r["bad"],
                       per_check=None if r["cnt"] is None else {k_: round(v, 3) for k_, v in r["cnt"].items()},
                       misses=round(r["ms"], 1), io_ms=round(r["io"], 1), paused=r["paused"],
                       mem=dict(now=round(footprint_now_gb(), 2), life=round(footprint_peak_gb(), 2)),
                       swap_gb=round(swap_gb(), 2), fan_cap=fan_cap(), refcheck=refcheck["equal"], C=C, pid=os.getpid(),
                       t=time.strftime("%H:%M:%S"))
            f.write(json.dumps(row) + "\n"); f.flush()
            new_rows += 1
            log(f"R-56 {arm} {key}: P {len(ids)}, {n} tokens, equal {row['equal']}"
                + (f" (first diff {r['diff']})" if r["diff"] else "")
                + f", {r['checks']} checks, kept {r['kept']:.2f}, {row['tok_s']} tok/s (quiet mode: FAN_CAP {row['fan_cap']})"
                + (f", AR rows bitwise {r['cmp'] - r['bad']}/{r['cmp']}" if arm == "ref2" else "")
                + (f", per check {row['per_check']}" if row["per_check"] else "") + f", mem {row['mem']}")
    finish(new_rows)
