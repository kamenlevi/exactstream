"""RELAX (docs/PREREG_relax_2026-11-02.md; plan: docs/PLAN_relax_2026-11-02.md sections 6-7): quality-defined modes on
the 16-row live drafting loop, free-running to max_tokens. Base: R-56d's ref2to_gate16 (REF2-T + OOO + the fused
combine, BIGMOE_MAXK 160 on build_next), with live16b's loop, drafter (DFlash2 L1-20 + Leap's l120_ft) and commit.
Per set an arm sets (cf threshold, copy gate, R1 delta) (ARMS):
  - the cf gate: a row whose drafter cf for the draft it verifies is >= the threshold takes its top-2 experts at the
    model's own weights (no renorm). Its ranks 2-9 point at its own rank-0 expert at weight 0 ("route"; the numerics of
    Q1's masking, but the 16-row id union the loader posts shrinks, so fewer experts stream). EVALSUITE_RELAX_GATE=mask
    keeps their own ids at weight 0 instead (Q1's map; the smoke's bitwise check of route == mask).
  - the copy gate: a row the cf gate leaves exact whose last 8 input tokens occur in the prompt takes top-2 too;
  - R1(delta): draft j is kept while logit(d_j) >= max logit - delta on its row; the bonus is row k's argmax.
Row 15 is always exact; the commit keeps k+1 rows (unchanged). cf = the selector's softmax over its 16 candidates at the
pick (MTP Head's d02b_walk chain_cf; the picks are live16b's). One arm per process:
  `evalsuite gpu relax ref --arm A --keys FILE` (EVALSUITE_RELAX_TAG: a suffix for the output dir, smoke runs;
  EVALSUITE_RELAX_HASH=1: every committed row's hash in the check log, for the smoke's route == mask check).
Step 0 in every process: humaneval/HumanEval/27 with the gate and the rule off must equal results/ref2t's row token for token
(else VOID, exit 5); its row hashes go to step0/ for the replay's bitwise check.
Out (results/relax_<arm>[_tag]/): decode.jsonl (the decode format: `evalsuite report --variant relax_<arm> --decode
--ref ref2t` scores it; resumable), checks/<key>.jsonl.gz (one line per check: the replay's input, the gate rows, the
acceptance, G's top-64 log-probs on the committed rows, expert-union bytes, SSD bytes, times), chunks.jsonl (one line
per process: bytes per expert, dense bytes)."""
from __future__ import annotations
import gzip, hashlib, json, os, sys, threading, time
import numpy as np
import mlx.core as mx
from . import env, data as D, model as EM
from .decode import diskread_bytes, footprint_now_gb, footprint_peak_gb
from .relax_core import (R, R1, R2, COPY_N, CodeScan, comment_rows, committed, gate_keep_mx, gate_rows, keep_mask,
                         kept_rows, prompt_grams, union_bytes)

LEAP = os.path.expanduser("~/Leap")
K = 16                                                   # drafter candidates per position (live16b's K)
KTOP = 10                                                # the 125B's experts per row
EOS = {248044, 248046}
TAPS = (5, 15, 25, 35, 45)
TOPM = 64                                                # G's top log-probs kept per committed row (the KLD)
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
FAN_CAP = os.path.expanduser("~/.local/share/quiet/FAN_CAP")
OOO_ENV = {"BIGMOE_TIER": "4433333333", "BIGMOE_TIER_GLU": "3c", "BIGMOE_TIER_CL": "112", "BIGMOE_TIER_COMBINE": "1",
           "BIGMOE_OOO": "1"}
SETS = ("math", "ifeval", "humaneval", "multi", "agent", "humanevalx", "agentlong",   # + the drift re-check's (Fri)
        "humanevalt")                                                               # + its code-only top-up (amend. 2)


def S_(thr=None, copy=False, rules=(), loop=None, union=False):
    """a set's setting: the cf gate's threshold, the copy gate, the relaxed rules every row may use, the comment rule,
    and whether the gate is Q1-UNION (a sure row keeps its rank >= 2 experts that are in the check's union) or Q1"""
    return dict(thr=thr, copy=copy, rules=tuple(rules), loop=loop, union=union)


OFF = S_()
G99 = dict(thr=0.9, copy=True, union=True)              # revision 3: Q1-UNION at cf 0.9 + copy (was Q1 at 0.99)
ARMS = {                                                 # PLAN_relax sec 9 (cloud task 14 merged), PREREG revision 1
    "control": dict(hc8=False, sets={s: OFF for s in SETS}),
    "A": dict(hc8=False, sets=dict(humaneval=S_(**G99), math=S_(**G99), agent=S_(**G99))),            # the gate alone
    "B": dict(hc8=True, sets=dict(humaneval=S_(**G99, loop=R2), math=S_(**G99), agent=S_(**G99),       # the mode
                                  humanevalx=S_(**G99, loop=R2), agentlong=S_(**G99),     # PREREG_relax_drift
                                  humanevalt=S_(**G99, loop=R2))),                        # its amendment 2
    "E": dict(hc8=False, sets=dict(ifeval=S_(rules=[R1]), multi=S_(rules=[R1]))),                     # chat R1
    "E8": dict(hc8=True, sets=dict(ifeval=S_(rules=[R1]), multi=S_(rules=[R1]))),                     # E + HC8 (Fri)
    "F": dict(hc8=False, sets=dict(ifeval=S_(rules=[R1, R2]), multi=S_(rules=[R1, R2]))),             # + R2
    "C": dict(hc8=True, sets=dict(humaneval=S_(**G99, rules=[R1], loop=R2), math=S_(**G99, rules=[R1]))),   # B + R1
    "G": dict(hc8=False, sets=dict(ifeval=S_(0.999, True, [R1]), multi=S_(0.999, True, [R1]))),       # E + the gate
    "T3ALL": dict(hc8=False, tier="3333333333",                                                       # NewIdeas 02:42:
                  sets=dict(ifeval=OFF, humaneval=OFF, math=OFF)),   # every rank at 3 bits (M-26), exact acceptance
}
REF_TIER = "4433333333"                                  # REF2-T's tiers: the step-0 rows and the replay's E
STEP0 = "humaneval/HumanEval/27"
GATE = dict(verify=False, sure=None, union=False, mode="route", cap=None)   # read by the streamed switch's wrapper
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


PAUSED_RC = 76                                           # the stage exited on a long PAUSE (runners: not a failure)


def pause_wait(limit_s=300):
    """wait out a heat PAUSE before a prompt; after limit_s exit instead, so no loaded process (and no bigmoe_ext loader
    thread, which spins while it waits: Breakthrough 22:3x) keeps the Mac warm. The runner waits in bash and relaunches."""
    t0 = time.time()
    while os.path.exists(PAUSE):
        if time.time() - t0 > limit_s:
            print(f"RELAX: PAUSE for over {limit_s} s: the process exits (rc {PAUSED_RC}); the runner relaunches after it",
                  flush=True)
            raise SystemExit(PAUSED_RC)
        time.sleep(5)


def rowhash(x):
    """sha1 of a logits row's bytes (dtype and shape included): live16b's"""
    return hashlib.sha1(f"{x.dtype}{tuple(x.shape)}".encode()
                        + np.asarray(mx.view(mx.contiguous(x).reshape(-1), mx.uint8)).tobytes()).hexdigest()[:20]


def swap_gb():
    import re, subprocess
    m = re.search(r"used = ([\d.]+)M", subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout)
    return float(m.group(1)) / 1024 if m else 0.0


def out_dir(arm):
    tag = os.environ.get("EVALSUITE_RELAX_TAG", "")
    return os.path.join(env.ROOT, "results", f"relax_{arm}" + (f"_{tag}" if tag else ""))


def install_gate(model):
    """wrap the streamed switch class's __call__ (the outermost call of every MoE layer): inside a 16-row verify,
    apply GATE's keep mask to the fused combine's scores (and, in route mode, point the dropped ranks at the row's
    rank-0 expert), and capture the ids before / after for the union bytes. Returns per-layer bytes per expert."""
    lay = model.language_model.model.layers
    sws = [l.mlp.switch_mlp for l in lay if getattr(getattr(l, "mlp", None), "switch_mlp", None) is not None]
    sws = [sw for sw in sws if getattr(sw, "_bm", None) is not None]
    S = type(sws[0])
    assert all(type(sw) is S for sw in sws) and S.__name__.endswith("Streamed"), (S, {type(sw) for sw in sws})
    per = lambda d: sum(a.nbytes / a.shape[0] for f in d.values() for a in f.values())
    bb, bs, r4s = {}, {}, set()
    for sw in sws:
        t, L = sw._bm["tier"], sw._bm["L"]
        bb[L], bs[L] = per(t["base"]), per(t["sideP"]) if t.get("sideP") else 0.0
        lo4 = [s_ for s_ in t["segs"] if s_[2] == 4]
        r4s.add(lo4[-1][1] if lo4 else 0)
    assert len(r4s) == 1, r4s
    if getattr(S, "_relax", False):
        return bb, bs, r4s.pop()
    orig = S.__call__

    def call(self, x, indices, *a, **kw):
        if GATE["verify"] and indices.ndim >= 2 and indices.shape[-2] == R:
            i0, sure = indices, GATE["sure"]
            keep = gate_keep_mx(indices, sure, GATE["union"]) if sure is not None else None   # Q1 / Q1-UNION
            if keep is not None:
                t = self._bm["tier"]
                assert t.get("scores") is not None, "the gate needs the fused combine's scores (BIGMOE_TIER_COMBINE)"
                t["scores"] = mx.where(keep, t["scores"], mx.zeros_like(t["scores"]))
                if GATE["mode"] == "route":
                    indices = mx.where(keep, indices, indices[..., :1])
            if GATE["cap"] is not None:
                GATE["cap"].append((self._bm["L"], i0, indices))
        return orig(self, x, indices, *a, **kw)

    S.__call__ = call
    S._relax = True
    return bb, bs, r4s.pop()


def dense_bytes(model):
    """non-expert weight bytes one check reads: every parameter outside the experts, less the gathered embeddings"""
    from mlx.utils import tree_flatten
    pp = tree_flatten(model.parameters())
    allb = sum(v.nbytes for k_, v in pp if "switch_mlp" not in k_)
    emb = sum(v.nbytes for k_, v in pp if "switch_mlp" not in k_ and "embed" in k_)
    return int(allb - emb), int(emb)


def main(rt, st, a, log):
    arm, chunk_s, C = a.arm, (a.chunk_minutes * 60) or 540, a.C
    assert arm in ARMS, f"--arm must be one of {sorted(ARMS)}"
    for k_ in ("MTPLX_FUSED_GATE_UP", "BIGMOE_VERIFY_MIDLOOP"):
        assert k_ not in os.environ, f"{k_} is set"
    ext_ok = (os.environ.get("BIGMOE_EXT_BUILD") == "build_next"     # the Oct 30 gate build; since BigMoE's Nov 4 build
              or (os.path.basename(env.BIGMOE_CODE.rstrip("/")) != "bigmoe-6c64148"   # ext/build serves MAXK 160 too
                  and os.environ.get("BIGMOE_EXT_BUILD") in (None, "build")))           # (MAXK is runtime)
    assert os.environ.get("BIGMOE_MAXK") == "160" and ext_ok, \
        "RELAX runs on the gate: BIGMOE_MAXK=160 and the gate's ext (build_next, or ext/build after Nov 4)"
    TIER = ARMS[arm].get("tier", REF_TIER)
    got, want = {k_: os.environ.get(k_) for k_ in OOO_ENV}, dict(OOO_ENV, BIGMOE_TIER=TIER)
    assert got == want, f"tier env {got}, arm {arm} wants {want}"
    GATE["mode"] = os.environ.get("EVALSUITE_RELAX_GATE", "route")
    assert GATE["mode"] in ("route", "mask"), GATE["mode"]
    OUT = out_dir(arm)
    for d in ("checks", "step0"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    threading.Thread(target=_tick, daemon=True).start()
    t_start, swap0, HARD = time.time(), swap_gb(), chunk_s + 300
    sys.path.insert(0, os.path.join(LEAP, "third_party"))   # the drafter (dflash_model_mlx)
    rv = EM.variant(os.environ.get("EVALSUITE_RELAX_VARIANT", "ref2t"))   # ref2tf0 on the converted store (Fri)
    ref2 = EM.ref2_import(rv)
    assert ref2.CFG.get("key_buckets") is True, ref2.CFG
    import bigmoe.stream as BS
    assert BS._SCRATCH_KEEP, "BIGMOE_SCRATCH_KEEP is off"
    gate_n = int(BS.GATE_N[0]) if hasattr(BS, "GATE_N") else None
    assert gate_n == 160, f"GATE_N {gate_n}"
    import bigmoe.tier as BT
    assert BT._COMBINE and BT._OOO and BT._GLU == "3c", (BT._COMBINE, BT._OOO, BT._GLU)
    import dflash_model_mlx as DF
    from mtplx import cache_state as CS
    Q, model = ref2.Q, rt.model
    HC8 = ARMS[arm]["hc8"]
    REF = {d["key"]: d for d in map(json.loads, open(os.path.join(env.ROOT, "results", "ref2t", "decode.jsonl")))}
    REF0 = {d["key"]: d for d in map(json.loads, open(os.path.join(env.ROOT, "results", "ref2th8" if HC8 else "ref2t",
                                                                   "decode.jsonl")))}   # step 0's reference rows
    items = {it["key"]: it for s in SETS for it in D.load("dec_" + s)}
    keys = open(a.keys).read().split()
    bad = [k_ for k_ in keys if ARMS[arm]["sets"].get(items[k_]["set"]) is None]
    assert not bad, f"arm {arm} has no setting for {bad[:3]} ..."
    path = os.path.join(OUT, "decode.jsonl")
    done = {json.loads(l)["key"] for l in open(path)} if os.path.exists(path) else set()
    ref2_sha = hashlib.sha256(open(ref2.__file__, "rb").read()).hexdigest()[:12]
    if HC8:                                       # Q3 HC8 as model.load runs it: before ref2.apply() (ref2 caches the mixers)
        assert not any(os.environ.get(k_) == "1" for k_ in ("BIGMOE_COMPILE_HC", "BIGMOE_HC_VREAD", "BIGMOE_HC_PACK"))
        log("hc8_fake: %d weights at 8 bits, largest |W - W8| %.3g" % EM.hc8_fake(model))
    ref2.apply()
    if HC8:                                       # no cached concatenation of a model mixer from before the swap
        gr = {id(m): m for _, m in EM._text(model).named_modules() if type(m).__name__ == "GatedResidual"}
        for k_, w in ref2._WCAT.items():
            d_ = gr[k_].input_mix_weight_down.weight if k_ in gr else None
            assert d_ is None or mx.array_equal(w[:d_.shape[0]], d_).item(), "ref2._WCAT holds a pre-swap mixer"
    selftest = ref2.self_test()
    ref2.remove(); ref2.reset_counts()
    assert all(selftest.values()), f"ref2.self_test failed: {selftest}"
    BB, BSD, R4 = install_gate(model)
    DENSE, EMB = dense_bytes(model)
    log(f"RELAX {arm}: ref2.py {ref2_sha}, gate mode {GATE['mode']}, C {C}, R4 {R4}, bytes per expert base "
        f"{med(list(BB.values())) / 1e6:.2f} MB / sidecar {med(list(BSD.values())) / 1e6:.2f} MB, dense {DENSE / 1e9:.2f} GB "
        f"(+ {EMB / 1e9:.2f} GB embeddings), BigMoE code {env.BIGMOE_CODE}")

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

    # ------------------------------------------------ taps, drafter, proposals: live16b's, + cf at each pick
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
        """live16b's chain (the same picks) + cf: the selector's softmax over the 16 candidates at the pick"""
        sel = draft.candidate_selector
        pred, path_, conf = blk[:, 0], [], []
        for p in range(hp.shape[1]):
            sc = unary[:, p] + mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None]
                                      * sel.successor_codebook(cand[:, p]), axis=-1)
            s_ = mx.argmax(sc, axis=-1)[:, None]
            pred = mx.take_along_axis(cand[:, p], s_, axis=-1)[:, 0]
            conf.append(mx.take_along_axis(mx.softmax(sc.astype(mx.float32), axis=-1), s_, axis=-1)[:, 0])
            path_.append(pred)
        return mx.stack(path_, axis=1), mx.stack(conf, axis=1)

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

    tok_ = rt.tokenizer
    dec1 = lambda t: tok_.decode([int(t)])               # one token's text (the comment rule's scan, as mission 25)

    def run(ids, cap, cfg, hashes=False):
        """one prompt, free-running: (tokens, per-check records, timings)"""
        thr, copy, rules0, loop, union = cfg["thr"], cfg["copy"], tuple(cfg["rules"]), cfg["loop"], cfg.get("union", False)
        grams = prompt_grams(ids) if copy else None
        scan = CodeScan() if loop else None
        pause_wait()
        paused, cut = False, False
        arm_on(True)
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
        ctx, a_ = T[start:P], P
        tok = int(mx.argmax(last).item())
        out, recs, cyc = [tok], [], []
        if scan is not None:
            scan.feed(dec1(tok))
        s0, d0 = stats(), diskread_bytes()
        td = c_prev = time.perf_counter()
        while len(out) < cap and tok not in EOS:
            if el() > HARD:
                cut = True
                break
            c0 = time.perf_counter()
            if out[1:]:
                cyc.append(c0 - c_prev)
            c_prev = c0
            ref2.remove()
            pm, cm = propose(mx.array([[tok] + [MASK] * (R - 1)]), ctx[None].astype(mx.bfloat16), dcache)
            dr, cf = np.array(pm[0]).tolist(), np.array(cm[0]).tolist()
            if (trim := dcache[0].offset - a_) > 0:
                DF._trim_recent_cache(dcache, trim)
            c1 = time.perf_counter()
            tail = out[-COPY_N:] if len(out) >= COPY_N else ids[len(out) - COPY_N:] + out
            g = gate_rows(cf, tail, dr, thr, grams)
            km = keep_mask(g, KTOP)
            if scan is not None:                         # the comment rule: the scan state before each draft
                dtx, pre, s_ = [dec1(x) for x in dr], [], scan.copy()
                for x in dtx:
                    pre.append((s_.in_comment(), s_.ctext))
                    s_.feed(x)
            ref2.apply()
            snap = CS.snapshot_untrimmable_cache_lazy(cache)
            tap_reset(); TAP["on"] = True
            GATE.update(verify=True, sure=mx.array([x > 0 for x in g]) if km is not None else None, union=union, cap=[])
            try:
                with ref2.verify_scope(model):
                    lg = model(mx.array([[tok] + dr]), cache=cache)
            finally:
                cap_ = GATE["cap"]
                GATE.update(verify=False, sure=None, cap=None)
            TAP["on"] = False
            l0 = lg[0].astype(mx.float32)
            am = mx.argmax(lg[0], axis=-1)
            ld = mx.take_along_axis(l0[:R - 1], mx.array(dr, dtype=mx.int32)[:, None], axis=-1)   # each draft's logit
            dl = ld[:, 0] - l0[:R - 1].max(axis=-1)
            drk = (l0[:R - 1] > ld).sum(axis=-1)                # the draft's rank on its row (0 = the argmax)
            lse = mx.logsumexp(l0, axis=-1)
            ti = mx.argpartition(-l0, kth=TOPM - 1, axis=-1)[:, :TOPM]
            tl = mx.take_along_axis(l0, ti, axis=-1) - lse[:, None]
            tp = taps()
            io_ = mx.stack([c_[1].reshape(R, -1) for c_ in cap_])
            in_ = mx.stack([c_[2].reshape(R, -1) for c_ in cap_])
            mx.eval(am, tp, lg, dl, drk, ti, tl, io_, in_)
            c2 = time.perf_counter()
            am, dlm, drk = am.tolist(), dl.tolist(), drk.tolist()
            cm = comment_rows(pre, dtx, [dec1(x) for x in am[:R - 1]]) if scan is not None else []
            kx, k = kept_rows(dr, am, dlm, drk, [rules0 + ((loop,) if j in cm else ()) for j in range(R)])
            model.commit_verified_window(cache, snap.states, keep_tokens=k + 1, verified_tokens=R)
            mx.async_eval(states(cache))                 # SB: the commit replay submitted alone
            new = committed(dr, am, k, EOS, cap - len(out))
            if scan is not None:
                scan.feed("".join(dec1(x) for x in new))
            c3 = time.perf_counter()
            Ls = [c_[0] for c_ in cap_]
            bn = union_bytes(np.array(in_), R4, [BB[L] for L in Ls], [BSD[L] for L in Ls])
            bx = union_bytes(np.array(io_), R4, [BB[L] for L in Ls], [BSD[L] for L in Ls])
            s1, d1 = stats(), diskread_bytes()
            n = len(new)
            rec = dict(c=len(recs), pos=len(out), inp=[tok] + dr, cf=[round(x, 6) for x in cf], g=g, am=am, k=k, kx=kx,
                       n=n, dlm=[round(x, 3) for x in dlm], drk=drk, cm=cm, ti=np.array(ti)[:n].tolist(),
                       tl=np.round(np.array(tl)[:n], 4).tolist(), ub=bn[0], ubx=bx[0], ne=bn[1], nex=bx[1],
                       ssd=int(d1 - d0), ms=int(s1[0] - s0[0]), io=round(float(1e3 * (s1[1] - s0[1])), 2),
                       t=[round(1e3 * (c1 - c0), 2), round(1e3 * (c2 - c1), 2), round(1e3 * (c3 - c2), 2)])
            if hashes:
                rec["h"] = [rowhash(lg[0, j]) for j in range(n)]
            recs.append(rec)
            s0, d0 = s1, d1
            out += new
            tok, ctx, a_ = out[-1], tp[:k + 1], a_ + k + 1
            paused = paused or os.path.exists(PAUSE)
        mx.eval(states(cache))
        dec = time.perf_counter() - td
        arm_on(False)
        del cache, dcache
        mx.clear_cache()
        return dict(out=out[:cap], recs=recs, pf=pf, dec=dec, cyc=1e3 * med(cyc), paused=paused, cut=cut)

    # ------------------------------------------------ step 0: the gate and the rule off reproduce REF2-T
    if TIER != REF_TIER:                          # T3ALL: no exact rows exist for this plan; check the plan took effect
        assert BT.PLAN == TIER and R4 == 0, (BT.PLAN, R4)
        log(f"RELAX {arm} step 0: plan {TIER} has no exact rows; the plan is in effect (bigmoe.tier PLAN {BT.PLAN}, "
            f"4-bit ranks {R4}); the replay checks itself on the control's step-0 rows")
        ok0 = None
    it0 = items[STEP0] if TIER == REF_TIER else None
    if it0 is not None:
        r0 = run(list(it0["prompt_ids"]), it0["max_tokens"], OFF, hashes=True)
        ok0 = r0["out"] == REF0[STEP0]["completion_ids"]
        json.dump(dict(key=STEP0, out=r0["out"], recs=r0["recs"], equal=ok0, pid=os.getpid(), t=time.strftime("%H:%M:%S")),
                  open(os.path.join(OUT, "step0", f"{os.getpid()}.json"), "w"))
        log(f"RELAX {arm} step 0: {STEP0} gate and rule off, {len(r0['out'])} tokens, equal to results/"
            f"{'ref2th8' if HC8 else 'ref2t'}: {ok0}")
        if not ok0:
            log(f"RELAX {arm}: VOID: step 0 does not reproduce {'REF2-T + HC8' if HC8 else 'REF2-T'}")
            raise SystemExit(5)

    # ------------------------------------------------ the prompts
    new_rows, gate_, yielded = 0, getattr(st, "gate", None), False
    with open(path, "a") as f:
        for key in keys:
            if key in done:
                continue
            if new_rows and el() > chunk_s:
                log(f"RELAX {arm}: chunk budget ({chunk_s:.0f} s running): {new_rows} new row(s)")
                break
            yflag = os.environ.get("EVALSUITE_YIELD_FLAG")
            if yflag and os.path.exists(yflag):          # a quiet ticket waits (cli --share): stop at this row boundary
                log(f"RELAX {arm}: yielding to a quiet ticket after {new_rows} new row(s)")
                yielded = True
                break
            it = items[key]
            cfg = ARMS[arm]["sets"][it["set"]]
            ids, cap = list(it["prompt_ids"]), it["max_tokens"]
            t0 = time.time()
            r = run(ids, cap, cfg, hashes=os.environ.get("EVALSUITE_RELAX_HASH") == "1")   # smoke: row hashes
            if r["cut"]:
                log(f"RELAX {arm} {key}: cut at {len(r['out'])} tokens by the hard limit ({HARD:.0f} s running): not saved")
                break
            if gate_ is not None:
                gate_.check()
            out, recs = r["out"], r["recs"]
            cp = os.path.join(OUT, "checks", key.replace("/", "__") + ".jsonl.gz")
            with gzip.open(cp + ".tmp", "wt") as g:
                for rec in recs:
                    g.write(json.dumps(rec) + "\n")
            os.replace(cp + ".tmp", cp)
            nc = max(len(recs), 1)
            gs = [x for rec in recs for x in rec["g"]]
            row = dict(key=key, set=it["set"], n_prompt=len(ids), completion_ids=out,
                       finish="eos" if out and out[-1] in EOS else "length",
                       text=tok_.decode([x for x in out if x not in EOS]), seconds=round(time.time() - t0, 2),
                       arm=arm, cfg=cfg, hc8=HC8, tier=TIER, gate_mode=GATE["mode"], sha=sha(out),
                       ref_sha=sha(REF[key]["completion_ids"]) if key in REF else None,
                       equal_ref=out == REF[key]["completion_ids"] if key in REF else None, checks=len(recs),
                       kept=round(float(np.mean([rec["k"] + 1 for rec in recs])) if recs else 0.0, 3),
                       kept_x=round(float(np.mean([rec["kx"] + 1 for rec in recs])) if recs else 0.0, 3),
                       g_cf=round(gs.count(1) / max(len(gs), 1), 4), g_copy=round(gs.count(2) / max(len(gs), 1), 4),
                       comment_rows=sum(len(x["cm"]) for x in recs),
                       tok_s=round(len(out) / r["dec"], 2) if r["dec"] else None, cyc_ms=round(r["cyc"], 1),
                       prefill_s=round(r["pf"], 2), dec_s=round(r["dec"], 2),
                       first_check_ms=round(sum(recs[0]["t"]), 1) if recs else None,   # AT14's read-out (TTFT = prefill_s)
                       ub_gb=round(sum(x["ub"] for x in recs) / nc / 1e9, 3),
                       ubx_gb=round(sum(x["ubx"] for x in recs) / nc / 1e9, 3),
                       ssd_gb=round(sum(x["ssd"] for x in recs) / nc / 1e9, 3),
                       misses=round(sum(x["ms"] for x in recs) / nc, 1), io_ms=round(sum(x["io"] for x in recs) / nc, 1),
                       verify_ms=round(med([x["t"][1] for x in recs]), 1), draft_ms=round(med([x["t"][0] for x in recs]), 1),
                       paused=r["paused"], mem=dict(now=round(footprint_now_gb(), 2), life=round(footprint_peak_gb(), 2)),
                       swap_gb=round(swap_gb(), 2), fan_cap=open(FAN_CAP).read().strip() if os.path.exists(FAN_CAP) else None,
                       C=C, pid=os.getpid(), t=time.strftime("%H:%M:%S"))
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            new_rows += 1
            log(f"RELAX {arm} {key}: {len(out)} tokens ({row['finish']}), equal to REF2-T {row['equal_ref']}, kept "
                f"{row['kept']:.2f} (exact rule {row['kept_x']:.2f}), gated rows cf {row['g_cf']:.1%} copy {row['g_copy']:.1%}, "
                f"union {row['ub_gb']:.2f} GB/check (exact {row['ubx_gb']:.2f}), SSD {row['ssd_gb']:.2f} GB/check, "
                f"{row['tok_s']} tok/s, mem {row['mem']}")
    rec = dict(arm=arm, pid=os.getpid(), start=time.strftime("%H:%M:%S", time.localtime(t_start)), end=time.strftime("%H:%M:%S"),
               C=C, hc8=HC8, tier=TIER, gate_mode=GATE["mode"], ref2_sha=ref2_sha, selftest=selftest, gate_n=gate_n, r4=R4,
               b_base=BB, b_side=BSD, dense=DENSE, emb=EMB, step0=ok0, rows=new_rows, running_s=round(el()),
               wall_s=round(time.time() - t_start), life_gb=round(footprint_peak_gb(), 2), swap0=round(swap0, 2),
               swap1=round(swap_gb(), 2), provenance=env.provenance())
    with open(os.path.join(OUT, "chunks.jsonl"), "a") as g:
        g.write(json.dumps(rec) + "\n")
    log(f"RELAX {arm}: process done: {new_rows} new row(s), lifetime peak {rec['life_gb']} GB, {rec['running_s']} s running")
    return "yield" if yielded else None
