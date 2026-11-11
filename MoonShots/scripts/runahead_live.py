"""M-16 RUNAHEAD step 2 (MoonShots, Sun 2026-11-01; QUEUE M-16, lines written first): the live dual pass on the 45 GB default.
Per check of the live REF2-T loop (R57_PRESET=45: tiers 4433333333 / 3c, R-55 gate MAXK 160, L1-14 truncation t 0.3, OOO +
TIER_COMBINE, DFlash2 L1-20 drafting, greedy): first a RUNAHEAD pass over the drafted rows that computes only the expert pairs
resident at the check's start (non-resident pairs contribute 0; no gate post, no SSD read, no LRU touch; slots from the ext's
side-effect-free lookup), rolled back with mtplx's rollback_after_verify; then the normal exact pass (unchanged path; its
tokens must stay == S45-T's REF2-T reference). Measured per check:
  U1  recall: the exact pass's misses on the rows the next pass would re-check (rows 0..k) that the runahead requested
      (main MSB-3 pool and the 4-bit ranks' LSB side pool), and the runahead's requests the exact rows did not need;
  U2  provisional verdict == exact verdict (accepted length and bonus token), and per-row argmax agreement;
  U3  the next block drafted from the runahead taps (rows 0..k) == the block drafted from the exact taps (LCP, and the
      next check's acceptance it would keep);
  U4  runahead pass ms (resident-only compute, no SSD) vs rows; the exact check + commit ms, misses, demand SSD ms.
Model (QUEUE M-16): pipelined pass = draft + max(compute(A + B), SSD at deep queue for B's requests) + commit + residual
stalls; compute(A + B) from the runahead ms-vs-rows fit; a check whose provisional verdict differs costs one extra pass with
no tokens; U3 trims the next check's tokens to what the runahead-drafted block would keep. Today = draft + exact check + commit
as measured here. Lines: PASS (-> BIG + a BigMoE build proposal) = modelled tok/s >= +40 % at C 224; KILL < +15 %.
Harness = BigMoE's r57_screen.py (read-only imports of BigMoE / EvalSuite / Leap; nothing in those repos is written).
usage (tonight, a quiet 45 GB turn): R57_PRESET=45 BIGMOE_EXT_BUILD=build [BigMoE venv] python runahead_live.py [C=224]
       [CAP=256] -> ~/MoonShots/data/runahead/ra_<C>.txt + ra_<C>_checks.jsonl"""
import json, os, sys, threading, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)
os.environ.pop("BIGMOE_TIER", None)
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
assert os.environ.get("R57_PRESET") == "45", "run on the 45 GB default (R57_PRESET=45)"
os.environ["BIGMOE_TIER"] = "4433333333"; os.environ["BIGMOE_TIER_GLU"] = "3c"
os.environ.setdefault("BIGMOE_TIER_CL", "112")
os.environ.setdefault("BIGMOE_TIER_COMBINE", "1")
os.environ.setdefault("BIGMOE_OOO", "1")
os.environ.setdefault("BIGMOE_MAXK", "160")
assert os.environ.get("BIGMOE_MAXK") == "160"
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
sys.path.insert(1, os.path.expanduser("~/Leap/third_party"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
from evalsuite.decode import footprint_now_gb, footprint_peak_gb
import numpy as np
import mlx.core as mx
import dflash_model_mlx as DF

OUT = os.path.expanduser("~/MoonShots/data/runahead"); os.makedirs(OUT, exist_ok=True)
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
C = int(sys.argv[1]) if len(sys.argv) > 1 else 224
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 256
R, K = 16, 16
EOS = {248044, 248046}
TAPS = (5, 15, 25, 35, 45)
TRUNC = 0.3
EARLY = int(os.environ.get("RA_EARLY", "8"))             # hybrid8: layers 0..EARLY-1 wait for every miss
PICK = (("humaneval", "humaneval/HumanEval/68"), ("math", "math500/test/prealgebra/1640.json"),
        ("humaneval", "humaneval/HumanEval/91"), ("math", "math500/test/algebra/694.json"),
        ("math", "math500/test/algebra/1282.json"), ("math", "gsm8k/708"))
CHAT = int(os.environ.get("RA_CHAT", "3"))               # + the first CHAT chat prompts (ifeval, ifeval, multi): no REF2-T
PICK = PICK[:int(os.environ.get("RA_PROMPTS", "6"))]     # reference there; the exact path is the reference by construction
HARD = float(os.environ.get("RA_HARD", "1000"))          # s of running time: no new check after this (turn cap 1200 s)
ACT = [0.0]
def _tick():
    last = time.time()
    while True:
        time.sleep(1)
        now = time.time()
        ACT[0] += min(now - last, 3.0)
        last = now
threading.Thread(target=_tick, daemon=True).start()
el = lambda: ACT[0]
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
med = lambda v: float(np.median(v)) if len(v) else float("nan")

t0 = time.time()
rt, st = EM.load(EM.variant("ref"), C=C)
model = rt.model
import ref2
import bigmoe.stream as BS
import bigmoe.gated as BG
import bigmoe.tier as BT
from mtplx import cache_state as CS
from mtplx.models import qwen4_exp as QX
Q = ref2.Q
TG = BT._tier_glu_mod()
assert BT._COMBINE and BT._GLU == "3c" and BT._OOO, (BT._COMBINE, BT._GLU, BT._OOO)
PLAN = BT.PLAN
RA = {"on": False, "mode": "skip", "main": None, "side": None, "main0": None, "side0": None, "cap": None, "calls": 0}
_ev_block = QX.SparseMoeBlock.__call__


def _runahead_moe(x, inds, scores, bm):
    """REF2-T's tier_glu_phase on the pairs resident at the check's start (phase 1 of _ooo_call with the ext's lookup in
    place of the gate's post); the other pairs give 0 in the combine. Reads no record, posts nothing, touches no stamp."""
    t, L = bm["tier"], bm["L"]
    RA["calls"] += 1
    Kk = inds.shape[-1]
    x2 = x.reshape(-1, x.shape[-1])
    M = x2.shape[0]
    ids = inds.reshape(M, Kk)
    lo4 = [s for s in t["segs"] if s[2] == 4]
    R4 = lo4[-1][1] if lo4 else 0
    pre_m = mx.take(RA["main"][L], ids.reshape(-1)).reshape(M, Kk)       # slots now
    ok = pre_m >= 0
    pre_l = None
    if R4:
        pre_l = mx.take(RA["side"][L], ids[:, :R4].reshape(-1)).reshape(M, R4)
        ok = ok & mx.concatenate([pre_l >= 0, mx.ones((M, Kk - R4), mx.bool_)], axis=-1)
    if RA["mode"] == "hybrid8" and EXT[L] < EARLY:       # hybrid8: every miss of the first EARLY layers is waited for
        pass                                             # (all pairs with a slot now are computed)
    elif RA["mode"] in ("hybrid", "hybrid8"):            # step 1's hybrid: an expert missing at the check's start is computed
        res0 = mx.take(RA["main0"][L], ids.reshape(-1)).reshape(M, Kk) >= 0   # only if some row needs it at a 4-bit rank
        if R4:                                           # (the runahead would wait for those reads); the rest stays skipped
            res0 = res0 & mx.concatenate([mx.take(RA["side0"][L], ids[:, :R4].reshape(-1)).reshape(M, R4) >= 0,
                                          mx.ones((M, Kk - R4), mx.bool_)], axis=-1)
            idi = ids.astype(mx.int32)                   # routing ids are uint32: -1 needs a signed type (23:28 crash)
            low = mx.where(res0[:, :R4], -1, idi[:, :R4]).reshape(-1)
            wait = (idi[..., None] == low[None, None, :]).any(axis=-1)
        else:
            wait = mx.zeros((M, Kk), mx.bool_)
        ok = ok & (res0 | wait)
    phase = mx.where(ok, 1, 2).astype(mx.int32)
    y1 = TG.tier_glu_phase(x2, t["base"], t["sideP"], mx.maximum(pre_m, 0), mx.maximum(pre_l, 0) if R4 else None,
                           phase, 1, plan=PLAN)
    return TG.combine_phase(y1, mx.zeros_like(y1), phase, scores.reshape(-1, Kk)).reshape(*x.shape[:-1], -1)


def _block(self, x):
    """r57_screen.py's fused block (EvalSuite's sorted top-k, combine in tier_call at 1-16 rows), + the runahead branch"""
    bm = getattr(self.switch_mlp, "_bm", None)
    if (bm is None or "tier" not in bm or not EM.STATE.sorted or x.ndim < 2 or int(np.prod(x.shape[:-1])) > 16):
        return _ev_block(self, x)
    gates = mx.softmax(self.gate(x), axis=-1, precise=True)
    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    order = mx.argsort(-scores, axis=-1)
    inds, scores = mx.take_along_axis(inds, order, axis=-1), mx.take_along_axis(scores, order, axis=-1)
    if RA["cap"] is not None:
        RA["cap"].setdefault(bm["L"], []).append(inds)
    shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
    if RA["on"]:
        return _runahead_moe(x, inds, scores, bm) + shared
    bm["tier"]["scores"], bm["tier"]["shared"] = scores, shared
    try:
        y = self.switch_mlp(x, inds)
    finally:
        bm["tier"].pop("scores", None); bm["tier"].pop("shared", None)
    if y.ndim == inds.ndim + 1:
        y = (y * scores[..., None]).sum(axis=-2)
    return y + shared
QX.SparseMoeBlock.__call__ = _block

lay = model.language_model.model.layers
NL = len(lay)
SW_ = [l.mlp.switch_mlp for l in lay]
TIERL = [L for L in range(NL) if getattr(SW_[L], "_bm", None) and "tier" in SW_[L]._bm]
EXT = {SW_[L]._bm["L"]: L for L in TIERL}               # ext layer index (bm["L"], the block's key) -> model layer
SIDE = {e: SW_[L]._bm["tier"]["side"] for e, L in EXT.items()}
say(f"M-16 RUNAHEAD step 2: full 125B via load_streamed (C = {C}, REF2-T {PLAN} / 3c, C_l {os.environ['BIGMOE_TIER_CL']}, "
    f"gate MAXK {BG.MAXK}, OOO {BT._OOO}, COMBINE {BT._COMBINE}), loaded in {time.time() - t0:.0f} s, "
    f"{mx.get_active_memory() / 1e9:.1f} GB active; {len(TIERL)} streamed tier layers; CAP {CAP}")


def tables():
    """slot tables now: main MSB-3 pool and the LSB side pool per tier layer (-1 = not resident); sets RA's slot tables"""
    mn, sd = {}, {}
    rng = list(range(512))
    for e in EXT:                                        # keyed by the ext index, as the block sees bm["L"]
        mn[e] = np.array(BS.ext.lookup(e, rng), np.int32)
        sd[e] = np.array(BS.ext.lookup(SIDE[e], rng), np.int32)
    RA["main"] = {e: mx.array(v) for e, v in mn.items()}
    RA["side"] = {e: mx.array(v) for e, v in sd.items()}
    return mn, sd


def stats():
    s = st.stats()
    return np.array([s["misses"], s["io_s"], s["slow_s"], s["slow_calls"]], dtype=np.float64)

# ---------------------------------------------------------------- taps for the drafter (r57_screen.py / Leap's S45)
TAP = {"on": False, "x": [[] for _ in TAPS]}
TAP_OF = {id(lay[L]): i for i, L in enumerate(TAPS)}
_layer_call = Q.DecoderLayer.__call__
def _layer(self, *a, **kw):
    out = _layer_call(self, *a, **kw)
    i = TAP_OF.get(id(self))
    if TAP["on"] and i is not None:
        o = out if isinstance(out, mx.array) else out[0]
        B, L_, W = o.shape
        TAP["x"][i].append(o.reshape(B, L_, 4, W // 4).astype(mx.float32).mean(axis=2)[0].astype(mx.float16))
    return out
Q.DecoderLayer.__call__ = _layer
def tap_reset():
    TAP["x"] = [[] for _ in TAPS]
def taps():
    f = [mx.concatenate(x) if len(x) > 1 else x[0] for x in TAP["x"]]
    return mx.stack(f, axis=1).reshape(f[0].shape[0], -1)

draft = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.load_weights(os.path.expanduser("~/Leap/data/l120_ft.safetensors"), strict=False)
draft.bind(model)
mx.eval(draft.parameters())
MASK, SWIN = int(draft.config.mask_token_id), int(draft.config.sliding_window)

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
    pred, path, conf = blk[:, 0], [], []
    for p in range(hp.shape[1]):
        sc = unary[:, p] + mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None]
                                  * sel.successor_codebook(cand[:, p]), axis=-1)
        s_ = mx.argmax(sc, axis=-1)
        pred = mx.take_along_axis(cand[:, p], s_[:, None], axis=-1)[:, 0]
        path.append(pred)
        conf.append(mx.take_along_axis(mx.softmax(sc.astype(mx.float32), axis=-1), s_[:, None], axis=-1)[:, 0])
    return mx.stack(path, axis=1), mx.stack(conf, axis=1)

def propose(tok, ctx, dcache):
    """r57's propose + the gate+t truncation; returns the drafted tokens (the drafter cache keeps the block rows: trim)"""
    hid = draft.hidden_states(mx.array([[tok] + [MASK] * (R - 1)]), ctx[None].astype(mx.bfloat16), dcache, 1)
    cand, unary = head2s(hid)
    pth, cf = chain(mx.array([[tok] + [MASK] * (R - 1)]), unary, cand, draft.candidate_selector.hidden_projection(hid))
    dr = np.array(pth[0]).tolist()
    return dr[:int((np.cumprod(np.array(cf[0])) >= TRUNC).sum())]

_midloop = Q._prefill_midloop_eval_layers
def ref_on(on):
    if on:
        ref2.apply()
        Q._prefill_midloop_eval_layers = lambda rows: _midloop(rows) or (4 if 2 <= int(rows) <= 16 else 0)   # SB
    else:
        ref2.remove()
        Q._prefill_midloop_eval_layers = _midloop

def states(cache):
    return [c.state for c in cache if c is not None and hasattr(c, "state")]

def pf1(on):
    if not on and BS._STAGE["bufs"] is not None:
        BS._stage_drop()
    BS._PF_PREFILL = on

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

def set_gate(n):
    BS.GATE_N[0] = n
    BS._PF_RANKED = False
    for sw in SW_:
        sw._bm.pop("next", None)

def capnp(cap):
    return {e: np.array(v[0]).reshape(-1, v[0].shape[-1]) for e, v in cap.items() if e in EXT}


def u1(capR, capE, mn, sd, k, early=0):
    """misses of the exact rows 0..k (main pool, all ranks; side pool, the 4-bit ranks) vs the runahead's requests (all its
    rows); requests the exact rows 0..k did not need; the exact pass's misses on all its rows (today's reads)"""
    R4 = PLAN.count("4")
    miss, cov, req, waste, today, wait = 0, 0, 0, 0, 0, 0
    for L in EXT:
        if L not in capR or L not in capE:
            continue
        ir, ie = capR[L], capE[L]
        rq = set(ir[mn[L][ir] < 0].tolist()) | {("s", e) for e in ir[:, :R4][sd[L][ir[:, :R4]] < 0].tolist()}
        if EXT[L] < early:                               # hybrid8: these reads are waited for, not queued
            wait += len(rq); miss_e = rq
        else:
            miss_e = None
        ea = ie[:k + 1]
        need = set(ea[mn[L][ea] < 0].tolist()) | {("s", e) for e in ea[:, :R4][sd[L][ea[:, :R4]] < 0].tolist()}
        allm = set(ie[mn[L][ie] < 0].tolist()) | {("s", e) for e in ie[:, :R4][sd[L][ie[:, :R4]] < 0].tolist()}
        lo = ir[:, :R4]
        low = {e for e in lo[(mn[L][lo] < 0) | (sd[L][lo] < 0)].tolist()}
        if miss_e is None:
            wait += len({e for e in low if mn[L][e] < 0}) + len({e for e in low if sd[L][e] < 0})
        miss += len(need); cov += len(need & rq); req += len(rq); waste += len(rq - need); today += len(allm)
    return miss, cov, req, waste, today, wait


def run(ids, name, cap=None):
    while os.path.exists(PAUSE):
        time.sleep(5)
    set_gate(40)                                         # the prefill as shipped (r57: slot arm), then the R-55 gate
    ref_on(True)
    cache, last, T = prefill(ids)
    set_gate(160)
    CAPN = cap or CAP
    P = len(ids)
    dcache = draft.make_cache()
    start = max(0, P - (SWIN - 1))
    for c in dcache:
        c.offset = start
    ctx, a = T[start:P], P
    tok = int(mx.argmax(last).item())
    out, recs, prevR, cut = [tok], [], {}, False
    while len(out) < CAPN and tok not in EOS:
        if el() > HARD:
            cut = True
            break
        c0 = time.perf_counter()
        ref2.remove()
        dr = propose(tok, ctx, dcache)
        if (trim := dcache[0].offset - a) > 0:
            DF._trim_recent_cache(dcache, trim)
        ref2.apply()
        c1 = time.perf_counter()
        rec = {"rows": len(dr) + 1, "draft_ms": 1e3 * (c1 - c0)}
        for nm, blk in (prevR or {}).items():             # U3 for the previous check: each variant's block vs this one
            lcp = next((i for i, (u, v) in enumerate(zip(blk, dr)) if u != v), min(len(blk), len(dr)))
            rec[{"skip": "u3", "hybrid": "u3h", "hybrid8": "u3h8"}[nm]] = {"lcp": lcp, "lenR": len(blk), "lenE": len(dr),
                                                                          "same": blk == dr}
        # ---- (1) runahead SKIP pass on the check-start residency (rolled back)
        mx.eval(states(cache)); mx.synchronize()          # Leap 12:5x: the previous commit replay (and any load it made)
        mn, sd = tables()                                 # must be done before the slot tables are read
        RA["main0"], RA["side0"] = RA["main"], RA["side"]
        def ra_pass(mode):
            s_r = stats()
            snapR = CS.snapshot_untrimmable_cache_lazy(cache)
            RA["cap"], RA["on"], RA["calls"], RA["mode"] = {}, True, 0, mode
            tap_reset(); TAP["on"] = True
            mx.synchronize()
            r0 = time.perf_counter()
            with ref2.verify_scope(model):
                lgR = model(mx.array([[tok] + dr]), cache=cache)
            amR, tpR = mx.argmax(lgR[0], axis=-1), taps()
            t2 = mx.topk(lgR[0].astype(mx.float32), 2, axis=-1)
            i2 = mx.argpartition(-lgR[0].astype(mx.float32), kth=1, axis=-1)[..., :2]   # the top-2 token ids (unordered)
            mx.eval(amR, tpR, t2, i2)
            r1 = time.perf_counter()
            TAP["on"] = False; RA["on"] = False
            capR = capnp(RA["cap"]); RA["cap"] = None
            CS.rollback_after_verify(cache, snapR, len(dr) + 1)
            model.clear_verify_capture(cache)
            mx.eval(states(cache))
            assert RA["calls"] >= len(EXT), ("runahead branch did not run on every streamed layer", RA["calls"])
            mg = np.abs(np.diff(np.array(t2), axis=-1))[:, 0].round(4).tolist()     # top-1 minus top-2 logit, per row
            a1, i2n = np.array(amR), np.array(i2)
            sec = np.where(i2n[:, 0] == a1, i2n[:, 1], i2n[:, 0]).tolist()          # the runahead's #2 token per row
            RA["second"] = sec
            return amR.tolist(), tpR, capR, 1e3 * (r1 - r0), float((stats() - s_r)[0]), mg
        amR, tpR, capR, ra_ms, ra_miss, mgR = ra_pass("skip")
        secR = RA["second"]
        # ---- (2) the exact pass (r57_screen.py's check: loads its misses), rolled back after the read-out
        s0 = stats()
        c2 = time.perf_counter()
        snap = CS.snapshot_untrimmable_cache_lazy(cache)
        RA["cap"] = {}
        tap_reset(); TAP["on"] = True
        with ref2.verify_scope(model):
            lg = model(mx.array([[tok] + dr]), cache=cache)
        TAP["on"] = False
        am, tp = mx.argmax(lg[0], axis=-1), taps()
        mx.eval(am, tp)
        c3 = time.perf_counter()
        sm = stats() - s0
        capE = capnp(RA["cap"]); RA["cap"] = None
        am = am.tolist()
        CS.rollback_after_verify(cache, snap, len(dr) + 1)
        model.clear_verify_capture(cache)
        mx.eval(states(cache))
        # ---- (3) runahead HYBRID pass: check-start residency decides the skips, today's slots compute the rest
        tables()
        amH, tpH, capH, hy_ms, hy_miss, mgH = ra_pass("hybrid")
        secH = RA["second"]
        am8, tp8, cap8, h8_ms, h8_miss, mg8 = ra_pass("hybrid8")
        sec8 = RA["second"]
        # ---- (4) the exact pass again, all resident now (its time = the exact compute of these rows), then the commit
        s1 = stats()
        snap2 = CS.snapshot_untrimmable_cache_lazy(cache)
        tap_reset(); TAP["on"] = True
        mx.synchronize()
        c4 = time.perf_counter()
        with ref2.verify_scope(model):
            lg2 = model(mx.array([[tok] + dr]), cache=cache)
        TAP["on"] = False
        am2, tp2 = mx.argmax(lg2[0], axis=-1), taps()
        t2e = mx.topk(lg2[0].astype(mx.float32), 2, axis=-1)
        mx.eval(am2, tp2, t2e)
        mgE = np.abs(np.diff(np.array(t2e), axis=-1))[:, 0].round(4).tolist()
        c5 = time.perf_counter()
        am2 = am2.tolist()
        same_bits = bool(mx.array_equal(lg2, lg).item())
        k = 0
        while k < len(dr) and dr[k] == am2[k]:
            k += 1
        model.commit_verified_window(cache, snap2.states, keep_tokens=k + 1, verified_tokens=len(dr) + 1)
        mx.async_eval(states(cache))
        res_miss = float((stats() - s1)[0])
        kE = 0
        while kE < len(dr) and dr[kE] == am[kE]:
            kE += 1
        def verdict(amX, secX):
            kX = 0
            while kX < len(dr) and dr[kX] == amX[kX]:
                kX += 1
            return dict(kR=kX, agree=(kX == k and amX[k] == am2[k]),
                        rows_eq=int(sum(x == y for x, y in zip(amX[:k + 1], am2[:k + 1]))),
                        second_is_exact=bool(kX == k and amX[k] != am2[k] and secX[k] == am2[k]))   # a #2 hedge would hit
        miss, cov, req, waste, today, _ = u1(capR, capE, mn, sd, k)
        missH, covH, reqH, wasteH, _, waitH = u1(capH, capE, mn, sd, k)
        miss8, cov8, req8, waste8, _, wait8 = u1(cap8, capE, mn, sd, k, early=EARLY)
        rec.update(k=k, rerun_equal=same_bits and kE == k, skip=verdict(amR, secR), hybrid=verdict(amH, secH), hybrid8=verdict(am8, sec8),
                   ra_ms=ra_ms, hy_ms=hy_ms, h8_ms=h8_ms, h8_misses=h8_miss,
                   u1h8=dict(miss=miss8, cov=cov8, req=req8, waste=waste8, wait=wait8),
                   res_ms=1e3 * (c5 - c4), ck_ms=1e3 * (c3 - c2), misses=float(sm[0]), io_ms=1e3 * float(sm[1]),
                   ra_misses=ra_miss, hy_misses=hy_miss, res_misses=res_miss,
                   u1=dict(miss=miss, cov=cov, req=req, waste=waste, today=today),
                   u1h=dict(miss=missH, cov=covH, req=reqH, waste=wasteH, wait=waitH))
        rec["agree"] = rec["skip"]["agree"]
        rec["marg"] = {"exact": mgE, "skip": mgR, "hybrid": mgH, "hybrid8": mg8}   # per row, rows 0..len(dr)
        tp = tp2
        am = am2
        new = dr[:k] + [am[k]]
        for j, x in enumerate(new):
            if x in EOS:
                new = new[:j + 1]
                break
        out += new
        prevR = {}
        for nm, tX in (("skip", tpR), ("hybrid", tpH), ("hybrid8", tp8)):   # U3: the next block from each variant's taps
            if rec[nm]["agree"] and out[-1] not in EOS:    # cache back where the normal proposal finds it
                ref2.remove()
                off0 = dcache[0].offset
                prevR[nm] = propose(am[k], tX[:k + 1], dcache)
                if (trim := dcache[0].offset - off0) > 0:
                    DF._trim_recent_cache(dcache, trim)
                ref2.apply()
        recs.append(rec)
        tok, ctx, a = out[-1], tp[:k + 1], a + k + 1
    mx.eval(states(cache))
    bad = ""
    try:
        st.gate.check()
    except RuntimeError as e:
        bad = f" INVALID ({e})"
    ref_on(False)
    del cache, dcache
    mx.clear_cache()
    if name is None:                                     # warm-up: not recorded
        return out, recs, cut, bad
    with open(os.path.join(OUT, f"ra_{C}_checks.jsonl"), "a") as f:
        for r in recs:
            f.write(json.dumps({"key": name, **r}) + "\n")
    return out[:CAP], recs, cut, bad


def model_tps(recs, var, alpha, beta, mpm, cap=None):
    """tokens / time: today (draft + exact check, as measured here) vs the pipelined runahead loop (QUEUE M-16's model):
    pass = draft + max(compute(A + B), SSD at deep queue for B's requests) + stalls; compute from the all-resident exact
    pass's ms-vs-rows fit; stalls = misses the runahead did not request (+ the hybrid's waited reads) at today's per-miss
    demand time; a check whose provisional verdict differs costs one more pass with no tokens; U3 trims tokens"""
    u, u3k = {"skip": ("u1", "u3"), "hybrid": ("u1h", "u3h"), "hybrid8": ("u1h8", "u3h8")}[var]
    tok_t, tok_p, t_t, t_p, prevk = 0, 0, 0.0, 0.0, None
    for r in recs:
        n = r["k"] + 1
        tok_t += n; t_t += r["draft_ms"] + r["ck_ms"]
        nA = (prevk + 1) if prevk is not None else 0
        per = r["io_ms"] / max(r["misses"], 1.0)
        rowsB, sc = r["rows"], 1.0
        if cap is not None:                              # NewIdeas 12:4x: A + B share one pass under REF2's row limit, so
            rowsB = max(1, min(r["rows"], cap - nA))     # B gets cap - |A| rows; a check cannot keep more than its rows
            n = min(n, rowsB)
            sc = (rowsB / r["rows"]) ** 0.67             # B's reads scale with its union (~rows^0.67)
        comp = alpha + beta * (rowsB + nA)
        ssd = mpm * r[u]["req"] * sc
        stall = (r[u]["miss"] - r[u]["cov"]) * per + r[u].get("wait", 0) * per
        t_p += r["draft_ms"] + max(comp, ssd) + stall
        if not r[var]["agree"]:
            t_p += r["draft_ms"] + alpha + beta * r["rows"]     # bubble: the next block's runahead pass, nothing to re-check
        x = r.get(u3k)
        if x is not None and not x["same"]:
            n = min(n, x["lcp"] + 1)
        tok_p += n
        prevk = n - 1
    return tok_t / t_t * 1e3, tok_p / t_p * 1e3


REFTOK = json.load(open(os.path.expanduser("~/BigMoE/data/r57/s45t_B_tokens.json")))
same = lambda a, b: next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
items = [next(x for x in ED.load("dec_" + s) if x["key"] == key) for s, key in PICK]
items += (ED.load("dec_ifeval")[:2] + ED.load("dec_multi")[:1])[:CHAT]
open(os.path.join(OUT, f"ra_{C}_checks.jsonl"), "w").close()
if os.environ.get("RA_WARM", "1") == "1":                # warm-up, discarded (S45-T's method)
    w = next(x for x in ED.load("dec_math") if x["key"] == "gsm8k/708")
    wo, wr, _, wb = run(list(w["prompt_ids"]), None, cap=64)
    say(f"  warm-up (discarded): gsm8k/708 {len(wo)} tokens in {len(wr)} checks{wb}")
ALL, ok_all = {}, True
for it in items:
    if el() > HARD:
        say(f"  time budget: stopped before {it['key']}")
        break
    key = it["key"]
    out, recs, cut, bad = run(list(it["prompt_ids"]), key)
    ref = REFTOK.get(key, {}).get("gate+t", [None])[0]
    eq = f"tokens == REF2-T reference {same(out, ref)} / {min(len(ref), len(out))}" if ref else "no REF2-T reference (chat)"
    if ref:
        ok_all &= same(out, ref) >= min(len(ref), len(out))
    ALL[key] = recs
    rr = recs
    def part(var, u, u3k):
        rc = sum(r[u]["cov"] for r in rr) / max(sum(r[u]["miss"] for r in rr), 1)
        u3s = [r[u3k]["same"] for r in rr if u3k in r]
        return (f"{var}: U2 {100 * np.mean([r[var]['agree'] for r in rr]):.1f} % (rows agree "
                f"{np.mean([r[var]['rows_eq'] / (r['k'] + 1) for r in rr]):.3f}), U1 recall {100 * rc:.1f} % (requests "
                f"{np.mean([r[u]['req'] for r in rr]):.0f}), U3 {100 * np.mean(u3s) if u3s else float('nan'):.1f} %")
    say(f"  {key}: {len(out)} tokens in {len(rr)} checks (kept {np.mean([r['k'] + 1 for r in rr]):.2f}); misses on rows 0..k "
        f"{np.mean([r['u1']['miss'] for r in rr]):.0f}, today's reads {np.mean([r['u1']['today'] for r in rr]):.0f} per check; "
        f"{part('skip', 'u1', 'u3')}; {part('hybrid', 'u1h', 'u3h')} (waits {np.mean([r['u1h']['wait'] for r in rr]):.0f}); "
        f"{part('hybrid8', 'u1h8', 'u3h8')} (waits {np.mean([r['u1h8']['wait'] for r in rr]):.0f}); "
        f"ms: exact check {med([r['ck_ms'] for r in rr]):.1f} (demand SSD {med([r['io_ms'] for r in rr]):.1f}), all-resident "
        f"re-run {med([r['res_ms'] for r in rr]):.1f}, runahead skip {med([r['ra_ms'] for r in rr]):.1f} / hybrid "
        f"{med([r['hy_ms'] for r in rr]):.1f}, draft {med([r['draft_ms'] for r in rr]):.1f}; checks: re-run bitwise == first "
        f"exact pass {sum(r['rerun_equal'] for r in rr)} / {len(rr)}, runahead reads "
        f"{sum(r['ra_misses'] + r['hy_misses'] + r['h8_misses'] for r in rr):.0f} "
        f"(must be 0), re-run reads {sum(r['res_misses'] for r in rr):.0f}; {eq}{'; cut' if cut else ''}{bad}")
recs = [r for v in ALL.values() for r in v]
rows = np.array([r["rows"] for r in recs], np.float64)
beta, alpha = np.polyfit(rows, np.array([r["res_ms"] for r in recs]), 1)
mpm = 2.16 / 13.8                                         # ms per MSB-3 record at the SSD's deep-queue rate (13.8 GB/s)
say(f"all-resident exact pass ms = {alpha:.1f} + {beta:.2f} x rows (fit over {len(recs)} checks; runahead skip median "
    f"{med([r['ra_ms'] for r in recs]):.1f} ms); SSD at deep queue {mpm:.3f} ms per read")
best = {}
for var in ("skip", "hybrid", "hybrid8"):
    for key, v in ALL.items():
        a_, b_ = model_tps(v, var, alpha, beta, mpm)
        say(f"  {var} {key}: today {a_:.1f} tok/s (draft + exact check as measured here) -> modelled RUNAHEAD {b_:.1f} "
            f"({b_ / a_ - 1:+.1%})")
    a_, b_ = model_tps(recs, var, alpha, beta, mpm)
    best[var] = b_ / a_ - 1
    _, b10 = model_tps(recs, var, alpha, beta, 2.16 / 10.0)  # sensitivity: per-layer bursts reach 7.8-10.1 GB/s (MoonShots SSD)
    _, b16 = model_tps(recs, var, alpha, beta, mpm, cap=16)
    say(f"  {var} pooled: {b_:.1f} vs today {a_:.1f} tok/s ({b_ / a_ - 1:+.1%}); with the SSD at 10 GB/s {b10:.1f} "
        f"({b10 / a_ - 1:+.1%}); A + B within REF2's 16 rows {b16:.1f} ({b16 / a_ - 1:+.1%})")
    for key, v in ALL.items():
        a1, b1 = model_tps(v, var, alpha, beta, mpm, cap=16)
        say(f"    {var} {key} within 16 rows: {b1:.1f} ({b1 / a1 - 1:+.1%})")
g = max(best.values())
v = "PASS" if g >= 0.40 else ("KILL" if g < 0.15 else "BETWEEN")
say(f"Lines (QUEUE M-16 step 2, C {C}): best variant {max(best, key=best.get)} modelled {g:+.1%} tok/s vs today -> {v} (PASS >= "
    f"+40 %, KILL < +15 %); exact path unchanged (every token == REF2-T reference): {ok_all}; all-resident re-runs bitwise == "
    f"the first exact pass: {sum(r['rerun_equal'] for r in recs)} / {len(recs)}")
say(f"  footprint now {footprint_now_gb():.1f} GB, lifetime peak {footprint_peak_gb():.1f} GB; {time.time() - t0:.0f} s")
open(os.path.join(OUT, f"ra_{C}.txt"), "w").write("\n".join(lines) + "\n")
