"""R-69L (Leap, Sun 2026-11-01; a copy of BigMoE's scripts/r57_screen.py via k62_live.py, read-only use of their engine): the
45 GB default's live loop (REF2-T 4433333333 / 3c on the R-55 gate + L1-14 truncation t 0.3, DFlash2) with BigMoE's
BIGMOE_TIER_COMBINE=1 + BIGMOE_OOO=1 (R-69: bitwise equal to today's tier path at S 1-16, tensor level) vs without.
Those flags are read at import, so one process per arm: the runner does A (default) - B (both flags) - A, tag per
process; R69_KEEP_COMBINE=1 keeps BIGMOE_TIER_COMBINE (the r57 harness pops it for REF2-T). Every token of B must equal
A's (same reference). Output per process: data/r69l/<tag>_screen.txt + tokens json.
Usage: [BigMoE venv] BIGMOE_MAXK=160 BIGMOE_EXT_BUILD=build_next R57_TIER=1 R57_TRUNC=0.3 R57_ARMS=gate+t R57_PROMPTS=4
       R57_REPS=1 R57_TAG=<tag> [R69_KEEP_COMBINE=1 BIGMOE_TIER_COMBINE=1 BIGMOE_OOO=1] k69_live.py 160 256"""
import json, os, sys, threading, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)            # the streamed engine has no fused gate | up
os.environ.pop("BIGMOE_TIER", None)
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)          # SB comes from the lambda below, as in S45
os.environ.setdefault("HF_HUB_OFFLINE", "1")
if os.environ.get("R57_PRESET") == "45":                 # THE 45 GB DEFAULT (NewIdeas 11-01 06:1x): REF2-T tiers 4433333333 / 3c
    os.environ.setdefault("R57_TIER", "1")               # + the R-55 gate (MAXK default 160) + L1-14 truncation t 0.3 + ref2's
    os.environ.setdefault("R57_ARMS", "gate+t")          # defaults (REF2b key buckets, GDNX); REF2b 4-bit exact = R57_PRESET unset
    os.environ.setdefault("R57_TRUNC", "0.3")
TIER = os.environ.get("R57_TIER") == "1"                 # S45-T (NewIdeas 18:2x): REF2-T on the gate at C's RAM (M-08 sizing)
if TIER:
    os.environ["BIGMOE_TIER"] = "4433333333"; os.environ["BIGMOE_TIER_GLU"] = "3c"
    os.environ.setdefault("BIGMOE_TIER_CL", "112")       # C_m follows from C's 4-bit bytes (stream.py's budget split)
    if os.environ.get("R69_KEEP_COMBINE") != "1":
        os.environ.pop("BIGMOE_TIER_COMBINE", None)
_PF_ARM = "gate+pf" in os.environ.get("R57_ARMS", "slot,gate,gate+pf")
os.environ.setdefault("BIGMOE_MAXK", "160")             # the default since R-55's adoption (gated.py); explicit for the asserts
assert os.environ.get("BIGMOE_MAXK") == "160", "see Usage"
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs5/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
sys.path.insert(1, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs5/Leap_third_party"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
if os.environ.get("R57_SPARE"):                          # R-57b: reserved prefetch slots per layer (EvalSuite forces 0); the
    os.environ["BIGMOE_SPARE"] = os.environ["R57_SPARE"]  # ext reads it when bigmoe loads (EM.load), so set it here
from evalsuite import model as EM, data as ED
from evalsuite.decode import footprint_now_gb, footprint_peak_gb
import numpy as np
import mlx.core as mx
import dflash_model_mlx as DF

OUT = os.path.expanduser("~/Leap/data/r69l"); os.makedirs(OUT, exist_ok=True)
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
C = int(sys.argv[1]) if len(sys.argv) > 1 else 192
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 256
PF_B = int(sys.argv[3]) if len(sys.argv) > 3 else 96
R, K = 16, 16
EOS = {248044, 248046}
TAPS = (5, 15, 25, 35, 45)
PICK = (("humaneval", "humaneval/HumanEval/68"), ("math", "math500/test/prealgebra/1640.json"),
        ("humaneval", "humaneval/HumanEval/91"), ("math", "math500/test/algebra/694.json"),        # S45's other prompts
        ("math", "math500/test/algebra/1282.json"), ("math", "gsm8k/708"))
# R-55L (the orchestrator 17:0x: headline the steady-state cycle; a clean slot re-run settles code): R57_ARMS=slot,gate
# R57_PROMPTS=6 R57_TAG=r55l runs only those arms on S45's six prompts and writes data/r57/r55l_screen.txt
ARMS_ = tuple(os.environ.get("R57_ARMS", "slot,gate,gate+pf").split(","))
PICK = PICK[:int(os.environ.get("R57_PROMPTS", "2"))]
TAG = os.environ.get("R57_TAG", "r57")
REPS = int(os.environ.get("R57_REPS", "2"))
TRUNC = float(os.environ.get("R57_TRUNC", "0.3"))
S10B_DIR = os.environ.get("R57_S10B_DIR")               # with BIGMOE_S10B=1 (R-57b data for Fit12): per run the gated calls'
if S10B_DIR:                                            # ids + MoE inputs (bigmoe/s10b.py) -> <dir>/<prompt>_<arm>_<n>.npz
    os.makedirs(S10B_DIR, exist_ok=True)        # arm gate+t: drafts cut where the running confidence < TRUNC              # runs per arm per prompt: 2 = ABBA order, 1 = one each
HARD = 600                                               # s of running time: no new check after this (job stop at 660)
ACT = [0.0]                                              # running time without SIGSTOP freezes (Leap's ticker)
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
from mtplx import cache_state as CS
Q = ref2.Q
assert BG.MAXK == 160 and BG.XLINES == 4 and (not _PF_ARM or PF_B <= 30 + 31 * BS._PRED_LINES), (PF_B, BS._PRED_LINES)
lay = model.language_model.model.layers
NL = len(lay)
SW_ = [l.mlp.switch_mlp for l in lay]
class _Corr:                                             # R-57b: Fit12's logit correction on the L+1 router (raw @ W + b),
    def __init__(self, r):                               # a callable in the router's place, so stream.py stays unchanged
        self.r, self.W, self.b = r, None, None
    def __call__(self, x):
        lg = self.r(x)
        return lg if self.W is None else lg.astype(mx.float32) @ self.W + self.b
CORR = {L: _Corr(lay[L + 1].mlp.gate) for L in range(NL - 1)}
NEXT = {L: [(CORR[L] if os.environ.get("R57_CORR") else lay[L + 1].mlp.gate, None, PF_B)] for L in range(NL - 1)
        if not SW_[L]._bm["full"] and not SW_[L + 1]._bm["full"]}   # stream.py's wiring for qwen4_exp (ratio None)
def corr_for(key):
    """Fit12's held-out 2-fold maps: each prompt gets the map fit on the other three (R57_CORR = the maps' directory)"""
    if not os.environ.get("R57_CORR"):
        return None
    d = os.environ["R57_CORR"]
    f = "corr_fit68_91_1282.npz" if any(k in key for k in ("1640", "694", "708")) else "corr_fit1640_694_708.npz"
    z = np.load(os.path.join(d, f))
    for L, c in CORR.items():
        T = L + 1
        c.W = mx.array(z[f"W{T}"].astype(np.float32)); c.b = mx.array(z[f"b{T}"].astype(np.float32))
    mx.eval([c.W for c in CORR.values()] + [c.b for c in CORR.values()])
    return f
import bigmoe.tier as _BT
say(f"R-69L flags: TIER_COMBINE {_BT._COMBINE}, OOO {_BT._OOO}, GLU {_BT._GLU}")
say(f"R-57 harness{' (REF2-T tiers ' + os.environ['BIGMOE_TIER'] + ', C_l ' + os.environ['BIGMOE_TIER_CL'] + ')' if TIER else ''}: full 125B via load_streamed (C = {C}), loaded in {time.time() - t0:.0f} s, {mx.get_active_memory() / 1e9:.1f} GB "
    f"active; REF2 + SB; gate rows MAXK {BG.MAXK} (ext {BG._BUILD}), PRED_LINES {BS._PRED_LINES} (ranked list <= "
    f"{30 + 31 * BS._PRED_LINES}), PF_B {PF_B}, PF_JIT {os.environ.get('BIGMOE_PF_JIT')}, PF_QD {os.environ.get('BIGMOE_PF_QD')}, "
    f"PF_VICTIM {os.environ.get('BIGMOE_PF_VICTIM')}; {len(NEXT)} look-ahead layers; CAP {CAP}")

def set_arm(arm):
    BS.GATE_N[0] = 40 if arm == "slot" else 160
    BS._PF_RANKED = arm == "gate+pf"
    for L, sw in enumerate(SW_):
        if arm == "gate+pf" and L in NEXT:
            sw._bm["next"] = NEXT[L]
        else:
            sw._bm.pop("next", None)

def pf1(on):
    if not on and BS._STAGE["bufs"] is not None:
        BS._stage_drop()
    BS._PF_PREFILL = on

def stats():
    s = st.stats()
    return np.array([s["misses"], s["io_s"]], dtype=np.float64)

# ---------------------------------------------------------------- taps for the drafter (Leap's S45 harness)
TAP = {"on": False, "x": [[] for _ in TAPS]}
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

def propose(blk, ctx, dcache):
    hid = draft.hidden_states(blk, ctx, dcache, 1)
    cand, unary = head2s(hid)
    return chain(blk, unary, cand, draft.candidate_selector.hidden_projection(hid))

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

def run(ids, arm, name=None):
    while os.path.exists(PAUSE):
        time.sleep(5)
    ref2.CFG["tier_nax4"] = arm.endswith("+nax")        # K-62L: the whole run (prefill + loop) on one tier kernel
    ref2.reset_counts()
    set_arm("slot")                                      # the prefill as shipped in every arm
    ref_on(True)
    p0, g0 = BG.PAUSED["pause_s"], BG.PAUSED["n_gap"]
    cache, last, T = prefill(ids)
    P = len(ids)
    dcache = draft.make_cache()
    start = max(0, P - (SWIN - 1))
    for c in dcache:
        c.offset = start
    ctx, a = T[start:P], P
    tok = int(mx.argmax(last).item())
    set_arm(arm)
    res0 = None
    if S10B_DIR:
        BS._S10B.drop()                                  # only the live loop's checks + commit replays
        res0 = np.array([BS.ext.lookup(L, list(range(512))) for L in range(NL)], np.int32)   # pools at loop start
    out, cyc, ck_t, kept, sm, wl, cut = [tok], [], [], [], [], [], False
    td = c_prev = time.perf_counter()
    while len(out) < CAP and tok not in EOS:
        if el() > HARD:
            cut = True
            break
        c0 = time.perf_counter()
        if out[1:]:
            cyc.append(c0 - c_prev)
        c_prev = c0
        ref2.remove()
        pth, cf = propose(mx.array([[tok] + [MASK] * (R - 1)]), ctx[None].astype(mx.bfloat16), dcache)
        dr = np.array(pth[0]).tolist()
        if arm.startswith("gate+t"):                     # Leap's S45c: L1-14 running-confidence truncation
            dr = dr[:int((np.cumprod(np.array(cf[0])) >= TRUNC).sum())]
        if (trim := dcache[0].offset - a) > 0:
            DF._trim_recent_cache(dcache, trim)
        ref2.apply()
        s0 = stats()
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
        while k < len(dr) and dr[k] == am[k]:
            k += 1
        model.commit_verified_window(cache, snap.states, keep_tokens=k + 1, verified_tokens=len(dr) + 1)
        mx.async_eval(states(cache))                     # SB: the commit replay submitted alone
        c2 = time.perf_counter()
        sm.append(stats() - s0)
        ck_t.append(c2 - c1); kept.append(k + 1); wl.append(c2 - c0)
        new = dr[:k] + [am[k]]
        for j, x in enumerate(new):
            if x in EOS:
                new = new[:j + 1]
                break
        out += new
        tok, ctx, a = out[-1], tp[:k + 1], a + k + 1
    mx.eval(states(cache))
    dec = time.perf_counter() - td
    if S10B_DIR and name:
        np.savez(os.path.join(S10B_DIR, name + ".npz"), **BS._S10B.flush(), tokens=np.array(out, np.int32), res0=res0)
    bad = ""
    try:
        st.gate.check()
    except RuntimeError as e:
        bad = f" INVALID ({e})"
    if BG.PAUSED["pause_s"] - p0 > 0.5 or BG.PAUSED["n_gap"] > g0:
        bad += f" PAUSED ({BG.PAUSED['pause_s'] - p0:.1f} s waited, {BG.PAUSED['n_gap'] - g0} gaps)"
    nax = ref2.counts()["nax4"]
    ref2.CFG["tier_nax4"] = False
    set_arm("slot")
    ref_on(False)
    del cache, dcache
    mx.clear_cache()
    sm = np.array(sm)
    return dict(out=out[:CAP], n=min(len(out), CAP), tps=min(len(out), CAP) / dec, cyc=1e3 * med(cyc), ck=1e3 * med(ck_t),
                kept=float(np.mean(kept)), checks=len(kept), miss=float(np.median(sm[:, 0])), io=1e3 * float(np.median(sm[:, 1])),
                tpsw=sum(kept[2:]) / max(sum(wl[2:]), 1e-9), cut=cut, bad=bad, nax=nax)   # tpsw: steady state, checks >= 2

REFROWS = {d["key"]: d for d in map(json.loads, open(os.path.expanduser("~/EvalSuite/results/ref2/decode.jsonl")))}
same = lambda a, b: next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
RES, ok_all, valid, TOK = {}, True, True, {}
if S10B_DIR:                                             # every layer's router, dequantized (Fit12's lead-1 forecast at every L)
    W = {}
    for T_ in range(1, NL):
        g = lay[T_].mlp.gate
        w = (mx.dequantize(g.weight, g.scales, getattr(g, "biases", None), g.group_size, g.bits,
                           **({"mode": g.mode} if hasattr(g, "mode") else {})) if hasattr(g, "scales") else g.weight)
        W[f"w{T_}"] = np.array(w.astype(mx.float16))
    np.savez(os.path.join(S10B_DIR, "routers.npz"), **W)
    say(f"  routers.npz: {len(W)} layers, {sum(v.nbytes for v in W.values()) / 1e6:.0f} MB")
if os.environ.get("R57_WARM") == "1":                    # warm-up, discarded (NewIdeas: S45-T's method)
    it = next(x for x in ED.load("dec_math") if x["key"] == "gsm8k/708")
    CAP, cap0 = 64, CAP
    for arm_w in ARMS_:                                   # K-62L: every arm warms up (NAX4's kernels compile here)
        r = run(list(it["prompt_ids"]), arm_w)
        say(f"  warm-up (discarded): gsm8k/708 [{arm_w}] {r['n']} tokens, {r['tps']:.1f} tok/s; NAX4 calls {r['nax']}{r['bad']}")
    CAP = cap0
for n_p, (s, key) in enumerate(PICK):
    it = next(x for x in ED.load("dec_" + s) if x["key"] == key)
    ids, want = list(it["prompt_ids"]), REFROWS[key]["completion_ids"]
    cf = corr_for(key)
    if cf:
        say(f"  {key}: forecast correction {cf} (held out)")
    arms = (ARMS_ + ARMS_[::-1]) if REPS == 2 else ARMS_
    arms = arms if n_p % 2 == 0 else arms[::-1]
    for arm in arms:
        if el() > HARD:
            say(f"  time budget: stopped before {key} [{arm}]")
            break
        r = run(ids, arm, f"{key.replace('/', '_').replace('.json', '')}_{arm}_{len(TOK.get(key, {}).get(arm, []))}")
        m = min(len(want), r["n"])
        agree = same(r["out"], want)
        ok = (agree >= m or TIER) and not r["cut"]          # REF2-T is its own reference (tokens compared across runs)
        TOK.setdefault(key, {}).setdefault(arm, []).append(r["out"])
        ok_all &= ok
        valid &= not r["bad"]
        if "PAUSED" not in r["bad"]:                     # a paused run's clock is wrong: out of the aggregates
            RES.setdefault(key, {}).setdefault(arm, []).append(r)
        say(f"  {key} [{arm}]: {r['n']} tokens in {r['checks']} checks (kept {r['kept']:.2f}), {r['tps']:.1f} tok/s "
            f"(steady state, checks >= 2: {r['tpsw']:.1f}), median "
            f"cycle {r['cyc']:.1f} ms, check + commit {r['ck']:.1f} ms; per check (medians) {r['miss']:.0f} misses, demand "
            f"SSD {r['io']:.1f} ms; tokens == REF2 row {agree} / {m}; NAX4 calls {r['nax']}{'; cut' if r['cut'] else ''}{r['bad']}")
eq_arms = all(len({tuple(r["out"]) for runs in v.values() for r in runs}) == 1 for v in RES.values())
eq_by = {arm: all(len({tuple(r["out"]) for r in v.get(arm, [])}) <= 1 for v in RES.values()) for arm in ARMS_}
def agg(arm, f):
    return med([med([f(r) for r in v[arm]]) for v in RES.values() if arm in v])
for arm in ARMS_:
    say(f"  {arm}: tok/s {agg(arm, lambda r: r['tps']):.1f} (steady state {agg(arm, lambda r: r['tpsw']):.1f}), cycle "
        f"{agg(arm, lambda r: r['cyc']):.1f} ms, check + commit {agg(arm, lambda r: r['ck']):.1f} ms, misses "
        f"{agg(arm, lambda r: r['miss']):.0f}, demand SSD {agg(arm, lambda r: r['io']):.1f} ms per check (medians over valid "
        f"runs, then prompts)")
if "slot" in ARMS_ and "gate" in ARMS_:
    for key, v in RES.items():
        if "slot" in v and "gate" in v:
            f = lambda arm, k: med([r[k] for r in v[arm]])
            say(f"    {key}: steady-state tok/s slot {f('slot', 'tpsw'):.1f} -> gate {f('gate', 'tpsw'):.1f} "
                f"({f('gate', 'tpsw') / f('slot', 'tpsw') - 1:+.0%}); whole run {f('slot', 'tps'):.1f} -> {f('gate', 'tps'):.1f}; "
                f"median cycle {f('slot', 'cyc'):.0f} -> {f('gate', 'cyc'):.0f} ms")
    cy_g, cy_s = agg("gate", lambda r: r["cyc"]), agg("slot", lambda r: r["cyc"])
    w_g, w_s = agg("gate", lambda r: r["tpsw"]), agg("slot", lambda r: r["tpsw"])
    say(f"R-55 live (gate vs slot): median cycle {cy_g:.1f} vs {cy_s:.1f} ms ({cy_g / cy_s - 1:+.1%}); steady-state tok/s "
        f"{w_g:.1f} vs {w_s:.1f} ({w_g / w_s - 1:+.1%}); whole-run tok/s {agg('gate', lambda r: r['tps']):.1f} vs "
        f"{agg('slot', lambda r: r['tps']):.1f}; tokens equal across every run of a prompt: {eq_arms}; every token == REF2: "
        f"{ok_all}; all runs valid: {valid}")
if "gate+pf" in ARMS_:
    io_g, io_p = agg("gate", lambda r: r["io"]), agg("gate+pf", lambda r: r["io"])
    ck_g, ck_p = agg("gate", lambda r: r["ck"]), agg("gate+pf", lambda r: r["ck"])
    cy_g, cy_p = agg("gate", lambda r: r["cyc"]), agg("gate+pf", lambda r: r["cyc"])
    d_io, d_ck, d_cy = io_p / io_g - 1, ck_p / ck_g - 1, cy_p / cy_g - 1
    v = ("FAIL (a token left REF2's row)" if not ok_all
         else "PASS" if d_io <= -0.5 and d_ck <= -0.25 else "KILL" if d_cy > -0.10 else "BETWEEN")
    say(f"R-57 verdict (PASS: demand SSD per check -50 % AND check + commit -25 %, gate+pf vs gate, every token == REF2; "
        f"KILL: cycle under -10 %): {v} (SSD {d_io:+.1%}, check + commit {d_ck:+.1%}, cycle {d_cy:+.1%})")
if "gate+t" in ARMS_ and "gate+t+nax" in ARMS_:
    for key, v in RES.items():
        if "gate+t" in v and "gate+t+nax" in v:
            f = lambda arm, k: med([r[k] for r in v[arm]])
            say(f"    {key}: steady-state tok/s 3c {f('gate+t', 'tpsw'):.1f} -> NAX4 {f('gate+t+nax', 'tpsw'):.1f} "
                f"({f('gate+t+nax', 'tpsw') / f('gate+t', 'tpsw') - 1:+.1%}); median cycle {f('gate+t', 'cyc'):.0f} -> "
                f"{f('gate+t+nax', 'cyc'):.0f} ms; check + commit {f('gate+t', 'ck'):.0f} -> {f('gate+t+nax', 'ck'):.0f} ms; "
                f"kept {f('gate+t', 'kept'):.2f} -> {f('gate+t+nax', 'kept'):.2f}")
    cy3, cyn = agg("gate+t", lambda r: r["cyc"]), agg("gate+t+nax", lambda r: r["cyc"])
    w3, wn = agg("gate+t", lambda r: r["tpsw"]), agg("gate+t+nax", lambda r: r["tpsw"])
    ck3, ckn = agg("gate+t", lambda r: r["ck"]), agg("gate+t+nax", lambda r: r["ck"])
    d_cy, d_w = cyn / cy3 - 1, wn / w3 - 1
    v = "PASS" if d_cy <= -0.08 and d_w >= 0.08 else "KILL" if d_cy > -0.03 else "BETWEEN"
    nax_on = all(r["nax"] > 0 for v_ in RES.values() for r in v_.get("gate+t+nax", [])) and \
        all(r["nax"] == 0 for v_ in RES.values() for r in v_.get("gate+t", []))
    say(f"K-62L verdict (PASS: cycle -8 % AND steady tok/s +8 %; KILL: cycle better by < 3 %): {v} (median cycle {cy3:.1f} -> "
        f"{cyn:.1f} ms, {d_cy:+.1%}; steady-state tok/s {w3:.1f} -> {wn:.1f}, {d_w:+.1%}; check + commit {ck3:.1f} -> {ckn:.1f} ms); "
        f"switch active only in the NAX4 arm: {nax_on}; tokens equal across runs within each arm: {eq_by}; all runs valid: {valid}")
say(f"R-69L: OOO calls {_BT.OOO_STATS['calls']} (flags TIER_COMBINE {_BT._COMBINE}, OOO {_BT._OOO})")
say(f"  footprint now {footprint_now_gb():.1f} GB, lifetime peak {footprint_peak_gb():.1f} GB; {time.time() - t0:.0f} s")
json.dump(TOK, open(os.path.join(OUT, f"{TAG}_tokens.json"), "w"))
open(os.path.join(OUT, f"{TAG}_screen.txt"), "w").write("\n".join(lines) + "\n")
