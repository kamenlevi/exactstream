"""R-40 STEP 3u (Leap, pre-registered Thu 2026-10-29 20:45 EET in ~/Leap/TABLE_350.md 'R-40 STEP 3u'): what in STEP 3z's
live loop breaks REF2's bit-identity with its AR? 3w / 3v: checks + partial commits + wrong rejected rows + many cycles
are exact WITHOUT the drafter. 3z (with the drafter) on prealgebra/1640: kept rows 40 / 48 bitwise == the AR.
Harness = STEP 3z's (EvalSuite's REF harness, C 128, PF1 for prefills only, REF2 + SB, taps hook, DFlash2 L1-20).
REF2's AR for 48 steps on prealgebra/1640, then four runs of the spec loop up to 48 tokens, each kept row vs the AR:
  U0  as STEP 3z: the drafter drafts with REF2 switched off (ref2.remove / apply around it), taps on
  U1  the drafter drafts with REF2 left ON
  U2  the drafter runs as in U0, but the check uses synthetic drafts from the AR tokens (m correct, the rest shifted)
  U3  no drafter forward; synthetic drafts; taps on; REF2 switched off / on between checks as in U0
Per run: equal kept rows / all, the first check with a differing row (its index, kept, rows), the tokens vs the AR.
A measurement; no lines.
Usage: [BigMoE venv] r40s3u_screen.py [C=128] -> ~/Leap/data/r40s3u_screen.txt"""
import json, os, sys, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)            # the streamed engine has no fused gate | up
os.environ.pop("BIGMOE_TIER", None)
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)          # SB is switched per arm below
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel0/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
sys.path.insert(1, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel0/Leap_third_party"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
from evalsuite.decode import footprint_now_gb
import numpy as np
import mlx.core as mx
import dflash_model_mlx as DF

D = os.path.expanduser("~/Leap/data")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
C = int(sys.argv[1]) if len(sys.argv) > 1 else 128
C = min(C, 128)   # memory (STEP 3y-fix's second attempt); numerics do not depend on C
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 256
R, K, NAR = 16, 16, 48
EOS = {248044, 248046}
TAPS = (5, 15, 25, 35, 45)
PICK = (("code", "humaneval", "humaneval/HumanEval/68"), ("math", "math", "math500/test/prealgebra/1640.json"),
        ("code", "humaneval", "humaneval/HumanEval/91"), ("math", "math", "math500/test/algebra/694.json"))
AR_KEYS = PICK[0][2], PICK[1][2]
BUDGET, LAST_RUN, HARD = 330, 420, 500                   # s: no new prompt / run / check after these (job stop at 540)
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
med = lambda v: float(np.median(v)) if len(v) else float("nan")
eqb = lambda a, b: a.shape == b.shape and bool(mx.all(mx.view(a.reshape(-1), mx.uint8) == mx.view(b.reshape(-1), mx.uint8)).item())

t0 = time.time()
rt, st = EM.load(EM.variant("ref"), C=C)                # EvalSuite's REF arm: load_streamed + its install + configure
model = rt.model
import ref2
import bigmoe.stream as BS
from mtplx import cache_state as CS
Q = ref2.Q
say(f"R-40 STEP 3u: full 125B via load_streamed (C = {C}), loaded in {time.time() - t0:.0f} s, "
    f"{mx.get_active_memory() / 1e9:.1f} GB active; EvalSuite's PF1 {BS._PF_PREFILL} (full scratch {BS._FULLSCR}): on "
    f"for prefills, off for checks / commits")
ref2.apply()
st_ok = ref2.self_test()
ref2.remove(); ref2.reset_counts()
say(f"  self-test: {st_ok}")

def pf1(on):
    """on: EvalSuite's PF1 prefill path; off: BigMoE's default _sync_call for calls with > MAXK ids. Switching off
    first does what the engine's first decode call does (_stage_drop: wait for the staging reads, warm the last layer)"""
    if not on and BS._STAGE["bufs"] is not None:
        BS._stage_drop()
    BS._PF_PREFILL = on

def stats():
    try:
        s = st.stats()
        return np.array([s["misses"], s["io_s"]], dtype=np.float64)
    except Exception:
        return np.zeros(2)

# ---------------------------------------------------------------- taps for the drafter (dfl2c_extract's, per forward)
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
def taps():                                              # [rows, 5 * 2560] fp16, rows of the forwards since the reset
    f = [mx.concatenate(x) if len(x) > 1 else x[0] for x in TAP["x"]]
    return mx.stack(f, axis=1).reshape(f[0].shape[0], -1)

# ---------------------------------------------------------------- the drafter (d02b_screen's eager loop)
draft = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.load_weights(os.path.join(D, "l120_ft.safetensors"), strict=False)
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
    pred, path = blk[:, 0], []
    for p in range(hp.shape[1]):
        sc = unary[:, p] + mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None]
                                  * sel.successor_codebook(cand[:, p]), axis=-1)
        pred = mx.take_along_axis(cand[:, p], mx.argmax(sc, axis=-1)[:, None], axis=-1)[:, 0]
        path.append(pred)
    return mx.stack(path, axis=1)

def propose(blk, ctx, dcache):
    hid = draft.hidden_states(blk, ctx, dcache, 1)
    cand, unary = head2s(hid)
    return chain(blk, unary, cand, draft.candidate_selector.hidden_projection(hid))

# ---------------------------------------------------------------- the two arms
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


NAR = 48
MS = (3, 0, 6, 1, 9, 2, 4, 0, 5, 1, 7, 2, 3, 6)          # synthetic drafts: correct ones per check (U2, U3)
KEY = PICK[1][2]
items = {it["key"]: it for it in ED.load("dec_math") if it["key"] == KEY}
ids = list(items[KEY]["prompt_ids"])

arm_on(True)
cache, last, _ = prefill(ids)
toks, ar_lg = [int(mx.argmax(last).item())], []
for _ in range(NAR + 16):
    lg = ref2.ar_step(model, cache, toks[-1])
    mx.eval(lg)
    ar_lg.append(lg.reshape(-1))
    toks.append(int(mx.argmax(lg).item()))
del cache
mx.clear_cache()
say(f"  {KEY}, P {len(ids)}: REF2 AR {NAR + 16} steps (ar_lg[j] = the logits after toks[j])")

def loop(name, drafter, ref_off_for_draft, synthetic):
    arm_on(True)
    cache, last, T = prefill(ids)
    P = len(ids)
    dcache = draft.make_cache()
    start = max(0, P - (SW - 1))
    for c in dcache:
        c.offset = start
    ctx, a, tok = T[start:P], 0, int(mx.argmax(last).item())
    eq, n, first, kept, ncheck = 0, 0, None, [], 0
    while a < NAR and ncheck < 40:
        if drafter:
            if ref_off_for_draft:
                ref2.remove()
            dr = np.array(propose(mx.array([[tok] + [MASK] * (R - 1)]), ctx[None].astype(mx.bfloat16), dcache)[0]).tolist()
            if (trim := dcache[0].offset - (P + a)) > 0:
                DF._trim_recent_cache(dcache, trim)
            if ref_off_for_draft:
                ref2.apply()
        elif ref_off_for_draft:
            ref2.remove(); ref2.apply()
        if synthetic:
            m = MS[ncheck % len(MS)]
            dr = toks[a + 1:a + 1 + m] + [toks[a + 1 + j + 1] for j in range(m, R - 1)]
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
        bad = [j for j in range(k + 1) if a + j < NAR + 16 and not eqb(lg[0, j], ar_lg[a + j])]
        eq += (k + 1) - len(bad); n += k + 1
        if bad and first is None:
            first = (ncheck, k + 1, bad, dr[:4])
        model.commit_verified_window(cache, snap.states, keep_tokens=k + 1, verified_tokens=R)
        mx.async_eval(states(cache))                     # SB: the commit replay submitted alone
        kept.append(k + 1)
        new = dr[:k] + [am[k]]
        a += k + 1
        tok, ctx = new[-1], tp[:k + 1]
        ncheck += 1
    mx.eval(states(cache))
    del cache, dcache
    mx.clear_cache()
    say(f"  {name}: {ncheck} checks, kept {kept}; kept rows bitwise == AR {eq} / {n}"
        + (f"; first differing check #{first[0]} (kept {first[1]}, rows {first[2]}, drafts {first[3]})" if first else ""))
    return eq == n

r = {}
r["U0"] = loop("U0 as STEP 3z (drafter, REF2 off while drafting, taps)", True, True, False)
r["U1"] = loop("U1 drafter with REF2 left on", True, False, False)
r["U2"] = loop("U2 drafter runs (REF2 off), synthetic drafts used", True, True, True)
r["U3"] = loop("U3 no drafter, synthetic drafts, taps, REF2 off/on between checks", False, True, True)
say(f"R-40 STEP 3u: exact per variant {r}; {time.time() - t0:.0f} s")
open(os.path.join(D, "r40s3u_screen.txt"), "w").write("\n".join(lines) + "\n")
