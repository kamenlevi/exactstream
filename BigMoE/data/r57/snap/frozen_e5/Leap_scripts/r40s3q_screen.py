"""R-40 STEP 3q (Leap, pre-registered Thu 2026-10-29 22:30 EET in ~/Leap/TABLE_350.md 'R-40 STEP 3q'): is REF2 bit-exact in the
live drafting loop once its commit runs in the verify phase? STEP 3t localized the break to the commit's lone PLE-layer
replay; MTPLX's attention_gate (the PLE gate) lowers sigmoid differently outside decode_verify, and the replay ran
outside the verify scopes. ref2.py now wraps commit_verified_window in attention_phase('decode_verify') +
model_forward_kind('target_verify') (CFG commit_phase) and runs <= 16-row calls on pool slots (R-50, CFG slot_call).
Harness = STEP 3z's (EvalSuite's REF harness, C 144, PF1 for prefills only, DFlash2 L1-20 drafting, REF2 + SB), with
ref2's defaults.
  Part L (live): HumanEval/68, prealgebra/1640, HumanEval/91, algebra/694 while the budget lasts: 48 in-process REF2
            AR logits, then the live spec loop up to 256 tokens. Every token vs EvalSuite's REF2 AR row; kept rows
            vs the AR logits for the first 48 tokens, bitwise.
  Part D (state): STEP 3t's synthetic loop on 1640 (64 tokens): every check start's 48-layer state digest vs the AR's.
  Part C (control): Part L's 1640 run with CFG commit_phase off (must leave the REF2 row as STEP 3z / 3u did).
PASS = every compared token and logits row equal in Part L AND every start state equal in Part D AND the control
breaks. FAIL = any difference in L or D. VOID = the control does not break (the 1640 case no longer tests the fix).
Usage: [BigMoE venv] r40s3q_screen.py [C=144] [CAP=256] -> ~/Leap/data/r40s3q_screen.txt"""
import hashlib, json, os, sys, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)            # the streamed engine has no fused gate | up
os.environ.pop("BIGMOE_TIER", None)
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)          # SB is switched per arm below
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_e5/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
sys.path.insert(1, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_e5/Leap_third_party"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
from evalsuite.decode import footprint_now_gb
import numpy as np
import mlx.core as mx
import dflash_model_mlx as DF

D = os.path.expanduser("~/Leap/data")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
C = int(sys.argv[1]) if len(sys.argv) > 1 else 144
C = min(C, 144)   # memory (3y-fix's second attempt); exactness does not depend on C
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 256
R, K, NAR = 16, 16, 48
EOS = {248044, 248046}
TAPS = (5, 15, 25, 35, 45)
PICK = (("code", "humaneval", "humaneval/HumanEval/68"), ("math", "math", "math500/test/prealgebra/1640.json"),
        ("code", "humaneval", "humaneval/HumanEval/91"), ("math", "math", "math500/test/algebra/694.json"))
AR_KEYS = PICK[0][2], PICK[1][2]
BUDGET, LAST_RUN, HARD = 380, 430, 500                   # s: no new prompt / run / check after these (job stop at 540)
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
med = lambda v: float(np.median(v)) if len(v) else float("nan")

t0 = time.time()
rt, st = EM.load(EM.variant("ref"), C=C)                # EvalSuite's REF arm: load_streamed + its install + configure
model = rt.model
import ref2
import bigmoe.stream as BS
from mtplx import cache_state as CS
Q = ref2.Q
say(f"R-40 STEP 3q: full 125B via load_streamed (C = {C}), loaded in {time.time() - t0:.0f} s, "
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

def run(ids, ref, ar=None):
    while os.path.exists(PAUSE):
        time.sleep(5)
    paused, cut = False, False
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
    s0 = stats()
    td = c_prev = time.perf_counter()
    while len(out) < CAP and tok not in EOS:
        if time.time() - t0 > HARD:
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
            mx.async_eval(states(cache))                 # SB: the commit replay submitted alone
        c2 = time.perf_counter()
        dr_t.append(c1 - c0); ck_t.append(c2 - c1); kept.append(k + 1)
        if ar is not None:                               # rows 0..k: inputs at positions a..a+k, all on the AR path
            pend += [(a - P + j, lg[0, j]) for j in range(k + 1) if a - P + j < len(ar)]   # compared after the timing
        new = dr[:k] + [am[k]]
        for j, x in enumerate(new):
            if x in EOS:
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
    out = out[:CAP]
    return dict(out=out, pf=pf, dec=dec, n=len(out), checks=len(kept), kept=float(np.mean(kept)) if kept else 0.0,
                cyc=1e3 * med(cyc), dr=1e3 * med(dr_t), ck=1e3 * med(ck_t), bad=bad, cmp=cmp, paused=paused, cut=cut,
                ms=ms, io=1e3 * io, cnt={k_: v / max(len(kept), 1) for k_, v in cnt.items() if v} if ref else None)


RES = os.path.expanduser("~/EvalSuite/results")
REFROWS = {d["key"]: d for d in map(json.loads, open(f"{RES}/ref2/decode.jsonl"))}
items = {}
for _, s, key in PICK:
    items.update({it["key"]: it for it in ED.load("dec_" + s) if it["key"] == key})
same = lambda a, b: next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
eqb = lambda a, b: a.shape == b.shape and bool(mx.all(mx.view(a.reshape(-1), mx.uint8) == mx.view(b.reshape(-1), mx.uint8)).item())

def live(key, phase=True):
    ref2.CFG["commit_phase"] = phase
    ids = list(items[key]["prompt_ids"])
    ar = ar_logits(ids, NAR)
    r = run(ids, True, ar)
    ref2.CFG["commit_phase"] = True
    want = REFROWS[key]["completion_ids"]
    m = min(len(want), r["n"])
    agree = same(r["out"], want)
    ok = agree >= m and r["bad"] == 0 and not r["cut"]
    say(f"  {key} [commit phase {'on' if phase else 'OFF (control)'}] P {len(ids)}: {r['n']} tokens in {r['checks']} checks "
        f"(kept {r['kept']:.2f}), {r['n'] / r['dec']:.1f} tok/s, median cycle {r['cyc']:.1f} ms; tokens == EvalSuite's "
        f"REF2 row {agree} / {m}; logits rows bitwise vs AR {r['cmp'] - r['bad']} / {r['cmp']}"
        + (f"; slot calls per check {r['cnt'].get('slot_calls', 0):.1f}" if r["cnt"] else "") + ("; cut" if r["cut"] else ""))
    return ok

# ---------------------------------------------------------------- Part L
L_ok, n_l = True, 0
for dom, s, key in PICK:
    if time.time() - t0 > BUDGET - 150:
        say(f"  Part L: time budget, {n_l} prompt(s) done")
        break
    L_ok = live(key) and L_ok
    n_l += 1

# ---------------------------------------------------------------- Part D: STEP 3t's synthetic loop, start states vs AR
sha = lambda x: hashlib.sha1(np.array(mx.view(mx.contiguous(x).reshape(-1), mx.uint8)).tobytes()).hexdigest()[:16]
def digest(cache):
    return [[None if x is None else (tuple(x.shape), sha(x)) for x in (c.state if c is not None and hasattr(c, "state") else ())]
            for c in cache]
MS = (3, 0, 6, 1, 9, 2, 4, 0, 5, 1, 7, 2, 3, 6)
KEYD = PICK[1][2]
ids = list(items[KEYD]["prompt_ids"])
starts, a = [], 0
for m in MS:
    starts.append(a); a += m + 1
arm_on(True)
cache, last, _ = prefill(ids)
toks, ar_dig = [int(mx.argmax(last).item())], {0: digest(cache)}
for s_ in range(64):
    lg = ref2.ar_step(model, cache, toks[-1])
    mx.eval(lg)
    toks.append(int(mx.argmax(lg).item()))
    if s_ + 1 in starts:
        ar_dig[s_ + 1] = digest(cache)
del cache
mx.clear_cache()
cache, last, _ = prefill(ids)
a, tok, D_ok, n_d = 0, toks[0], True, 0
for m in MS:
    if a + R + 1 > 64:
        break
    if a in ar_dig:
        n_d += 1
        if digest(cache) != ar_dig[a]:
            D_ok = False
            say(f"  Part D: the state at relative {a} differs from the AR's")
            break
    drafts = toks[a + 1:a + 1 + m] + [toks[a + 1 + j + 1] for j in range(m, R - 1)]
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    with ref2.verify_scope(model):
        lg = model(mx.array([[tok] + drafts]), cache=cache)
    am = mx.argmax(lg[0], axis=-1).tolist()
    k = 0
    while k < R - 1 and drafts[k] == am[k]:
        k += 1
    model.commit_verified_window(cache, snap.states, keep_tokens=k + 1, verified_tokens=R)
    mx.async_eval(states(cache))
    a += k + 1
    tok = am[k]
del cache
mx.clear_cache()
arm_on(False)
say(f"  Part D (synthetic loop on {KEYD}): start states == the AR's at {n_d} check starts: {D_ok}")

# ---------------------------------------------------------------- Part C: control (commit phase off)
C_broke = None
if time.time() - t0 < LAST_RUN:
    C_broke = not live(PICK[1][2], phase=False)
else:
    say("  Part C: time budget, skipped")
v = ("FAIL" if not (L_ok and D_ok) else "VOID (the control did not break)" if C_broke is False
     else "PASS" if C_broke else "PASS (control skipped)")
say(f"R-40 STEP 3q verdict (PASS: live loop + states bit-exact vs REF2's AR and the control breaks): {v} "
    f"(Part L exact {L_ok} over {n_l} prompt(s); Part D {D_ok}; control broke {C_broke}; {time.time() - t0:.0f} s)")
open(os.path.join(D, "r40s3q_screen.txt"), "w").write("\n".join(lines) + "\n")
