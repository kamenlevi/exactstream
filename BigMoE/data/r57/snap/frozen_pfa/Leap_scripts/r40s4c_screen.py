"""R-40 STEP 4c (Leap, pre-registered Fri 2026-10-30 02:45 EET in ~/Leap/TABLE_350.md 'R-40 STEP 4c'): does REF2 + key buckets
(scripts/ref2_kb.py: every 16-row attention call with >= 512 keys runs at the next multiple of 512, rows of another
class recomputed in theirs) make the live loop bit-exact past 512 keys? STEP 4a / 4b: without it, rows >= 2 of a check
past ~512 keys leave the AR at a QSA attention layer (MLX's full attention kernel's bits depend on the key length).
Harness = STEP 3k's (EvalSuite's REF harness, C 144, PF1 for prefills only, DFlash2 L1-20 drafting, REF2 + SB, ref2's
defaults), with ref2_kb installed. The reference is the in-process REF2 AR WITH key buckets (a new reference past 512
keys); agreement with EvalSuite's stored REF2 rows is reported, not required.
  Part L: geometry/702 (P 520; AR rows 520..679, all past 512) and HumanEval/68 (P 367; AR rows 367..586, crossing 512
          at 496): the live loop up to 256 tokens; every kept row with an AR row compared bitwise.
  Part C (control): 702 with key buckets off (ref2's 128-key split alone, loop and AR): must break as in 4a.
PASS = every compared row equal in Part L AND the control breaks. FAIL = any difference in L. VOID = the control does
not break.
Usage: [BigMoE venv] r40s4c_screen.py [C=144] [CAP=256] -> ~/Leap/data/r40s4c_screen.txt"""
import hashlib, json, os, sys, threading, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)            # the streamed engine has no fused gate | up
os.environ.pop("BIGMOE_TIER", None)
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)          # SB is switched per arm below
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
sys.path.insert(1, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_third_party"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
from evalsuite.decode import footprint_now_gb, footprint_peak_gb
import numpy as np
import mlx.core as mx
import dflash_model_mlx as DF

D = os.path.expanduser("~/Leap/data")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
C = int(sys.argv[1]) if len(sys.argv) > 1 else 144
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 256
SPEED = len(sys.argv) > 3 and sys.argv[3] == "speed"   # S45: the live loop's speed at the given C, no AR / Part D / Part C
if not SPEED:
    C = min(C, 144)   # memory (3y-fix's second attempt); exactness does not depend on C
R, K, NAR = 16, 16, 48
EOS = {248044, 248046}
TAPS = (5, 15, 25, 35, 45)
PICK = (("math", "math", "math500/test/geometry/702.json"), ("code", "humaneval", "humaneval/HumanEval/68"))
NARK = {"math500/test/geometry/702.json": 160, "humaneval/HumanEval/68": 220}   # AR rows per prompt
AR_KEYS = PICK[0][2], PICK[1][2]
BUDGET, LAST_RUN, HARD = 380, 430, 500                   # s: no new prompt / run / check after these (job stop at 540)
ACT = [0.0]                                              # running time: a quiet window's SIGSTOP freezes this ticker too, and
def _tick():                                             # run_locked.sh's limit also skips stopped time (3k's first run lost
    last = time.time()                                   # prompts 5-6 and the control to an 84-min freeze, Thu 22:40)
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
rt, st = EM.load(EM.variant("ref"), C=C)                # EvalSuite's REF arm: load_streamed + its install + configure
model = rt.model
import ref2
import ref2_kb
ref2_kb.install()                                        # before ref2.apply(): it installs the key-bucket wrapper
import bigmoe.stream as BS
from mtplx import cache_state as CS
Q = ref2.Q
say(f"R-40 STEP 4c: full 125B via load_streamed (C = {C}), loaded in {time.time() - t0:.0f} s, "
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

def live(key, split=True):
    ref2.CFG["sdpa_classes"] = split
    ids = list(items[key]["prompt_ids"])
    ar = ar_logits(ids, NARK[key])                               # rows past 512 keys (and across it for HumanEval/68)
    r = run(ids, True, ar)
    ref2.CFG["sdpa_classes"] = True
    want = REFROWS[key]["completion_ids"]
    m = min(len(want), r["n"])
    agree = same(r["out"], want)
    ok = r["bad"] == 0 and r["cmp"] > 0 and not r["cut"]          # vs the in-process AR with key buckets
    say(f"  {key} [SDPA classes {'on' if split else 'OFF (control)'}] P {len(ids)}: {r['n']} tokens in {r['checks']} checks "
        f"(kept {r['kept']:.2f}), {r['n'] / r['dec']:.1f} tok/s, median cycle {r['cyc']:.1f} ms; tokens == EvalSuite's "
        f"REF2 row {agree} / {m}; logits rows bitwise vs AR {r['cmp'] - r['bad']} / {r['cmp']}"
        + (f"; slot calls per check {r['cnt'].get('slot_calls', 0):.1f}, SDPA splits per check {r['cnt'].get('sdpa_split', 0):.2f}"
           if r["cnt"] else "") + ("; cut" if r["cut"] else "") + ("; paused" if r["paused"] else ""))
    if SPEED:
        say(f"    S45: per cycle (medians) drafter {r['dr']:.1f} ms + check and commit {r['ck']:.1f} ms; per check {r['ms']:.1f} "
            f"misses, SSD reads {r['io']:.1f} ms; prefill {r['pf']:.1f} s")
        SP.append((key, r["n"] / r["dec"], r["cyc"], r["kept"], r["ms"], r["io"]))
    return ok

# ---------------------------------------------------------------- Part L
L_ok, n_l, SP = True, 0, []
for dom, s, key in PICK:
    if el() > BUDGET - 60:
        say(f"  Part L: time budget, {n_l} prompt(s) done")
        break
    L_ok = live(key) and L_ok
    n_l += 1

if SPEED:
    mean = lambda v: float(np.mean(v)) if v else float("nan")
    code, math_ = [x for x in SP if "humaneval" in x[0]], [x for x in SP if "humaneval" not in x[0]]
    say(f"S45 (REF2's exact path at C = {C}, DFlash2 L1-20 drafting, {CAP} tokens): tok/s code {mean([x[1] for x in code]):.1f} "
        f"({len(code)} prompts), math {mean([x[1] for x in math_]):.1f} ({len(math_)}); median cycle {med([x[2] for x in SP]):.1f} ms; "
        f"kept {mean([x[3] for x in SP]):.2f} per check; misses {mean([x[4] for x in SP]):.0f} and SSD reads "
        f"{mean([x[5] for x in SP]):.1f} ms per check (means); every token == EvalSuite's REF2 row: {L_ok}; footprint now "
        f"{footprint_now_gb():.1f} GB, lifetime peak {footprint_peak_gb():.1f} GB, MLX peak {mx.get_peak_memory() / 1e9:.1f} GB; "
        f"{el():.0f} s running ({time.time() - t0:.0f} s wall)")
    open(os.path.join(D, "s45_speed.txt" if C == 192 else f"s45_speed_c{C}.txt"), "w").write("\n".join(lines) + "\n")
    sys.exit(0)

# ---------------------------------------------------------------- Part C: control (key buckets off, 702)
C_broke = None
if el() < LAST_RUN:
    ref2_kb.KB["on"] = False
    C_broke = not live(PICK[0][2])
    ref2_kb.KB["on"] = True
else:
    say("  Part C: time budget, skipped")
say(f"  key-bucket counts: {ref2_kb.COUNTS}")
v = ("FAIL" if not L_ok else "VOID (the control did not break)" if C_broke is False
     else "PASS" if C_broke else "PASS (control skipped)")
say(f"R-40 STEP 4c verdict (PASS: live loop bit-exact vs the key-bucket AR past 512 keys and the control breaks): {v} "
    f"(Part L exact {L_ok} over {n_l} prompt(s); control broke {C_broke}; {el():.0f} s running, {time.time() - t0:.0f} s wall)")
open(os.path.join(D, "r40s4c_screen.txt"), "w").write("\n".join(lines) + "\n")
