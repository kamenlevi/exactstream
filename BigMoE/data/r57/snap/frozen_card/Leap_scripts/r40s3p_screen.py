"""R-40 STEP 3p (Leap, pre-registered Thu 2026-10-29 22:25 EET in ~/Leap/TABLE_350.md 'R-40 STEP 3p'): where does REF2's live
loop leave its AR on algebra/694? STEP 3q: with the commit-phase fix, /68, prealgebra/1640 and /91 are bit-exact in
the live drafting loop (256 tokens, 48 / 48 rows each), but algebra/694 (P 98) keeps only 14 / 48 rows and leaves the
REF2 row at token 216. A second, separate break.
Harness = STEP 3q's (EvalSuite's REF harness, C 144, PF1 for prefills only, REF2 + SB with ref2's defaults: slot call,
commit in the verify phase; DFlash2 L1-20 drafting; taps).
  AR: prefill + 64 REF2 AR steps: logits, per-layer row records (sha1 per record), and every position's 48-layer
      cache-state digest (3w's).
  Loop A (live drafter, as 3q) and Loop B (3t's synthetic drafts), each up to 64 tokens: at every check start the
      state digest vs the AR's at that position; every kept row vs the AR logits; at the first check with a differing
      kept row its row records vs the AR step's (earliest differing record); the first check whose start state
      differs (layer / slot).
A measurement; no lines.
Usage: [BigMoE venv] r40s3p_screen.py [C=144] -> ~/Leap/data/r40s3p_screen.txt"""
import hashlib, json, os, sys, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)            # the streamed engine has no fused gate | up
os.environ.pop("BIGMOE_TIER", None)
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)          # SB is switched per arm below
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_card/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
sys.path.insert(1, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_card/Leap_third_party"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
from evalsuite.decode import footprint_now_gb
import numpy as np
import mlx.core as mx
import dflash_model_mlx as DF

D = os.path.expanduser("~/Leap/data")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
C = int(sys.argv[1]) if len(sys.argv) > 1 else 144
C = min(C, 144)   # memory; exactness does not depend on C
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

t0 = time.time()
rt, st = EM.load(EM.variant("ref"), C=C)                # EvalSuite's REF arm: load_streamed + its install + configure
model = rt.model
import ref2
import bigmoe.stream as BS
from mtplx import cache_state as CS
Q = ref2.Q
say(f"R-40 STEP 3p: full 125B via load_streamed (C = {C}), loaded in {time.time() - t0:.0f} s, "
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


eqb = lambda a, b: a.shape == b.shape and bool(mx.all(mx.view(a.reshape(-1), mx.uint8) == mx.view(b.reshape(-1), mx.uint8)).item())
sha = lambda x: hashlib.sha1(np.array(mx.view(mx.contiguous(x).reshape(-1), mx.uint8)).tobytes()).hexdigest()[:16]
NAR = 64
KEY = PICK[3][2]                                         # math500/test/algebra/694.json
items = {it["key"]: it for it in ED.load("dec_math") if it["key"] == KEY}
ids = list(items[KEY]["prompt_ids"])
lay = model.language_model.model.layers
KIND = ["ple" if "ple" in l else ("gdn" if l.is_linear else "qsa") for l in lay]

REC = {"on": False, "r": []}
def hook(klass, part):
    orig = klass.__call__
    def call(self, *a, **kw):
        y = orig(self, *a, **kw)
        if REC["on"]:
            o = y if isinstance(y, mx.array) else y[0]
            REC["r"].append((part, o.reshape(-1, o.shape[-1])))
        return y
    klass.__call__ = call
for klass, part in ((Q.GatedDeltaNet, "gdn"), (Q.Attention, "attn"), (Q.SparseMoeBlock, "moe"), (Q.PLELayer, "ple"),
                    (Q.GatedResidual, "hc")):
    hook(klass, part)                                    # DecoderLayer already carries the taps wrapper
def digest(cache):
    return [[None if x is None else (tuple(x.shape), sha(x)) for x in (c.state if c is not None and hasattr(c, "state") else ())]
            for c in cache]
def diff(a, b):
    out = []
    for i, (ra, rb) in enumerate(zip(a, b)):
        bad = [j for j in range(max(len(ra), len(rb))) if (ra[j] if j < len(ra) else "-") != (rb[j] if j < len(rb) else "-")]
        if bad:
            out.append((i, KIND[i], bad))
    return out
rowrec = lambda rec, row: [(p, sha(o[row])) for p, o in rec]

# ---------------------------------------------------------------- the AR
arm_on(True)
cache, last, _ = prefill(ids)
toks, ar_lg, ar_rec, ar_dig = [int(mx.argmax(last).item())], [], [], [digest(cache)]
for s_ in range(NAR):
    REC["r"] = []; REC["on"] = True
    lg = ref2.ar_step(model, cache, toks[-1])
    mx.eval(lg)
    REC["on"] = False
    ar_rec.append(rowrec(REC["r"], 0))
    ar_lg.append(lg.reshape(-1))
    toks.append(int(mx.argmax(lg).item()))
    ar_dig.append(digest(cache))                         # state after toks[0 .. s_]
REC["r"] = []
del cache
mx.clear_cache()
say(f"  {KEY}, P {len(ids)}: REF2 AR {NAR} steps, {len(ar_rec[0])} records per step, digests at every position")

MS = (3, 0, 6, 1, 9, 2, 4, 0, 5, 1, 7, 2, 3, 6)
def loop(name, drafter):
    arm_on(True)
    cache, last, T = prefill(ids)
    P = len(ids)
    dcache = draft.make_cache()
    st0 = max(0, P - (SW - 1))
    for c in dcache:
        c.offset = st0
    ctx, a, tok = T[st0:P], 0, int(mx.argmax(last).item())
    first_state, first_row, kept, n_c, eq, n = None, None, [], 0, 0, 0
    while a + R + 1 <= NAR and n_c < 40:
        if first_state is None:
            d = diff(ar_dig[a], digest(cache))
            if d:
                first_state = (n_c, a, d[:8])
        if drafter:
            ref2.remove()
            dr = np.array(propose(mx.array([[tok] + [MASK] * (R - 1)]), ctx[None].astype(mx.bfloat16), dcache)[0]).tolist()
            if (trim := dcache[0].offset - (P + a)) > 0:
                DF._trim_recent_cache(dcache, trim)
            ref2.apply()
        else:
            m = MS[n_c % len(MS)]
            dr = toks[a + 1:a + 1 + m] + [toks[a + 1 + j + 1] for j in range(m, R - 1)]
        snap = CS.snapshot_untrimmable_cache_lazy(cache)
        REC["r"] = []; REC["on"] = True
        tap_reset(); TAP["on"] = True
        with ref2.verify_scope(model):
            lg = model(mx.array([[tok] + dr]), cache=cache)
        TAP["on"] = False
        am, tp = mx.argmax(lg[0], axis=-1), taps()
        mx.eval(lg, am, tp)
        REC["on"] = False
        am = am.tolist()
        k = 0
        while k < R - 1 and dr[k] == am[k]:
            k += 1
        bad = [j for j in range(k + 1) if not eqb(lg[0, j], ar_lg[a + j])]
        eq += k + 1 - len(bad); n += k + 1
        if bad and first_row is None:
            r = bad[0]
            mine, ref = rowrec(REC["r"], r), ar_rec[a + r]
            e = next((i for i in range(min(len(mine), len(ref))) if mine[i] != ref[i]), None)
            where = None if e is None else (e, mine[e][0], sum(1 for x in range(e) if mine[x][0] == "hc") // 2)
            first_row = (n_c, a, k + 1, bad, where)
        REC["r"] = []
        model.commit_verified_window(cache, snap.states, keep_tokens=k + 1, verified_tokens=R)
        mx.async_eval(states(cache))
        kept.append(k + 1)
        a += k + 1
        tok, ctx = (dr[:k] + [am[k]])[-1], tp[:k + 1]
        n_c += 1
        if tok != toks[a]:
            break
    del cache, dcache
    mx.clear_cache()
    say(f"  {name}: {n_c} checks, kept {kept}; kept rows == AR {eq} / {n}")
    say("    first check whose START state differs: "
        + (f"#{first_state[0]} at relative {first_state[1]}: {first_state[2]}" if first_state else "none"))
    say("    first check with a differing kept row: "
        + (f"#{first_row[0]} at relative {first_row[1]} (kept {first_row[2]}, rows {first_row[3]}); its row {first_row[3][0]}'s "
           f"earliest differing record: " + (f"record {first_row[4][0]} ({first_row[4][1]}, ~layer {first_row[4][2]})"
                                             if first_row[4] else "none (all records equal)") if first_row else "none"))

loop("Loop A (live drafter)", True)
loop("Loop B (synthetic drafts)", False)
say(f"R-40 STEP 3p done in {time.time() - t0:.0f} s")
open(os.path.join(D, "r40s3p_screen.txt"), "w").write("\n".join(lines) + "\n")
