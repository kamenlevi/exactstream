"""R-40 STEP 3 (NewIdeas Wed 21:2x, ~/NewIdeas/QUEUE.md row R-40 'STEP 3'; Leap, pre-registered Thu 2026-10-29 01:55 EET
in ~/Leap/TABLE_350.md 'R-40 STEP 3'): REF2 + SB vs today's shipped exact check in BigMoE's streamed engine, all 48
layers, with live drafting.
Harness: EvalSuite's REF harness (evalsuite.model.load('ref'), C = 176: under the 36 GB rule with the drafter; numerics do
not depend on C). Drafter: DFlash2 L1-20 (bf16, block 16, the official 1-step loop, R-06's exact two-stage head) on
taps from the verify forward (mean of the 4 HC streams after zero-based layers 5 / 15 / 25 / 35 / 45), always on stock
kernels. Greedy loop: 16-row check, keep = accepted + 1, commit_verified_window.
  today:     stock kernels, the engine's exact verify scopes, the commit left lazy (folds into the next check)
  REF2 + SB: ref2 on (prefill and checks, as ref2.generate); SB = the mid-loop async_eval every 4 layers (R-37's
             BIGMOE_VERIFY_MIDLOOP, patched per arm) + the commit replay submitted alone
Prompts: HumanEval/68, math500 prealgebra/1640, HumanEval/91, algebra/694 (EvalSuite's dec_ sets), <= 256 new tokens,
ABBA / BAAB; no new prompt after 420 s (the verdict then uses the prompts done).
Exact: every REF2 run's tokens == EvalSuite's REF2 AR rows (results/ref2, Leap 86ca915 = these kernel files) and the
REF2 runs agree; on HumanEval/68 and prealgebra/1640 every kept check row's logits == in-process ref2.ar_step logits
(first 48 tokens), bitwise.
Lines (NewIdeas): PASS = exact AND >= 40 ms saved per 48-layer check (median cycle, today - REF2, per prompt, median
over prompts); KILL = < 20 ms saved; BETWEEN otherwise.
Usage: [BigMoE venv] r40s3_screen.py [C=176] [CAP=256] -> ~/Leap/data/r40s3_screen.txt"""
import json, os, sys, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)            # the streamed engine has no fused gate | up
os.environ.pop("BIGMOE_TIER", None)
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)          # SB is switched per arm below
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
sys.path.insert(1, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/Leap_third_party"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
from evalsuite.decode import footprint_now_gb
import numpy as np
import mlx.core as mx
import dflash_model_mlx as DF

D = os.path.expanduser("~/Leap/data")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
C = int(sys.argv[1]) if len(sys.argv) > 1 else 176
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 256
R, K, NAR = 16, 16, 48
EOS = {248044, 248046}
TAPS = (5, 15, 25, 35, 45)
PICK = (("code", "humaneval", "humaneval/HumanEval/68"), ("math", "math", "math500/test/prealgebra/1640.json"),
        ("code", "humaneval", "humaneval/HumanEval/91"), ("math", "math", "math500/test/algebra/694.json"))
BUDGET = 420                                             # s: no new prompt after this (run_locked stops the job at 540)
AR_KEYS = {"humaneval/HumanEval/68", "math500/test/prealgebra/1640.json"}
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
med = lambda v: float(np.median(v)) if len(v) else float("nan")

t0 = time.time()
rt, st = EM.load(EM.variant("ref"), C=C)                # EvalSuite's REF arm: load_streamed + its install + configure
model = rt.model
import ref2
from mtplx import cache_state as CS
Q = ref2.Q
say(f"R-40 STEP 3: full 125B via load_streamed (C = {C}), loaded in {time.time() - t0:.0f} s, "
    f"{mx.get_active_memory() / 1e9:.1f} GB active")
ref2.apply()
st_ok = ref2.self_test()
ref2.remove(); ref2.reset_counts()
say(f"  self-test: {st_ok}")

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
    """ref2.prefill's calls (chunks of 1024, eval per chunk), plus the taps"""
    cache, lg = model.make_cache(), None
    tap_reset(); TAP["on"] = True
    for a in range(0, len(ids), chunk):
        lg = model(mx.array([ids[a:a + chunk]]), cache=cache)
        mx.eval(lg, states(cache), [x[-1] for x in TAP["x"]])
    TAP["on"] = False
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
    paused = False
    arm_on(ref)
    ref2.reset_counts()
    mx.synchronize()
    t0 = time.perf_counter()
    cache, last, T = prefill(ids)
    pf = time.perf_counter() - t0
    P = len(ids)
    dcache = draft.make_cache()
    start = max(0, P - (SW - 1))
    for c in dcache:
        c.offset = start
    ctx, a = T[start:P], P
    tok = int(mx.argmax(last).item())
    out, cyc, dr_t, ck_t, kept, pend = [tok], [], [], [], [], []
    td = c_prev = time.perf_counter()
    while len(out) < CAP and tok not in EOS:
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
                cyc=1e3 * med(cyc), dr=1e3 * med(dr_t), ck=1e3 * med(ck_t), bad=bad, cmp=cmp, paused=paused,
                cnt={k_: v / max(len(kept), 1) for k_, v in cnt.items() if v} if ref else None)

# ---------------------------------------------------------------- the runs
RES = os.path.expanduser("~/EvalSuite/results")
REFROWS = {v: {d["key"]: d for d in map(json.loads, open(f"{RES}/{v}/decode.jsonl"))} for v in ("ref2", "ref.r2")}
items = {}
for _, s, key in PICK:
    items.update({it["key"]: it for it in ED.load("dec_" + s) if it["key"] == key})
same = lambda a, b: next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
OUT, exact, ar_bad, ar_cmp = [], True, 0, 0
for n_p, (dom, s, key) in enumerate(PICK):
    if time.time() - t0 > BUDGET:
        say(f"  time budget: {time.time() - t0:.0f} s used, the remaining {len(PICK) - n_p} prompt(s) skipped")
        break
    ids = list(items[key]["prompt_ids"])
    ar = None
    if key in AR_KEYS:
        t1 = time.time()
        ar = ar_logits(ids, NAR)
        say(f"  {key}: REF2 AR logits for {NAR} tokens in {time.time() - t1:.0f} s")
    order = (False, True, True, False) if n_p % 2 == 0 else (True, False, False, True)
    got = {False: [], True: []}
    for j, ref in enumerate(order):
        r = run(ids, ref, ar if (ref and ar is not None and not got[True]) else None)
        got[ref].append(r)
        want = REFROWS["ref2" if ref else "ref.r2"][key]["completion_ids"]
        m = min(len(want), r["n"])
        agree = same(r["out"], want)
        if ref:
            exact = exact and agree >= m and r["bad"] == 0
            ar_bad += r["bad"]; ar_cmp += r["cmp"]
        say(f"  {key} [{'REF2 + SB' if ref else 'today'}] P {len(ids)}: prefill {r['pf']:.2f} s; {r['n']} tokens in "
            f"{r['checks']} checks (kept {r['kept']:.2f}), {r['n'] / r['dec']:.1f} tok/s; median cycle {r['cyc']:.1f} ms "
            f"(draft {r['dr']:.1f}, check + commit {r['ck']:.1f}); tokens == {'REF2' if ref else 'stock'} AR rows for "
            f"{agree} / {m}" + (f"; logits rows bitwise vs AR: {r['cmp'] - r['bad']} / {r['cmp']}" if r["cmp"] else "")
            + (f"; REF2 calls per check {r['cnt']}" if r["cnt"] else "") + ("; PAUSE seen" if r["paused"] else "")
            + f"; footprint {footprint_now_gb():.1f} GB")
    a2 = got[True]
    if a2[0]["out"] != a2[1]["out"]:
        exact = False
        say(f"  {key}: the two REF2 runs differ at token {same(a2[0]['out'], a2[1]['out'])}")
    OUT.append(dict(key=key, dom=dom, today=got[False], ref=got[True],
                    save=np.mean([r["cyc"] for r in got[False]]) - np.mean([r["cyc"] for r in got[True]])))

say("  per prompt, saved per 48-layer check (median cycle, today - REF2 + SB): "
    + ", ".join(f"{o['key'].split('/')[-1]} {o['save']:.1f} ms" for o in OUT))
for dom in ("code", "math"):
    oo = [o for o in OUT if o["dom"] == dom]
    if not oo:
        continue
    tps = {a: np.mean([r["n"] / r["dec"] for o in oo for r in o[a]]) for a in ("today", "ref")}
    say(f"  {dom}: tok/s today {tps['today']:.1f}, REF2 + SB {tps['ref']:.1f}; kept per check "
        f"{np.mean([r['kept'] for o in oo for r in o['today'] + o['ref']]):.2f}; prefill s today "
        f"{np.mean([r['pf'] for o in oo for r in o['today']]):.2f}, REF2 {np.mean([r['pf'] for o in oo for r in o['ref']]):.2f}")
clean = [o for o in OUT if not any(r["paused"] for r in o["today"] + o["ref"])]   # a PAUSE freezes the process
save = med([o["save"] for o in clean])
paused = len(clean) < len(OUT)
v = ("VOID (every prompt saw a PAUSE)" if not clean else "PASS" if exact and save >= 40 else "KILL" if save < 20
     else "BETWEEN" if exact else "FAIL (not exact)")
say(f"  exact: {exact} (REF2 tokens == EvalSuite's REF2 AR rows, REF2 runs agree, logits rows bitwise vs AR "
    f"{ar_cmp - ar_bad} / {ar_cmp})" + (f"; {len(OUT) - len(clean)} prompt(s) saw a PAUSE, left out of the saving"
                                         if paused else ""))
say(f"R-40 STEP 3 verdict (PASS: exact AND >= 40 ms saved per 48-layer check; KILL: < 20 ms): {v} "
    f"(median saving {save:.1f} ms per check)")
open(os.path.join(D, "r40s3_screen.txt"), "w").write("\n".join(lines) + "\n")
raise SystemExit(0 if exact else 1)
