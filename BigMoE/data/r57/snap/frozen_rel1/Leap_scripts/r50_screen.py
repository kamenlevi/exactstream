"""R-50 SLOTCALL (NewIdeas GO Thu 2026-10-29 18:59, from Leap's 18:48 proposal; Leap, pre-registered Thu 19:2x EET in
~/Leap/TABLE_350.md 'R-50 SLOTCALL'): 16-row checks on BigMoE's plain _sync_call path (PF1 off) gather each layer's
needed experts into a scratch pool (mx.take of up to ~158 experts x 9 pieces) before the MoE call. REF2's expert kernel
computes every (row, expert) pair on its own, so it can run on the pool itself with slot ids, as the gated path does:
ensure the experts, map ids -> slots, call; no gather, no scratch, no clear_cache.
Harness as STEP 3z (EvalSuite's REF harness, C = 176, PF1 for prefills only, DFlash2 L1-20 drafting live, greedy,
keep = accepted + 1). Both arms REF2 + SB:
  gather: BigMoE's _sync_call as shipped (scratch gather)
  slot:   the same, but REF2-on calls with <= 0.9 C distinct experts call the experts kernel on pool slots
Prompts: HumanEval/68 ABBA, prealgebra/1640 BAAB, <= 160 tokens; no new run after 420 s.
Lines (NewIdeas): PASS = every token equal between the arms AND each run's first-check logits hash equal AND >= 40 ms
saved per check (median cycle, gather - slot, per prompt, median over prompts); KILL = under 15 ms.
Usage: [BigMoE venv] r50_screen.py [C=176] [CAP=160] -> ~/Leap/data/r50_screen.txt"""
import hashlib, json, os, sys, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)            # the streamed engine has no fused gate | up
os.environ.pop("BIGMOE_TIER", None)
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)          # SB is switched per arm below
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
sys.path.insert(1, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_third_party"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
from evalsuite.decode import footprint_now_gb
import numpy as np
import mlx.core as mx
import dflash_model_mlx as DF

D = os.path.expanduser("~/Leap/data")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
C = int(sys.argv[1]) if len(sys.argv) > 1 else 176
C = min(C, 144)   # Thu 20:08: two loads at C 176 tripped run_locked's +2 GB swap guard (13 GB compressor,
                      # 19:15 and 20:06); a smaller pool cuts the footprint ~4.3 GB; slot calls need <= 0.9 C = 129 experts
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 160
R, K, NAR = 16, 16, 48
EOS = {248044, 248046}
TAPS = (5, 15, 25, 35, 45)
PICK = (("code", "humaneval", "humaneval/HumanEval/68"), ("math", "math", "math500/test/prealgebra/1640.json"),
        ("code", "humaneval", "humaneval/HumanEval/91"), ("math", "math", "math500/test/algebra/694.json"))
AR_KEYS = PICK[0][2], PICK[1][2]
LAST_RUN, HARD = 420, 500                                # s: no new run / check after these (job stop at 540)
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
say(f"R-50 SLOTCALL: full 125B via load_streamed (C = {C}), loaded in {time.time() - t0:.0f} s, "
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
    first_lg = None
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
        if first_lg is None:
            first_lg = lg[0]
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
    fh = hashlib.sha1(np.array(mx.view(first_lg.reshape(-1), mx.uint8)).tobytes()).hexdigest()[:16]
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
                ms=ms, io=1e3 * io, fh=fh, cnt={k_: v / max(len(kept), 1) for k_, v in cnt.items() if v} if ref else None)


# ---------------------------------------------------------------- the slot call (this process only; BigMoE's file is untouched)
ref2.CFG["slot_call"] = False                           # ref2.py has R-50 built in since 22:1x: keep the
_orig_sync = BS._sync_call                               # gather arm on BigMoE's shipped path
SLOT = {"on": False, "calls": 0, "fallback": 0}
def _slot_call(cls, self, x, indices, bm, *a, **kw):
    if not SLOT["on"] or "tier" in bm or BS._PF_PREFILL or bm.get("mtp") or not ref2.enabled():
        return _orig_sync(cls, self, x, indices, bm, *a, **kw)
    L, C = bm["L"], bm["C"]
    ids = np.array(indices)                              # the same id sync as the shipped path
    uniq = np.unique(ids)
    if len(uniq) > int(C * 0.9):                         # the shipped path would split into groups: leave it to it
        SLOT["fallback"] += 1
        return _orig_sync(cls, self, x, indices, bm, *a, **kw)
    BS.SYNC["calls"] += 1; BS.SYNC["rows"] += ids.size // ids.shape[-1]
    slots = np.array(BS.ext.ensure(L, mx.array(uniq.astype(np.int32))))
    sl = slots[np.searchsorted(uniq, ids)].astype(np.int32)
    y = cls.__call__(self, x, mx.array(sl).astype(indices.dtype), *a, **kw)
    mx.eval(y)                                           # as the shipped path: done before a later ensure of L can evict
    SLOT["calls"] += 1
    return y
BS._sync_call = _slot_call

RES = os.path.expanduser("~/EvalSuite/results")
PICK2 = PICK[:2]
items = {}
for _, s, key in PICK2:
    items.update({it["key"]: it for it in ED.load("dec_" + s) if it["key"] == key})
same = lambda a, b: next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
OUT, equal = [], True
for n_p, (dom, s, key) in enumerate(PICK2):
    ids = list(items[key]["prompt_ids"])
    order = (False, True, True, False) if n_p % 2 == 0 else (True, False, False, True)   # slot arm = True
    got = {False: [], True: []}
    for slot in order:
        if time.time() - t0 > LAST_RUN:
            say(f"  {key}: time budget, {len(got[False]) + len(got[True])} of 4 runs done")
            break
        SLOT["on"] = slot
        c0, f0 = SLOT["calls"], SLOT["fallback"]
        r = run(ids, True)
        SLOT["on"] = False
        r["slot_calls"] = (SLOT["calls"] - c0) / max(r["checks"], 1); r["fb"] = SLOT["fallback"] - f0
        if not r["cut"]:
            got[slot].append(r)
        say(f"  {key} [{'slot' if slot else 'gather'}] P {len(ids)}: prefill {r['pf']:.2f} s; {r['n']} tokens in "
            f"{r['checks']} checks (kept {r['kept']:.2f}), {r['n'] / r['dec']:.1f} tok/s; median cycle {r['cyc']:.1f} ms "
            f"(draft {r['dr']:.1f}, check + commit {r['ck']:.1f}); misses per check {r['ms']:.1f} (io {r['io']:.0f} ms); "
            f"first-check logits {r['fh']}; slot calls per check {r['slot_calls']:.1f}, fallbacks {r['fb']}"
            + ("; PAUSE seen" if r["paused"] else "") + ("; cut by the time limit" if r["cut"] else ""))
    runs = got[False] + got[True]
    if runs:
        tk = all(r["out"] == runs[0]["out"] for r in runs)
        fh = all(r["fh"] == runs[0]["fh"] for r in runs)
        if not (tk and fh):
            equal = False
            say(f"  {key}: arms differ: tokens equal {tk} (first difference at {min(same(r['out'], runs[0]['out']) for r in runs)}), "
                f"first-check logits equal {fh}")
    if got[False] and got[True]:
        OUT.append(dict(key=key, dom=dom, g=got[False], s=got[True],
                        save=np.mean([r["cyc"] for r in got[False]]) - np.mean([r["cyc"] for r in got[True]])))
say("  per prompt, saved per check (median cycle, gather - slot): "
    + ", ".join(f"{o['key'].split('/')[-1]} {o['save']:.1f} ms" for o in OUT))
clean = [o for o in OUT if not any(r["paused"] for r in o["g"] + o["s"])]
save = med([o["save"] for o in clean])
v = ("VOID (no prompt without a PAUSE)" if not clean else "FAIL (arms not bit-equal)" if not equal
     else "PASS" if save >= 40 else "KILL" if save < 15 else "BETWEEN")
say(f"R-50 SLOTCALL verdict (PASS: tokens + first-check logits equal AND >= 40 ms saved per check; KILL: < 15 ms): {v} "
    f"(median saving {save:.1f} ms over {len(clean)} prompt(s); arms bit-equal {equal}; {time.time() - t0:.0f} s)")
open(os.path.join(D, "r50_screen.txt"), "w").write("\n".join(lines) + "\n")
