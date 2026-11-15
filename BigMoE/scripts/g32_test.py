#!/usr/bin/env python3
"""G32 test (Sun 15 Nov 2026; Leap's 32-row exact-check spec a821edc, harness ~/Leap/scripts/week98/c3_v32.py): the 32-row
exact check through BigMoE's widened gate (ext/build_g32, BIGMOE_MAXK 320) on the FULL streamed 125B (the runtime path, C
as given), against <= 16-row checks.
- ref2 runs with exact-45's env and forms, then v32 (V32_MOE1=1: at 17-32 rows in decode_verify, the MoE block keeps its
  verify lowering and the experts go as ONE call into the streamed SwitchGLU, i.e. through the G32 gate).
- A wide call takes the sync-free gate when its distinct-expert bound (gate_req) fits the pool. Per window, gate_req = the
  routing log's max distinct experts over layers for those tokens (bank routes, the same tokens on the runtime path).
- T1 forward: one M-row verify vs <= 16-row verifies of the same rows (full accepts between), logits per row bitwise,
  M = 32 / 17 / 24 / 31.
- T2 commit: a 32-row verify, commit keep k, then a 4-row probe; vs k rows as <= 16-row checks, then the same probe
  (k = 1, 9, 16, 17, 24, 31).
Pass: all bitwise.
usage: DLOG_ROWS=32 BIGMOE_MAXK=320 BIGMOE_EXT_BUILD=build_g32 V32_MOE1=1 [venv] g32_test.py C KEY [KEY...]
"""
import os
for k_, v_ in (("REF2_SB_FIRST", "1"), ("REF2_PLE_LATE", "1"), ("REF2_COMMIT_SUB", "9"), ("REF2_KEEP_CACHES", "1"), ("V32_MOE1", "1")):
    os.environ.setdefault(k_, v_)                       # exact-45 env for ref2 (as c3_v32.py) + one-call experts
import json, os, random, re, shutil, sys, threading, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)            # the streamed engine has no fused gate | up
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)          # SB comes from the lambda below, as in S45
os.environ.pop("BIGMOE_MTP_STREAM", None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["BIGMOE_TIER"] = "4433333333"; os.environ["BIGMOE_TIER_GLU"] = "3c"   # k69_live R57_PRESET=45 (REF2-T)
os.environ.setdefault("BIGMOE_TIER_CL", "112")
os.environ.pop("BIGMOE_TIER_COMBINE", None); os.environ.pop("BIGMOE_OOO", None)  # D-23's preset (D23_NEWDEF=0)
os.environ.setdefault("BIGMOE_MAXK", str(10 * int(os.environ.get("DLOG_ROWS", "16"))))   # E2: 10 x rows (160 as shipped)
os.environ.setdefault("BIGMOE_EXT_BUILD", "build_next")
assert os.environ.get("BIGMOE_MAXK") == str(10 * int(os.environ.get("DLOG_ROWS", "16")))
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

H = os.path.expanduser
PAUSE = H("~/.local/share/quiet/PAUSE")
C = int(sys.argv[1]) if len(sys.argv) > 1 else 100
TRUNC, R, K, TOPK = 0.3, 16, 16, 32
ROWS = int(os.environ.get("DLOG_ROWS", "16"))           # E2: rows per check (16 = as shipped; fewer at small C, exact)
EOS = {248044, 248046}
TAPS = (5, 15, 25, 35, 45)
PRESET = "45 (REF2-T 4433333333/3c, C_l 112, MAXK 160, build_next, L1-20 + cut 0.3; D-23's)"

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
def say(s):
    print(f"[dlog {time.strftime('%H:%M:%S')}] {s}", flush=True)

if os.environ.get("G32_DRY") == "1":                    # head-test: the launch, env, module asserts, ext build, windows; no model
    import bigmoe.gated as BG, bigmoe.stream as BS, bigmoe.tier as _BT
    assert BG.MAXK == int(os.environ["BIGMOE_MAXK"]) and BG.XLINES == (0 if BG.MAXK <= 61 else min(9, (BG.MAXK - 61 + 30) // 31)), (BG.MAXK, BG.XLINES)
    assert not _BT._COMBINE and not _BT._OOO, "D-23's pre-OOO preset expected"
    assert "build_g32" in BG.ext.__file__ or os.environ.get("BIGMOE_EXT_BUILD") != "build_g32", BG.ext.__file__
    import ref2
    sys.path.insert(0, H("~/Leap/scripts/week98"))
    import v32
    assert v32.ST["moe1"] and hasattr(v32, "install") and "moe1" in v32.CNT, v32.ST
    for key in sys.argv[2:]:
        bz = np.load(H(f"~/datasets/deepdraft/bank/{key}.npz")); rz = np.load(H(f"~/datasets/deepdraft/routes/{key}.npz"))
        gen, R = int(bz["gen"]), rz["routes"]
        G = max(len(np.unique(R[gen:gen + 36, l])) for l in range(R.shape[1]))
        say(f"DRY {key}: prompt {gen}, ids {len(bz['ids'])}, routes {R.shape}, gate_req {G} (C {C}: {'fits' if G <= C else 'DOES NOT FIT'})")
        assert G <= C and len(bz["ids"]) >= gen + 36
    say(f"DRY OK: MAXK {BG.MAXK}, XLINES {BG.XLINES}, ROW {BG.ROW} B, GATE_N {BS.GATE_N}, ext {BG.ext.__file__.split('/ext/')[-1]}, "
        f"V32_MOE1 {os.environ.get('V32_MOE1')}, ref2 {ref2.__file__.split('/')[-1]}")
    sys.exit(0)

# ---------------------------------------------------------------- the default's model (D-23's preset; no MTP head needed)
t0 = time.time()
rt, st = EM.load(EM.variant("ref"), C=C)
model = rt.model
try:                                                     # E2 small-C mode: rows x top-10 must fit every layer's base pool
    _cm = min(sw._bm["C"] for sw in st._tier_sws)
    if ROWS * 10 > _cm:
        say(f"E2: smallest base pool C_m {_cm} < {ROWS} rows x 10: rows -> {_cm // 10}")
        ROWS = _cm // 10
    say(f"E2: C {C}, base pool C_m >= {_cm}, rows per check {ROWS}, BIGMOE_MAXK {os.environ.get('BIGMOE_MAXK')}")
except Exception as e:
    say(f"E2: C_m check skipped ({e!r})")
import ref2
import bigmoe.stream as BS
import bigmoe.gated as BG
import bigmoe.tier as _BT
from mtplx import cache_state as CS
Q = ref2.Q
assert BG.MAXK == int(os.environ["BIGMOE_MAXK"]) and BG.XLINES == (0 if BG.MAXK <= 61 else min(9, (BG.MAXK - 61 + 30) // 31)), \
    (BG.MAXK, BG.XLINES)                                # 160 / 4 as shipped; G32: 320 / 9 (gated.py caps at 9 lines)
assert not _BT._COMBINE and not _BT._OOO, "D-23's pre-OOO preset expected"
TXT = model.language_model.model
lay = TXT.layers
SW_ = [l.mlp.switch_mlp for l in lay]
TOKZ = rt.tokenizer
say(f"loaded in {time.time() - t0:.0f} s, {mx.get_active_memory() / 1e9:.1f} GB active; C {C}; preset {PRESET}")

def set_arm(arm):                                        # k69_live's: the prefill as shipped ('slot'), the loop on the gate
    BS.GATE_N[0] = min(40, BG.MAXK) if arm == "slot" else BG.MAXK   # 40 / 160 as shipped (E2 small-C: <= the pool)
    BS._PF_RANKED = False
    for sw in SW_:
        sw._bm.pop("next", None)

def pf1(on):
    if not on and BS._STAGE["bufs"] is not None:
        BS._stage_drop()
    BS._PF_PREFILL = on

# ---------------------------------------------------------------- taps for the drafter (Leap's S45 harness, as D-23)
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
draft.load_weights(H("~/Leap/data/l120_ft.safetensors"), strict=False)
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

# ---------------------------------------------------------------- the logger: reads arrays the loop already evaluated
def top32(lg):
    """Leap's dfl2c_extract recipe: the 125B's top-32 ids (best first) and log-probs (fp16) per row"""
    lg = lg.astype(mx.float32)
    top = mx.argpartition(-lg, TOPK - 1, axis=-1)[:, :TOPK]
    v = mx.take_along_axis(lg, top, axis=-1)
    o = mx.argsort(-v, axis=-1)
    top, v = mx.take_along_axis(top, o, axis=-1), mx.take_along_axis(v, o, axis=-1)
    return top.astype(mx.int32), (v - mx.logsumexp(lg, axis=-1, keepdims=True)).astype(mx.float16)

def hbits(tp):
    """the drafter's own context input for these tap rows, hidden_norm(fc(taps)) in bf16 (stored as uint16 bits in save)"""
    return draft.hidden_norm(draft.fc(tp.astype(mx.bfloat16)))

def prefill(ids, chunk=1024):
    """k69_live's prefill (chunks of 1024 on the PF1 path, taps) + each chunk's widened stream and logits' top-32"""
    pf1(True)
    cache, lg, ws, tks = model.make_cache(), None, [], []
    tap_reset(); TAP["on"] = True
    for a in range(0, len(ids), chunk):
        lg = model(mx.array([ids[a:a + chunk]]), cache=cache)
        w = TXT._last_widened
        mx.eval(lg, states(cache), [x[-1] for x in TAP["x"]], w)
        assert w.shape[1] == min(chunk, len(ids) - a), w.shape
        ws.append(w[0].astype(mx.float16))
    TAP["on"] = False
    pf1(False)
    return cache, lg[0, -1], taps(), mx.concatenate(ws) if len(ws) > 1 else ws[0]




# ---------------------------------------------------------------- G32: ref2 forms (as c3_v32.py), v32, the verify scope
import contextlib
from mtplx.attention_context import attention_phase, exact_verify, model_forward_kind
ref2.apply()
for k_ in ("att_rope", "idx_prep", "hc_write", "hc_gate", "gdn_inproj_x", "gdn_body", "hc_b"):
    ref2.CFG[k_] = True
sys.path.insert(0, H("~/Leap/scripts/week98"))
import v32
v32.install(ref2)
say(f"G32: BIGMOE_MAXK {os.environ.get('BIGMOE_MAXK')} (gate XLINES {BG.XLINES}), ext {os.environ.get('BIGMOE_EXT_BUILD')}, C {C}, V32_MOE1 {os.environ.get('V32_MOE1')}")


@contextlib.contextmanager
def verify_scope():
    with attention_phase("decode_verify"), model_forward_kind("target_verify"), exact_verify(True), model.verify_capture_scope():
        yield


def pre(ids, gen):
    set_arm("slot")
    cache, _l, _t, _w = prefill(ids[:gen])
    mx.eval(states(cache))
    set_arm("gate")
    return cache


def verify(cache, ids, pos, n, keep=None):
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    with verify_scope():
        lg = model(mx.array([ids[pos:pos + n]]), cache=cache)
    lg = lg[0].astype(mx.float32); mx.eval(lg)
    if not model.commit_verified_window(cache, snap.states, keep_tokens=n if keep is None else keep, verified_tokens=n):
        raise RuntimeError(f"commit refused (n {n}, keep {keep})")
    mx.eval(states(cache))
    return lg


def chunks(n):
    out, a = [], 0
    while a < n:
        out.append(min(16, n - a)); a += out[-1]
    return out


def cmp(a, b):
    eq = mx.all(a == b, axis=-1); d = mx.max(mx.abs(a - b)); mx.eval(eq, d)
    rows = np.array(eq).astype(bool)
    return int(rows.sum()), len(rows), float(d.item())


def fwd(cache, ids, pos, n):                             # a verify forward, NOT committed: the caller rolls back or commits
    with verify_scope():
        lg = model(mx.array([ids[pos:pos + n]]), cache=cache)
    lg = lg[0].astype(mx.float32); mx.eval(lg)
    return lg


def rowdiff(a, b):                                       # per row: max |diff|, argmax equal
    d = mx.max(mx.abs(a - b), axis=-1); am = mx.argmax(a, axis=-1) == mx.argmax(b, axis=-1); mx.eval(d, am)
    return [round(float(x), 3) for x in np.array(d)], int(np.array(am).sum())


def flat(t):
    return [y for x in t for y in flat(x)] if isinstance(t, (list, tuple)) else ([t] if isinstance(t, mx.array) else [])


if os.environ.get("G32_BISECT") == "1":                 # Sun 02:3x: T1 / T2 were 0-bitwise at EVERY M (17-32) with the same
    key = sys.argv[2]                                    # max |diff| 3.15, the keep-1 probe too: one prefill + rollbacks (no
    bz = np.load(H(f"~/datasets/deepdraft/bank/{key}.npz")); rz = np.load(H(f"~/datasets/deepdraft/routes/{key}.npz"))
    ids = [int(t) for t in bz["ids"]]; gen = int(bz["gen"]); RT = rz["routes"]   # second prefill in the comparisons)
    G = max(len(np.unique(RT[gen:gen + 36, l])) for l in range(RT.shape[1]))
    for sw in SW_:
        sw._bm["gate_req"] = G
    say(f"BISECT {key}: prompt {gen}, gate_req {G}")
    set_arm("slot")
    ca, la, ta, wa = prefill(ids[:gen]); mx.eval(states(ca))
    cb, lb, tb, wb = prefill(ids[:gen]); mx.eval(states(cb))
    set_arm("gate")
    sa, sb = flat(states(ca)), flat(states(cb))
    neq = sum(1 for x, y in zip(sa, sb) if x.shape != y.shape or not bool(mx.all(x == y).item()))
    say(f"B0 two prefills: last logits bitwise {bool(mx.all(la == lb).item())} (max |diff| {float(mx.max(mx.abs(la.astype(mx.float32) - lb.astype(mx.float32))).item()):.3g}); "
        f"taps bitwise {bool(mx.all(ta == tb).item())}; widened bitwise {bool(mx.all(wa == wb).item())}; cache arrays unequal {neq} / {len(sa)}")
    del cb, sb
    P = ca
    S0 = CS.snapshot_untrimmable_cache_lazy(P)
    def back(n, s=None):
        CS.rollback_after_verify(P, S0 if s is None else s, n); mx.eval(states(P))
    l16a = fwd(P, ids, gen, 16); back(16); l16b = fwd(P, ids, gen, 16); back(16)
    v32.CNT.update({k: 0 for k in v32.CNT})
    l32a = fwd(P, ids, gen, 32); c32 = dict(v32.CNT); back(32); l32b = fwd(P, ids, gen, 32); back(32)
    say(f"B1 determinism on one prefill: 16 rows twice bitwise {cmp(l16a, l16b)[:2]}; 32 rows twice bitwise {cmp(l32a, l32b)[:2]}; v32 {c32}")
    sn = CS.snapshot_untrimmable_cache_lazy(P)
    lA = fwd(P, ids, gen, 16)
    assert model.commit_verified_window(P, sn.states, keep_tokens=16, verified_tokens=16)
    mx.eval(states(P))
    S1 = CS.snapshot_untrimmable_cache_lazy(P)
    lB = fwd(P, ids, gen + 16, 16); back(16, S1); back(16)
    d1, a1 = rowdiff(l32a[:16], lA); d2, a2 = rowdiff(l32a[16:], lB)
    say(f"B2 one 32-row call vs 16 + 16 (one prefill): rows 0-15 bitwise {cmp(l32a[:16], lA)[:2]}, argmax equal {a1} / 16; "
        f"rows 16-31 bitwise {cmp(l32a[16:], lB)[:2]}, argmax equal {a2} / 16; 16-row call now vs before {cmp(lA, l16a)[:2]}")
    say(f"B2 per-row max |diff| rows 0-15 {d1}")
    say(f"B2 per-row max |diff| rows 16-31 {d2}")
    v32.ST["moe1"] = False
    v32.CNT.update({k: 0 for k in v32.CNT})
    l32o = fwd(P, ids, gen, 32); co = dict(v32.CNT); back(32)
    v32.ST["moe1"] = True
    do, ao = rowdiff(l32o, mx.concatenate([lA, lB]))
    say(f"B3 moe1 OFF (v32 splits the MoE block 2 x 16): vs 16 + 16 bitwise {cmp(l32o, mx.concatenate([lA, lB]))[:2]}, argmax "
        f"equal {ao} / 32; vs moe1 ON bitwise {cmp(l32o, l32a)[:2]}; v32 {co}; per-row max |diff| {do}")
    tap_reset(); TAP["on"] = True
    fwd(P, ids, gen, 32); TAP["on"] = False; t32 = taps(); back(32)
    tap_reset(); TAP["on"] = True
    sn = CS.snapshot_untrimmable_cache_lazy(P)
    fwd(P, ids, gen, 16); TAP["on"] = False             # taps off through the commit (in case it replays layers)
    assert model.commit_verified_window(P, sn.states, keep_tokens=16, verified_tokens=16)
    mx.eval(states(P)); S1 = CS.snapshot_untrimmable_cache_lazy(P)
    TAP["on"] = True
    fwd(P, ids, gen + 16, 16); TAP["on"] = False; tch = taps(); back(16, S1); back(16)
    assert t32.shape == tch.shape, (t32.shape, tch.shape)
    nl = len(TAPS); t32r, tchr = t32.reshape(32, nl, -1), tch.reshape(32, nl, -1)
    per = []
    for i, L in enumerate(TAPS):
        eq = mx.all(t32r[:, i] == tchr[:, i], axis=-1); dd = mx.max(mx.abs(t32r[:, i].astype(mx.float32) - tchr[:, i].astype(mx.float32)))
        mx.eval(eq, dd)
        per.append(f"L{L}: {int(np.array(eq).sum())} / 32 rows bitwise (max |diff| {float(dd.item()):.3g})")
    say("B4 taps (layer outputs, 4-way mean, fp16), one 32-row call vs 16 + 16: " + "; ".join(per))
    say(f"BISECT done; peak {mx.get_peak_memory() / 1e9:.1f} GB")
    sys.exit(0)

if os.environ.get("G32_VDUMP"):                         # K-NEXT verify dump (MoonShots 03:4x, ~/FableGod/deepdraft/LEVERS_A_MOONSHOTS.md
    import collections                                   # "K-next"): per check-end miss of t30b's walk (checkstart_walk.py's
    for set_ in os.environ["G32_VDUMP"].split(","):     # kmiss_<set>.json), the 125B's row(s) computed on the WRONG draft d_k
        MS = json.load(open(H(f"~/BigMoE/data/deepdraft/ceiling/kmiss_{set_}.json")))   # after the true prefix ids[:t]: top-8
        byf = collections.OrderedDict()                  # ids / log-probs per branch row (d_k, then d_(k+1), d_(k+2) where
        for m_ in MS:                                    # recoverable), on the bank's runtime path (prefill + <= 16-row commits)
            byf.setdefault(m_["file"], []).append(m_)
        outp = H(f"~/BigMoE/data/deepdraft/ceiling/kdump_{set_}.jsonl")
        fo = open(outp, "w"); nb = hit = 0
        t0_ = time.time()
        for wi_, (fn, ms_) in enumerate(byf.items()):
            bz = np.load(H(f"~/datasets/deepdraft/bank/{fn}"))
            ids = [int(t_) for t_ in bz["ids"]]; gen = int(bz["gen"])
            P = pre(ids, gen); pos = gen
            for m_ in sorted(ms_, key=lambda x_: x_["t"]):
                t = m_["t"]
                while pos < t:                           # commit the true tokens up to t (exclusive)
                    n_ = min(16, t - pos); verify(P, ids, pos, n_); pos += n_
                toks = [m_["d"]] + ([m_["d1"]] + ([m_["d2"]] if m_["d2"] is not None else []) if m_["d1"] is not None else [])
                sn_ = CS.snapshot_untrimmable_cache_lazy(P)
                with verify_scope():
                    lg_ = model(mx.array([toks]), cache=P)
                lg_ = lg_[0].astype(mx.float32)
                ls_ = lg_ - mx.logsumexp(lg_, axis=-1, keepdims=True)
                top_ = mx.argpartition(-lg_, 7, axis=-1)[:, :8]
                tv_ = mx.take_along_axis(ls_, top_, axis=-1)
                o_ = mx.argsort(-tv_, axis=-1)
                top_, tv_ = mx.take_along_axis(top_, o_, axis=-1), mx.take_along_axis(tv_, o_, axis=-1)
                mx.eval(top_, tv_)
                CS.rollback_after_verify(P, sn_, len(toks)); mx.eval(states(P))
                tops, tlps = np.array(top_).tolist(), np.round(np.array(tv_), 4).tolist()
                rec = dict(m_, branch_rows=len(toks), top8_ids=tops, top8_lp=tlps,
                           y2=ids[t + 2] if t + 2 < len(ids) else None, y3=ids[t + 3] if t + 3 < len(ids) else None,
                           argmax_eq_y=bool(tops[0][0] == m_["y"]), margin_row=float(tlps[0][0] - tlps[0][1]))
                fo.write(json.dumps(rec) + "\n"); nb += 1; hit += int(rec["argmax_eq_y"])
            del P; mx.clear_cache()
            if wi_ % 10 == 0:
                say(f"VDUMP {set_}: {wi_ + 1} / {len(byf)} windows, {nb} branches, P(argmax == y) so far {hit / max(nb, 1):.3f}, {time.time() - t0_:.0f} s")
        fo.close()
        say(f"VDUMP {set_} done: {nb} branches -> {outp}; P(row past the miss's argmax == y) {hit / max(nb, 1):.4f}; {time.time() - t0_:.0f} s")
    say(f"VDUMP-ALL done; peak {mx.get_peak_memory() / 1e9:.1f} GB")
    sys.exit(0)


def rowsplit(a, s, e):                                   # rows [s, e) of an array whose rows axis is the one of size 32 / 16
    ax = [i for i, d in enumerate(a.shape) if d in (32,)][0]
    return a[tuple(slice(None) if i != ax else slice(s, e) for i in range(a.ndim))], ax


def rowseq(a, b, ax):                                    # per-row bitwise + max |diff| (rows axis first)
    a2, b2 = mx.moveaxis(a, ax, 0), mx.moveaxis(b, ax, 0)
    n = a2.shape[0]
    a2, b2 = a2.reshape(n, -1), b2.reshape(n, -1)
    eq = mx.all(a2 == b2, axis=-1); d = mx.max(mx.abs(a2.astype(mx.float32) - b2.astype(mx.float32))); mx.eval(eq, d)
    return int(np.array(eq).sum()), n, float(d.item())


if os.environ.get("G32_BISECT") == "2":                 # Sun 02:3x: B4 put the drift in L0-5 (L5 max 0.0072). The streamed
    key = sys.argv[2]                                    # experts' one 32-row call is BigMoE's piece; v32 splits everything
    bz = np.load(H(f"~/datasets/deepdraft/bank/{key}.npz")); rz = np.load(H(f"~/datasets/deepdraft/routes/{key}.npz"))
    ids = [int(t) for t in bz["ids"]]; gen = int(bz["gen"]); RT = rz["routes"]   # else (or leaves it stock, C3-checked).
    G = max(len(np.unique(RT[gen:gen + 36, l])) for l in range(RT.shape[1]))
    for sw in SW_:
        sw._bm["gate_req"] = G
    set_arm("slot")
    P, _l, _t, _w = prefill(ids[:gen]); mx.eval(states(P))
    set_arm("gate")
    S0 = CS.snapshot_untrimmable_cache_lazy(P)
    LS = (0, 1, 2, 3, 4, 5)
    BLK = {id(lay[L].mlp): L for L in LS}; SWI = {id(SW_[L]): L for L in LS}
    RB, RS = {}, {}
    _bc = type(lay[0].mlp); _bcall = _bc.__call__
    _sc = type(SW_[0]); _scall = _sc.__call__
    def _rb(self, x, *a, **kw):
        L = BLK.get(id(self))
        if L is not None and L not in RB:
            RB[L] = x
        return _bcall(self, x, *a, **kw)
    def _rs(self, x, indices, *a, **kw):
        L = SWI.get(id(self))
        if L is not None and L not in RS:
            RS[L] = (x, indices, a, kw)
        return _scall(self, x, indices, *a, **kw)
    _bc.__call__ = _rb; _sc.__call__ = _rs
    try:
        fwd(P, ids, gen, 32)
    finally:
        _bc.__call__ = _bcall; _sc.__call__ = _scall
    mx.eval(list(RB.values()), [v[0] for v in RS.values()], [v[1] for v in RS.values()])
    CS.rollback_after_verify(P, S0, 32); mx.eval(states(P))
    say(f"BB recorded at L{sorted(RB)}: block x {RB[LS[0]].shape}; switch x {RS[LS[0]][0].shape}, ids {RS[LS[0]][1].shape}, extra args {len(RS[LS[0]][2])} / {sorted(RS[LS[0]][3])}")
    for L in LS:
        x, ind, a, kw = RS[L]; sw = SW_[L]
        with verify_scope():
            e32 = sw(x, ind, *a, **kw); mx.eval(e32)
            xa, ax = rowsplit(x, 0, 16); ia, _ = rowsplit(ind, 0, 16)
            xb, _ = rowsplit(x, 16, 32); ib, _ = rowsplit(ind, 16, 32)
            ea = sw(xa, ia, *a, **kw); mx.eval(ea)
            eb = sw(xb, ib, *a, **kw); mx.eval(eb)
        _, eax = rowsplit(e32, 0, 16)
        e16 = mx.concatenate([ea, eb], axis=eax)
        r_e = rowseq(e32, e16, eax)
        xb_ = RB[L]; blk = lay[L].mlp
        with verify_scope():
            y32 = blk(xb_); mx.eval(y32)
            ya_, bax = rowsplit(xb_, 0, 16); yb_, _ = rowsplit(xb_, 16, 32)
            ya = blk(ya_); mx.eval(ya)
            yb = blk(yb_); mx.eval(yb)
        _, yax = rowsplit(y32, 0, 16)
        r_b = rowseq(y32, mx.concatenate([ya, yb], axis=yax), yax)
        say(f"BB L{L}: streamed experts 32 vs 16 + 16 rows bitwise {r_e[0]} / {r_e[1]} (max |diff| {r_e[2]:.3g}), out {tuple(e32.shape)}; "
            f"whole MoE block {r_b[0]} / {r_b[1]} (max |diff| {r_b[2]:.3g})")
    yk = RS[LS[0]]
    if yk[1].shape[-1] != 10:
        say("BB combine test skipped (ids not top-10)"); sys.exit(0)
    with verify_scope():
        e32 = SW_[LS[0]](yk[0], yk[1], *yk[2], **yk[3]); mx.eval(e32)
    sc_ = mx.softmax(mx.random.normal((32, 10), key=mx.random.key(0)), axis=-1).astype(e32.dtype)
    if e32.size != 32 * 10 * e32.shape[-1]:
        say(f"BB combine test skipped (expert output {tuple(e32.shape)} is already combined)"); sys.exit(0)
    e2 = e32.reshape(32, 10, e32.shape[-1])
    c32 = (e2 * sc_[..., None]).sum(axis=-2)
    c16 = mx.concatenate([(e2[:16] * sc_[:16, :, None]).sum(axis=-2), (e2[16:] * sc_[16:, :, None]).sum(axis=-2)])
    r_c = rowseq(c32, c16, 0)
    say(f"BB MLX combine (y * scores).sum(-2) at 32 vs 16 + 16 rows: bitwise {r_c[0]} / {r_c[1]} (max |diff| {r_c[2]:.3g}), dtype {e32.dtype}")
    say(f"BISECT2 done; peak {mx.get_peak_memory() / 1e9:.1f} GB")
    sys.exit(0)

GEN = {}
for l in open(H("~/EvalSuite/cache/ref/tf_gen.jsonl")):
    r = json.loads(l); GEN[r["key"].replace("/", "_")] = len(r["prompt_ids"])
allok = True
for key in sys.argv[2:]:
    bz = np.load(H(f"~/datasets/deepdraft/bank/{key}.npz")); rz = np.load(H(f"~/datasets/deepdraft/routes/{key}.npz"))
    ids = [int(t) for t in bz["ids"]]; gen = int(bz["gen"])
    R = rz["routes"]
    G = max(len(np.unique(R[gen:gen + 36, l])) for l in range(R.shape[1]))
    for sw in SW_:
        sw._bm["gate_req"] = G
    say(f"{key}: prompt {gen}, window rows {gen}..{gen + 35}: max distinct experts over layers {G} -> gate_req {G}")
    for M in (32, 17, 24, 31):
        c1 = pre(ids, gen)
        v32.CNT.update({k: 0 for k in v32.CNT})
        lv = verify(c1, ids, gen, M); cnt = dict(v32.CNT)
        c2 = pre(ids, gen); parts, pos = [], gen
        for n in chunks(M):
            parts.append(verify(c2, ids, pos, n)); pos += n
        eq, tot, d = cmp(lv, mx.concatenate(parts, axis=0)); allok &= eq == tot
        say(f"  T1 M {M}: rows bitwise {eq} / {tot} (max |diff| {d:.3g}); v32 wrappers {cnt}")
        del c1, c2; mx.clear_cache()
    for k in (1, 9, 16, 17, 24, 31):
        c1 = pre(ids, gen); verify(c1, ids, gen, 32, keep=k); pv = verify(c1, ids, gen + k, 4)
        c2 = pre(ids, gen); pos = gen
        for n in chunks(k):
            verify(c2, ids, pos, n); pos += n
        pr = verify(c2, ids, gen + k, 4)
        eq, tot, d = cmp(pv, pr); allok &= eq == tot
        say(f"  T2 keep {k}: probe rows bitwise {eq} / {tot} (max |diff| {d:.3g})")
        del c1, c2; mx.clear_cache()
rc = {k: v for k, v in ref2.counts().items() if v}
say(f"ref2 forms that ran (counts): {rc}")
say(f"G32 verdict: {'ALL BITWISE: the 32-row check through the widened gate equals <= 16-row checks' if allok else 'NOT BITWISE: bisect before anything else'}; peak {mx.get_peak_memory() / 1e9:.1f} GB")
