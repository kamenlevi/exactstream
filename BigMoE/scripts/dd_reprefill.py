#!/usr/bin/env python3
"""DEEPDRAFT re-prefill (Sat 14 Nov 2026, WEEK98 step 2 of "freshness first"): run the 125B's prefill over already
captured sequences (prompt + answer ids) and save the RAW taps at full width on every row. The taps are layers
5/15/25/35/45 (the mean of the 4 HC streams), 5 x 2560 fp16, rows aligned with the capture's h / wd (N - 1 rows).
They serve the per-layer freshness tests (layer 5 alone, 5 + 15, ...: what a cheap partial trunk pass would give) and
the raw anchor taps. The model, preset, tap hooks and prefill are dd_capture.py's, copied verbatim (lines 30-51, 73-91,
162-277, 290-304). A check per file: hidden_norm(fc(raw)) against the captured h, and the prefill's widened stream
against the captured wd. The answer rows were captured on the verify path, so a small difference is expected there.
usage: [BigMoE venv] dd_reprefill.py C LIST   (LIST = a text file of capture .npz paths; RP_OUT = the output folder,
       default ~/datasets/deepdraft/rawtaps; resumable: a file with output is skipped)
"""
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
assert BG.MAXK == int(os.environ["BIGMOE_MAXK"]) and BG.XLINES == (0 if BG.MAXK <= 61 else min(4, (BG.MAXK - 61 + 30) // 31)), \
    (BG.MAXK, BG.XLINES)                                # 160 / 4 as shipped; E2 small-C mode: 10 x rows
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


# ---------------------------------------------------------------- driver: re-prefill the listed files, raw taps on every row
OUT = H(os.environ.get("RP_OUT", "~/datasets/deepdraft/rawtaps")); os.makedirs(OUT, exist_ok=True)
FILES = [l.strip() for l in open(sys.argv[2]) if l.strip()]
say(f"re-prefill: {len(FILES)} files -> {OUT}")
for f in FILES:
    key = os.path.basename(f)[:-4]
    dst = os.path.join(OUT, key + ".npz")
    if os.path.exists(dst):
        continue
    while os.path.exists(PAUSE):
        time.sleep(5)
    z = np.load(f)
    ids = [int(t) for t in z["ids"]]
    if os.environ.get("RP_PROMPT_ONLY"):                  # a check: the prompt alone, as the capture prefilled it
        ids = ids[:int(z["gen"])]
    n = len(ids) - 1
    set_arm("slot")                                      # as dd_capture.py's run(): the prefill as shipped, REF2 on
    ref_on(True)
    t1 = time.perf_counter()
    cache, _lg, tp, w = prefill(ids)
    ref2.remove()                                        # as run(): the logging ops (hbits) on the stock path
    raw = tp[:n]
    hb = hbits(raw)
    mx.eval(raw, hb, w)
    h_cap = (np.asarray(z["h"][:n]).astype(np.uint32) << 16).view(np.float32)
    h_new = np.array(hb.astype(mx.float32))
    nz = np.abs(h_cap).sum(-1) > 0
    gen = int(z["gen"])
    ans = np.zeros(n, bool); ans[min(n, max(0, gen - 1)):] = True
    def rel(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-9))
    wd_cap = np.asarray(z["wd"][:n]).astype(np.float32); wd_new = np.array(w[:n].astype(mx.float32))
    msg = (f"h rel diff prompt {rel(h_new[nz & ~ans], h_cap[nz & ~ans]):.2e}, answer {rel(h_new[nz & ans], h_cap[nz & ans]):.2e}; "
           f"wd rel diff prompt {rel(wd_new[~ans], wd_cap[~ans]):.2e}, answer {rel(wd_new[ans], wd_cap[ans]):.2e}")
    np.savez(dst + ".tmp.npz", raw=np.array(raw).astype(np.float16), wd=np.array(w[:n]).astype(np.float16), src=f)
    os.replace(dst + ".tmp.npz", dst)
    del cache
    mx.clear_cache()
    say(f"{key}: {len(ids)} tokens in {time.perf_counter() - t1:.1f} s; {msg}")
say("re-prefill done")
