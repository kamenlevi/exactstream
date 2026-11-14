#!/usr/bin/env python3
"""DEEPDRAFT feature BANK (Sat 14 Nov 2026; the orchestrator 18:0x: build once, so most screens never run the 125B again).
The runtime path, per sequence (prompt + answer ids):
  - the prompt alone through the shipped prefill on the 1,024 grid, REF2 on, as dd_capture.py's run() (its rows equal
    the capture's bit for bit);
  - then the answer through verify calls of ROWS rows (the gate arm, inside ref2.verify_scope), every row committed:
    a check's own path at this C (ROWS = 9 at C 100, as the capture's E2 small-C mode sets it).
Saved per file (~/datasets/deepdraft/bank/<key>.npz), rows 0..N-2 aligned with the capture's h / wd:
  raw [n, 5 x 2560] fp16 (taps 5/15/25/35/45, the mean of the 4 HC streams), wd [n, 10240] fp16, tk / lp [n, 32] (the
  125B's top-32 ids / log-probs on rows >= gen - 1, zero before), ids, gen, src, path.
Sources: n1112 capture files (dd__mc_*), MBPP (dlog__mbpp*), the 24 HumanEval+ windows (tf_code_HumanEval_*: gen from
EvalSuite's tf_gen records, as dd_d22_ttt.gen_of). The model, preset, tap hooks and prefill are dd_capture.py's, copied
verbatim (via dd_reprefill.py).
usage: [BigMoE venv] dd_bank.py C LIST   (resumable; the log goes to stdout)
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



# ---------------------------------------------------------------- driver: the runtime path, per file
OUT = H("~/datasets/deepdraft/bank"); os.makedirs(OUT, exist_ok=True)
FILES = [l.strip() for l in open(sys.argv[2]) if l.strip()]
GEN = {}
for l in open(H("~/EvalSuite/cache/ref/tf_gen.jsonl")):
    r = json.loads(l)
    GEN[r["key"].replace("/", "_")] = len(r["prompt_ids"])
say(f"bank: {len(FILES)} files -> {OUT}; verify calls of {ROWS} rows")


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-9))


for f in FILES:
    key = os.path.basename(f)[:-4]
    dst = os.path.join(OUT, key + ".npz")
    if os.path.exists(dst):
        continue
    while os.path.exists(PAUSE):
        time.sleep(5)
    z = np.load(f)
    ids = [int(t) for t in z["ids"]]
    gen = int(z["gen"]) if "gen" in z.files else GEN[key]
    N = len(ids); n = N - 1
    t1 = time.perf_counter()
    set_arm("slot"); ref_on(True)
    cache, last, T0, W0 = prefill(ids[:gen])
    ref2.remove()
    tk0, lp0 = top32(last[None])
    raws, wds, tks, lps = [T0], [W0], [tk0], [lp0]
    ref2.apply(); set_arm("gate")
    a = gen
    while a < n:
        ch = ids[a:min(a + ROWS, n)]
        snap = CS.snapshot_untrimmable_cache_lazy(cache)
        tap_reset(); TAP["on"] = True
        with ref2.verify_scope(model):
            lg = model(mx.array([ch]), cache=cache)
        TAP["on"] = False
        tp = taps(); wv = TXT._last_widened
        mx.eval(lg, tp, wv)
        model.commit_verified_window(cache, snap.states, keep_tokens=len(ch), verified_tokens=len(ch))
        mx.async_eval(states(cache))
        ref2.remove()
        tk_, lp_ = top32(lg[0, :len(ch)])
        ref2.apply()
        raws.append(tp[:len(ch)]); wds.append(wv[0, :len(ch)].astype(mx.float16)); tks.append(tk_); lps.append(lp_)
        a += len(ch)
    mx.eval(states(cache))
    ref2.remove()
    raw = np.concatenate([np.array(x) for x in raws])[:n]; wd = np.concatenate([np.array(x) for x in wds])[:n]
    tk = np.zeros((n, 32), np.int32); lp = np.zeros((n, 32), np.float16)
    tk[gen - 1:] = np.concatenate([np.array(x) for x in tks])[:n - gen + 1]; lp[gen - 1:] = np.concatenate([np.array(x) for x in lps])[:n - gen + 1]
    assert raw.shape == (n, 12800) and wd.shape == (n, 10240), (raw.shape, wd.shape, n)
    msg = ""
    if "wd" in z.files and z["wd"].shape[0] >= n:          # checks against the source's own features (d16w keeps N rows)
        wc = np.asarray(z["wd"][:n]).astype(np.float32)
        msg = f"wd rel diff prompt {rel(wd[:gen].astype(np.float32), wc[:gen]):.2e}, answer {rel(wd[gen:].astype(np.float32), wc[gen:]):.2e}"
    if "tk" in z.files:
        tkc = np.asarray(z["tk"]); m = slice(gen - 1, n)
        msg += f"; top-1 agrees with the source on {100 * float((tk[m, 0] == tkc[m, 0]).mean()):.2f} % of answer rows"
    am = tk[gen - 1:n, 0] == np.asarray(ids[gen:N])
    msg += f"; greedy check: top-1 = the next token on {100 * float(am.mean()):.2f} %"
    np.savez(dst + ".tmp.npz", raw=raw.astype(np.float16), wd=wd.astype(np.float16), tk=tk, lp=lp, ids=np.asarray(ids, np.int32),
             gen=gen, src=f, path=f"runtime: prompt prefill (1,024 grid) + {ROWS}-row verify calls, C {C}")
    os.replace(dst + ".tmp.npz", dst)
    del cache
    mx.clear_cache()
    say(f"{key}: {N} tokens (prompt {gen}) in {time.perf_counter() - t1:.1f} s; {msg}")
say("bank build finished")
