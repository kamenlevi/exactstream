#!/usr/bin/env python3
"""convcache on a stand-in loop (light GPU): the loop's globals as the cached prefill uses them, with a two-layer model
(one QSA attention + one GDN layer of qwen4_exp, random weights, the real cache classes) and an MTP head stand-in (one QSA
layer fed (hidden_i, id_i+1) pairs, as the loop's MTPHead.append). run() below follows the loop's run(): U3C footprint
on, prefill, the DFlash2 context T[P-2047:P] (+ A6 dwin's T[P-2048:P-1]) or the MTP head fed from ws, then decode-style
writes (16-row verify, rollback, commit 5) on the main cache and the MTP cache.

Requests follow Leap's recording (agent/sessions_real_1110.jsonl): conversation A (12,302 tokens, system + tools end at
10,019) turns 1-2, conversation B sharing 10,159 tokens (turn 1 DFlash2; turns 2-3 on the chat path = MTP head), a side
request (18 tokens), then A again. Each request is checked twice: convcache's own CONV_CACHE_CHECK (a cold prefill before
the warm one) and here, against a cold run of the loop's own prefill on a fresh cache (last logits, the drafter context,
the U3C footprint, the MTP head's cache after its prompt pairs).
usage: CONV_CACHE_CHECK=1 ~/BigMoE/.venv/bin/python conv/test_mock.py [out.json]"""
import hashlib, json, os, sys, time
from types import SimpleNamespace

os.environ.setdefault("CONV_CACHE_CHECK", "1")
os.environ.setdefault("CONV_CACHE_GB", "4")
import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import ArraysCache

mx.set_cache_limit(256 << 20)                             # give freed buffers back (beside a big job: keep the footprint small)

from mtplx import cache_state as CS
from mtplx.models import qwen4_exp as Q

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convcache as CC

ROOT = os.path.expanduser("~/BigMoE")
cfg = json.load(open(os.path.join(ROOT, "models/bare/config.json")))
args = Q.TextArgs.from_dict(cfg.get("text_config", cfg))
RATIO, H, NE, IM = args.indexer_compress_ratio or 4, args.hidden_size, 4099, 151644
mx.random.seed(0)
att, gdn, att_m = Q.Attention(args), Q.GatedDeltaNet(args), Q.Attention(args)
for m_ in (att, gdn, att_m):
    m_.set_dtype(mx.bfloat16)
E = (mx.random.normal((NE, H)) * 0.5).astype(mx.bfloat16)
WO = (mx.random.normal((H, 64)) * 0.05).astype(mx.bfloat16)
mx.eval(att.parameters(), gdn.parameters(), att_m.parameters(), E, WO)
TAP = {"on": False, "x": [[] for _ in range(2)]}
FG = dict(cap=None)
EM = SimpleNamespace(STATE=SimpleNamespace(capture=None))


def sha(x):
    if x.dtype == mx.bfloat16:
        x = x.view(mx.uint16)
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()[:16]


class Layer:                                               # a DecoderLayer stand-in, with the loop's _layer TAP wrapper
    def __init__(self, kind, mod, L):
        self.kind, self.mod, self.L, self.is_linear = kind, mod, L, kind == "gdn"

    def __call__(self, h, *, input_ids, ssm_mask, cache):
        x = h[..., :H]
        y = self.mod(x, cache) if self.kind == "att" else self.mod(x, None, cache)
        if FG["cap"] is not None:                          # the U3C hook: this layer's row, its routed "experts"
            FG["cap"][self.L, np.unique(np.array(input_ids) % 512)] = True
        out = h + mx.tile(y, (1, 1, 4))
        if TAP["on"]:
            B, L_, W = out.shape
            TAP["x"][self.L].append(out.reshape(B, L_, 4, W // 4).astype(mx.float32).mean(axis=2)[0].astype(mx.float16))
        return out


class TM:
    def __init__(self):
        self.layers = [Layer("att", att, 0), Layer("gdn", gdn, 1)]
        self.args = SimpleNamespace(hc_count=4)
        self._ple_stage_idx, self.ssm_idx, self._last_widened = None, 1, None

    def embed_tokens(self, x):
        return E[x % NE]

    def hyper_connection_mixer(self, h):
        B, L_, W = h.shape
        return h.reshape(B, L_, 4, W // 4).mean(axis=2)

    def __call__(self, inputs, cache):
        def fwd():
            h = mx.tile(self.embed_tokens(inputs), (1, 1, 4))
            for layer, c in zip(self.layers, cache):
                h = layer(h, input_ids=inputs, ssm_mask=None, cache=c)
            self._last_widened = h
            return self.hyper_connection_mixer(h)
        if Q.qwen4_opdiet_enabled("rope"):
            with Q._rope_table_scope():
                return fwd()
        return fwd()


class LM:
    def __init__(self):
        self.model = TM()

    def _head_logits(self, out):
        return out @ WO


class Model:
    def __init__(self):
        self.language_model = LM()

    def make_cache(self):
        return [Q.QSACache(RATIO), ArraysCache(size=2)]

    def make_mtp_cache(self):
        return [Q.QSACache(RATIO)]

    def __call__(self, x_ids, cache, emit_logits=True, logits_keep=None):
        return self.language_model._head_logits(self.language_model.model(x_ids, cache))


model = Model()
TXT = model.language_model.model
PFLM = int(os.environ.get("MOCK_PFLM", "0"))
import types
_bm, _bt = types.ModuleType("bigmoe"), types.ModuleType("bigmoe.tier")   # pflm's sweep calls bigmoe.tier._reuse_drop()
_bt._reuse_drop, _bm.tier = (lambda: None), _bt
sys.modules["bigmoe"], sys.modules["bigmoe.tier"] = _bm, _bt
sys.path.insert(0, os.path.expanduser("~/Leap/release/out/pkg_110f/engine/loop"))
import pflm as PFLMM                                       # Leap's real PF-A, for the loop's own (orig) path


def tap_reset():
    TAP["x"] = [[] for _ in range(2)]


def taps():
    f = [mx.concatenate(x) if len(x) > 1 else x[0] for x in TAP["x"]]
    return mx.stack(f, axis=1).reshape(f[0].shape[0], -1)


def states(cache):
    return [c.state for c in cache if c is not None and hasattr(c, "state")]


def prefill(ids, want_w=False, chunk=1024):                # the loop's prefill (r57_screen_rel1.py:986): PF-A or chunk-major
    if PFLM and len(ids) > chunk:
        tap_reset(); TAP["on"] = True
        ws = []

        def _onc(k_, h_):
            if want_w:
                mx.eval(h_)
                ws.append((k_ * chunk, h_))
        cache, lg = PFLMM.prefill_layer_major(model, ids, chunk, states, on_chunk=_onc, on_layer=None)
        mx.eval(lg, states(cache), [x[-1] for x in TAP["x"]])
        TAP["on"] = False
        return cache, lg[0, -1], taps(), ws
    cache, lg, ws = model.make_cache(), None, []
    tap_reset(); TAP["on"] = True
    for a in range(0, len(ids), chunk):
        lg = model(mx.array([ids[a:a + chunk]]), cache=cache)
        mx.eval(lg, states(cache), [x[-1] for x in TAP["x"]])
        if want_w:
            w = TXT._last_widened
            mx.eval(w)
            ws.append((a, w))
    TAP["on"] = False
    return cache, lg[0, -1], taps(), ws


class MTPHead:
    def __init__(self):
        self.cache = model.make_mtp_cache()
        self.logits = self.h = None

    def append(self, hidden, next_ids):
        x = hidden[..., :H] + E[mx.array(np.asarray(next_ids) % NE)][None]
        if x.shape[1] > 1:
            att_m(x[:, :-1], self.cache[0])
        self.h = att_m(x[:, -1:], self.cache[0])
        self.logits = self.h
        mx.eval(self.h, self.cache[0].state)


class _Tk:
    def token_to_id(self, t):
        return IM if t == "<|im_start|>" else None


g = dict(prefill=prefill, MTPHead=MTPHead, model=model, TAP=TAP, TXT=TXT, FG=FG, EM=EM, A6=set(), A6CAP={}, PFLM=PFLM,
         PFLMM=PFLMM,
         PFNOLOG=False, pf1=lambda on: None, tap_reset=tap_reset, taps=taps, states=states, SWIN=2048, _TKZ=_Tk())


def mtp_state(head):
    return [sha(x) for x in head.cache[0].state if x is not None]


def run(ids, mtp, rng, n_dec=60):
    """the loop's run() as far as the cache goes; returns what a cold run must reproduce"""
    FG["cap"] = np.zeros((48, 512), bool)
    cache, last, T, ws = g["prefill"](ids, mtp, chunk=1024)
    cap, FG["cap"] = FG["cap"], None
    P, out = len(ids), {}
    out["last"], out["cap"] = sha(last), hashlib.sha256(cap.tobytes()).hexdigest()[:16]
    tok = int(mx.argmax(last).item())
    if not mtp:
        out["ctx"] = sha(T[max(0, P - 2047):P])
        out["dwin"] = sha(T[max(0, P - 2048):P - 1])
    else:
        head, full = g["MTPHead"](), list(ids) + [tok]
        for a_, w in ws:
            head.append(w, full[a_ + 1:a_ + 1 + w.shape[1]])
        out["mtp"] = mtp_state(head)
    done = 0                                               # decode-style writes on the live caches
    while done < n_dec:
        snap = CS.snapshot_untrimmable_cache_lazy(cache)
        rows = rng.integers(0, 248320, 16).tolist()
        model(mx.array([rows]), cache=cache)
        mx.eval(states(cache))
        CS.rollback_after_verify(cache, snap, 16)
        model(mx.array([rows[:5]]), cache=cache)
        mx.eval(states(cache))
        if mtp:
            head.append(mx.zeros((1, 5, 4 * H), mx.bfloat16), rows[:5])
        done += 5
    return out


def cold(ids, mtp):
    """reference: the loop's own prefill on a fresh cache, and a fresh MTP head fed all its pairs"""
    FG["cap"] = np.zeros((48, 512), bool)
    cache, last, T, ws = prefill(ids, mtp, chunk=1024)
    cap, FG["cap"] = FG["cap"], None
    P, out = len(ids), {"last": sha(last), "cap": hashlib.sha256(cap.tobytes()).hexdigest()[:16]}
    tok = int(mx.argmax(last).item())
    if not mtp:
        out["ctx"] = sha(T[max(0, P - 2047):P])
        out["dwin"] = sha(T[max(0, P - 2048):P - 1])
    else:
        head, full = MTPHead(), list(ids) + [tok]
        for a_, w in ws:
            head.append(w, full[a_ + 1:a_ + 1 + w.shape[1]])
        out["mtp"] = mtp_state(head)
    return out


def turn(prev, rng, n_new):
    return prev + rng.integers(0, 248320, n_new).tolist()


if __name__ == "__main__":
    dst = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data/conv/b1_mock.json")
    t0 = time.time()
    assert CC.install(g)
    rng = np.random.default_rng(5)
    D = 8000 if os.environ.get("MOCK_SMALL") == "1" else 0   # MOCK_SMALL=1: the same shape 8,000 tokens shorter (~1.5 GB)
    sysp = rng.integers(0, 248320, 10019 - D).tolist()
    head_ = [IM] + rng.integers(0, 248320, 4).tolist()      # "<|im_start|>assistant\n<think>\n\n</think>\n\n" stand-in
    a1 = sysp + rng.integers(0, 248320, 140).tolist() + rng.integers(0, 248320, 12302 - 10019 - 140 - 5).tolist() + head_
    a2 = turn(a1, rng, 189) + head_                         # turn 1 + reply + tool result, then the generation prompt
    b1 = a1[:10159 - D] + rng.integers(0, 248320, 12300 - 10159 - 5).tolist() + head_
    b2 = turn(b1, rng, 99) + head_
    b3 = turn(b2, rng, 218) + head_
    side = rng.integers(0, 248320, 18).tolist()
    a3 = turn(a2, rng, 300) + head_
    plan = [("A turn 1", a1, False, dict(sys_end=10019 - D, lane="main", conv="A")),
            ("A turn 2", a2, False, dict(sys_end=10019 - D, lane="main", conv="A")),
            ("B turn 1 (new conversation, same system + tools)", b1, False, dict(sys_end=10019 - D, lane="main", conv="B")),
            ("B turn 2 (chat path: MTP head)", b2, True, dict(sys_end=10019 - D, lane="main", conv="B")),
            ("B turn 3 (chat path: MTP head)", b3, True, dict(sys_end=10019 - D, lane="main", conv="B")),
            ("side request (title)", side, False, dict(sys_end=0, lane="side", side=True)),
            ("A turn 3 (back to A)", a3, False, dict(sys_end=10019 - D, lane="main", conv="A"))]
    rows = []
    for name, ids, mtp, h in plan:
        CC.hint(**h)
        t1 = time.time()
        w = run(ids, mtp, rng)
        info = CC.last()
        ref = cold(ids, mtp)
        same = {k: w.get(k) == ref.get(k) for k in ref}
        rows.append(dict(case=name, N=len(ids), mtp=mtp, hit=info["hit"], c=info["c"], src=info["src"],
                         prefilled=info["prefilled"], miss=info["miss"], check=info["check"], vs_cold=same,
                         PASS=all(same.values()) and info["check"] in ("", "PASS"), entries=info.get("entries"),
                         gb=info.get("gb"), edges=info.get("edges"), s=round(time.time() - t1, 1)))
        print(json.dumps(rows[-1]), flush=True)
    e_end = ((12302 - D - 5) // 1024) * 1024
    expect = [(False, 0), (True, e_end), (True, (min(10019 - D, 10159 - D - 1) // 1024) * 1024), (False, 0), (True, e_end),
              (False, 0), (True, e_end)]
    if os.environ.get("CONV_CACHE_BRANCH") == "0":         # the fallback: the branch (case 3) is a cold miss
        expect[2] = (False, 0)
    got = [(r["hit"], r["c"]) for r in rows]
    R = dict(what="convcache B1 on a stand-in loop (1 QSA + 1 GDN layer, random weights)", pflm=PFLM, small=D, rows=rows,
             hits_as_expected=got == expect, expected=expect, got=got,
             PASS=all(r["PASS"] for r in rows) and got == expect, peak_GB=round(mx.get_peak_memory() / 1e9, 3),
             wall_s=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(R, open(dst, "w"), indent=1)
    print("PASS" if R["PASS"] else "FAIL", "hits", got, "expected", expect, "peak", R["peak_GB"], "GB, wall", R["wall_s"], "s")
