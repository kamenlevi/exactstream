"""c55_profile.py (Breakthrough, Sun 15 Nov 2026; CHECK55 step 0 for Leap's plan ~/Leap/data/week98/check55_plan.md 76817d3):
where an exact 16- and 32-row check spends its time, on Leap's C6 setup (12-layer resident slice GGGA x 3, read-only imports
of ~/Leap/scripts ref2 / v32 and MTPLX) in the exact-45 engine's 32-row configuration (BigMoE G32T / C32m 03:1x: v32 +
V32_MOE1, no mr2, no v32mr2, no hc32, no DENSE-N; ref2 forms att_rope, idx_prep, hc_write (K-73) + hc_gate, gdn_inproj_x
(K-83), gdn_body off as in the 45 preset; SB every 4 layers on 2-32-row checks). No xctrace: libmtlprof (mtlprof.m), an
in-process hook on the Metal classes, counts every dispatch / barrier / encoder / command buffer and reads each command
buffer's own GPU start / end times.
Arms (each on its own cache from a 1,024-token prefill, all advancing 32 tokens per round as C6):
  A16   one 16-row check (+ an untimed second to keep pace);  C32  one 32-row check;
  with C55_K73R=1 also A16k / C32k: the same with the engine's K-73 MoE block (router_fused, shared_cat, moe_tail) ported to
  the resident experts (the slice has no tier switch, so the engine's own K-73 block falls back to MTPLX's stock block).
Pass T (timing; production scheduling, no extra flushes): 2 warm-up + C55_ROUNDS (12) rounds, arm order rotated. Per check:
  wall in three phases (forward call = Python build with SB's mid-loop submits; read = argmax().tolist(); commit =
  commit_verified_window + eval + synchronize), thread CPU per phase (waiting costs none: tested), the encode CPU inside
  SB's mx.async_eval calls, GPU busy per phase (union of command-buffer GPU intervals), dispatches / barriers / command
  buffers, and the kernels by pipeline name.
Pass K (attribution; separate rounds): every op-class boundary flushes (mx.async_eval) so each command buffer holds one
  class. Per class: dispatches, GPU ms (its command buffers' GPU time), Python build CPU, encode CPU; per layer index too.
  Classes: other (embed, tile, glue), ple, hc_read, hc_write, gdn_proj, gdn_core, att_proj, att_core, moe_router,
  moe_experts, moe_shared, moe_tail, head_mixer, head_lm, read, commit. The routing's union per layer gives the floors.
Usage: [venv] python -I -B c55_profile.py [smoke]   (MEM ~23 GB at 12 layers, ~7 GB at 4; quiet ticket for the real run)
Output: ~/Breakthrough/data/m1/check55/<tag>.json + .txt (C55_TAG, default c55_<HHMM>)."""
import contextlib
import glob
import json
import os
import sys
import time

for k_, v_ in (("REF2_SB_FIRST", "1"), ("REF2_PLE_LATE", "1"), ("REF2_COMMIT_SUB", "9"), ("REF2_KEEP_CACHES", "1"),
               ("V32_MOE1", "1")):
    os.environ.setdefault(k_, v_)
os.environ.pop("REF2_DENSE_NATIVE", None)                # exact-45 has no native copies
os.environ.pop("V32_MR2", None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mtlprof as MP

MP.load(int(os.environ.get("C55_PROFLIB", "2")))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx_lm.utils as mu

sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts/week98"))
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
SBN = int(os.environ.get("C55_SB", "4"))
ROUNDS = 3 if SMOKE else int(os.environ.get("C55_ROUNDS", "12"))
KROUNDS = 2 if SMOKE else int(os.environ.get("C55_KROUNDS", "4"))
SB32 = os.environ.get("C55_SB32") == "1"              # pass S: ref2's SB_FIRST + PLE_LATE seams widened to 17-32 rows
KO = os.environ.get("C55_KO") == "1" or SB32          # the causal knock-outs (pass X): +CPU spin, LM head as one call
K73R = os.environ.get("C55_K73R", "1") == "1" and not KO
SPIN_MS = float(os.environ.get("C55_SPIN_MS", "2.0"))
GDNB = os.environ.get("C55_GDNB") == "1"
TAG = os.environ.get("C55_TAG", "c55_" + time.strftime("%H%M") + ("_smoke" if SMOKE else ""))
OUT = os.path.expanduser("~/Breakthrough/data/m1/check55")
os.makedirs(OUT, exist_ok=True)
t00 = time.time()
LOG = []


def say(s):
    print(s, flush=True)
    LOG.append(s)


MD = os.path.expanduser("~/BigMoE/models/bare")
NL = 4 if SMOKE else 12
orig = mu.load_model


def lazy_load_model(path, *a, **kw):
    a = list(a)
    if a:
        a[0] = True
    else:
        kw["lazy"] = True
    model, cfg = orig(path, *a, **kw)
    if hasattr(model, "language_model"):
        inner = model.language_model.model
        inner.layers = inner.layers[:NL]
        mx.eval(model.parameters())
    return model, cfg


mu.load_model = lazy_load_model
try:
    from mtplx.runtime import load
    rt = load(MD, mtp=False)
finally:
    mu.load_model = orig
import mtplx.models.qwen4_exp as Q
import mtplx.cache_state as CS
from mlx_lm.models import switch_layers as SL
from mtplx.attention_context import attention_phase, exact_verify, model_forward_kind, current_attention_phase

if SBN:                                                  # the engine's SB (r57 ref_on), on 2-32-row checks: scheduling only
    _mid = Q._prefill_midloop_eval_layers
    Q._prefill_midloop_eval_layers = lambda rows: _mid(rows) or (SBN if 2 <= int(rows) <= 32 else 0)
model = rt.model
inner = model.language_model.model
KINDS = "".join("G" if l.is_linear else "A" for l in inner.layers)
import ref2

ref2.apply()
for k_ in ("att_rope", "idx_prep", "hc_write", "hc_gate", "gdn_inproj_x"):   # K-73 + K-83: the exact-45 forms
    ref2.CFG[k_] = True
ref2.CFG["gdn_body"] = GDNB                              # off in the 45 preset (R57_GDNB unset)
import v32

v32.install(ref2)
v32.ST.update(on=True, moe1=True, mr2=False)
VOCAB = int(model.language_model.lm_head.weight.shape[0]) if hasattr(model.language_model, "lm_head") else -1

# ---- K-73's block on the resident experts (C55_K73R): r57_screen_fast.py's _exact_block without the tier checks
from router_fused import router_fused
from glue_fused import moe_tail, shared_cat

_BLK_V32 = Q.SparseMoeBlock.__call__                     # v32's _moeblock over MTPLX's stock block
BLK = {"k73": False}
K73N = [0]


def _k73r_block(self, x):
    rows = 1
    for d in x.shape[:-1]:
        rows *= d
    if not BLK["k73"] or rows > 32 or current_attention_phase() != "decode_verify" or not self.norm_topk_prob:
        return _BLK_V32(self, x)
    K73N[0] += 1
    _mark("moe_router", x)
    inds, scores = router_fused(self.gate(x), self.top_k)
    _mark("moe_shared", inds, scores)
    g, s = shared_cat(self, x)
    _mark("moe_experts", g, s)
    y = self.switch_mlp(x, inds)
    _mark("moe_tail", y)
    if y.ndim == inds.ndim + 1:
        y = (y * scores[..., None]).sum(axis=-2)
    out = moe_tail(y, g, s, exp="precise")
    return out


Q.SparseMoeBlock.__call__ = _k73r_block

XO = {"head1": False, "spin": 0.0, "sb32": False}
_QMM_V32 = mx.quantized_matmul


def _qmm_ko(*a, **k):
    if XO["head1"] and int(a[1].shape[0]) == VOCAB:      # KO: the LM head as ONE 32-row MLX call (ref2 -> stock MLX above 16
        return v32._W["qmm"](*a, **k)                    # rows; not exact, timing only): one read of the 358 MB head
    return _QMM_V32(*a, **k)


mx.quantized_matmul = _qmm_ko
_DEC0 = Q.DecoderLayer.__call__


def _dec_spin(self, *a, **k):
    if XO["spin"] > 0:                                   # KO: extra Python CPU per layer (thread-CPU busy loop)
        t = time.thread_time()
        while time.thread_time() - t < XO["spin"]:
            pass
    return _DEC0(self, *a, **k)


Q.DecoderLayer.__call__ = _dec_spin
_DEC1 = Q.DecoderLayer.__call__


def _dec_sb32(self_, x, *a, **k):
    """pass S: ref2 _dec_call's seams (b) SB_FIRST and (d) PLE_LATE at 17-32 rows (ref2 itself: 5-16 rows only)"""
    if (XO["sb32"] and 17 <= int(x.shape[1]) <= 32 and ref2._ON[0] and getattr(self_, "ple", None) is not None
            and not ref2._IN_COMMIT[0]):
        if ref2.CFG["sb_first"]:
            mx.async_eval(x)
            SBN_CNT[0] += 1
        if ref2._PLE_DEFER[0] is not None:
            emb_, args_ = ref2._PLE_DEFER[0]
            ref2._PLE_DEFER[0] = None
            ref2._ORIG["ple_stage"](emb_, *args_)
            SBN_CNT[1] += 1
    return _DEC1(self_, x, *a, **k)


SBN_CNT = [0, 0]
Q.DecoderLayer.__call__ = _dec_sb32
_STAGE0 = Q.NGramEmbedding.stage


def _stage_sb32(self_, input_ids, cache, state_idx):
    """pass S: ref2 _ple_stage's deferral at 17-32 rows"""
    if (XO["sb32"] and 17 <= int(input_ids.shape[1]) <= 32 and ref2._ON[0] and ref2.CFG["ple_late"]
            and not ref2._IN_COMMIT[0]):
        ref2._PLE_DEFER[0] = (self_, (input_ids, cache, state_idx))
        return None
    return _STAGE0(self_, input_ids, cache, state_idx)


Q.NGramEmbedding.stage = _stage_sb32

# ---- attribution machinery (pass K only; inert otherwise)
CLASSES = ["other", "ple", "hc_read", "hc_write", "gdn_proj", "gdn_core", "att_proj", "att_core", "moe_router",
           "moe_experts", "moe_shared", "moe_tail", "head_mixer", "head_lm", "read", "commit"]
TI = {c: i for i, c in enumerate(CLASSES)}
ATTR = [False]
CUR = [0]
LAYER = [-1]
LASTU = [0.0]
ACC = {}                                                 # (layer, tag) -> [build s, encode s, dispatches, flushes]
FLUSHES = []                                             # (cb i0, cb i1, layer, tag)
UNION = {}                                               # layer -> [distinct experts per check]
_AE = mx.async_eval


def _collect(objs, out):
    for o in objs:
        if isinstance(o, mx.array):
            out.append(o)
        elif isinstance(o, (tuple, list)):
            _collect(o, out)
        elif isinstance(o, dict):
            _collect(list(o.values()), out)
        elif o is not None and not isinstance(o, nn.Module) and hasattr(o, "state"):
            out.extend(Q._midloop_state_arrays([o]))     # exactly what production's SB submit evaluates (recurrent states +
    return out                                           # boundary captures; KV entries are consumed by attention itself)


def _acc(key):
    a = ACC.get(key)
    if a is None:
        a = ACC[key] = [0.0, 0.0, 0, 0]
    return a


def _flush(objs):
    arrs = _collect(objs, [])
    key = (LAYER[0], CUR[0])
    a = _acc(key)
    if arrs:
        c0 = MP.counts()
        u = time.thread_time()
        _AE(*arrs)
        a[1] += time.thread_time() - u
        c1 = MP.counts()
        a[2] += c1["disp"] - c0["disp"]
        a[3] += 1
        if c1["ncb"] > c0["ncb"]:
            FLUSHES.append((c0["ncb"], c1["ncb"], LAYER[0], CUR[0]))


def _switch(tag, objs):
    """close the current class (its Python time, then flush its graph under its own tag) and open `tag`"""
    u = time.thread_time()
    _acc((LAYER[0], CUR[0]))[0] += u - LASTU[0]
    _flush(objs)
    CUR[0] = tag
    MP.tag(tag)
    LASTU[0] = time.thread_time()


def _mark(name, *objs):
    if ATTR[0]:
        _switch(TI[name], objs)


def _leaf(name, extra=None, after=None):
    """wrap f: flush the inputs under the caller's class, run f under `name`, flush the outputs (+ extra(args)) under it"""
    def deco(f):
        def w(*a, **k):
            if not ATTR[0]:
                return f(*a, **k)
            prev = CUR[0]
            nm = name(a) if callable(name) else name
            if nm is None:
                return f(*a, **k)
            _switch(TI[nm], a)
            out = f(*a, **k)
            back = after(prev) if after else prev
            _switch(back, (out,) + (tuple(extra(a, k)) if extra else ()))
            return out
        w.__name__ = getattr(f, "__name__", "w")
        w._c55_orig = f
        return w
    return deco


MIXER = inner.hyper_connection_mixer
LAYER_IDX = {id(l): i for i, l in enumerate(inner.layers)}
SAVED = {}


def _qmm_name(a, k):
    x, w = a[0], a[1]
    N = int(w.shape[0]) if k.get("transpose", True) else -1
    c = CLASSES[CUR[0]]
    if N == VOCAB:
        return "head_lm"
    if c in ("gdn_core",):
        return "gdn_proj"
    if c in ("att_core",):
        return "att_proj"
    if c in ("moe_router", "moe_tail") and N == 1:
        return "moe_shared"
    return None


def install_attr():
    """wrap the installed (ref2 / v32 / K-73r) callables at the class boundaries; uninstall_attr() restores them"""
    def put(obj, attr, w):
        SAVED[(obj, attr)] = getattr(obj, attr)
        setattr(obj, attr, w)
    qmm0 = mx.quantized_matmul

    def qmm_w(*a, **k):
        if not ATTR[0]:
            return qmm0(*a, **k)
        nm = _qmm_name(a, k)
        if nm is None:
            return qmm0(*a, **k)
        prev = CUR[0]
        _switch(TI[nm], a)
        out = qmm0(*a, **k)
        _switch(prev, (out,))
        return out
    put(mx, "quantized_matmul", qmm_w)
    lin0 = nn.Linear.__call__

    def lin_w(self, x):
        if not ATTR[0] or CLASSES[CUR[0]] not in ("gdn_core", "att_core"):
            return lin0(self, x)
        prev = CUR[0]
        _switch(TI["gdn_proj" if CLASSES[prev] == "gdn_core" else "att_proj"], (x,))
        out = lin0(self, x)
        _switch(prev, (out,))
        return out
    put(nn.Linear, "__call__", lin_w)
    put(Q.PLELayer, "__call__", _leaf("ple")(Q.PLELayer.__call__))
    put(Q.GatedResidual, "__call__", _leaf(lambda a: "head_mixer" if a[0] is MIXER else "hc_read")(Q.GatedResidual.__call__))
    put(Q, "_hyper_residual_write", _leaf("hc_write")(Q._hyper_residual_write))
    put(Q.GatedDeltaNet, "__call__", _leaf("gdn_core", extra=lambda a, k: tuple(a[3:4]) + ((k["cache"],) if "cache" in k else ()))(Q.GatedDeltaNet.__call__))
    put(Q.Attention, "__call__", _leaf("att_core", extra=lambda a, k: tuple(a[2:3]) + ((k["cache"],) if "cache" in k else ()))(Q.Attention.__call__))
    put(Q.SparseMoeBlock, "__call__", _leaf("moe_router")(Q.SparseMoeBlock.__call__))
    se_cls = type(inner.layers[0].mlp.shared_expert)
    put(se_cls, "__call__", _leaf(lambda a: "moe_shared" if CLASSES[CUR[0]].startswith("moe") else None)(se_cls.__call__))
    for cls in (Q._FusedGateUpSwitchGLU, SL.SwitchGLU):
        f0 = cls.__call__

        def sw_w(self, x, indices, *a, _f0=f0, **k):
            if ATTR[0]:
                try:
                    UNION.setdefault(LAYER[0], []).append(indices)
                except Exception:
                    pass
            return _leaf("moe_experts", after=lambda p: TI["moe_tail"] if CLASSES[p] in ("moe_router", "moe_experts") else p)(_f0)(
                self, x, indices, *a, **k)
        put(cls, "__call__", sw_w)
    dec0 = Q.DecoderLayer.__call__

    def dec_w(self, *a, **k):
        if not ATTR[0]:
            return dec0(self, *a, **k)
        prevL = LAYER[0]
        _switch(CUR[0], a + tuple(k.values()))           # what came before this layer, under its class and layer
        LAYER[0] = LAYER_IDX.get(id(self), -1)
        out = dec0(self, *a, **k)
        _switch(CUR[0], (out,))
        LAYER[0] = prevL
        return out
    put(Q.DecoderLayer, "__call__", dec_w)


def uninstall_attr():
    for (obj, attr), f in SAVED.items():
        setattr(obj, attr, f)
    SAVED.clear()


# ---- encode CPU inside SB's mid-loop submits (pass T)
ENC = [0.0]


def _ae_timed(*a, **k):
    u = time.thread_time()
    r = _AE(*a, **k)
    ENC[0] += time.thread_time() - u
    return r


mx.async_eval = _ae_timed

say(f"C55 {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; slice {KINDS} ({NL} layers), "
    f"{mx.get_active_memory() / 1e9:.1f} GB active; SB {SBN or 'off'}; gdn_body {GDNB}; K-73r arms {K73R}; vocab {VOCAB}")
files = sorted(glob.glob(os.path.expanduser("~/Leap/scripts/*.py")))
ids = list(rt.tokenizer.encode("".join(open(f).read() for f in files)))
P = 1024


@contextlib.contextmanager
def verify_scope():
    with attention_phase("decode_verify"), model_forward_kind("target_verify"), exact_verify(True), \
            model.verify_capture_scope():
        yield


def states(cache):
    return [c.state for c in cache if hasattr(c, "state")]


def prefill():
    BLK["k73"] = False
    cache = model.make_cache()
    lg = model(mx.array([ids[0:P]]), cache=cache)
    mx.eval(lg, states(cache))
    return cache


ARMTAG = [0]                                             # pass T: each arm's checks carry tag 32 + its index


def check(cache, pos, n, rec=False):
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    mx.synchronize()
    if not ATTR[0]:
        MP.tag(ARMTAG[0] if rec else 63)                 # only the timed checks carry the arm's tag (63 = untagged)
    c0 = MP.counts()
    g0 = MP.tag_counts()
    ENC[0] = 0.0
    if ATTR[0]:
        CUR[0] = TI["other"]
        MP.tag(CUR[0])
        LAYER[0] = -1
    t0 = MP.now(); w0 = time.perf_counter(); u0 = time.thread_time()
    LASTU[0] = u0
    with verify_scope():
        lg = model(mx.array([ids[pos:pos + n]]), cache=cache)
    tf = MP.now(); w1 = time.perf_counter(); u1 = time.thread_time()
    enc_f = ENC[0]
    if ATTR[0]:
        _switch(TI["read"], (lg,))
    mx.argmax(lg[0], axis=-1).tolist()
    tr = MP.now(); w2 = time.perf_counter(); u2 = time.thread_time()
    if ATTR[0]:
        _switch(TI["commit"], ())
    ok = model.commit_verified_window(cache, snap.states, keep_tokens=n, verified_tokens=n)
    assert ok
    mx.eval(states(cache))
    mx.synchronize()
    if ATTR[0]:
        _switch(TI["other"], ())
    t3 = MP.now(); w3 = time.perf_counter(); u3 = time.thread_time()
    c1 = MP.counts()
    g1 = MP.tag_counts()
    if not ATTR[0]:
        MP.tag(63)
    if not rec:
        return None
    return dict(t=(t0, tf, tr, t3), w=(w1 - w0, w2 - w1, w3 - w2), u=(u1 - u0, u2 - u1, u3 - u2), enc_f=enc_f,
                c={k: c1[k] - c0[k] for k in c1}, cb=(c0["ncb"], c1["ncb"]), tagdisp=(g1 - g0).tolist())


ARMS = {"A16": (16, False), "C32": (32, False)}
if K73R:
    ARMS.update({"A16k": (16, True), "C32k": (32, True)})
if KO and not SB32:
    ARMS.update({"A16cpu": (16, False), "C32cpu": (32, False), "C32h1": (32, False)})
if SB32:
    ARMS.update({"C32sb": (32, False)})
XO_ARM = {"A16cpu": dict(spin=1e-3 * SPIN_MS / NL), "C32cpu": dict(spin=1e-3 * SPIN_MS / NL), "C32h1": dict(head1=True),
          "C32sb": dict(sb32=True)}
C = {a: prefill() for a in ARMS}
POS = {a: P for a in ARMS}


def run_arm(a, rec=False):
    n, k73 = ARMS[a]
    BLK["k73"] = k73
    ARMTAG[0] = 32 + list(ARMS).index(a)
    c, p = C[a], POS[a]
    XO.update(head1=False, spin=0.0, sb32=False)
    XO.update(XO_ARM.get(a, {}))
    r = check(c, p, n, rec)
    XO.update(head1=False, spin=0.0, sb32=False)
    if n < 32:                                           # untimed: keep pace (32 tokens per round), outside attribution
        was = ATTR[0]
        ATTR[0] = False
        check(c, p + n, 32 - n)
        ATTR[0] = was
    POS[a] += 32
    BLK["k73"] = False
    return r


# ---- bits: the K-73r block equals MTPLX's stock block (info; same cache state, one forward each, then rolled back)
def logits_once(a, k73):
    BLK["k73"] = k73
    n = ARMS[a][0]
    sn = CS.snapshot_untrimmable_cache_lazy(C[a])
    with verify_scope():
        lg = model(mx.array([ids[POS[a]:POS[a] + n]]), cache=C[a]).astype(mx.float32)
    mx.eval(lg)
    CS.rollback_after_verify(C[a], sn, n)
    mx.eval(states(C[a]))
    BLK["k73"] = False
    return lg


for i in range(2):                                       # warm-up (and the shared_cat concats are made here)
    for a in ARMS:
        run_arm(a)
BITS = {}
if SB32:
    def logits_sb(a, on):
        XO["sb32"] = on
        sn = CS.snapshot_untrimmable_cache_lazy(C[a])
        with verify_scope():
            lg = model(mx.array([ids[POS[a]:POS[a] + 32]]), cache=C[a]).astype(mx.float32)
        mx.eval(lg)
        CS.rollback_after_verify(C[a], sn, 32)
        mx.eval(states(C[a]))
        XO["sb32"] = False
        return lg
    c0_ = SBN_CNT[:]
    l0, l1 = logits_sb("C32", False), logits_sb("C32", True)
    eq = mx.all(l0 == l1, axis=-1)
    mx.eval(eq)
    BITS["C32sb_vs_C32"] = (int(np.array(eq).sum()), int(eq.shape[-1]))
    say(f"bits: C32sb (SB_FIRST + PLE_LATE at 32 rows) vs C32, rows bitwise {BITS['C32sb_vs_C32']}; seams fired (b, d) "
        f"{SBN_CNT[0] - c0_[0]}, {SBN_CNT[1] - c0_[1]}")
if K73R:
    for a in ("A16", "C32"):
        l0, l1 = logits_once(a, False), logits_once(a, True)
        eq = mx.all(l0 == l1, axis=-1)
        mx.eval(eq)
        BITS[a] = (int(np.array(eq).sum()), int(eq.shape[-1]))
    say(f"bits: K-73r block vs MTPLX's stock block, rows bitwise {BITS} (info only)")

# ---- calibration: one flush of one tiny kernel (encode + commit CPU; the command buffer's own GPU time)
z = mx.zeros((16, 2560), mx.bfloat16)
mx.eval(z)
mx.synchronize()
cal_u, cal_n0 = [], MP.counts()["ncb"]
for i in range(300):
    z = z + 1
    u = time.thread_time()
    _AE(z)
    cal_u.append(time.thread_time() - u)
mx.synchronize()
time.sleep(0.1)
cal_cb = MP.cbs(cal_n0, MP.counts()["ncb"])
cal_d = (cal_cb["end"] - cal_cb["start"])[cal_cb["ready"] == 1]
CAL = dict(flush_cpu_us=1e6 * float(np.median(cal_u[50:])), cb_gpu_us_med=1e6 * float(np.median(cal_d)),
           cb_gpu_us_p10=1e6 * float(np.percentile(cal_d, 10)), ncb=int(len(cal_cb)))
say(f"calibration: one flush of one tiny kernel costs {CAL['flush_cpu_us']:.1f} us thread CPU; its command buffer "
    f"{CAL['cb_gpu_us_med']:.1f} us GPU (p10 {CAL['cb_gpu_us_p10']:.1f}) over {CAL['ncb']} buffers")

# ---- pass T
names = list(ARMS)
RT = {a: [] for a in ARMS}
PS0 = MP.psos()
for r in range(ROUNDS):
    order = names[r % len(names):] + names[:r % len(names)]
    for a in order:
        RT[a].append(run_arm(a, rec=True))
PS1 = MP.psos()
time.sleep(0.2)
ncb_end = MP.counts()["ncb"]
CBALL = MP.cbs(0, ncb_end)


def cb_slice(i0, i1):
    return CBALL[i0:i1]


def summarize(recs):
    out = {"wall": [], "fwd": [], "read": [], "commit": [], "cpu_build": [], "cpu_enc_sb": [], "cpu_read": [],
           "cpu_commit": [], "gpu": [], "gpu_fwd": [], "gpu_read": [], "gpu_commit": [], "gpu_first": [], "idle": [],
           "disp": [], "bar": [], "enc": [], "ncb": [], "fwait": []}
    for x in recs:
        t0, tf, tr, t3 = x["t"]
        cb = cb_slice(*x["cb"])
        W = sum(x["w"])
        g = MP.busy(cb, t0, t3)
        out["wall"].append(W); out["fwd"].append(x["w"][0]); out["read"].append(x["w"][1]); out["commit"].append(x["w"][2])
        out["cpu_build"].append(x["u"][0] - x["enc_f"]); out["cpu_enc_sb"].append(x["enc_f"])
        out["cpu_read"].append(x["u"][1]); out["cpu_commit"].append(x["u"][2])
        out["gpu"].append(g); out["gpu_fwd"].append(MP.busy(cb, t0, tf)); out["gpu_read"].append(MP.busy(cb, tf, tr))
        out["gpu_commit"].append(MP.busy(cb, tr, t3))
        st = [s for s, rd in zip(cb["start"], cb["ready"]) if rd and s >= t0]
        out["gpu_first"].append((min(st) - t0) if st else float("nan"))
        out["idle"].append(W - g)
        for k in ("disp", "bar", "enc", "fwait"):
            out[k].append(x["c"][k])
        out["ncb"].append(x["c"]["ncb"])
    return {k: float(np.median(v)) for k, v in out.items()}, {k: [float(z) for z in v] for k, v in out.items()}


ST_T, RAW_T = {}, {}
for a in names:
    ST_T[a], RAW_T[a] = summarize(RT[a])
KERN = MP.pso_delta(PS0, PS1)                            # all arms together over pass T
nchecks = ROUNDS * len(names)
SLOTNAME = {k: v[0] for k, v in MP.psos().items()}
KERN_ARM = {}
for i, a in enumerate(names):                            # per arm (tag 32 + i), per check; the untimed pace checks excluded
    tp = MP.tag_pso(32 + i)
    agg = {}
    for sl in np.nonzero(tp)[0]:
        agg[SLOTNAME.get(int(sl), "?")] = agg.get(SLOTNAME.get(int(sl), "?"), 0) + int(tp[sl])
    KERN_ARM[a] = {k: v / ROUNDS for k, v in agg.items()}   # per timed check

# ---- KO report (pass X): paired per-round deltas against the base arm, then stop (no pass K)
if KO:
    m_ = lambda v: 1e3 * v
    say(f"pass X (knock-outs): {ROUNDS} rounds x {len(names)} arms, order rotated; per check medians, paired deltas per round")
    say("| arm | wall | CPU total | GPU busy | GPU idle | dispatches |")
    say("|---|---|---|---|---|---|")
    for a in names:
        s_ = ST_T[a]
        cpu_ = s_["cpu_build"] + s_["cpu_enc_sb"] + s_["cpu_read"] + s_["cpu_commit"]
        say(f"| {a} | {m_(s_['wall']):.2f} | {m_(cpu_):.2f} | {m_(s_['gpu']):.2f} | {m_(s_['idle']):.2f} | {s_['disp']:.0f} |")
    KOD = {}
    for a, b in ((("C32sb", "C32"),) if SB32 else (("A16cpu", "A16"), ("C32cpu", "C32"), ("C32h1", "C32"))):
        for f in ("wall", "gpu"):
            d_ = np.array(RAW_T[a][f]) - np.array(RAW_T[b][f])
            KOD[f"{a}-{b}:{f}"] = (m_(np.median(d_)), m_(np.percentile(d_, 25)), m_(np.percentile(d_, 75)))
        cpu_a = np.array(RAW_T[a]["cpu_build"]) + np.array(RAW_T[a]["cpu_enc_sb"]) + np.array(RAW_T[a]["cpu_read"])
        cpu_b = np.array(RAW_T[b]["cpu_build"]) + np.array(RAW_T[b]["cpu_enc_sb"]) + np.array(RAW_T[b]["cpu_read"])
        d_ = cpu_a - cpu_b
        KOD[f"{a}-{b}:cpu"] = (m_(np.median(d_)), m_(np.percentile(d_, 25)), m_(np.percentile(d_, 75)))
        d_ = np.array(RAW_T[a]["gpu_first"]) - np.array(RAW_T[b]["gpu_first"])
        KOD[f"{a}-{b}:gpu_first"] = (m_(np.median(d_)), m_(np.percentile(d_, 25)), m_(np.percentile(d_, 75)))
        say(f"  {a} - {b}: GPU first start {KOD[a + '-' + b + ':gpu_first'][0]:+.2f} ms (median {m_(ST_T[a]['gpu_first']):.2f} vs "
            f"{m_(ST_T[b]['gpu_first']):.2f})")
        say(f"  {a} - {b}: wall {KOD[a + '-' + b + ':wall'][0]:+.2f} ms [IQR {KOD[a + '-' + b + ':wall'][1]:+.2f}, "
            f"{KOD[a + '-' + b + ':wall'][2]:+.2f}]; CPU {KOD[a + '-' + b + ':cpu'][0]:+.2f}; GPU busy {KOD[a + '-' + b + ':gpu'][0]:+.2f}")
    say(f"done in {time.time() - t00:.0f} s; peak {mx.get_peak_memory() / 1e9:.1f} GB")
    json.dump(dict(tag=TAG, when=time.strftime("%a %d %b %Y %H:%M:%S"), ko=True, sb32=SB32, bits=BITS, sbn_cnt=SBN_CNT, spin_ms=SPIN_MS, rounds=ROUNDS, T=ST_T, T_raw=RAW_T,
                   deltas=KOD, kernels_T_arm=KERN_ARM, log=LOG), open(os.path.join(OUT, TAG + ".json"), "w"), indent=1, default=str)
    open(os.path.join(OUT, TAG + ".txt"), "w").write("\n".join(LOG) + "\n")
    say(f"wrote {OUT}/{TAG}.json / .txt")
    sys.exit(0)

# ---- pass K
install_attr()
ATTR[0] = True
TP0 = {c: MP.tag_pso(TI[c]) for c in CLASSES}
ACC.clear(); FLUSHES.clear(); UNION.clear()
RK = {a: [] for a in names}
ACCK = {}
FLK = {}
UNK = {}
PSO_K = {}
for r in range(1 + KROUNDS):
    order = names[r % len(names):] + names[:r % len(names)]
    for a in order:
        ACC.clear(); FLUSHES.clear(); UNION.clear()
        tp0 = {c: MP.tag_pso(TI[c]) for c in CLASSES}
        x = run_arm(a, rec=True)
        if r == 0:
            continue
        tp1 = {c: MP.tag_pso(TI[c]) for c in CLASSES}
        RK[a].append(x)
        ACCK.setdefault(a, []).append({k: list(v) for k, v in ACC.items()})
        FLK.setdefault(a, []).append(list(FLUSHES))
        u = {}
        for L, lst in UNION.items():
            ids_ = np.concatenate([np.array(t.astype(mx.int32)).reshape(-1) for t in lst])
            u[L] = int(len(np.unique(ids_)))
        UNK.setdefault(a, []).append(u)
        pk = PSO_K.setdefault(a, {})
        for c in CLASSES:
            d = tp1[c] - tp0[c]
            for sl in np.nonzero(d)[0]:
                pk.setdefault(c, {})
                pk[c][int(sl)] = pk[c].get(int(sl), 0) + int(d[sl])
ATTR[0] = False
uninstall_attr()
MP.tag(0)
time.sleep(0.2)
CBALL = MP.cbs(0, MP.counts()["ncb"])
SLOTNAME = {k: v[0] for k, v in MP.psos().items()}


def attr_table(a):
    """per class, per check (mean over the measured rounds): dispatches (the hook's per-tag count), GPU ms (its command
    buffers' GPU time, by the tag at commit), GPU ms net of one empty buffer's time each, build ms, encode ms (flush
    thread CPU), command buffers; and dispatches per layer by layer kind (G / A) from the flush records"""
    n = len(ACCK[a])
    T = {c: [0.0] * 7 for c in CLASSES}                  # disp, gpu s, gpu net s, build s, enc s, flushes, cbs
    TL = {}
    for x in RK[a]:
        td = x["tagdisp"]
        for c in CLASSES:
            T[c][0] += td[TI[c]]
        cb = CBALL[x["cb"][0]:x["cb"][1]]
        ok = cb["ready"] == 1
        dur = (cb["end"] - cb["start"])
        for c in CLASSES:
            sel = ok & (cb["tag"] == TI[c])
            T[c][1] += float(np.sum(dur[sel]))
            T[c][2] += float(np.sum(np.maximum(dur[sel] - 1e-6 * CAL["cb_gpu_us_p10"], 0)))
            T[c][6] += int(sel.sum())
    for acc in ACCK[a]:
        for (L, t), (b, e, d, nf) in acc.items():
            c = CLASSES[t]
            T[c][3] += b; T[c][4] += e; T[c][5] += nf
            kind = KINDS[L] if 0 <= L < len(KINDS) else "-"
            TL.setdefault((kind, c), 0)
            TL[(kind, c)] += d
    T = {c: [v[0] / n, 1e3 * v[1] / n, 1e3 * v[2] / n, 1e3 * v[3] / n, 1e3 * v[4] / n, v[5] / n, v[6] / n] for c, v in T.items()}
    nk = {"G": KINDS.count("G"), "A": KINDS.count("A"), "-": 1}
    TL = {f"{k}:{c}": v / n / nk[k] for (k, c), v in TL.items()}
    return T, TL


def kern_by_class(a, top=8):
    out = {}
    for c, d in PSO_K.get(a, {}).items():
        n = len(ACCK[a])
        agg = {}
        for sl, v in d.items():
            nm = SLOTNAME.get(sl, "?")
            agg[nm] = agg.get(nm, 0) + v / n
        out[c] = sorted(agg.items(), key=lambda z: -z[1])[:top]
    return out


# ---- floors: distinct bytes read per check at 280 GB/s (dense once; the routed union per layer)
from mlx.utils import tree_flatten

dense_b, exp_b = 0, {}
for k_, v_ in tree_flatten(model.parameters()):
    if not isinstance(v_, mx.array):
        continue
    if ".switch_mlp." in k_:
        L = int(k_.split(".layers.")[1].split(".")[0])
        exp_b[L] = exp_b.get(L, 0) + v_.nbytes
    elif "embed_tokens" in k_ or "ngram" in k_ or "ple_embedding" in k_:
        continue
    else:
        dense_b += v_.nbytes
NEXP = int(inner.layers[0].mlp.switch_mlp.num_experts) if hasattr(inner.layers[0].mlp.switch_mlp, "num_experts") else 512
FLOOR = {}
for a in names:
    un = UNK.get(a, [])
    per_L = {L: float(np.mean([u.get(L, 0) for u in un])) for L in exp_b} if un else {}
    eb = sum(per_L[L] * exp_b[L] / NEXP for L in per_L)
    FLOOR[a] = dict(dense_gb=dense_b / 1e9, experts_gb=eb / 1e9, union_mean=float(np.mean(list(per_L.values()))) if per_L else 0,
                    floor_ms=1e3 * (dense_b + eb) / 280e9)

# ---- report
m = lambda v: 1e3 * v
say(f"pass T: {ROUNDS} rounds x {len(names)} arms (contexts {P} -> {max(POS.values())}); medians per check")
say("| arm | wall | fwd / read / commit | CPU build | CPU encode (SB) | CPU read | CPU commit | GPU busy (fwd / read / commit) "
    "| GPU idle | GPU first start | dispatches | barriers | encoders | cmd buffers | floor |")
say("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for a in names:
    s = ST_T[a]
    say(f"| {a} | {m(s['wall']):.2f} | {m(s['fwd']):.2f} / {m(s['read']):.2f} / {m(s['commit']):.2f} | {m(s['cpu_build']):.2f} | "
        f"{m(s['cpu_enc_sb']):.2f} | {m(s['cpu_read']):.2f} | {m(s['cpu_commit']):.2f} | {m(s['gpu']):.2f} ({m(s['gpu_fwd']):.2f} / "
        f"{m(s['gpu_read']):.2f} / {m(s['gpu_commit']):.2f}) | {m(s['idle']):.2f} | {m(s['gpu_first']):.2f} | {s['disp']:.0f} | "
        f"{s['bar']:.0f} | {s['enc']:.0f} | {s['ncb']:.0f} | {FLOOR[a]['floor_ms']:.2f} |")
for a in names:
    T, TL = attr_table(a)
    kd = float(np.mean([x["c"]["disp"] for x in RK[a]]))
    say(f"pass K {a} ({KROUNDS} rounds; per check) | class | dispatches | GPU ms | GPU ms net | build ms | encode ms (flushes) | flushes | cmd buffers |")
    tot = [0.0] * 7
    for c in CLASSES:
        v = T[c]
        if v[0] or v[1] or v[3] > 0.01:
            say(f"|   {c} | {v[0]:.1f} | {v[1]:.2f} | {v[2]:.2f} | {v[3]:.2f} | {v[4]:.2f} | {v[5]:.1f} | {v[6]:.1f} |")
            for j in range(7):
                tot[j] += v[j]
    say(f"|   total | {tot[0]:.1f} | {tot[1]:.2f} | {tot[2]:.2f} | {tot[3]:.2f} | {tot[4]:.2f} | {tot[5]:.1f} | {tot[6]:.1f} | "
        f"(pass K counter {kd:.1f} dispatches per check; pass T {ST_T[a]['disp']:.0f} dispatches, GPU busy {m(ST_T[a]['gpu']):.2f} ms)")
    say("    dispatches per layer: " + ", ".join(f"{k} {v:.1f}" for k, v in sorted(TL.items()) if v >= 0.5))
say("floors: " + "; ".join(f"{a} dense {FLOOR[a]['dense_gb']:.2f} GB + experts {FLOOR[a]['experts_gb']:.2f} GB (union "
                           f"{FLOOR[a]['union_mean']:.1f} / layer) = {FLOOR[a]['floor_ms']:.2f} ms" for a in names))
for a in names:
    say(f"top kernels, {a} (pass T, dispatches per check): " + "; ".join(
        f"{k[:56]} {v:.1f}" for k, v in sorted(KERN_ARM[a].items(), key=lambda z: -z[1])[:14]))
say(f"done in {time.time() - t00:.0f} s; peak {mx.get_peak_memory() / 1e9:.1f} GB")
res = dict(tag=TAG, when=time.strftime("%a %d %b %Y %H:%M:%S"), smoke=SMOKE, kinds=KINDS, rounds=ROUNDS, krounds=KROUNDS,
           sbn=SBN, gdn_body=GDNB, k73r=K73R, bits=BITS, T=ST_T, T_raw=RAW_T, K={a: attr_table(a)[0] for a in names},
           K_layers={a: attr_table(a)[1] for a in names}, K_kernels={a: kern_by_class(a) for a in names},
           K_union=UNK, floors=FLOOR, kernels_T={k: v / nchecks for k, v in KERN.items()}, k73n=K73N[0], cal=CAL,
           kernels_T_arm=KERN_ARM, K_raw={a: [dict(c=x["c"], tagdisp=x["tagdisp"]) for x in RK[a]] for a in names},
           v32_cnt=dict(v32.CNT), log=LOG, peak_gb=mx.get_peak_memory() / 1e9)
json.dump(res, open(os.path.join(OUT, TAG + ".json"), "w"), indent=1, default=str)
open(os.path.join(OUT, TAG + ".txt"), "w").write("\n".join(LOG) + "\n")
say(f"wrote {OUT}/{TAG}.json / .txt")
