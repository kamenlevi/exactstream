"""Decision map of the 125B (FableGod mission 9): teacher-forced replays with one part of the computation perturbed,
measuring how often the ARGMAX changes, by domain (code / IFEval / multi) and by the 125B's own confidence class
(sure p1 >= 0.99 / tie p1 - p2 <= 0.1 / mid). Runs inside the BigMoE venv like ~/TrainLab/sticky/d02a_extract.py.

ADAPT (the only names a runner must check, all taken from ref2.py / d02a_extract.py as read on 2026-11-01):
  A1 model loading: bigmoe.loader.load_streamed(...) with the 45 GB preset, as d02a_extract.py does (copy its call).
  A2 forward: FORWARD(model, ids) -> logits [T, V] for a token window, chunked prefill with a fresh cache; d02a_extract's
     patched _text / mtplx.generation pattern. Replace the body of forward_logits() with that call.
  A3 layer discovery: model.layers[i] holds Q.GatedDeltaNet or Q.Attention (mixer), Q.GatedResidual (HC read),
     Q._FusedGateUpSwitchGLU or SL.SwitchGLU (routed experts), and a shared-expert module whose attribute name contains
     "shared". discover_layers() walks modules by class; fix the names if the walk prints "missing".
  A4 windows: d02b_walk_{code,d02a_ifeval,d02a_multi}.npz hold names, {s}_ids, {s}_off, {s}_gen (prompt | greedy answer).
  A5 the routed-experts call signature: __call__(self, x, indices) (ref2.py _switch); the expert weights are per-row
     top-10 ids in `indices` with gate scores applied inside. Rank cuts are applied by masking indices (see dm_ablations).

WIRED (Breakthrough, Mon 2026-11-02 01:xx; Fable's originals kept as *.orig):
  A1 d02a_extract.py's own call and env: load_streamed(~/BigMoE/models/bare, C 51, mtp=False), compiled paths off, PAUSE-
     aware passes (Leap's pausewait). That is D-02a's configuration (4-bit experts, exact prefill, ~20 GB peak), NOT the
     3-bit-tier 45 GB preset: the gate below compares with D-02a's stored top-1, which only D-02a's own configuration can
     reproduce; C only changes speed in an exact prefill. DM_C overrides C.
  A2 d02a_extract.py's _text hook: cold G.restore_or_prefill_prompt_state per window (no session bank, so no prefix
     restore), the lm_head on each 256-row slice of the final hidden, top-32 by argpartition + sort as D-02a, top-8 kept.
     run_window() returns (top-8 ids, log-probs) for rows 0..N-2 (row j predicts token j+1).
  A3 by attribute: layers[i].linear_attn (GDN) / .self_attn (attention), .attn_hyper_connection + .mlp_hyper_connection
     (both HC reads), .mlp (SparseMoeBlock: routed experts + shared expert).
  A4 as written (12 windows per set; all have off 0).
  A5 WRONG in the original: the block's top-10 ids come from argpartition, in NO rank order, and the scores are applied
     OUTSIDE the switch (moe_prefill_combine for >= 32-row chunks, the parent's (y * scores).sum for narrower ones), so
     index masking would keep 2 arbitrary experts of the 10 and break the combine's shapes. MoE perturbations act on the
     SCORES instead (PERT below): the wrapper ranks each row's 10 scores, zeroes ranks >= k ("mass": the kept experts
     keep the model's own weights) or also rescales them to the row's original routed weight ("renorm"), then runs the
     model's own combine. The dropped experts are still computed and weighted 0 (bit-exact zero contribution).
"""
import os, sys, json, time
for _k, _v in (("MTPLX_NGRAM_PREWARM", "off"), ("MTPLX_COMPILED_VERIFY", "0"), ("MTPLX_QWEN4EXP_COMPILE", "0"),
               ("MTPLX_COMPILED_GDN", "0"), ("MTPLX_COMPILE_AR_FORWARD", "0"), ("MTPLX_COMPILED_TARGET_PREFIX", "0")):
    os.environ[_k] = _v                                  # = d02a_extract.py (A1)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(1, os.path.expanduser("~/Leap/scripts"))      # pausewait (read-only import, as d02a_extract.py)
sys.path.insert(2, os.path.expanduser("~/TrainLab/sticky"))
from mtplx.models import qwen4_exp as Q
from mtplx import generation as G
import mtplx.kernels.qwen4_moe_prefill_combine as PC

TD = os.path.expanduser("~/TrainLab/data")
MD = os.path.expanduser("~/BigMoE/models/bare")
OUT = os.path.expanduser(os.environ.get("DM_OUT", "~/FableGod/decision_map/out")); os.makedirs(OUT, exist_ok=True)
SETS = [("code", "d02b_walk_code.npz"), ("ifeval", "d02b_walk_d02a_ifeval.npz"), ("multi", "d02b_walk_d02a_multi.npz")]
TOPK = 8
CHUNK = 512
RT = {}

def load_windows(max_per_set=12):
    """[(domain, name, ids(np.int32), off, gen)] : positions >= gen+1 (the answer) are the measured ones."""
    W = []
    for dom, wf in SETS:
        z = np.load(os.path.join(TD, wf))
        for s, nm in enumerate(z["names"]):
            if s >= max_per_set: break
            W.append((dom, str(nm), z[f"{s}_ids"].astype(np.int32), int(z[f"{s}_off"]), int(z[f"{s}_gen"])))
    return W

def load_model():
    """A1: d02a_extract.py's call (see WIRED). Returns (model, None); the runtime and streamer stay in RT."""
    from bigmoe.loader import load_streamed
    import pausewait                                     # noqa: F401 (BigMoE passes wait while PAUSE exists)
    t0 = time.time()
    rt, st = load_streamed(MD, int(os.environ.get("DM_C", "51")), mtp=False)
    model = rt.model if hasattr(rt, "model") else rt
    RT.update(rt=rt, st=st, head=model.language_model.lm_head)
    print(f"[dm] loaded in {time.time() - t0:.0f}s (C {os.environ.get('DM_C', '51')})", flush=True)
    return model, None

# ---- A2: the capture hook (d02a_extract.py's _text) ------------------------------------------------------------
REC = {"on": False, "tk": [], "lp": []}
_text_call = Q.Qwen4ExpTextModel.__call__
def _text(self, inputs, cache=None, input_embeddings=None):
    out = _text_call(self, inputs, cache, input_embeddings)
    if REC["on"]:
        h = out[0]
        for i in range(0, h.shape[0], 256):
            lg = RT["head"](h[i:i + 256]).astype(mx.float32)
            top = mx.argpartition(-lg, 31, axis=-1)[:, :32]
            v = mx.take_along_axis(lg, top, axis=-1)
            o = mx.argsort(-v, axis=-1)
            top, v = mx.take_along_axis(top, o, axis=-1)[:, :TOPK], mx.take_along_axis(v, o, axis=-1)[:, :TOPK]
            lp = v - mx.logsumexp(lg, axis=-1, keepdims=True)
            mx.async_eval(top, lp)
            REC["tk"].append(top.astype(mx.int32)); REC["lp"].append(lp)
    return out
Q.Qwen4ExpTextModel.__call__ = _text

def forward_logits(model, ids):
    raise NotImplementedError("wired as run_window(): the [T, V] logits are reduced to top-8 inside the hook (A2)")

def run_window(model, ids):
    """Cold teacher-forced prefill of ids; returns (top-8 ids int64 [N-1, 8], log-probs float32 [N-1, 8])."""
    REC["tk"], REC["lp"] = [], []
    PREV.clear(); CUR["pos"] = {}; CUR["kpos"] = {}; CAPR["L"] = {}; CAPR["X"] = {}; CAPR["W"] = {}   # fresh per window
    REC["on"] = True
    try:
        G.restore_or_prefill_prompt_state(RT["rt"], [int(t) for t in ids], base_hidden_variant=None,
                                          mtp_history_policy="cycle", capture_hidden=False)
    finally:
        REC["on"] = False
    RT["st"].gate.check()
    tk, lp = np.array(mx.concatenate(REC["tk"])).astype(np.int64), np.array(mx.concatenate(REC["lp"])).astype(np.float32)
    N = len(ids)
    assert tk.shape[0] in (N - 1, N), f"captured rows {tk.shape[0]}, want {N - 1}"
    mx.clear_cache()
    return tk[:N - 1], lp[:N - 1]

def topk_stats(logits, k=TOPK):
    lp = mx.log(mx.softmax(logits.astype(mx.float32), axis=-1))
    o = mx.argsort(-lp, axis=-1)[:, :k]
    v = mx.take_along_axis(lp, o, axis=-1)
    return np.array(o).astype(np.int64), np.array(v).astype(np.float32)

def classes(lp):
    p1, p2 = np.exp(lp[:, 0]), np.exp(lp[:, 1])
    c = np.where(p1 >= 0.99, 0, np.where(p1 - p2 <= 0.1, 1, 2))   # 0 sure, 1 tie, 2 mid
    return c

# ---- layer discovery and perturbation registry -----------------------------------------------------------------
LAYERS = {}   # id(module) -> (layer_index, kind)
ACTIVE = {}   # kind -> callable(layer_index, self, args, y_out) -> y_out' ; set by dm_ablations (att / gdn / hc)
PREV = {}     # per (kind, layer) previous-position output for "reuse previous position" perturbations
PERT = {}     # MoE block perturbation (A5): topk k, renorm, shared_zero, reuse (None / "all" / set of layers), q (bits)

def discover_layers(model):
    """A3: by attribute (see WIRED)."""
    LAYERS.clear(); missing = []
    layers = model.language_model.model.layers
    for i, layer in enumerate(layers):
        mix = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
        if isinstance(mix, Q.GatedDeltaNet): LAYERS[id(mix)] = (i, "gdn")
        elif isinstance(mix, Q.Attention): LAYERS[id(mix)] = (i, "att")
        else: missing.append((i, "mixer"))
        for nm in ("attn_hyper_connection", "mlp_hyper_connection"):
            m = getattr(layer, nm, None)
            if isinstance(m, Q.GatedResidual): LAYERS[id(m)] = (i, "hc")
            else: missing.append((i, nm))
        if isinstance(getattr(layer, "mlp", None), Q.SparseMoeBlock): LAYERS[id(layer.mlp)] = (i, "moe")
        else: missing.append((i, "mlp"))
    kinds = {}
    for i, k in LAYERS.values(): kinds[k] = kinds.get(k, 0) + 1
    print(f"[dm] discover_layers: {len(layers)} layers, {kinds}; missing {missing[:8]}", flush=True)
    assert not missing, missing
    blk = layers[0].mlp
    print(f"[dm] MoE block: top_k {blk.top_k}, norm_topk_prob {blk.norm_topk_prob!r}", flush=True)
    return LAYERS

def _moe_on():
    return bool(PERT)

def _rank_mask(scores, k):
    order = mx.argsort(-scores.astype(mx.float32), axis=-1)
    return mx.argsort(order, axis=-1) < k                 # rank of each of the row's 10 scores < k

def _row_k(n):
    """DAY9 Q1 / Q2 (Breakthrough, Mon 2026-11-02): this call's n rows' expert counts from CUR['k'] (the window's per-
    row k, 10 = exact), in window order: CUR['kpos'] counts each layer's rows since the window started."""
    li = CUR["L"]; p0 = CUR["kpos"].get(li, 0); CUR["kpos"][li] = p0 + n
    kk = np.full(n, 10, np.int32); kr = CUR["k"]
    if PERT.get("k_layers") is not None and li not in PERT["k_layers"]: return kk   # the gate acts on these layers only
    m = max(0, min(n, len(kr) - p0)); kk[:m] = kr[p0:p0 + m]
    return kk

def _row_rm(n):
    """Q1-UNION (11-02): this call's n rows' keep-by-rank flags [n, 10] from CUR['rm'] (the window's [rows, 48, 10]
    mask, rank-sorted), in window order on CUR['kpos'] like _row_k; rows past the mask keep all."""
    li = CUR["L"]; p0 = CUR["kpos"].get(li, 0); CUR["kpos"][li] = p0 + n
    kp = np.ones((n, 10), bool); rm = CUR["rm"]
    m = max(0, min(n, rm.shape[0] - p0)); kp[:m] = rm[p0:p0 + m, li]
    if PERT.get("t3keep") and CUR.get("k") is not None:   # rows served from records3 (dm_t3) post nothing on the stock path
        kr = CUR["k"]; mm = max(0, min(n, len(kr) - p0)); kp[:mm][kr[p0:p0 + mm] == 0] = False
    return kp

def _pert_scores(scores):
    k = PERT.get("topk")
    if CUR.get("rm") is not None:                         # Q1-UNION: keep the ranks the window's mask keeps (mass)
        kp = _row_rm(scores.shape[-2])
        if kp.all(): return scores
        order = mx.argsort(-scores.astype(mx.float32), axis=-1); rank = mx.argsort(order, axis=-1)
        kpa = mx.array(kp).reshape(rank.shape)            # [n, 10] -> the scores' own shape ([1, n, 10] on some paths)
        return scores * mx.take_along_axis(kpa, rank, axis=-1).astype(scores.dtype)
    if CUR.get("k") is not None:                          # per-row k (the drafter gate); no topk on top of it
        assert not k, "topk and a per-row gate together are not wired"
        kk = _row_k(scores.shape[-2])
        if (kk >= scores.shape[-1]).all(): return scores
        order = mx.argsort(-scores.astype(mx.float32), axis=-1)
        s = scores * (mx.argsort(order, axis=-1) < mx.array(kk)[:, None]).astype(scores.dtype)
    elif not k: return scores
    else: s = scores * _rank_mask(scores, k).astype(scores.dtype)
    if PERT.get("renorm"):                                # keep the row's routed weight: x sum(all 10) / sum(kept)
        s = (s.astype(mx.float32) * (scores.astype(mx.float32).sum(axis=-1, keepdims=True)
                                     / s.astype(mx.float32).sum(axis=-1, keepdims=True))).astype(scores.dtype)
    return s

def _reuse_here(li):
    r = PERT.get("reuse")
    return r is not None and (r == "all" or li in r)

def _shift(z, axis):
    """Each position takes the previous position's value (position 0 of the call keeps its own)."""
    n = z.shape[axis]
    if n < 2: return z
    a = mx.take(z, mx.array([0] + list(range(n - 1))), axis=axis)
    return a

def _fq(z, bits):
    s = mx.max(mx.abs(z), axis=-1, keepdims=True) / (2 ** (bits - 1) - 1)
    return mx.round(z / mx.maximum(s, 1e-12)) * s

CUR = {"L": None, "pos": {}, "skip": None, "k": None, "kpos": {}}

def _reuse(li, z):
    """Night 2 (Breakthrough, Mon 2026-11-02 07:xx): MoE routed output z [rows, D] of layer li for this call's rows, in
    window order (CUR['pos'] counts each layer's rows since the window started, so chunks line up).
    Lag k (PERT 'lag', default 1): every row t takes the COMPUTED output of row t - k of the window, carried across
    chunk calls in PREV (rows t < k keep their own). Night 1's _shift restarted at each call; this does not.
    Chain (PERT 'chain'): rows flagged in CUR['skip'] (the measured rows whose BASELINE class is sure, the 125B's own
    class as an oracle) take the output of the last unflagged row, which computes as usual; the chain runs back across
    calls, so a run of sure rows all reuse one computed row (what an engine that skips sure rows would see)."""
    assert z.ndim == 2, z.shape
    n = z.shape[0]
    p0 = CUR["pos"].get(li, 0); CUR["pos"][li] = p0 + n
    if PERT.get("chain"):
        skip, s = (CUR["skip_all"] if li in PERT.get("chain_all", ()) else CUR["skip"]), np.zeros(n, bool)   # reuse map
        if skip is not None:
            m = max(0, min(n, len(skip) - p0)); s[:m] = skip[p0:p0 + m]
        carry = PREV.get(("chain", li))                     # [1, D]: the last computed row of earlier calls
        if not s.all(): PREV[("chain", li)] = z[int(np.nonzero(~s)[0][-1])][None]
        if not s.any(): return z
        idx = np.maximum.accumulate(np.where(~s, np.arange(n), -1))
        if (idx < 0).any():
            assert carry is not None, f"chain: layer {li} row {p0} skips with no computed row before it"
            idx[idx < 0] = n; z_ext = mx.concatenate([z, carry.astype(z.dtype)], axis=0)
        else: z_ext = z
        return mx.take(z_ext, mx.array(idx.astype(np.int32)), axis=0)
    k = int(PERT.get("lag", 1))
    prev = PREV.get(("lag", li))                            # the last <= k computed rows of this window's earlier calls
    m = 0 if prev is None else prev.shape[0]
    z_ext = z if prev is None else mx.concatenate([prev.astype(z.dtype), z], axis=0)
    if k: PREV[("lag", li)] = z_ext[-k:]
    idx = np.arange(n) + m - k
    idx[idx < 0] = (np.arange(n) + m)[idx < 0]             # window rows t < k keep their own output
    return mx.take(z_ext, mx.array(idx.astype(np.int32)), axis=0)
_combine = PC.moe_prefill_combine
def _combine_p(y_sorted, inv_order, scores, shared):
    """>= 32-row chunks (SparseMoeBlock._prefill_call imports this name at each call)."""
    if not _moe_on(): return _combine(y_sorted, inv_order, scores, shared)
    li = CUR["L"]
    scores = _pert_scores(scores)
    sh = shared * 0 if PERT.get("shared_zero") else shared
    if _reuse_here(li) or PERT.get("q"):
        routed = _combine(y_sorted, inv_order, scores, shared * 0)     # [rows, D]
        if _reuse_here(li): routed = _reuse(li, routed)
        if PERT.get("q"): routed = _fq(routed, PERT["q"])
        return (routed + sh).astype(routed.dtype)
    return _combine(y_sorted, inv_order, scores, sh)

_ORIG = {}
def install():
    """att / gdn / hc: wrap the class __call__ once (ACTIVE[kind] for the layer index of `self`). MoE: wrap
    SparseMoeBlock.__call__ (after BigMoE's own patches) and the prefill combine (PERT)."""
    def wrap(cls, kind):
        if (cls, kind) in _ORIG: return
        orig = cls.__call__; _ORIG[(cls, kind)] = orig
        def call(self, *a, **kw):
            li = LAYERS.get(id(self), (None, None))[0]
            pre = ACTIVE.get(kind + "_pre")
            if pre is not None and li is not None: a = pre(li, self, a)
            y = orig(self, *a, **kw)
            post = ACTIVE.get(kind)
            return post(li, self, a, y) if (post is not None and li is not None) else y
        cls.__call__ = call
    wrap(Q.GatedDeltaNet, "gdn"); wrap(Q.Attention, "att"); wrap(Q.GatedResidual, "hc")
    if "moe" in _ORIG: return
    orig = Q.SparseMoeBlock.__call__; _ORIG["moe"] = orig
    def moe_call(self, x):
        li = LAYERS.get(id(self), (None, None))[0]
        CUR["L"] = li
        add = _t3_add(self, li, x) if (PERT.get("t3") and CUR["k"] is not None and li is not None) else None
        y = moe_inner(self, x, li)
        return y if add is None else y + add.reshape(y.shape).astype(y.dtype)
    def moe_inner(self, x, li):
        if CAPR["on"] and li is not None:                 # the block's own router, ranked (argpartition has no order)
            g = self.gate(x); g = g.reshape(-1, g.shape[-1])
            CAPR["L"].setdefault(li, []).append(mx.argsort(-g.astype(mx.float32), axis=-1)[:, :10].astype(mx.int16))
            if CAPR.get("x"):                             # sketch-on-miss (11-02): the routed experts' input + router weights
                gg = mx.softmax(g.astype(mx.float32), axis=-1, precise=True); o = mx.argsort(-gg, axis=-1)[:, :10]
                wv = mx.take_along_axis(gg, o, axis=-1)
                if self.norm_topk_prob: wv = wv / wv.sum(axis=-1, keepdims=True)
                CAPR["X"].setdefault(li, []).append(x.reshape(-1, x.shape[-1]).astype(mx.float16))
                CAPR["W"].setdefault(li, []).append(wv.astype(mx.float16))
        if not _moe_on() or li is None or Q._moe_prefill_combine_applies(self, x):
            return orig(self, x)                          # stock path; >= 32 rows meet _combine_p inside _prefill_call
        # < 32-row chunks: the parent's own ops (mlx_lm Qwen3NextSparseMoeBlock) with the perturbed scores
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        y = self.switch_mlp(x, inds)
        y = (y * _pert_scores(scores)[..., None]).sum(axis=-2)
        if _reuse_here(li):
            assert y.ndim == 2 or y.shape[0] == 1, y.shape     # one sequence: rows in window order
            y = _reuse(li, y.reshape(-1, y.shape[-1])).reshape(y.shape)
        if PERT.get("q"): y = _fq(y, PERT["q"])
        shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        if PERT.get("shared_zero"): shared = shared * 0
        return y + shared
    Q.SparseMoeBlock.__call__ = moe_call
    PC.moe_prefill_combine = _combine_p

def _t3_add(block, li, x):
    """M-18 arm (dm_t3.py): this call's gated rows (k == 0 in CUR['k']) -> [rows, D] with their top-2 routed output from
    records3 at PERT['t3'] bits, zeros elsewhere (peeks the window's row counter; the stock path's _pert_scores advances it)."""
    import dm_t3
    rows = x.reshape(-1, x.shape[-1]); n = rows.shape[0]
    p0 = CUR["kpos"].get(li, 0); kr = CUR["k"]
    kk = np.full(n, 10, np.int32); m = max(0, min(n, len(kr) - p0)); kk[:m] = kr[p0:p0 + m]
    G = np.flatnonzero(kk == 0)
    if not G.size: return None
    keep = None
    if PERT.get("t3keep"):                                # the gated rows' own rank mask at this layer (rank-sorted)
        rm = CUR["rm"]; keep = np.ones((len(G), 10), bool); gi = p0 + G; ok_ = gi < rm.shape[0]
        keep[ok_] = rm[gi[ok_], li]
    yg = dm_t3.routed_topk(block, li, rows[mx.array(G.astype(np.int32))], bits=PERT["t3"], K=int(PERT.get("t3k", 2)), keep=keep)
    pos = np.full(n, len(G), np.int32); pos[G] = np.arange(len(G))       # ungated rows take an appended zero row
    return mx.take(mx.concatenate([yg, mx.zeros((1, yg.shape[-1]), yg.dtype)]), mx.array(pos), axis=0)

CAPR = {"on": False, "L": {}, "x": False, "X": {}, "W": {}}  # DAY9 bytes side ('capture' run): per layer, rank-sorted top-10 expert ids per call
TEARDOWN = []   # undo callables registered by a setup that changes the model itself (hc8w); run by clear()
def clear():
    ACTIVE.clear(); PREV.clear(); PERT.clear(); CUR["k"] = None; CUR["rm"] = None; CUR["skip"] = None; CAPR["on"] = False; CAPR["x"] = False
    while TEARDOWN: TEARDOWN.pop()()

_BASE = {}
_GATE = {}
def _gate(nm, n):
    """The realizable gate (dm_gate.py: DFlash2's cf for the draft each row verifies; NaN = none) over rows 0 .. n-1."""
    if not _GATE:
        import dm_gate; _GATE.update(dm_gate.load())
    g = _GATE[nm]; assert len(g) == n - 1, (nm, len(g), n)
    return np.r_[np.nan_to_num(g, nan=-1.0), -1.0]

_CERT = {}
def _cert(nm, n, src):
    """The two-pass check's certificate (Fable mission 20, cloud Fable's design) over rows 0 .. n-1: each measured row's
    own p1 under the cut, read from the all-rows run src (topk2 / topk5 = pass 1 at top-k1 on every row); -1 elsewhere."""
    if src not in _CERT: _CERT[src] = json.load(open(os.path.join(OUT, src + ".json")))
    r = _CERT[src][nm]; p = np.exp(np.array(r["lp"])[:, 0]); a = r["a"]
    assert a + len(p) == n - 1, (nm, src, a, len(p), n)
    return np.r_[np.full(a, -1.0), p, -1.0]

def _acert(nm, n, src, lo, hi, s):
    """The adaptive certificate (cloud Fable's 12 GB v3; dm_acert.py) over rows 0 .. n-1: p1 under the cut >= hi, or >= lo
    where the drafter is sure (cf >= s) and its draft equals pass 1's argmax (pass 1 = the all-rows run src)."""
    import dm_acert
    if src not in _CERT: _CERT[src] = json.load(open(os.path.join(OUT, src + ".json")))
    r = _CERT[src][nm]; p = np.exp(np.array(r["lp"])[:, 0]); a1 = np.array(r["tk"])[:, 0]; a = r["a"]
    assert a + len(p) == n - 1, (nm, src, a, len(p), n)
    return np.r_[np.zeros(a, bool), dm_acert.acert(nm, a, p, a1, lo, hi, s), False]

def _gate_k(nm, n):
    """Per-row k from PERT['gate']: ("top", thr, k) = top-k on rows whose drafter cf >= thr, the rest exact (Q1);
    ("bands",) = NEW-1: cf >= 0.999 top-2, 0.99-0.999 top-4, 0.9-0.99 top-8, else exact (Q2; all experts are 4-bit
    here, so band 3's "ranks 0-3 at 4 bits" is plain top-8); ("cert", src, thr, k1) = the two-pass check: top-k1 on
    rows whose own p1 under the cut (run src) is >= thr, the rest exact (pass 2); ("acert", src, lo, hi, s, k1) = the same
    with the adaptive certificate (_acert)."""
    spec, k = PERT["gate"], np.full(n, 10, np.int32)
    if spec[0] == "cert": k[_cert(nm, n, spec[1]) >= spec[2]] = spec[3]; return k
    if spec[0] == "acert": k[_acert(nm, n, *spec[1:5])] = spec[5]; return k
    if spec[0] == "t3cert": k[_cert(nm, n, spec[1]) >= spec[2]] = 0; return k   # 3-bit pass 1: certified rows via dm_t3
    if spec[0] in ("t3mask", "t3keep"): return k        # set from the rank mask in measure()
    g = _gate(nm, n)
    if spec[0] == "top": k[g >= spec[1]] = spec[2]
    elif spec[0] == "t3": k[g >= spec[1]] = 0           # dm_t3 adds their top-2 (records3, PERT['t3'] bits)
    elif spec[0] == "t3all": k[:] = 0                   # t3_all / t4_all: every row, all ranks from records3
    elif spec[0] == "bands": k[g >= 0.9] = 8; k[g >= 0.99] = 4; k[g >= 0.999] = 2
    else: raise ValueError(spec)
    return k

def _chain_skip(nm, a, n):
    """Chain mode's skip mask over the window's n rows: measured rows (j >= a) whose BASELINE class is sure, or with
    PERT['chain_gate'] = thr, whose drafter cf >= thr (the realizable gate)."""
    if not _BASE: _BASE.update(json.load(open(os.path.join(OUT, "baseline.json"))))
    b = _BASE[nm]; assert b["a"] == a, (nm, b["a"], a)
    c = classes(np.array(b["lp"]))
    skip = np.zeros(n, bool); skip[a:a + len(c)] = c == 0
    if PERT.get("chain_gate"): skip[a:a + len(c)] = _gate(nm, n)[a:a + len(c)] >= PERT["chain_gate"]
    return bound_runs(skip, int(PERT.get("chain_max") or 0))

def bound_runs(skip, S):
    """Stage 2 (bounded chain, S >= 1): inside each run of skipping rows, every (S + 1)-th row computes instead (a
    refresh), so no row reuses an output more than S rows old. S = 0: unbounded (stage 1's chain)."""
    if not S: return skip
    r = np.zeros(len(skip), int)                          # 1-based position inside the current run
    for j in range(len(skip)): r[j] = (r[j - 1] + 1 if j else 1) if skip[j] else 0
    return skip & (r % (S + 1) != 0)

def measure(model, windows, name, setup):
    """setup(): registers ACTIVE / PERT perturbations (or nothing for the baseline). Writes out/<name>.json with per-
    position top-8 ids and log-probs for the answer positions."""
    clear(); setup()
    res = {}
    t0 = time.time()
    for dom, nm, ids, off, gen in windows:
        a = gen + 1 - off                                   # first answer position inside ids
        CUR["skip"] = _chain_skip(nm, a, len(ids)) if PERT.get("chain") else None
        if PERT.get("chain_all"):                           # the reuse map (11-02): these layers chain on EVERY answer row
            sa = np.zeros(len(ids), bool); sa[a:len(ids) - 1] = True; CUR["skip_all"] = bound_runs(sa, int(PERT.get("chain_max") or 0))
        CUR["k"] = _gate_k(nm, len(ids)) if PERT.get("gate") else None
        CUR["rm"] = np.load(os.path.join(OUT, PERT["rmask"], nm))["keep"] if PERT.get("rmask") else None
        if PERT.get("gate") == ("t3keep",):                 # P1-K1 at 3 bits (Tue 15:0x): the rows the mask cuts at all
            kk = np.full(len(ids), 10, np.int32); r = CUR["rm"]; m = min(len(kk), r.shape[0])
            kk[:m][~r[:m].all(axis=(1, 2))] = 0; CUR["k"] = kk
        if PERT.get("gate") == ("t3mask",):                 # the t3 rows = the rows whose mask drops rank 0 (Tue 11:0x): a
            kk = np.full(len(ids), 10, np.int32); r = CUR["rm"]; m = min(len(kk), r.shape[0])   # live cut's exact bonus
            kk[:m][~r[:m, :, 0].any(axis=1)] = 0; CUR["k"] = kk                                 # rows are not served twice
        tk, lp = run_window(model, ids)
        res[nm] = dict(dom=dom, a=int(a), off=int(off), tk=tk[a:].tolist(), lp=lp[a:].tolist())
        extra = ""
        if PERT.get("reuse") is not None:                   # every reused layer saw each row exactly once, in order
            seen = {CUR["pos"].get(L) for L in (range(48) if PERT["reuse"] == "all" else PERT["reuse"])}
            if not (len(seen) == 1 and next(iter(seen)) in (len(ids) - 1, len(ids))):
                extra = f"; WARNING reuse row counts {sorted(seen, key=str)} for {len(ids)} ids (rows misaligned)"
                res[nm]["warn"] = extra
        if CAPR["on"]:                                      # out/routing/<window>.npz: ids [n - 1, 48, 10], rank-sorted
            R = [np.array(mx.concatenate(CAPR["L"][L])) for L in range(48)]
            assert len({r.shape[0] for r in R}) == 1 and R[0].shape[0] in (len(ids) - 1, len(ids)), [r.shape for r in R]
            os.makedirs(os.path.join(OUT, "routing"), exist_ok=True)
            np.savez(os.path.join(OUT, "routing", nm), ids=np.stack(R, axis=1)[:len(ids) - 1], tokens=ids, gen=gen)
            extra += f"; routing {R[0].shape[0]} rows"
            if CAPR.get("x"):                               # out/xcap/<window>.npz: answer rows' MoE input x [m, 48, 2560] and
                X = np.stack([np.array(mx.concatenate(CAPR["X"][L]))[a:len(ids) - 1] for L in range(48)], axis=1)   # weights
                W = np.stack([np.array(mx.concatenate(CAPR["W"][L]))[a:len(ids) - 1] for L in range(48)], axis=1)
                os.makedirs(os.path.join(OUT, "xcap"), exist_ok=True)
                np.savez(os.path.join(OUT, "xcap", nm), x=X, w=W, ids=np.stack(R, axis=1)[a:len(ids) - 1], a=a)
                extra += f"; x {X.shape}"
        if CUR["k"] is not None or CUR.get("rm") is not None:   # every MoE layer saw each row exactly once, in order
            seen = {CUR["kpos"].get(L) for L in range(48)}
            if not (len(seen) == 1 and next(iter(seen)) in (len(ids) - 1, len(ids))):
                extra += f"; WARNING gate row counts {sorted(seen, key=str)} for {len(ids)} ids (rows misaligned)"
                res[nm]["warn"] = extra
            if CUR["k"] is not None:
                kk = CUR["k"][a:len(ids) - 1]
                extra += "; k " + " ".join(f"{v}:{int((kk == v).sum())}" for v in (2, 4, 8, 10) if (kk == v).any())
            else:
                rk = CUR["rm"][a:len(ids) - 1]; extra += f"; kept experts per row-layer {rk.sum() / max(rk.shape[0] * 48, 1):.2f}"
        if CUR["skip"] is not None:
            s = CUR["skip"][a:]; runs = np.diff(np.flatnonzero(np.diff(np.r_[0, s.astype(np.int8), 0])))[::2]
            extra += f"; chain skips {int(s.sum())}/{len(ids) - 1 - a} answer rows, longest run {int(runs.max()) if len(runs) else 0}"
        print(f"{name} {dom} {nm} {len(ids) - 1 - a} positions {time.time() - t0:.0f}s{extra}", flush=True)
    json.dump(res, open(os.path.join(OUT, name + ".json"), "w"))
    clear()
    return res

def gate(base_name="baseline", need=0.999):
    """Baseline top-1 vs D-02a's stored top-1 (ifeval / multi: ~/TrainLab/data/d02a/<name>; code: ~/Leap/data/l118_tgt,
    D-02a's code reference) on every measured position. Writes out/gate.txt; returns (agreement, ok)."""
    base = json.load(open(os.path.join(OUT, base_name + ".json")))
    win = {nm: (ids, off) for dom, nm, ids, off, gen in load_windows(10 ** 6)}
    eq = n = 0; per = {}
    for nm, b in base.items():
        f = os.path.join(os.path.expanduser("~/Leap/data/l118_tgt") if b["dom"] == "code" else os.path.join(TD, "d02a"), nm)
        z = np.load(f)
        ref = z["tk"][:, 0] if "tk" in z else z["top1"]
        roff = int(z["off"]) if "off" in z else 0
        ids, off = win[nm]                                    # the same text as the reference (absolute positions)
        assert off == b["off"] and np.array_equal(np.asarray(z["ids"])[off - roff:off - roff + len(ids)], ids), nm
        mine = np.array(b["tk"])[:, 0]
        pos = b["off"] + b["a"] + np.arange(len(mine))       # absolute row index (row j predicts token j + 1)
        r = ref[pos - roff]
        e = int((r == mine).sum()); eq += e; n += len(mine)
        d = per.setdefault(b["dom"], [0, 0]); d[0] += e; d[1] += len(mine)
    ag = eq / max(n, 1)
    txt = (f"GATE baseline vs D-02a stored top-1: {eq}/{n} = {100 * ag:.3f} % (need >= {100 * need:.1f} %): "
           + ("PASS" if ag >= need else "FAIL") + "; " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in per.items()))
    open(os.path.join(OUT, "gate.txt"), "w").write(txt + "\n"); print(txt, flush=True)
    return ag, ag >= need

def report(base_name="baseline", names=None):
    base = json.load(open(os.path.join(OUT, base_name + ".json")))
    names = names or [f[:-5] for f in sorted(os.listdir(OUT)) if f.endswith(".json") and f[:-5] != base_name]
    rows = ["| perturbation | code sure / tie / mid flip % (n) | ifeval sure / tie / mid | multi sure / tie / mid | mean dmargin |", "|---|---|---|---|---|"]
    for nm in names:
        r = json.load(open(os.path.join(OUT, nm + ".json"))); cells = []; dm = []
        if not isinstance(r, dict) or not all(isinstance(v, dict) and "tk" in v for v in list(r.values())[:1]): continue   # not a run (gate_dfl2.json)
        for dom in ("code", "ifeval", "multi"):
            flips = np.zeros(3); n = np.zeros(3)
            for w, b in base.items():
                if b["dom"] != dom or w not in r: continue
                btk, blp = np.array(b["tk"]), np.array(b["lp"]); rtk, rlp = np.array(r[w]["tk"]), np.array(r[w]["lp"])
                m = min(len(btk), len(rtk)); btk, blp, rtk, rlp = btk[:m], blp[:m], rtk[:m], rlp[:m]
                c = classes(blp)
                f = rtk[:, 0] != btk[:, 0]
                dm.append(((rlp[:, 0] - rlp[:, 1]) - (blp[:, 0] - blp[:, 1])).mean())
                for k in range(3): flips[k] += f[c == k].sum(); n[k] += (c == k).sum()
            cells.append(" / ".join(f"{100 * flips[k] / max(n[k], 1):.2f}" for k in range(3)) + f" ({int(n.sum())})")
        rows.append(f"| {nm} | " + " | ".join(cells) + f" | {np.mean(dm) if dm else 0:+.3f} |")
    txt = "\n".join(rows); open(os.path.join(OUT, "report.md"), "w").write(txt + "\n"); print(txt)
