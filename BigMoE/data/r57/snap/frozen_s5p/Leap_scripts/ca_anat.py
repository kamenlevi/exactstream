"""CA (NewIdeas Mon 13:46, QUEUE row CA; Leap, pre-registered Mon 2026-10-26 in ~/Leap/TABLE_350.md 'CA'): anatomy of
one exact resident check of the 125B at 8 / 16 / 32 rows, on real weights.
A resident 12-layer slice (layers 0-11 = 9 GDN incl. the one PLE layer + 3 QSA; the model is GGGA x 12), loaded as in
b5_slice.py, runs MTPLX's own forward (read only) under the engine's verify scopes, with the engine's per-check
bookkeeping: a lazy pre-verify snapshot, the verify forward (capture on), the greedy acceptance read (argmax .tolist(),
the engine's sync), and a capture-commit of a partial accept (keep = KEEP[rows] ~ L1-20's tokens per check), whose
lazy replay lands in the next check, as in the engine. Real code tokens after a 1K or 8K-token prefill.
Modes (one process each; ca_batch.sh):
  smoke  one config, A and P untraced: P's tokens must equal A's, every commit must land, labels per check printed
  time   A = eager, as the engine runs 16 rows (build, then one sync per check); B = async_eval every 4 layers (b5's
         B). Wall per check. Plus the DFlash2 drafter's 1-step draft at block 16 / 32 (refine19's path, real features)
  prof   run under xctrace (Metal System Trace) with MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER huge, so a command
         buffer ends only where this script evaluates. B2 = B (end-to-end GPU busy); P = one command buffer per part,
         labelled in order: each quantized matmul (dense, labelled by weight shape), the MoE's routing sort / expert
         kernels / unsort, HC reads and writes, PLE, the head, the commit replay, the acceptance read, and the ops
         between hooked calls inside the attention / GDN / MoE modules (attention core incl. the indexer, GDN core,
         routing, combine). ca_report.py aligns the trace's command buffers with data/ca_labels.json.
  diff   (run 3, no trace: xctrace's temp files stopped runs 1 and 2) knockout differences in one process. Checks cycle
         through variants in rounds after one prefill: A (engine), M (= time's B), D (B through layer_diff's call
         sites; D-dense_gdn ... D-all) and L0 (no layers: embedding, PLE staging, head, acceptance read). A knocked-out
         call still builds its graph but returns its own output from an earlier check, so its GPU work is pruned from
         the graph while its inputs are kept alive (the rest of the graph runs as in D). A class's GPU time = the
         paired wall difference D - D-class (GPU-bound: wall ~= busy). Cache writes a knockout leaves lazy are flushed
         after the acceptance read; the commit replay is evaluated and timed on its own. diffsmoke = one short config.
  commit (R-12 step 1, Mon 22:38): the commit replay's parts after real 16-row checks at 1K, by keep length 1 / 4 /
         10 / 15: the pure-GDN recurrence replays, the PLE layer's full replay and the rest (QSA trims), each evaluated
         and timed alone; GDN state bytes per layer (what keeping every row's state would write).
Usage: [BigMoE venv] ca_anat.py smoke|time|prof|diff|diffsmoke|commit"""
import contextlib, glob, json, os, sys, time
MODE = sys.argv[1] if len(sys.argv) > 1 else "smoke"
if MODE == "prof":                                      # read once at device init: must precede the mlx import
    os.environ["MLX_MAX_OPS_PER_BUFFER"] = "1000000"
    os.environ["MLX_MAX_MB_PER_BUFFER"] = "1000000"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")    # the 29.8 GB PLE n-gram table stays file-backed, as at 45 GB
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx_lm.utils as mu

MD = os.path.expanduser("~/BigMoE/models/bare")
D = os.path.expanduser("~/Leap/data")
NL = 12
CTXS = (1024,) if MODE in ("smoke", "diffsmoke", "commit") else (1024, 8192)
ROWS = (16,) if MODE in ("smoke", "diffsmoke", "commit") else (8, 16, 32)
KEEP = {8: 7, 16: 10, 32: 11}                           # accepted + 1 per check: block 8 6.9, L1-20 9.54 / 10.5
N_WARM, N_CHK = (1, 3) if MODE == "smoke" else (3, 10)
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

orig = mu.load_model
def lazy_load_model(path, *a, **kw):
    a = list(a)
    if a: a[0] = True
    else: kw["lazy"] = True
    model, cfg = orig(path, *a, **kw)
    if hasattr(model, "language_model"):
        inner = model.language_model.model
        inner.layers = inner.layers[:NL]               # layers 12-47 are never read
        mx.eval(model.parameters())
    return model, cfg
mu.load_model = lazy_load_model
t_load = time.time()
try:
    from mtplx.runtime import load
    rt = load(MD, mtp=False)
finally:
    mu.load_model = orig
import mtplx.models.qwen4_exp as Q
import mtplx.cache_state as CS
import mtplx.moe_sorted_gather as MSG
import mlx_lm.models.switch_layers as SL
from mtplx.attention_context import attention_phase, exact_verify, model_forward_kind

model = rt.model
tm = model.language_model
inner = tm.model
KINDS = "".join("G" if l.is_linear else "A" for l in inner.layers)
say(f"CA {MODE}: {len(inner.layers)} layers resident ({KINDS}), {mx.get_active_memory() / 1e9:.1f} GB active, loaded "
    f"in {time.time() - t_load:.0f} s, pid {os.getpid()}")

files = sorted(glob.glob(os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5p/Leap_scripts/*.py")))
ids = list(rt.tokenizer.encode("".join(open(f).read() for f in files)))
assert len(ids) >= max(CTXS) + 40 * (N_WARM + N_CHK) + 64, len(ids)

@contextlib.contextmanager
def verify_scope():
    with attention_phase("decode_verify"), model_forward_kind("target_verify"), exact_verify(True), \
            model.verify_capture_scope():
        yield

def rope_scope():
    return Q._rope_table_scope() if Q.qwen4_opdiet_enabled("rope") else contextlib.nullcontext()

def leaves(x, out=None):
    out = [] if out is None else out
    if isinstance(x, mx.array):
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for y in x:
            leaves(y, out)
    elif isinstance(x, dict):
        for y in x.values():
            leaves(y, out)
    return out

def states(cache):
    return [c.state for c in cache if hasattr(c, "state")]

med = lambda v: float(np.median(v))

# ---------------------------------------------------------------- P: one command buffer per part
PROF = [False]
LAB, CUR, LAYER, DEPTH = [], ["top"], [-1], [0]

def cut(label, *arrays):
    """end the current part: evaluate everything it produced (async) in one command buffer, plus a 1-element marker
    op so the buffer is never empty, and record the label"""
    if not PROF[0]:
        return
    ls = [a for a in leaves(list(arrays)) if a.size > 0]
    if not ls:
        return
    a = ls[0]
    mk = (a[(0,) * a.ndim] if a.ndim else a) + 1
    mx.async_eval(*ls, mk)
    LAB.append((LAYER[0], CUR[0], label))

@contextlib.contextmanager
def part(name):
    prev = CUR[0]; CUR[0] = name
    try:
        yield
    finally:
        CUR[0] = prev

def hooked(fn, kind, shaped=False):
    def h(*a, **kw):
        if not PROF[0] or DEPTH[0]:
            return fn(*a, **kw)
        cut(f"pre_{kind}", *[x for x in a if isinstance(x, mx.array)][:1])   # the part's ops since the last cut
        DEPTH[0] += 1
        try:
            y = fn(*a, **kw)
        finally:
            DEPTH[0] -= 1
        w = a[1] if len(a) > 1 and isinstance(a[1], mx.array) else kw.get("w")
        cut(f"{kind}{tuple(int(s) for s in w.shape)}" if shaped and w is not None else kind, y)
        return y
    return h

if MODE in ("smoke", "prof"):                            # time mode runs the engine's code untouched
    mx.quantized_matmul = hooked(mx.quantized_matmul, "qmm", True)
    mx.gather_qmm = hooked(mx.gather_qmm, "gqmm", True)
    MSG.sort_rows = hooked(MSG.sort_rows, "route_sort")
    MSG.swiglu_rows = hooked(MSG.swiglu_rows, "exp_gu")
    MSG.switch_linear = hooked(MSG.switch_linear, "exp_down")
    SL._scatter_unsort = hooked(SL._scatter_unsort, "unsort")

# ---------------------------------------------------------------- diff: knockouts
DIFF, DEAD, KO_ON = [False], [0], set()
STORE, ORD, KA = {}, {}, []
QMM_CLS = {"gdn": "dense_gdn", "qsa": "dense_qsa", "moe": "dense_moe"}

def KO(cls, kind, thunk, inputs):
    """run the call; if its class is knocked out, return the same call's output from an earlier check instead (its
    own GPU work becomes dead and is pruned) and keep its inputs alive (upstream work is not pruned). Calls inside a
    knocked-out call are dead too: they neither store nor knock out."""
    if not DIFF[0] or cls is None:
        return thunk()
    k = (LAYER[0], CUR[0], kind)
    n = ORD.get(k, 0); ORD[k] = n + 1
    if DEAD[0]:
        return thunk()
    key = k + (n,)
    if cls in KO_ON and key in STORE:
        DEAD[0] += 1
        try:
            thunk()
        finally:
            DEAD[0] -= 1
        KA.extend(leaves(list(inputs)))
        return STORE[key]
    y = thunk()
    STORE[key] = y
    return y

def ko_fn(fn, kind, cls_of):
    def h(*a, **kw):
        return KO(cls_of(), kind, lambda: fn(*a, **kw), [x for x in list(a) + list(kw.values()) if isinstance(x, mx.array)])
    return h

if MODE in ("diff", "diffsmoke"):
    import mlx_lm.models.gated_delta as GD
    mx.quantized_matmul = ko_fn(mx.quantized_matmul, "qmm", lambda: QMM_CLS.get(CUR[0]) if LAYER[0] >= 0 else None)
    mx.gather_qmm = ko_fn(mx.gather_qmm, "gqmm", lambda: "experts" if CUR[0] == "moe" else None)
    SL._gather_sort = ko_fn(SL._gather_sort, "sort", lambda: "gather" if CUR[0] == "moe" else None)
    SL._scatter_unsort = ko_fn(SL._scatter_unsort, "unsort", lambda: "gather" if CUR[0] == "moe" else None)
    GD.gated_delta_update = ko_fn(GD.gated_delta_update, "rec", lambda: "gdn_rec" if CUR[0] == "gdn" else None)

def layer_diff(layer, hidden, x, ssm_mask, c):
    """DecoderLayer.__call__ (as layer_prof) with the call-site knockouts: PLE, HC read / read-write / write, QSA"""
    kind = "gdn" if layer.is_linear else "qsa"
    if "ple" in layer:
        with part("ple"):
            hidden = hidden + KO("ple", "ple", lambda: layer.ple(hidden, x, c), (hidden,))
    with part("hc_read"):
        mixed, hyper, inject = KO("hc", "read", lambda: layer.attn_hyper_connection(hidden), (hidden,))
    with part(kind):
        if layer.is_linear:
            block_out = layer.linear_attn(mixed, ssm_mask, c)
        else:
            block_out = KO("attn", "attn", lambda: layer.self_attn(mixed, c), (mixed,))
    with part("hc_rw"):
        mixed, hyper, inject = KO("hc", "rw", lambda: layer.mlp_hyper_connection(hyper, pending=(block_out, inject)),
                                  (hyper, block_out, inject))
    with part("moe"):
        block_out = layer.mlp(mixed)
    with part("hc_write"):
        hidden = KO("hc", "write", lambda: Q._hyper_residual_write(hyper, block_out, inject), (hyper, block_out, inject))
    return hidden

def layers_diff(h, x, cache, ssm_mask, every=4):
    """layers_mid with layer_diff, knocked-out calls' inputs evaluated with their group"""
    last = len(inner.layers) - 1
    for i, (layer, c) in enumerate(zip(inner.layers, cache)):
        LAYER[0] = i
        layers_capture(h, x, cache, i, layer, c)
        h = layer_diff(layer, h, x, ssm_mask, c)
        if i < last and (i + 1) % every == 0:
            mx.async_eval(h, *Q._midloop_state_arrays(cache[i + 1 - every:i + 1]), *KA)
            KA.clear()
    return h

def layer_prof(layer, hidden, x, ssm_mask, c):
    """DecoderLayer.__call__, verbatim, with a cut after each part"""
    kind = "gdn" if layer.is_linear else "qsa"
    if "ple" in layer:
        with part("ple"):
            hidden = hidden + layer.ple(hidden, x, c)
            cut("tail", hidden)
    with part("hc_read"):
        mixed, hyper, inject = layer.attn_hyper_connection(hidden)
        cut("tail", mixed, inject)
    with part(kind):
        block_out = layer.linear_attn(mixed, ssm_mask, c) if layer.is_linear else layer.self_attn(mixed, c)
        cut("tail", block_out)
    with part("hc_rw"):
        mixed, hyper, inject = layer.mlp_hyper_connection(hyper, pending=(block_out, inject))
        cut("tail", mixed, hyper, inject)
    with part("moe"):
        block_out = layer.mlp(mixed)
        cut("tail", block_out)
    with part("hc_write"):
        hidden = Q._hyper_residual_write(hyper, block_out, inject)
        cut("tail", hidden)
    return hidden

def body(x, cache, layers_fn):
    """Qwen4ExpTextModel._forward + TextModel's head (as b5_slice.py), with the layer loop swapped"""
    LAYER[0] = -1
    with part("embed"):
        h = inner.embed_tokens(x)
        ssm_mask = Q.create_ssm_mask(h, cache[inner.ssm_idx])
        if inner._ple_stage_idx is not None and Q._COMPILED_VERIFY_PLE.get() is None:
            ple = inner.layers[inner._ple_stage_idx].ple
            ple.ple_embedding.stage(x, cache[inner._ple_stage_idx], ple.NGRAM_IDX)
        h = mx.tile(h, (1, 1, inner.args.hc_count))
        cut("tail", h)
    h = layers_fn(h, x, cache, ssm_mask)
    inner._last_widened = h
    LAYER[0] = -1
    with part("head"):
        def head():
            out = inner.hyper_connection_mixer(h)
            cut("mixer", out)
            lg = inner.embed_tokens.as_linear(out) if tm.args.tie_word_embeddings else tm.lm_head(out)
            cut("tail", lg)
            return lg
        lg = KO("head", "head", head, (h,))
    return lg

def layers_capture(h, x, cache, i, layer, c):
    if Q._VERIFY_CAPTURE.get() and c is not None and getattr(layer, "ple", None) is not None:
        c._mtplx_verify_ple = (h, x)

def layers_mid(h, x, cache, ssm_mask, every=4):
    last = len(inner.layers) - 1
    for i, (layer, c) in enumerate(zip(inner.layers, cache)):
        layers_capture(h, x, cache, i, layer, c)
        h = layer(h, input_ids=x, ssm_mask=ssm_mask, cache=c)
        if i < last and (i + 1) % every == 0:
            mx.async_eval(h, *Q._midloop_state_arrays(cache[i + 1 - every:i + 1]))
    return h

def layers_prof(h, x, cache, ssm_mask):
    for i, (layer, c) in enumerate(zip(inner.layers, cache)):
        LAYER[0] = i
        layers_capture(h, x, cache, i, layer, c)
        h = layer_prof(layer, h, x, ssm_mask, c)
    return h

def forward(variant, x, cache):
    with verify_scope():
        if variant == "A":
            return model(x, cache=cache)
        with rope_scope():                               # as b5_slice.py's body(); the model enters it itself in A
            return body(x, cache, {"P": layers_prof, "D": layers_diff, "L0": lambda h, *a: h}.get(variant, layers_mid))

def prefill(P):
    cache = model.make_cache()
    for a in range(0, P, 1024):
        lg = model(mx.array([ids[a:min(P, a + 1024)]]), cache=cache)
        mx.eval(lg, states(cache))
    return cache

def run(variant, P, R):
    """N_WARM + N_CHK checks of R rows after a P-token prefill; returns per-check wall (ms), tokens, labels, ok"""
    cache = prefill(P)
    mx.synchronize()
    pos, walls, toks, labs, oks, spans, ple = P, [], [], [], [], [], []
    PROF[0] = variant == "P"
    time.sleep(0.3)                                      # a GPU gap marks the block in the trace
    for k in range(N_WARM + N_CHK):
        LAB.clear()
        LAYER[0] = -1
        p0 = Q._PLE_STAGE_SECONDS[0]
        t0 = time.perf_counter()
        with part("commit"):                             # the previous check's lazy commit replay (P: its own buffer)
            cut("replay", *leaves(states(cache)))
        x = mx.array([ids[pos:pos + R]])
        snap = CS.snapshot_untrimmable_cache_lazy(cache)
        lg = forward(variant, x, cache)
        with part("accept"):
            g = mx.argmax(lg[0], axis=-1).tolist()       # the engine's acceptance read: the per-check sync
            if PROF[0]:
                LAB.append((-1, "accept", "argmax"))
        t1 = time.perf_counter()
        prof, PROF[0] = PROF[0], False                   # the commit is lazy (its PLE replay calls hooked matmuls):
        ok = model.commit_verified_window(cache, snap.states, keep_tokens=KEEP[R], verified_tokens=R)
        PROF[0] = prof                                   # it lands in the next check's 'replay' buffer, as in the engine
        pos += KEEP[R]
        walls.append(1e3 * (t1 - t0)); toks.append(g); labs.append(list(LAB)); oks.append(bool(ok))
        spans.append((t0, t1)); ple.append(1e3 * (Q._PLE_STAGE_SECONDS[0] - p0))
    PROF[0] = False
    mx.eval(states(cache))
    mx.synchronize()
    time.sleep(0.3)
    del cache
    PLE[0] = med(ple[N_WARM:])
    return walls, toks, labs, oks, spans

PLE = [0.0]                                              # median PLE n-gram staging (CPU, before the graph) of the last run

DRAFT = {}

def draft_times():
    """DFlash2's 1-step draft (refine19.py's path) at block 16 / 32, on real features (one dfl2d_code sample)"""
    sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5p/Leap_third_party"))
    import dflash_model_mlx as DF
    class Target(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.QuantizedEmbedding(248320, 2560, group_size=64, bits=4)
            self.lm_head = nn.QuantizedLinear(2560, 248320, bias=False, group_size=64, bits=4)
    w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
    tgt = Target()
    tgt.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales",
                      "biases")] + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales",
                      "biases")])
    del w
    draft = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
    draft.bind(tgt)
    draft.load_weights(os.path.join(D, "l120_ft.safetensors"), strict=False)
    mx.eval(draft.parameters(), tgt.parameters())
    MASK = int(draft.config.mask_token_id)
    def select(block, hidden):
        logits = draft.compute_logits(hidden)
        sel = draft.candidate_selector
        cand = mx.argpartition(logits, -sel.top_k, axis=-1)[..., -sel.top_k:]
        unary = mx.take_along_axis(logits, cand, axis=-1)
        hp = sel.hidden_projection(hidden)
        pred, path = block[:, 0], []
        for p in range(hp.shape[1]):
            sc = unary[:, p] + mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None]
                                      * sel.successor_codebook(cand[:, p]), axis=-1)
            pred = mx.take_along_axis(cand[:, p], mx.argmax(sc, axis=-1)[:, None], axis=-1)[:, 0]
            path.append(pred)
        return mx.stack(path, axis=1)
    path = sorted(glob.glob(os.path.join(D, "dfl2d_code", "*.npz")))[0]
    z = np.load(path)
    f, sid, off, gen = z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])
    for bs in (16, 32):
        cache = draft.make_cache()
        start = max(off, gen - (draft.config.sliding_window - 1))
        for c in cache:
            c.offset = start
        fed, a, ts = start, gen, []
        while a < off + len(sid) - 1 and len(ts) < 60:
            ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
            blk = mx.array([[int(sid[a - off])] + [MASK] * (bs - 1)])
            mx.eval(ctx, blk); mx.synchronize()
            t0 = time.perf_counter()
            d = np.array(select(blk, draft.hidden_states(blk, ctx, cache, 1))[0]).tolist()
            ts.append(1e3 * (time.perf_counter() - t0))
            fed = a
            true = sid[a - off + 1:a - off + bs]
            k = 0
            while k < len(true) and d[k] == true[k]:
                k += 1
            if (trim := cache[0].offset - a) > 0:
                DF._trim_recent_cache(cache, trim)
            a = a + k + 1
        say(f"  draft block {bs}: {med(ts[5:]):.2f} ms per 1-step draft (median of {len(ts) - 5}; min {min(ts[5:]):.2f}) "
            f"incl. the selector's per-position loop")
        DRAFT[bs] = med(ts[5:])

CLASSES = ["dense_gdn", "dense_qsa", "dense_moe", "experts", "gather", "hc", "gdn_rec", "attn", "ple", "head"]
VARS = ["A", "M", "D"] + CLASSES + ["all", "L0"]

def run_diff(P, R, rounds, warm=1):
    """rounds of every variant, one check each, after one prefill. Per check: forward wall (t0 -> the acceptance read),
    CPU build (t0 -> the graph built), the flush of lazy cache writes, then the commit, evaluated and timed alone"""
    cache = prefill(P)
    mx.synchronize()
    STORE.clear()
    pos, rec, oks = P, {v: [] for v in VARS}, []
    for r in range(warm + rounds):
        for v in VARS:
            ORD.clear(); KA.clear(); KO_ON.clear()
            KO_ON.update(CLASSES if v == "all" else [v] if v in CLASSES else [])
            x = mx.array([ids[pos:pos + R]])
            mx.eval(x)
            p0 = Q._PLE_STAGE_SECONDS[0]
            t0 = time.perf_counter()
            snap = CS.snapshot_untrimmable_cache_lazy(cache) if v != "L0" else None
            DIFF[0] = v not in ("A", "M")
            lg = forward("A" if v == "A" else "M" if v == "M" else "L0" if v == "L0" else "D", x, cache)
            if v != "A":                                 # A is the engine's eager path: build, then one sync
                mx.async_eval(lg, *KA)
            KA.clear()
            tb = time.perf_counter()
            g = mx.argmax(lg[0], axis=-1).tolist()
            t1 = time.perf_counter()
            DIFF[0] = False
            ok = True
            if v != "L0":                                # L0 leaves the cache alone (no layer ran)
                mx.eval(*leaves(states(cache)))
                t2 = time.perf_counter()
                ok = model.commit_verified_window(cache, snap.states, keep_tokens=KEEP[R], verified_tokens=R)
                mx.eval(*leaves(states(cache)))
                mx.synchronize()
                t3 = time.perf_counter()
                pos += KEEP[R]
            else:
                t2 = t3 = t1
            oks.append(bool(ok))
            if r >= warm:
                rec[v].append(dict(wall=1e3 * (t1 - t0), build=1e3 * (tb - t0), flush=1e3 * (t2 - t1),
                                   commit=1e3 * (t3 - t2), ple=1e3 * (Q._PLE_STAGE_SECONDS[0] - p0)))
    mx.synchronize()
    del cache
    STORE.clear()
    return rec, all(oks)

def run_commit(P, R, keeps=(1, 4, 10, 15), n=5, warm=2):
    """R-12 step 1: per check, the engine's forward (A) and acceptance read, the flush, then the commit built (lazy)
    and its parts evaluated one at a time: pure-GDN replays, the PLE layer's replay, the rest"""
    cache = prefill(P)
    mx.synchronize()
    L = inner.layers
    gdn = [i for i, l in enumerate(L) if l.is_linear and getattr(l, "ple", None) is None]
    ple = [i for i, l in enumerate(L) if getattr(l, "ple", None) is not None]
    rec, pos, oks = {k: [] for k in keeps}, P, []
    for r in range(warm + n):
        for keep in keeps:
            x = mx.array([ids[pos:pos + R]])
            mx.eval(x)
            snap = CS.snapshot_untrimmable_cache_lazy(cache)
            lg = forward("A", x, cache)
            mx.argmax(lg[0], axis=-1).tolist()
            mx.eval(*leaves(states(cache)))
            mx.synchronize()
            t0 = time.perf_counter()
            ok = model.commit_verified_window(cache, snap.states, keep_tokens=keep, verified_tokens=R)
            t1 = time.perf_counter()
            mx.eval(*leaves([cache[i].state for i in gdn])); mx.synchronize(); t2 = time.perf_counter()
            mx.eval(*leaves([cache[i].state for i in ple])); mx.synchronize(); t3 = time.perf_counter()
            mx.eval(*leaves(states(cache))); mx.synchronize(); t4 = time.perf_counter()
            pos += keep
            oks.append(bool(ok))
            if r >= warm:
                rec[keep].append(dict(build=1e3 * (t1 - t0), gdn=1e3 * (t2 - t1), ple=1e3 * (t3 - t2), rest=1e3 * (t4 - t3)))
    sb = [cache[i].state[1].nbytes for i in gdn]
    cb = [cache[i].state[0].nbytes for i in gdn]
    mx.synchronize()
    del cache
    return rec, all(oks), len(gdn), len(ple), float(np.mean(sb)), float(np.mean(cb))

if MODE == "commit":
    assert len(ids) >= 1024 + 7 * 30 + 64
    rec, ok, ng, npl, sb, cb = run_commit(1024, 16)
    say(f"R-12 step 1: commit parts after 16-row checks at 1K, 12-layer slice ({ng} pure-GDN layers, {npl} PLE layer); "
        f"commits landed {ok}; GDN state {sb / 1e6:.2f} MB + conv {cb / 1e6:.3f} MB per layer")
    for k, v in rec.items():
        m = {f: med([c[f] for c in v]) for f in ("build", "gdn", "ple", "rest")}
        est = m["gdn"] * 35 / ng + m["ple"] + 4 * m["rest"]
        say(f"  keep {k:2d}: build {m['build']:.2f}, GDN replays {m['gdn']:.2f} ({m['gdn'] / ng:.3f} per layer), PLE layer "
            f"{m['ple']:.2f}, rest {m['rest']:.2f} ms; 48 layers: ~{est:.2f} ms (+ build)")
    ks = list(rec)
    g1, g15 = (med([c["gdn"] for c in rec[k]]) for k in (ks[0], ks[-1]))
    say(f"  GDN replay time keep {ks[0]} / keep {ks[-1]}: {g1 / g15:.2f} (1 = launch-bound, ~0.07 = row-bound)")
    for rows, name in ((16, "every row"), (4, "every 4th row")):
        say(f"  keeping {name}'s GDN states in the verify writes {rows * sb * 36 / 1e9:.2f} GB per check "
            f"= {rows * sb * 36 / 250e9 * 1e3:.1f} ms at 250 GB/s")
    open(os.path.join(D, "ca_commit.txt"), "w").write("\n".join(lines) + "\n")
    raise SystemExit(0 if ok else 1)

if MODE in ("diff", "diffsmoke"):
    rounds = 1 if MODE == "diffsmoke" else 6
    need = max(CTXS) + (rounds + 1) * len(VARS) * max(KEEP[r] for r in ROWS) + 64
    assert len(ids) >= need, (len(ids), need)
    OUT = []
    for P in CTXS:
        for R in ROWS:
            rec, ok = run_diff(P, R, rounds)
            m = {v: {k: med([c[k] for c in rec[v]]) for k in rec[v][0]} for v in VARS}
            dif = {v: med([b["wall"] - c["wall"] for b, c in zip(rec["D"], rec[v])]) for v in CLASSES + ["all"]}
            OUT.append(dict(ctx=P, rows=R, med=m, diff=dif, ok=ok, raw=rec))
            say(f"  ctx {P:5d}, {R:2d} rows (12 layers; commits landed {ok}): A {m['A']['wall']:.2f}, M {m['M']['wall']:.2f}, "
                f"D {m['D']['wall']:.2f} ms (CPU build {m['D']['build']:.2f}), L0 {m['L0']['wall']:.2f}, all {m['all']['wall']:.2f}; "
                f"commit {m['D']['commit']:.2f}, flush {m['D']['flush']:.2f}, PLE stage {m['D']['ple']:.2f}")
            say("    D - D-class (ms): " + ", ".join(f"{c} {dif[c]:.2f}" for c in CLASSES + ["all"]))
            parts = sum(dif[c] for c in CLASSES if c != "dense_qsa")
            say(f"    disjoint classes sum {parts:.2f} vs D - D-all {dif['all']:.2f} (additivity {parts - dif['all']:+.2f} ms)")
    json.dump(dict(kinds=KINDS, rounds=rounds, out=OUT), open(os.path.join(D, f"ca_{MODE}.json"), "w"))
    open(os.path.join(D, f"ca_{MODE}.txt"), "w").write("\n".join(lines) + "\n")
    if not all(o["ok"] for o in OUT):
        raise SystemExit("a commit did not land")
elif MODE == "smoke":
    wa, ta, _, oa, _ = run("A", CTXS[0], ROWS[0])
    wp, tp, lp, op, _ = run("P", CTXS[0], ROWS[0])
    say(f"  A walls {[round(w, 1) for w in wa]} ms; P walls {[round(w, 1) for w in wp]} ms; PLE staging {PLE[0]:.2f} ms")
    say(f"  tokens P == A: {tp == ta}; commits landed A {all(oa)}, P {all(op)}")
    say(f"  P cuts per check: {[len(l) for l in lp]}; first check's labels:")
    for l in lp[-1]:
        say(f"    {l}")
    if tp != ta or not (all(oa) and all(op)):
        raise SystemExit("smoke FAILED")
elif MODE == "time":
    TREC = []
    for P in CTXS:
        for R in ROWS:
            res = {}
            for v in ("A", "B"):
                w, t, _, ok, _ = run(v, P, R)
                res[v] = (w, t, ok, PLE[0])
            same = res["A"][1] == res["B"][1]
            TREC.append(dict(ctx=P, rows=R, A=med(res["A"][0][N_WARM:]), B=med(res["B"][0][N_WARM:]), same=same,
                             ple_A=res["A"][3], ple_B=res["B"][3]))
            say(f"  ctx {P:5d}, {R:2d} rows: A eager {med(res['A'][0][N_WARM:]):6.2f} ms [min {min(res['A'][0][N_WARM:]):.2f}], "
                f"B async/4 {med(res['B'][0][N_WARM:]):6.2f} ms [min {min(res['B'][0][N_WARM:]):.2f}] per 12-layer check; "
                f"tokens B == A {same}; commits {all(res['A'][2]) and all(res['B'][2])}; PLE staging (CPU, in the build) "
                f"{res['A'][3]:.2f} / {res['B'][3]:.2f} ms")
    draft_times()
    json.dump(dict(checks=TREC, draft=DRAFT), open(os.path.join(D, "ca_time.json"), "w"))
    open(os.path.join(D, "ca_time.txt"), "w").write("\n".join(lines) + "\n")
elif MODE == "prof":
    blocks = []
    for P in CTXS:
        for R in ROWS:
            for v in ("B2", "P"):
                w, t, labs, ok, spans = run("P" if v == "P" else "B", P, R)
                n = [len(l) for l in labs] if v == "P" else None
                blocks.append(dict(variant=v, ctx=P, rows=R, walls=w, labels=labs, ok=ok, spans=spans, n_warm=N_WARM,
                                   ple_ms=PLE[0]))
                say(f"  {v} ctx {P} rows {R}: wall {med(w[N_WARM:]):.2f} ms; cuts per check {n[-1] if n else '-'}; "
                    f"commits {all(ok)}")
    json.dump(dict(pid=os.getpid(), kinds=KINDS, blocks=blocks), open(os.path.join(D, "ca_labels.json"), "w"))
    open(os.path.join(D, "ca_prof.txt"), "w").write("\n".join(lines) + "\n")
