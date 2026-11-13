"""R57_PFLM=1|2 (Sun 8 Nov; Kamen's MUST prefill >= 500 tok/s; NewIdeas' PF-A, Leap's advice, MoonShots'
patches/layer_major_prefill/DESIGN.md): layer-major prefill with the screen's chunk boundaries, plus a per-layer expert scratch
so each layer's experts are read from records3 ONCE per prompt instead of once per 1,024-row chunk.

- Today (45 GB r57, R57_TIER=1): every prefill MoE call goes stream._sync_call -> tier.prefill_call, which builds a scratch of
  exactly the call's unique experts, fills it from records3 with the ext's CPU rebuild (an SSD read of every one of them) and
  frees it. A 1,024-row chunk routes to ~all 512 experts, so each chunk re-reads ~1.5 GB per layer (~75 GB per chunk): a 7.9K
  prompt reads ~600 GB (N45: TTFT 64 s, 124 tok/s).
- Order: `for layer: for chunk:` instead of `for chunk: for layer:`. Each (layer, chunk) step gets exactly today's inputs (its own
  hyper stream, the layer's KV / GDN cache after chunks 0..k-1), the same chunk sizes and the same kernels (MoonShots DESIGN.md:
  bit-identical when the per-call computation is unchanged; Leap: a one-call prefill is NOT bitwise, 0 / 9 hashes).
- Reads: install() wraps tier.prefill_call. While LMC["on"], a layer's first call allocates nothing new beyond one full-E scratch
  (E rows x every 4-bit piece, rows = expert ids, ~1.57 GB: about today's per-call scratch, which is freed meanwhile) and reads
  the call's experts into their rows; later chunks of the same layer read only experts not yet there. The scratch is reused
  across layers (a new layer only resets which rows are valid) and dropped after the prefill (the decode peak is unchanged).
  - R57_PFLM=1: the call runs on the full-E scratch with the ORIGINAL ids (no copy). The kernel sees E = 512 weight rows instead
    of len(unique): tier.prefill_call's docstring warns that this changed MLX's gather kernel choice on SHORT prompts (B7 prose
    sha); layer-major only runs prompts > 1 chunk. Exactness is an empirical question: R57_LOGHASH vs the chunk-major arm.
  - R57_PFLM=2: bitwise by construction: each call gathers its unique rows into a compact copy (mx.take, sorted ids) and runs on
    compact ids, exactly today's call shape. Costs one extra ~1.5 GB copy per call (the 45.0 GB line decides).
- Logits: only the last chunk goes through the mixer + head, its last 256 rows (as R57_PFNOLOG).
- R57_PFLM=3 (PF-B, Sun 08:4x, Leap's plan; mode 1's scratch on original ids): each layer is split at its MoE. Every chunk's
  attention part (PLE, attention hyper read, attention / GDN, MLP hyper read) runs in chunk order and goes to the GPU with
  mx.async_eval; meanwhile the CPU reads ALL the layer's experts into the scratch (R57_PFLM_AHEAD=1, the default: PFX routed
  93-99 % of them, so ~1-7 % more bytes; the ext holds the GIL, so the overlap is CPU read || GPU attention). Then ONE MoE
  call per group of chunks (<= R57_PFLM_TOK tokens, default 4096) on Leap's padded SwitchGLU (MLX 0.32.2's sorted gather
  leaves rows unwritten past 32,767 sorted rows unless M % 64 == 0), then each chunk's residual write. The router and the
  shared expert run once per group (per-token, batch-independent per Leap; R57_LOGHASH vs mode 1 decides). The screen passes
  on_layer (its TAP for the drafter, which layer(...) did in modes 1 / 2). R57_PFLM_CC=1: mx.clear_cache() after each layer.
"""
import os
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import json as _json, time as _time
# Leap 11-11 (prefill 1.5K, stage split): a copy of the engine's pflm.py with timers in PF-B's _split_layers and around
# the whole prefill. Off unless R57_PFLM_PROF=<jsonl path>. R57_PFLM_PROF_SYNC=1: a barrier after every sub-step (GPU time
# per stage; timing only, the tokens are not the point); 0: timestamps only (the real overlapped run: host build vs waits).
_PROF = os.environ.get("R57_PFLM_PROF") or None
_PSYNC = os.environ.get("R57_PFLM_PROF_SYNC") == "1"
# Leap 11-11 14:xx, prefill 1.5K step 1 (exact host work): R57_PFLM_PIPE=1 (with R57_PFLM=3) pipelines PF-B across layers:
# two full-E scratches; the main thread reads layer L+1's experts into the idle one while the GPU runs layer L; the MoE call
# builds lazily (no np.array(indices) sync, no eval: all E experts are already in the active scratch); the only waits are
# for the layer's own read and, before reading L+1 into the buffer layer L-1 used, for layer L-1's outputs. Same ops on
# the same inputs in the same stream order -> bitwise equal by construction (to be checked against R57_PFLM_PIPE=0).
_PIPE = os.environ.get("R57_PFLM_PIPE") == "1"


class _T:
    """per-layer accumulators: wall (perf_counter) and this thread's CPU (thread_time) per key"""
    def __init__(self):
        self.d = {}
    def add(self, k, w, c):
        a = self.d.setdefault(k, [0.0, 0.0]); a[0] += w; a[1] += c


def _clk():
    return _time.perf_counter(), _time.thread_time()


def _ev(*a):
    if _PSYNC:
        mx.eval(*[x for x in a if isinstance(x, mx.array)])

LMC = {"on": False, "mode": 1, "t": None, "have": set(), "sc": None, "reads": 0, "hits": 0, "calls": 0, "tier": None}


class _SegPerChunk(nn.Module):
    """PF-B's exactness fix (Leap 16:5x, 6553fcb; PrefillKernels FINDINGS N3): the MoE block's shared_expert_gate is a q4
    matmul with N = 1, which MLX 0.32.2 runs through qmm_splitk with split_k = 512 / ceil(M / 32): its bits depend on M. Over a
    group of chunks it runs once per chunk slice (PF-A's M), concatenated on the token axis; everything else stays batched."""

    def __init__(self, f, sizes):
        super().__init__()
        self.f = f
        self.sizes = list(sizes)

    def __call__(self, x):
        outs, o = [], 0
        for n_ in self.sizes:
            outs.append(self.f(x[:, o:o + n_]))
            o += n_
        return mx.concatenate(outs, axis=1)


def switch_glu_padded(m, x, indices, align=64):
    """Leap's /Users/kamen/Leap/release/cleanclone/cache/clone/kernels/pfb_fix.py (Sun 08:3x): SwitchGLU's forward (sort, up / gate / down sorted gather, SwiGLU,
    unsort) with the sorted rows padded to a multiple of 64 (copies of the last row, same expert), so MLX 0.32.2's
    affine_gather_qmm_rhs_nax takes its aligned path. Bitwise == the 1,024-token calls (Leap, 1.6K / 4.7K / 7.9K)."""
    from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort
    xe = mx.expand_dims(x, (-2, -3))
    xs, idx, inv = _gather_sort(xe, indices)
    R = idx.shape[0]
    pad = (-R) % align
    if pad:
        xs = mx.concatenate([xs, mx.repeat(xs[-1:], pad, axis=0)], axis=0)
        idx = mx.concatenate([idx, mx.repeat(idx[-1:], pad, axis=0)], axis=0)
    h = m.activation(m.up_proj(xs, idx, sorted_indices=True), m.gate_proj(xs, idx, sorted_indices=True))
    y = m.down_proj(h, idx, sorted_indices=True)
    if pad:
        y = y[:R]
    return _scatter_unsort(y, inv, indices.shape).squeeze(-2)


def _scratch(t):
    """The full-E scratch for layer t's piece shapes (reallocated only when the shapes change) and, on a new layer, no valid
    rows yet. Returns E."""
    E = int(t["E"])
    shp = [((E,) + tuple(sh), dt) for sh, dt in t["shapes4"]]
    if LMC["sc"] is None or [(tuple(s.shape), s.dtype) for s in LMC["sc"]] != shp:
        LMC["sc"] = None
        mx.clear_cache()
        LMC["sc"] = [mx.zeros(sh, dt) for sh, dt in shp]
        mx.eval(*LMC["sc"])
        LMC["t"] = None
    if LMC["t"] is not t:                                # a new layer: no row of the scratch is valid for it yet
        LMC["t"], LMC["have"] = t, set()
    return E


def _read_all(t):
    """PF-B: every expert of layer t not yet in the scratch, read now (CPU) while the GPU runs the queued attention parts."""
    E = _scratch(t)
    need = [e for e in range(E) if e not in LMC["have"]]
    if need:
        LMC["tier"]._read_rebuilt(t, need, need, LMC["sc"])
        LMC["have"].update(need)
        LMC["reads"] += len(need)


def install(tier, mode=1):
    """Wrap bigmoe.tier.prefill_call (stream._sync_call looks it up at call time). Inactive unless LMC['on']."""
    LMC["mode"] = int(mode)
    LMC["tier"] = tier
    if getattr(tier, "_pflm_installed", False):
        return
    orig = tier.prefill_call
    from bigmoe.stream import PIECES

    def prefill_call(cls, self, x, indices, bm, *a, **kw):
        if not LMC["on"]:
            return orig(cls, self, x, indices, bm, *a, **kw)
        t = bm["tier"]
        tier.STATS["prefill_calls"] += 1
        if LMC.get("pipe"):                              # PIPE: all E experts already in the active scratch, original ids
            saved = [getattr(getattr(self, proj), field) for proj, field in PIECES]
            for (proj, field), arr in zip(PIECES, LMC["sc"]):
                setattr(getattr(self, proj), field, arr)
            try:
                return switch_glu_padded(self, x, indices)   # lazy: no sync here
            finally:
                for (proj, field), arr in zip(PIECES, saved):
                    setattr(getattr(self, proj), field, arr)
        ids = np.array(indices)
        uniq = np.unique(ids)
        _scratch(t)
        need = [int(e) for e in uniq if int(e) not in LMC["have"]]
        if need:                                         # the same ext rebuild as today, into those experts' own rows
            tier._read_rebuilt(t, need, need, LMC["sc"])
            LMC["have"].update(need)
            LMC["reads"] += len(need)
        LMC["hits"] += len(uniq) - len(need)
        LMC["calls"] += 1
        if LMC["mode"] == 2:                             # today's call shape: a compact sorted scratch + order-preserving ids
            gi = mx.array(uniq.astype(np.int32))
            scr = [mx.take(s, gi, axis=0) for s in LMC["sc"]]
            mx.eval(*scr)
            idx = mx.array(np.searchsorted(uniq, ids).astype(np.int32)).astype(indices.dtype)
        else:
            scr, idx = LMC["sc"], indices
        saved = [getattr(getattr(self, proj), field) for proj, field in PIECES]
        for (proj, field), arr in zip(PIECES, scr):
            setattr(getattr(self, proj), field, arr)
        try:
            y = switch_glu_padded(self, x, idx) if LMC["mode"] == 3 else cls.__call__(self, x, idx, *a, **kw)
            mx.eval(y)
        finally:
            for (proj, field), arr in zip(PIECES, saved):
                setattr(getattr(self, proj), field, arr)
        if LMC["mode"] == 2:
            del scr
        return y

    tier.prefill_call = prefill_call
    tier._pflm_installed = True


def _split_layers_pipe(tm, xs, hs, cache, states, stage, on_layer):
    """R57_PFLM=3 + R57_PFLM_PIPE=1: PF-B pipelined across layers (see _PIPE). Returns False (and does nothing) when a
    layer's piece shapes differ from layer 0's, so the caller runs the plain loop."""
    import mtplx.models.qwen4_exp as Q
    tok = int(os.environ.get("R57_PFLM_TOK", "4096") or 4096)
    seg = os.environ.get("R57_PFLM_SEG", "1") != "0"
    layers = list(tm.layers)
    ts = [l_.mlp.switch_mlp._bm["tier"] for l_ in layers]
    E = int(ts[0]["E"])
    shp = [((E,) + tuple(sh), dt) for sh, dt in ts[0]["shapes4"]]
    if any(int(t_["E"]) != E or [((E,) + tuple(sh), dt) for sh, dt in t_["shapes4"]] != shp for t_ in ts):
        return False
    for l_ in layers:
        bm_ = l_.mlp.switch_mlp._bm
        assert "tier" in bm_ and not bm_["full"], "R57_PFLM=3: every layer must take the tier path (none fully resident)"
    groups, cur, n = [], [], 0
    for k, x in enumerate(xs):
        if cur and n + x.shape[1] > tok:
            groups.append(cur)
            cur, n = [], 0
        cur.append(k)
        n += x.shape[1]
    groups.append(cur)
    LMC["sc"] = None
    mx.clear_cache()
    bufs = []
    for _ in range(2):
        b_ = [mx.zeros(sh, dt) for sh, dt in shp]
        mx.eval(*b_)
        bufs.append(b_)
    allE = list(range(E))

    def read(L):
        LMC["tier"]._read_rebuilt(ts[L], allE, allE, bufs[L % 2])
        LMC["reads"] += E
        return L

    read(0)                                              # layer 0: before its attention (nothing queued yet)
    prev_out = None
    LMC["pipe"] = True
    try:
        for L, (layer, c) in enumerate(zip(layers, cache)):
            pend = []
            T_ = _T() if _PROF else None
            for k, x in enumerate(xs):
                if L == 0:
                    assert Q.create_ssm_mask(hs[k], cache[tm.ssm_idx]) is None, "PFLM: an ssm mask (padding?)"
                sync = stage and L == tm._ple_stage_idx
                if T_: w0, c0 = _clk()
                if sync:
                    ple = tm.layers[tm._ple_stage_idx].ple
                    ple.ple_embedding.stage(x, cache[tm._ple_stage_idx], ple.NGRAM_IDX)
                    if T_: w1, c1 = _clk(); T_.add("ple_stage", w1 - w0, c1 - c0)
                h = hs[k]
                if T_: w0, c0 = _clk()
                if "ple" in layer:
                    h = h + layer.ple(h, x, c)
                mixed, hyper, inject = layer.attn_hyper_connection(h)
                bo = layer.linear_attn(mixed, None, c) if layer.is_linear else layer.self_attn(mixed, c)
                mixed, hyper, inject = layer.mlp_hyper_connection(hyper, pending=(bo, inject))
                pend.append((mixed, hyper, inject))
                outs = [a for a in (mixed, hyper, inject) if isinstance(a, mx.array)] + list(states([c]))
                (mx.eval if sync else mx.async_eval)(*outs)
                if T_: w1, c1 = _clk(); T_.add("attn_submit", w1 - w0, c1 - c0)
            if T_: w0, c0 = _clk()
            LMC["sc"], LMC["t"], LMC["have"] = bufs[L % 2], ts[L], set(allE)
            outs_L = []
            for g in groups:
                M = pend[g[0]][0] if len(g) == 1 else mx.concatenate([pend[k][0] for k in g], axis=1)
                fix = len(g) > 1 and seg
                if fix:
                    mlp_ = layer.mlp
                    seg0 = mlp_.shared_expert_gate
                    mlp_.shared_expert_gate = _SegPerChunk(seg0, [pend[k][0].shape[1] for k in g])
                try:
                    Y = layer.mlp(M)
                finally:
                    if fix:
                        mlp_.shared_expert_gate = seg0
                off = 0
                for k in g:
                    n_ = pend[k][0].shape[1]
                    hs[k] = Q._hyper_residual_write(pend[k][1], Y if len(g) == 1 else Y[:, off:off + n_], pend[k][2])
                    off += n_
                    if on_layer is not None:
                        on_layer(layer, hs[k])
                    outs_L.append(hs[k])
                del M, Y
            mx.async_eval(*outs_L)
            if T_: w1, c1 = _clk(); T_.add("moe_submit", w1 - w0, c1 - c0); w0, c0 = w1, c1
            if L + 1 < len(layers):
                if prev_out is not None:
                    mx.eval(*prev_out)                   # layer L-1 done: its buffer, (L+1) % 2, is free; L stays queued
                if T_: w1, c1 = _clk(); T_.add("prev_wait", w1 - w0, c1 - c0); w0, c0 = w1, c1
                read(L + 1)                              # main thread (the ext holds the GIL); the GPU runs layer L meanwhile
                if T_: w1, c1 = _clk(); T_.add("read", w1 - w0, c1 - c0)
            if T_:
                with open(_PROF, "a") as f_:
                    f_.write(_json.dumps({"L": L, "linear": bool(layer.is_linear), "chunks": len(xs), "groups": len(groups),
                                          "sync": _PSYNC, "pipe": True,
                                          "t": {k: [round(v[0], 6), round(v[1], 6)] for k, v in T_.d.items()}}) + "\n")
            prev_out = outs_L
            del pend
        mx.eval(*prev_out)
    finally:
        LMC["pipe"] = False
        LMC["sc"] = None
        del bufs
    return True


def _split_layers(tm, xs, hs, cache, states, stage, on_layer):
    """R57_PFLM=3 (PF-B): each layer split at its MoE (qwen4_exp DecoderLayer.__call__'s steps verbatim; REF2's patched
    hyper read / attention / write are looked up at call time as there; its _dec_call only acts on 5-16-row passes)."""
    import mtplx.models.qwen4_exp as Q
    tok = int(os.environ.get("R57_PFLM_TOK", "4096") or 4096)
    ahead = os.environ.get("R57_PFLM_AHEAD", "1") != "0"
    cc = os.environ.get("R57_PFLM_CC") == "1"
    seg = os.environ.get("R57_PFLM_SEG", "1") != "0"     # =0: the batched gate (pb3 / pb38 at d44ec7a: not bitwise)
    assert not seg or all(hasattr(l_.mlp, "shared_expert_gate") for l_ in tm.layers), "R57_PFLM_SEG: no shared_expert_gate"
    groups, cur, n = [], [], 0                           # contiguous chunk groups of <= tok tokens (at least one chunk)
    for k, x in enumerate(xs):
        if cur and n + x.shape[1] > tok:
            groups.append(cur)
            cur, n = [], 0
        cur.append(k)
        n += x.shape[1]
    groups.append(cur)
    for layer in tm.layers:                              # a fully resident layer would run the unpadded stock call
        bm_ = layer.mlp.switch_mlp._bm
        assert "tier" in bm_ and not bm_["full"], "R57_PFLM=3: every layer must take the tier path (none fully resident)"
    for L, (layer, c) in enumerate(zip(tm.layers, cache)):
        t = layer.mlp.switch_mlp._bm["tier"]
        pend = []
        T_ = _T() if _PROF else None
        for k, x in enumerate(xs):
            if L == 0:                                   # as _forward: the ssm mask of a B=1 unpadded chunk is None
                assert Q.create_ssm_mask(hs[k], cache[tm.ssm_idx]) is None, "PFLM: an ssm mask (padding?)"
            sync = stage and L == tm._ple_stage_idx      # the PLE stage feeds chunk k's PLE read: consume it before k+1's
            if T_: w0, c0 = _clk()
            if sync:
                ple = tm.layers[tm._ple_stage_idx].ple
                ple.ple_embedding.stage(x, cache[tm._ple_stage_idx], ple.NGRAM_IDX)
                if T_: w1, c1 = _clk(); T_.add("ple_stage", w1 - w0, c1 - c0)
            h = hs[k]
            if T_: w0, c0 = _clk()
            if "ple" in layer:
                h = h + layer.ple(h, x, c)
                if T_: _ev(h); w1, c1 = _clk(); T_.add("ple", w1 - w0, c1 - c0); w0, c0 = w1, c1
            mixed, hyper, inject = layer.attn_hyper_connection(h)
            if T_: _ev(mixed, hyper, inject); w1, c1 = _clk(); T_.add("hc_read", w1 - w0, c1 - c0); w0, c0 = w1, c1
            bo = layer.linear_attn(mixed, None, c) if layer.is_linear else layer.self_attn(mixed, c)
            if T_: _ev(bo); w1, c1 = _clk(); T_.add("gdn" if layer.is_linear else "qsa", w1 - w0, c1 - c0); w0, c0 = w1, c1
            mixed, hyper, inject = layer.mlp_hyper_connection(hyper, pending=(bo, inject))
            pend.append((mixed, hyper, inject))
            outs = [a for a in (mixed, hyper, inject) if isinstance(a, mx.array)] + list(states([c]))
            (mx.eval if sync else mx.async_eval)(*outs)
            if T_: _ev(*outs); w1, c1 = _clk(); T_.add("hc_write", w1 - w0, c1 - c0)
        if T_: w0, c0 = _clk()
        if ahead:
            _read_all(t)                                 # CPU read || the GPU's queued attention parts
        if T_: w1, c1 = _clk(); T_.add("read", w1 - w0, c1 - c0)
        for g in groups:
            if T_: w0, c0 = _clk()
            M = pend[g[0]][0] if len(g) == 1 else mx.concatenate([pend[k][0] for k in g], axis=1)
            fix = len(g) > 1 and seg                     # R57_PFLM_SEG (default 1): the shared expert's gate per chunk slice
            if fix:
                mlp_ = layer.mlp
                seg0 = mlp_.shared_expert_gate
                mlp_.shared_expert_gate = _SegPerChunk(seg0, [pend[k][0].shape[1] for k in g])
            try:
                Y = layer.mlp(M)                         # router + shared expert + the padded SwitchGLU (tier wrapper)
            finally:
                if fix:
                    mlp_.shared_expert_gate = seg0
            off = 0
            for k in g:
                n_ = pend[k][0].shape[1]
                hs[k] = Q._hyper_residual_write(pend[k][1], Y if len(g) == 1 else Y[:, off:off + n_], pend[k][2])
                off += n_
                if on_layer is not None:
                    on_layer(layer, hs[k])
            if T_: w1, c1 = _clk(); T_.add("moe_build", w1 - w0, c1 - c0); w0, c0 = w1, c1
            mx.eval(*[hs[k] for k in g])
            if T_: w1, c1 = _clk(); T_.add("moe_eval", w1 - w0, c1 - c0)
            del M, Y
        if T_:
            with open(_PROF, "a") as f_:
                f_.write(_json.dumps({"L": L, "linear": bool(layer.is_linear), "chunks": len(xs), "groups": len(groups),
                                      "sync": _PSYNC, "t": {k: [round(v[0], 6), round(v[1], 6)] for k, v in T_.d.items()}}) + "\n")
        del pend
        if cc:
            mx.clear_cache()


def prefill_layer_major(model, ids, chunk, states, on_chunk=None, on_layer=None):
    """The screen's chunk-major prefill, layer-major. Returns (cache, logits of the last chunk's last 256 rows).
    on_chunk(k, h_k) runs after the LAST layer for each chunk k (the widened stream, for the MTP head). on_layer(layer, h_k):
    R57_PFLM=3 only, after each (layer, chunk) step (modes 1 / 2 call layer(...), whose screen wrapper taps itself)."""
    import mtplx.models.qwen4_exp as Q
    import bigmoe.tier as BT
    lm = model.language_model
    tm = lm.model
    cache = model.make_cache()
    xs = [mx.array([ids[a:a + chunk]]) for a in range(0, len(ids), chunk)]

    def run():
        if _PROF: _w0, _c0 = _clk()
        hs = []
        for x in xs:
            h = tm.embed_tokens(x)
            hs.append(mx.tile(h, (1, 1, tm.args.hc_count)))
        mx.eval(*hs)
        stage = tm._ple_stage_idx is not None and Q._COMPILED_VERIFY_PLE.get() is None
        BT._reuse_drop()
        LMC.update(on=True, t=None, have=set())
        try:
            if LMC["mode"] == 3:
                if not (_PIPE and _split_layers_pipe(tm, xs, hs, cache, states, stage, on_layer)):
                    _split_layers(tm, xs, hs, cache, states, stage, on_layer)
            for L, (layer, c) in enumerate(zip(tm.layers, cache) if LMC["mode"] != 3 else ()):
                for k, x in enumerate(xs):
                    if L == 0:                           # as _forward: the ssm mask of a B=1 unpadded chunk is None
                        assert Q.create_ssm_mask(hs[k], cache[tm.ssm_idx]) is None, "PFLM: an ssm mask (padding?)"
                    if stage and L == tm._ple_stage_idx:   # as _forward stages chunk k's n-grams before its layers: the
                        ple = tm.layers[tm._ple_stage_idx].ple   # stage only touches the PLE layer's cache, which holds
                        ple.ple_embedding.stage(x, cache[tm._ple_stage_idx], ple.NGRAM_IDX)   # chunks 0..k-1 here too
                    hs[k] = layer(hs[k], input_ids=x, ssm_mask=None, cache=c)
                    mx.eval(hs[k], *states([c]))
        finally:
            LMC.update(on=False, t=None, have=set(), sc=None)
            mx.clear_cache()
        if on_chunk is not None:
            for k, h in enumerate(hs):
                on_chunk(k, h)
        tm._last_widened = hs[-1]
        out = tm.hyper_connection_mixer(hs[-1])[:, -256:]   # TextModel(logits_keep=256): the mixer over the whole chunk
        lg_ = lm._head_logits(out)
        if _PROF:
            mx.eval(lg_); _w1, _c1 = _clk()
            with open(_PROF, "a") as f_:
                f_.write(_json.dumps({"total": [round(_w1 - _w0, 4), round(_c1 - _c0, 4)], "tokens": len(ids), "chunk": chunk,
                                      "mode": LMC["mode"], "sync": _PSYNC}) + "\n")
        return lg_

    if Q.qwen4_opdiet_enabled("rope"):
        with Q._rope_table_scope():
            lg = run()
    else:
        lg = run()
    return cache, lg


# ---------------------------------------------------------------- PF-D (R57_PFD=1)
# (a quote from the project owner, kept private)
# prefill reads EVERY routed expert-layer from records3 (bytes = routed x 2.79 MB), although 65.5 % of them have their MSB-3
# base and 25.1 % their LSB side part in the 45 GB pools; TTFT = 0.32 s + bytes / 7.33 GB/s. Strata's prompt path (MIT,
# ~/StrataEval/src: stream.cpp at_layer) reads only the routed experts its arena lacks. Ours needs the 4-bit pieces in a
# scratch (tier.prefill_call: the same gather_qmm call on the call's sorted unique experts), so PF-D only changes where each
# scratch row's bytes come from and keeps the call itself:
#   A  base + side resident: rebuilt on the GPU from the two pool slots;
#   B  base resident only: the LSB part (0.72 of 2.79 MB) read from records3 (pread, F_NOCACHE, threads), rebuilt on the GPU;
#   C  base absent: tier._read_rebuilt as today (the ext's CPU rebuild of the full record).
# The GPU rebuild is the ext's, op for op (bigmoe_ext.cpp unnative / unp11 / rebuild_w4 / halve_bf16, native+p11 layout 2):
# a pool slot holds its record part's pieces verbatim. R57_PFD_CHECK=1 compares every rebuilt row with tier._read_rebuilt's
# (extra reads: timing void) and counts mismatching rows of A and B.
import fcntl
from concurrent.futures import ThreadPoolExecutor

import time
PFD = {"calls": 0, "A": 0, "B": 0, "C": 0, "ssd": 0, "checked": 0, "bad_A": 0, "bad_B": 0, "fallback": 0,
       "tB": 0.0, "tC": 0.0, "tG": 0.0, "tM": 0.0,           # s: side-part reads, today's reads, GPU rebuild (+ scatter), the call
       "tZ": 0.0, "tP": 0.0, "bad_C": 0,                      # scratch zeros; the whole wrapped call; v3 CHECK of copied C rows
       "k4": 0, "k5": 0,
       "pB": 0, "pC": 0, "hB": 0, "hC": 0, "mB": 0, "mC": 0, "tPred": 0.0}   # read-ahead: predicted / hit / read on demand                                      # FAST rows at 3 bits: base pooled (4) / base read (5)
_and, _or, _shl, _shr = mx.bitwise_and, mx.bitwise_or, mx.left_shift, mx.right_shift


def _u(v):
    return mx.array(v, dtype=mx.uint32)


_R32 = np.arange(32)
_WI, _SH = (11 * _R32) >> 5, (11 * _R32) & 31            # unp11: code r at bits 11r .. 11r + 10 of an 11-word block
_NEED = _SH > 21


def _spread8(x, l):                                      # 8 weights: 24 code bits + 8 low bits -> 8 nibbles (ext spread8)
    a = _or(_and(x, _u(0xFFF)), _shl(_and(x, _u(0xFFF000)), _u(4)))
    a = _or(_and(a, _u(0x003F003F)), _shl(_and(a, _u(0x0FC00FC0)), _u(2)))
    a = _or(_and(a, _u(0x07070707)), _shl(_and(a, _u(0x38383838)), _u(1)))
    y = _and(l, _u(0xFF))
    y = _and(_or(y, _shl(y, _u(12))), _u(0x000F000F))
    y = _and(_or(y, _shl(y, _u(6))), _u(0x03030303))
    y = _and(_or(y, _shl(y, _u(3))), _u(0x11111111))
    return _or(_shl(a, _u(1)), y)


def _unnative(a, O, KG, cw):                             # native chunk (t*KG + g)*32 + r -> row-major (t*32 + r)*KG + g
    n = a.shape[0]
    return a.reshape(n, O // 32, KG, 32, cw).transpose(0, 1, 3, 2, 4).reshape(n, O, KG * cw)


def _unp11(wd, O, KG):                                   # (n, words) p11 piece -> (n, O, KG) bf16 bits (in uint32)
    n, nb = wd.shape[0], O // 32 * KG
    q = wd[:, :11 * nb].reshape(n, nb, 11)
    lo = _shr(mx.take(q, mx.array(_WI.astype(np.int32)), axis=2), _u(_SH.astype(np.uint32)))
    hi = _shl(mx.take(q, mx.array(np.minimum(_WI + 1, 10).astype(np.int32)), axis=2),
              _u(np.where(_NEED, 32 - _SH, 0).astype(np.uint32)))
    c = mx.where(mx.array(_NEED), _or(lo, hi), lo)
    base = _shl(wd[:, 11 * nb], _u(7)).reshape(n, 1, 1)
    v = _or(_shl(_and(c, _u(0x400)), _u(5)), _and(c, _u(0x3FF)) + base)
    return v.reshape(n, O // 32, KG, 32).transpose(0, 1, 3, 2).reshape(n, O, KG)


def _w4(w3, l1, O, KG):                                  # row-major w3 (n, O, 6KG) + l1 (n, O, 2KG) -> 4-bit (n, O, 8KG)
    n = w3.shape[0]
    w = w3.reshape(n, O, 2 * KG, 3)
    a, b, c = w[..., 0], w[..., 1], w[..., 2]
    o0 = _spread8(_and(a, _u(0xFFFFFF)), l1)
    o1 = _spread8(_or(_shr(a, _u(24)), _shl(_and(b, _u(0xFFFF)), _u(8))), _shr(l1, _u(8)))
    o2 = _spread8(_or(_shr(b, _u(16)), _shl(_and(c, _u(0xFF)), _u(16))), _shr(l1, _u(16)))
    o3 = _spread8(_shr(c, _u(8)), _shr(l1, _u(24)))
    return mx.stack([o0, o1, o2, o3], axis=-1).reshape(n, O, 8 * KG)


def _bf16(v):
    return v.astype(mx.uint16).view(mx.bfloat16)


# PF-D v2 (17:4x; v1's MLX-op rebuild cost ~0.85 ms per expert, slower than the SSD read it saves): the same rebuild as two
# fused Metal kernels, one pass each. W4: one thread per (expert, row, 64-weight group): 6 native w3 words + 2 l1 words -> 8
# row-major 4-bit words (the ext's spread8 / rebuild_w4). P11: one thread per (expert, block, code): an 11-bit code -> its bf16
# bits at the row-major (row, group) place, halved for the scales (unp11 + halve_bf16). R57_PFD_KERNEL=0 keeps v1's ops.
_W4_HDR = """
inline uint spread8(uint x, uint l) {
  uint a = (x & 0xFFFu) | ((x & 0xFFF000u) << 4);
  a = (a & 0x003F003Fu) | ((a & 0x0FC00FC0u) << 2);
  a = (a & 0x07070707u) | ((a & 0x38383838u) << 1);
  uint y = l & 0xFFu;
  y = (y | (y << 12)) & 0x000F000Fu;
  y = (y | (y << 6)) & 0x03030303u;
  y = (y | (y << 3)) & 0x11111111u;
  return (a << 1) | y;
}
"""
_W4_SRC = """
  uint gid = thread_position_in_grid.x;
  uint per = O_ * KG_;
  uint i = gid / per, rem = gid - i * per;
  uint row = rem / KG_, g = rem - row * KG_;
  uint cn = ((row >> 5) * KG_ + g) * 32u + (row & 31u);
  const device uint* a3 = w3 + (size_t)i * per * 6u + (size_t)cn * 6u;
  const device uint* a1 = l1 + (size_t)i * per * 2u + (size_t)cn * 2u;
  device uint* o = out + (size_t)i * per * 8u + (size_t)rem * 8u;
  for (uint h = 0; h < 2u; h++) {
    uint a = a3[3u * h], b = a3[3u * h + 1u], c = a3[3u * h + 2u], l = a1[h];
    o[4u * h + 0u] = spread8(a & 0xFFFFFFu, l);
    o[4u * h + 1u] = spread8((a >> 24) | ((b & 0xFFFFu) << 8), l >> 8);
    o[4u * h + 2u] = spread8((b >> 16) | ((c & 0xFFu) << 16), l >> 16);
    o[4u * h + 3u] = spread8(c >> 8, l >> 24);
  }
"""
_P11_SRC = """
  uint gid = thread_position_in_grid.x;
  uint per = NB_ * 32u;
  uint i = gid / per, rem = gid - i * per;
  uint b = rem >> 5, r = rem & 31u;
  const device uint* w = wd + (size_t)i * WD_;
  const device uint* q = w + (size_t)b * 11u;
  uint bit = 11u * r, wi = bit >> 5, sh = bit & 31u;
  uint c = q[wi] >> sh;
  if (sh > 21u) c |= q[wi + 1u] << (32u - sh);
  uint v = (((c & 0x400u) << 5) | ((c & 0x3FFu) + (w[11u * NB_] << 7))) & 0xFFFFu;
  if (HALVE_) v = ((v & 0x7FFFu) == 0u) ? v : ((v - 0x80u) & 0xFFFFu);
  out[(size_t)i * per + (b / KG_) * 32u * KG_ + (b % KG_) + r * KG_] = (uint16_t)v;
"""
# PF-D v3 (18:1x; pfd_bench.py: v2's tG = gathers 2.3 + kernels 2.8 + zeros 2.0 + scatter 3.8 ms per layer, ~2x more in-run with
# fresh buffers): two ASSEMBLE kernels write the call's whole scratch in one pass. rowmap[row] = (kind, i1, i2): kind 0 = A
# (base pool slot i1, side pool slot i2), 1 = B (base pool slot i1, read side part i2), 2 = C (row i1 of the ext-rebuilt rows).
# They read the pools in place (no gathers, no intermediates, no scatter).
_P11_HDR = """
inline uint p11code(const device uint* w, uint NB, uint b, uint r, uint halve) {
  const device uint* q = w + (size_t)b * 11u;
  uint bit = 11u * r, wi = bit >> 5, sh = bit & 31u;
  uint c = q[wi] >> sh;
  if (sh > 21u) c |= q[wi + 1u] << (32u - sh);
  uint v = (((c & 0x400u) << 5) | ((c & 0x3FFu) + (w[11u * NB] << 7))) & 0xFFFFu;
  if (halve) v = ((v & 0x7FFFu) == 0u) ? v : ((v - 0x80u) & 0xFFFFu);
  return v;
}
"""
_AW4_SRC = """
  uint gid = thread_position_in_grid.x;
  uint per = O_ * KG_;
  uint row = gid / per, rem = gid - row * per;
  int kind = rowmap[3u * row], i1 = rowmap[3u * row + 1u], i2 = rowmap[3u * row + 2u];
  device uint* o = out + (size_t)row * per * 8u + (size_t)rem * 8u;
  if (kind == 2) {
    const device uint* s = cw + (size_t)i1 * per * 8u + (size_t)rem * 8u;
    for (uint k = 0; k < 8u; k++) o[k] = s[k];
    return;
  }
  uint orow = rem / KG_, g = rem - orow * KG_;
  uint cn = ((orow >> 5) * KG_ + g) * 32u + (orow & 31u);
  const device uint* a3 = bw3 + (size_t)i1 * per * 6u + (size_t)cn * 6u;
  const device uint* a1 = (kind == 0 ? sl1 : bl1) + (size_t)i2 * per * 2u + (size_t)cn * 2u;
  for (uint h = 0; h < 2u; h++) {
    uint a = a3[3u * h], b = a3[3u * h + 1u], c = a3[3u * h + 2u], l = a1[h];
    o[4u * h + 0u] = spread8(a & 0xFFFFFFu, l);
    o[4u * h + 1u] = spread8((a >> 24) | ((b & 0xFFFFu) << 8), l >> 8);
    o[4u * h + 2u] = spread8((b >> 16) | ((c & 0xFFu) << 16), l >> 16);
    o[4u * h + 3u] = spread8(c >> 8, l >> 24);
  }
"""
_AP11_SRC = """
  uint gid = thread_position_in_grid.x;
  uint per = NB_ * 32u;
  uint row = gid / per, rem = gid - row * per;
  uint b = rem >> 5, r = rem & 31u;
  int kind = rowmap[3u * row], i1 = rowmap[3u * row + 1u], i2 = rowmap[3u * row + 2u];
  size_t at = (b / KG_) * 32u * KG_ + (b % KG_) + r * KG_;
  if (kind == 2) {
    os[(size_t)row * per + at] = cs[(size_t)i1 * per + at];
    ob[(size_t)row * per + at] = cb[(size_t)i1 * per + at];
    return;
  }
  os[(size_t)row * per + at] = (uint16_t)p11code(bs3 + (size_t)i1 * WS_, NB_, b, r, 1u);
  const device uint* bw = (kind == 0 ? sb + (size_t)i2 * WB_ : bb + (size_t)i2 * WB_);
  ob[(size_t)row * per + at] = (uint16_t)p11code(bw, NB_, b, r, 0u);
"""
_KER = {}


def _kernels_v3():
    if "aw4" not in _KER:
        _KER["aw4"] = mx.fast.metal_kernel(name="pfd_aw4", input_names=["bw3", "sl1", "bl1", "cw", "rowmap"],
                                           output_names=["out"], source=_AW4_SRC, header=_W4_HDR)
        _KER["ap11"] = mx.fast.metal_kernel(name="pfd_ap11", input_names=["bs3", "sb", "bb", "cs", "cb", "rowmap"],
                                            output_names=["os", "ob"], source=_AP11_SRC, header=_P11_HDR)
    return _KER["aw4"], _KER["ap11"]


# (a quote from the project owner, kept private)
#   3 = C read raw: the full record (staged words rr, stride RW) rebuilt on the GPU (exact; no CPU rebuild, no zeroed scratch,
#       and every read GIL-free, so reads can run beside the GPU);
#   4 = FAST base only: base pool slot i1 at 3 bits (q = 2m, s = s3 / 2, b = b3: the base-only approximation, no side read);
#   5 = FAST base read: the MSB part only (staged words rm5, stride MW) at 3 bits.
_AW4V4_SRC = """
  uint gid = thread_position_in_grid.x;
  uint per = O_ * KG_;
  uint row = gid / per, rem = gid - row * per;
  int kind = rowmap[3u * row], i1 = rowmap[3u * row + 1u], i2 = rowmap[3u * row + 2u];
  device uint* o = out + (size_t)row * per * 8u + (size_t)rem * 8u;
  if (kind == 2) {
    const device uint* s = cw + (size_t)i1 * per * 8u + (size_t)rem * 8u;
    for (uint k = 0; k < 8u; k++) o[k] = s[k];
    return;
  }
  uint orow = rem / KG_, g = rem - orow * KG_;
  size_t cn = ((size_t)(orow >> 5) * KG_ + g) * 32u + (orow & 31u);
  const device uint* a3;
  const device uint* a1 = bw3;                          // placeholder; hasl says whether it is read
  bool hasl = true;
  if (kind == 0) { a3 = bw3 + (size_t)i1 * per * 6u + cn * 6u; a1 = sl1 + (size_t)i2 * per * 2u + cn * 2u; }
  else if (kind == 1) { a3 = bw3 + (size_t)i1 * per * 6u + cn * 6u; a1 = rl + (size_t)i2 * LW_ + OL1L_ + cn * 2u; }
  else if (kind == 3) { const device uint* rec = rr + (size_t)i1 * RW_; a3 = rec + OW3_ + cn * 6u; a1 = rec + OL1_ + cn * 2u; }
  else if (kind == 4) { a3 = bw3 + (size_t)i1 * per * 6u + cn * 6u; hasl = false; }
  else { a3 = rm5 + (size_t)i1 * MW_ + OW3_ + cn * 6u; hasl = false; }
  for (uint h = 0; h < 2u; h++) {
    uint a = a3[3u * h], b = a3[3u * h + 1u], c = a3[3u * h + 2u], l = hasl ? a1[h] : 0u;
    o[4u * h + 0u] = spread8(a & 0xFFFFFFu, l);
    o[4u * h + 1u] = spread8((a >> 24) | ((b & 0xFFFFu) << 8), l >> 8);
    o[4u * h + 2u] = spread8((b >> 16) | ((c & 0xFFu) << 16), l >> 16);
    o[4u * h + 3u] = spread8(c >> 8, l >> 24);
  }
"""
_AP11V4_SRC = """
  uint gid = thread_position_in_grid.x;
  uint per = NB_ * 32u;
  uint row = gid / per, rem = gid - row * per;
  uint b = rem >> 5, r = rem & 31u;
  int kind = rowmap[3u * row], i1 = rowmap[3u * row + 1u], i2 = rowmap[3u * row + 2u];
  size_t at = (b / KG_) * 32u * KG_ + (b % KG_) + r * KG_;
  if (kind == 2) {
    os[(size_t)row * per + at] = cs[(size_t)i1 * per + at];
    ob[(size_t)row * per + at] = cb[(size_t)i1 * per + at];
    return;
  }
  const device uint* s3w;
  const device uint* bw;
  if (kind == 0) { s3w = bs3 + (size_t)i1 * WS_; bw = sb + (size_t)i2 * WB_; }
  else if (kind == 1) { s3w = bs3 + (size_t)i1 * WS_; bw = rl + (size_t)i2 * LW_ + OBL_; }
  else if (kind == 3) { s3w = rr + (size_t)i1 * RW_ + OS3_; bw = rr + (size_t)i1 * RW_ + OB_; }
  else if (kind == 4) { s3w = bs3 + (size_t)i1 * WS_; bw = bb3 + (size_t)i1 * WS_; }
  else { s3w = rm5 + (size_t)i1 * MW_ + OS3_; bw = rm5 + (size_t)i1 * MW_ + OB3_; }
  os[(size_t)row * per + at] = (uint16_t)p11code(s3w, NB_, b, r, 1u);
  ob[(size_t)row * per + at] = (uint16_t)p11code(bw, NB_, b, r, 0u);
"""


def _kernels_v4():
    if "aw4v4" not in _KER:
        _KER["aw4v4"] = mx.fast.metal_kernel(name="pfd_aw4v4", input_names=["bw3", "sl1", "rl", "cw", "rr", "rm5", "rowmap"],
                                             output_names=["out"], source=_AW4V4_SRC, header=_W4_HDR)
        _KER["ap11v4"] = mx.fast.metal_kernel(name="pfd_ap11v4",
                                              input_names=["bs3", "bb3", "sb", "rl", "cs", "cb", "rr", "rm5", "rowmap"],
                                              output_names=["os", "ob"], source=_AP11V4_SRC, header=_P11_HDR)
    return _KER["aw4v4"], _KER["ap11v4"]


_V4PH = []


def v4_placeholder():
    """An unused v4 input (no rows of its kind). MLX passes a custom kernel's inputs of fewer than 8 elements in the
    'constant' address space and the v4 kernels assign inputs to 'device' pointers, so a placeholder must be larger:
    Sun 8 Nov 23:5x, the in-run v4 / FAST arms died compiling with a one-element placeholder (the unit test's inputs were
    all real arrays). 1,024 words, never read."""
    if not _V4PH:
        _V4PH.append(mx.zeros((1, 1024), mx.uint32))
    return _V4PH[0]


def assemble_v4(t, rm, base, side, rl, scrC, rr, rm5):
    """v4: assemble with kinds 0-5 (see above). rl (n1, lsb bytes / 4): staged raw LSB parts (kind 1); rr (n3, recsz / 4) /
    rm5 (n5, msb_sz / 4) uint32: staged raw records / MSB parts; one-row placeholders when unused; scrC: kind 2's rows."""
    kw, kp = _kernels_v4()
    N = int(rm.shape[0])
    rmx = mx.array(rm.astype(np.int32))
    RW, MW, LW, msbw = int(rr.shape[1]), int(rm5.shape[1]), int(rl.shape[1]), int(t["msb_sz"]) // 4
    out = []
    for p, p_ in enumerate(("gate_proj", "up_proj", "down_proj")):
        O, KG = int(t["rows3"][p]), int(t["shapes4"][3 * p][0][1]) // 8
        nb = O // 32 * KG
        o = [int(t["offs"][i]) // 4 for i in (3 * p, 3 * p + 1, 3 * p + 2, 9 + 2 * p, 10 + 2 * p)]   # w3, s3, b3, l1, b words
        W = kw(inputs=[base[p_]["w3"], side[p_]["l1"], rl, scrC[3 * p], rr, rm5, rmx],
               template=[("O_", O), ("KG_", KG), ("RW_", RW), ("MW_", MW), ("LW_", LW), ("OW3_", o[0]), ("OL1_", o[3]),
                         ("OL1L_", o[3] - msbw)],
               grid=(N * O * KG, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(N, O, 8 * KG)], output_dtypes=[mx.uint32])[0]
        WS = int(base[p_]["s3"].shape[1])
        assert int(base[p_]["b3"].shape[1]) == WS, "PF-D v4: base s3 / b3 pieces of different widths"
        S, B_ = kp(inputs=[base[p_]["s3"], base[p_]["b3"], side[p_]["b"], rl, scrC[3 * p + 1].view(mx.uint16),
                           scrC[3 * p + 2].view(mx.uint16), rr, rm5, rmx],
                   template=[("NB_", nb), ("KG_", KG), ("WS_", WS), ("WB_", int(side[p_]["b"].shape[1])), ("RW_", RW),
                             ("MW_", MW), ("LW_", LW), ("OS3_", o[1]), ("OB3_", o[2]), ("OB_", o[4]), ("OBL_", o[4] - msbw)],
                   grid=(N * nb * 32, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(N, O, KG), (N, O, KG)],
                   output_dtypes=[mx.uint16, mx.uint16])
        out += [W, S.view(mx.bfloat16), B_.view(mx.bfloat16)]
    return out


def assemble(t, rm, base, side, bl1s, bbs, scrC):
    """v3: the call's whole 4-bit scratch (9 pieces, PIECES order) from rowmap rm (N, 3) int32: A rows from the pools in place,
    B rows from the base pool + read side parts (bl1s / bbs per projection), C rows copied from scrC (the ext's rebuild)."""
    kw, kp = _kernels_v3()
    N = int(rm.shape[0])
    rmx = mx.array(rm.astype(np.int32))
    out = []
    for p, p_ in enumerate(("gate_proj", "up_proj", "down_proj")):
        O, KG = int(t["rows3"][p]), int(t["shapes4"][3 * p][0][1]) // 8
        nb = O // 32 * KG
        W = kw(inputs=[base[p_]["w3"], side[p_]["l1"], bl1s[p], scrC[3 * p], rmx], template=[("O_", O), ("KG_", KG)],
               grid=(N * O * KG, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(N, O, 8 * KG)], output_dtypes=[mx.uint32])[0]
        S, B_ = kp(inputs=[base[p_]["s3"], side[p_]["b"], bbs[p], scrC[3 * p + 1].view(mx.uint16),
                           scrC[3 * p + 2].view(mx.uint16), rmx],
                   template=[("NB_", nb), ("KG_", KG), ("WS_", int(base[p_]["s3"].shape[1])), ("WB_", int(side[p_]["b"].shape[1]))],
                   grid=(N * nb * 32, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(N, O, KG), (N, O, KG)],
                   output_dtypes=[mx.uint16, mx.uint16])
        out += [W, S.view(mx.bfloat16), B_.view(mx.bfloat16)]
    return out


def _kernels():
    if not _KER:
        _KER["w4"] = mx.fast.metal_kernel(name="pfd_w4", input_names=["w3", "l1"], output_names=["out"], source=_W4_SRC,
                                          header=_W4_HDR)
        _KER["p11"] = mx.fast.metal_kernel(name="pfd_p11", input_names=["wd"], output_names=["out"], source=_P11_SRC)
    return _KER["w4"], _KER["p11"]


def rebuild_parts(t, w3s, s3s, l1s, bs):
    """The 9 4-bit pieces (PIECES order) of n experts from their part pieces: per projection w3 (n, O, 6KG) and l1 (n, O, 2KG)
    in native order, s3 / b as (n, words) p11 pieces. Exactly the ext's read_expert on a native+p11 record."""
    if os.environ.get("R57_PFD_KERNEL", "1") == "0":
        return rebuild_parts_ops(t, w3s, s3s, l1s, bs)
    kw4, kp11 = _kernels()
    out = []
    for p in range(3):
        O, KG = int(t["rows3"][p]), int(t["shapes4"][3 * p][0][1]) // 8
        n = int(w3s[p].shape[0])
        W = kw4(inputs=[w3s[p], l1s[p]], template=[("O_", O), ("KG_", KG)], grid=(n * O * KG, 1, 1), threadgroup=(256, 1, 1),
                output_shapes=[(n, O, 8 * KG)], output_dtypes=[mx.uint32])[0]
        sb = []
        for wd, hv in ((s3s[p], 1), (bs[p], 0)):
            nb = O // 32 * KG
            sb.append(kp11(inputs=[wd], template=[("NB_", nb), ("WD_", int(wd.shape[1])), ("KG_", KG), ("HALVE_", hv)],
                           grid=(n * nb * 32, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(n, O, KG)],
                           output_dtypes=[mx.uint16])[0].view(mx.bfloat16))
        out += [W, sb[0], sb[1]]
    return out


def rebuild_parts_ops(t, w3s, s3s, l1s, bs):
    """v1: the same rebuild in MLX ops (exact, ~0.85 ms per expert)."""
    out = []
    for p in range(3):
        O, KG = int(t["rows3"][p]), int(t["shapes4"][3 * p][0][1]) // 8
        W = _w4(_unnative(w3s[p], O, KG, 6), _unnative(l1s[p], O, KG, 2), O, KG)
        h = _unp11(s3s[p], O, KG)
        S = mx.where(_and(h, _u(0x7FFF)) == 0, h, h - _u(0x80))   # halve_bf16: s = s3 / 2 (exponent - 1; zeros kept)
        out += [W, _bf16(S), _bf16(_unp11(bs[p], O, KG))]
    return out


def lsb_pieces(t, buf):
    """records3 LSB parts (n, lsb bytes) -> per projection the l1 (n, O, 2KG) and b (n, words) pieces (index offsets)."""
    r, msb, n = t["r"], int(t["msb_sz"]), buf.shape[0]
    l1s, bs = [], []
    for p in range(3):
        O, KG = int(t["rows3"][p]), int(t["shapes4"][3 * p][0][1]) // 8
        o1, z1 = int(t["offs"][9 + 2 * p]) - msb, int(r["lsb_piece_size"][2 * p])
        o2, z2 = int(t["offs"][10 + 2 * p]) - msb, int(r["lsb_piece_size"][2 * p + 1])
        l1s.append(mx.array(np.ascontiguousarray(buf[:, o1:o1 + z1]).view(np.uint32)).reshape(n, O, 2 * KG))
        bs.append(mx.array(np.ascontiguousarray(buf[:, o2:o2 + z2]).view(np.uint32)))
    return l1s, bs


def msb_pieces(t, buf):
    """records3 MSB parts (n, msb bytes) -> per projection the w3 (n, O, 6KG) and s3 (n, words) pieces (unit test)."""
    r, n = t["r"], buf.shape[0]
    w3s, s3s = [], []
    for p in range(3):
        O, KG = int(t["rows3"][p]), int(t["shapes4"][3 * p][0][1]) // 8
        o1, z1 = int(t["offs"][3 * p]), int(r["msb_piece_size"][3 * p])
        o2, z2 = int(t["offs"][3 * p + 1]), int(r["msb_piece_size"][3 * p + 1])
        w3s.append(mx.array(np.ascontiguousarray(buf[:, o1:o1 + z1]).view(np.uint32)).reshape(n, O, 6 * KG))
        s3s.append(mx.array(np.ascontiguousarray(buf[:, o2:o2 + z2]).view(np.uint32)))
    return w3s, s3s


_PFD_FD, _PFD_POOL = {}, []


def read_parts(t, ids, off, size):
    """bytes [off, off + size) of each record ids[k] (F_NOCACHE preads on R57_PFD_THREADS threads) -> (n, size) uint8."""
    buf = np.empty((len(ids), size), np.uint8)
    fd = _PFD_FD.get(t["path"])
    if fd is None:
        fd = _PFD_FD[t["path"]] = os.open(t["path"], os.O_RDONLY)
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
    if not _PFD_POOL:
        _PFD_POOL.append(ThreadPoolExecutor(max_workers=int(os.environ.get("R57_PFD_THREADS", "12"))))

    def one(k):
        mv, base, got = memoryview(buf[k]), int(ids[k]) * int(t["recsz"]) + off, 0
        while got < size:
            n_ = os.preadv(fd, [mv[got:]], base + got)
            if n_ <= 0:
                raise OSError(f"PF-D: short read of record {int(ids[k])} at {base + got}")
            got += n_
    list(_PFD_POOL[0].map(one, range(len(ids))))
    if os.environ.get("R57_PFD_POOL") == "call":         # diagnostic (18:3x): no reader threads alive outside the call
        _PFD_POOL.pop().shutdown(wait=True)
    return buf


def install_pfd(tier, check=False, layers=None):
    """Wrap tier.prefill_call (after PF-A's install, before the screen's PFSTAT wrapper). Falls back to the wrapped call
    for PF-A's layer-major prefill (LMC on), the grouped prefill, a layer without a sidecar, or a layout other than 2."""
    if getattr(tier, "_pfd_installed", False):
        return
    orig = tier.prefill_call
    from bigmoe.stream import PIECES
    V3 = os.environ.get("R57_PFD_V", "3") == "3"         # R57_PFD_V=2: v2 (gathers + fused rebuild + scatter)
    V4 = os.environ.get("R57_PFD_V") == "4"              # v4 exact: absent bases read raw, rebuilt in the assemble pass
    FASTP = os.environ.get("R57_PFD_FAST") == "1"        # FAST (lossy, gated): side-absent experts at 3 bits, MSB-only reads
    CC = os.environ.get("R57_PFD_CC", "1") != "0"        # =0: keep MLX's buffer cache across layer calls (no re-faulting)
    IDLE = os.environ.get("R57_PFD_IDLE") == "1"         # =1: installed but inactive (decode-slowdown diagnostic)
    AHEAD = os.environ.get("R57_PFD_AHEAD") == "1"       # v4 / FAST: read the next layer's predicted misses during this layer
    LAYERS = list(layers or [])
    if AHEAD:
        assert LAYERS and all(hasattr(l_.mlp, "gate") for l_ in LAYERS), "R57_PFD_AHEAD needs install_pfd(layers=...) with mlp.gate"
    STAGE, GRAVE, LAST = {}, [], [-1]                    # per layer: {"B" / "C": staging}; arrays with reads still landing

    def pread_into(t_, e, off, size, dst):
        fd = _PFD_FD.get(t_["path"])
        if fd is None:
            fd = _PFD_FD[t_["path"]] = os.open(t_["path"], os.O_RDONLY)
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        mv, base, got = memoryview(dst), int(e) * int(t_["recsz"]) + off, 0
        while got < size:
            n_ = os.preadv(fd, [mv[got:]], base + got)
            if n_ <= 0:
                raise OSError(f"PF-D: short read of record {int(e)} at {base + got}")
            got += n_

    def pool_():
        if not _PFD_POOL:
            _PFD_POOL.append(ThreadPoolExecutor(max_workers=int(os.environ.get("R57_PFD_THREADS", "12"))))
        return _PFD_POOL[0]

    def stage_reads(t_, ids, off, size, spare):
        """an MLX staging array (len(ids) + spare rows of size / 4 words); the preads land in its buffer (zero-copy)."""
        n = len(ids)
        arr = mx.zeros((n + spare, size // 4), mx.uint32)
        mx.eval(arr)
        view = np.asarray(memoryview(arr)).view(np.uint8).reshape(n + spare, size)
        futs = {e: pool_().submit(pread_into, t_, e, off, size, view[k]) for k, e in enumerate(ids)}
        return [arr, {e: k for k, e in enumerate(ids)}, futs, view, n]

    def gather(stage, t_, ids, off, size, tag):
        """rows of ids in one staging array: prefetched ones + the rest read now into its spare rows (or a second array,
        concatenated). Waits only for the needed reads; arrays with unneeded reads in flight stay referenced in GRAVE."""
        ids = [int(e) for e in ids]
        if stage is None:
            arr, emap, futs, view, free = stage_reads(t_, ids, off, size, 0)
            PFD["m" + tag] += len(ids)
        else:
            arr, emap, futs, view, free = stage
            miss = [e for e in ids if e not in emap]
            PFD["h" + tag] += len(ids) - len(miss)
            PFD["m" + tag] += len(miss)
            if miss and free + len(miss) <= arr.shape[0]:
                for e in miss:
                    emap[e] = free
                    futs[e] = pool_().submit(pread_into, t_, e, off, size, view[free])
                    free += 1
            elif miss:
                a2 = stage_reads(t_, miss, off, size, 0)
                for e in ids:
                    f_ = futs.get(e) or a2[2].get(e)
                    if f_ is not None:
                        f_.result()
                GRAVE.append((arr, [f_ for e_, f_ in futs.items() if e_ not in ids]))
                n0 = int(arr.shape[0])
                arr = mx.concatenate([arr, a2[0]], axis=0)
                for e in miss:
                    emap[e] = n0 + a2[1][e]
                futs = {}
        for e in ids:
            f_ = futs.get(e)
            if f_ is not None:
                f_.result()
        rest = [f_ for e_, f_ in futs.items() if e_ not in set(ids)]
        if rest:
            GRAVE.append((arr, rest))
        return arr, np.array([emap[e] for e in ids], np.int32)

    def predict(L1, x, k):
        """the next layer's router on this layer's MoE input (R-57's forecast): its predicted misses start reading now."""
        lay_ = LAYERS[L1]
        t1 = (getattr(lay_.mlp.switch_mlp, "_bm", None) or {}).get("tier")
        if t1 is None or t1.get("side") is None or t1.get("layout") != 2:
            return
        top = mx.argpartition(-lay_.mlp.gate(x), kth=k - 1, axis=-1)[..., :k]
        pred = np.unique(np.array(top))
        u = pred.astype(np.int32).tolist()
        sb1 = np.array(t1["ext"].lookup(L1, u), np.int64)
        ss1 = np.array(t1["ext"].lookup(t1["side"], u), np.int64)
        Bp, Cp = pred[(sb1 >= 0) & (ss1 < 0)], pred[sb1 < 0]
        m1, r1 = int(t1["msb_sz"]), int(t1["recsz"])
        st = {}
        if len(Bp) and not FASTP:
            st["B"] = stage_reads(t1, [int(e) for e in Bp], m1, r1 - m1, max(8, len(Bp) // 2))
        if len(Cp):
            st["C"] = stage_reads(t1, [int(e) for e in Cp], 0, m1 if FASTP else r1, max(8, len(Cp) // 2))
        STAGE[L1] = st
        PFD["pB"] += 0 if FASTP else len(Bp)
        PFD["pC"] += len(Cp)

    def v4_scratch(t_, bm_, x, k, uniq, sb, ss, A, B, C):
        msb_, rec_ = int(t_["msb_sz"]), int(t_["recsz"])
        L = int(bm_["L"])
        mine = {}
        GRAVE[:] = [(a_, fs) for a_, fs in GRAVE if not all(f_.done() for f_ in fs)]
        if AHEAD:
            if L <= LAST[0]:                             # a new pass over the layers (next prompt / chunk): stale stages out
                for st_ in STAGE.values():
                    for v in st_.values():
                        GRAVE.append((v[0], list(v[2].values())))
                STAGE.clear()
            LAST[0] = L
            mine = STAGE.pop(L, {})
            tq = time.perf_counter()
            if L + 1 < len(LAYERS):
                predict(L + 1, x, k)
            PFD["tPred"] += time.perf_counter() - tq
        ph = v4_placeholder()                            # >= 8 elements: 'device', as the kernels need (23:5x)
        rl = rr = rm5 = ph
        t0 = time.perf_counter()
        rowB = rowC = None
        if len(B) and not FASTP:
            rl, rowB = gather(mine.get("B"), t_, uniq[B], msb_, rec_ - msb_, "B")
        t1 = time.perf_counter()
        if len(C):
            arr, rowC = gather(mine.get("C"), t_, uniq[C], 0, msb_ if FASTP else rec_, "C")
            if FASTP:
                rm5 = arr
            else:
                rr = arr
        t2 = time.perf_counter()
        for st_ in (mine.get("B"), mine.get("C")):       # this layer's own stage: unneeded prefetches may still be landing
            if st_ is not None:
                GRAVE.append((st_[0], [f_ for f_ in st_[2].values() if not f_.done()]))
        PFD["tB"] += t1 - t0
        PFD["tC"] += t2 - t1
        rm = np.zeros((len(uniq), 3), np.int32)
        rm[A, 1], rm[A, 2] = sb[A], ss[A]
        if FASTP:
            rm[B, 0], rm[B, 1] = 4, sb[B]
        elif len(B):
            rm[B, 0], rm[B, 1], rm[B, 2] = 1, sb[B], rowB
        if len(C):
            rm[C, 0], rm[C, 1] = (5 if FASTP else 3), rowC
        scrC = [mx.zeros((1, *sh), dt) for sh, dt in t_["shapes4"]]
        scr = assemble_v4(t_, rm, t_["base"], t_["sideP"], rl, scrC, rr, rm5)
        PFD["k4"] += int((rm[:, 0] == 4).sum())
        PFD["k5"] += int((rm[:, 0] == 5).sum())
        return scr, rm


    def prefill_call(cls, self, x, indices, bm, *a, **kw):
        tp_ = time.perf_counter()
        if IDLE:                                         # diagnostic (18:3x): installed, every call takes the wrapped path
            PFD["fallback"] += 1
            return orig(cls, self, x, indices, bm, *a, **kw)
        t = bm["tier"]
        ids = np.array(indices)
        uniq = np.unique(ids)
        G = getattr(tier, "_PF_GROUP", 0)
        if (LMC["on"] or t.get("layout") != 2 or t.get("sideP") is None or t.get("side") is None
                or (G and len(uniq) > G and ids.size >= 16 and ids.size >= 4 * len(uniq))):
            PFD["fallback"] += 1
            return orig(cls, self, x, indices, bm, *a, **kw)
        tier.STATS["prefill_calls"] += 1
        u32 = uniq.astype(np.int32).tolist()
        sb = np.array(t["ext"].lookup(bm["L"], u32), np.int64)
        ss = np.array(t["ext"].lookup(t["side"], u32), np.int64)
        rows = np.arange(len(uniq))
        A, B, C = rows[(sb >= 0) & (ss >= 0)], rows[(sb >= 0) & (ss < 0)], rows[sb < 0]
        msb, lsz = int(t["msb_sz"]), int(t["recsz"]) - int(t["msb_sz"])
        if V4 or FASTP:                                  # v4 / FAST: zero-copy staged reads (+ read-ahead), one assemble pass
            tier._reuse_drop()
            scr, rm = v4_scratch(t, bm, x, int(indices.shape[-1]), uniq, sb, ss, A, B, C)
            t4_ = time.perf_counter()
            mx.eval(*scr)
            PFD["tG"] += time.perf_counter() - t4_
            if check:
                _pfd_check_rows(tier, t, uniq, scr, rm[:, 0])
            A_B = ()
        t0_ = time.perf_counter()
        lsb = read_parts(t, uniq[B], msb, lsz) if len(B) and not (V4 or FASTP) else None
        t1_ = time.perf_counter()
        if not (V4 or FASTP):
            tier._reuse_drop()
        if V4 or FASTP:
            pass
        elif V3:                                         # v3: C rows into their own small scratch, then one assemble pass
            nC = len(C)
            scrC = [mx.zeros((max(nC, 1), *sh), dt) for sh, dt in t["shapes4"]]
            mx.eval(*scrC)
            t2_ = time.perf_counter()
            if nC:
                tier._read_rebuilt(t, uniq[C].astype(np.int32).tolist(), list(range(nC)), scrC)
            t3_ = time.perf_counter()
            rm = np.zeros((len(uniq), 3), np.int32)
            rm[A, 1], rm[A, 2] = sb[A], ss[A]
            rm[B, 0], rm[B, 1], rm[B, 2] = 1, sb[B], np.arange(len(B))
            rm[C, 0], rm[C, 1] = 2, np.arange(nC)
            if lsb is not None:
                bl1s, bbs = lsb_pieces(t, lsb)
            else:                                        # never read (no kind 1 rows): one-row placeholders
                bl1s = [mx.zeros((1, *t["sideP"][p_]["l1"].shape[1:]), mx.uint32) for p_ in ("gate_proj", "up_proj", "down_proj")]
                bbs = [mx.zeros((1, *t["sideP"][p_]["b"].shape[1:]), mx.uint32) for p_ in ("gate_proj", "up_proj", "down_proj")]
            scr = assemble(t, rm, t["base"], t["sideP"], bl1s, bbs, scrC)
            t4_ = time.perf_counter()
            mx.eval(*scr)                                # the assemble pass, timed apart from the call
            PFD["tG"] += time.perf_counter() - t4_
            PFD["tZ"] += t2_ - t1_
            PFD["tB"] += t1_ - t0_
            PFD["tC"] += t3_ - t2_
            if check:
                _pfd_check_rows(tier, t, uniq, scr, rm[:, 0])
            del scrC, bl1s, bbs
            A_B = ()
        else:
            scr = [mx.zeros((len(uniq), *sh), dt) for sh, dt in t["shapes4"]]
            mx.eval(*scr)
            t2_ = time.perf_counter()
            if len(C):                                   # C: today's path, into those rows
                tier._read_rebuilt(t, uniq[C].astype(np.int32).tolist(), C.astype(np.int32).tolist(), scr)
            t3_ = time.perf_counter()
            PFD["tB"] += t1_ - t0_
            PFD["tZ"] += t2_ - t1_
            PFD["tC"] += t3_ - t2_
            A_B = np.concatenate([A, B])
        AB = A_B
        if len(AB):
            gi = mx.array(sb[AB].astype(np.int32))
            w3s = [mx.take(t["base"][p_]["w3"], gi, axis=0) for p_ in ("gate_proj", "up_proj", "down_proj")]
            s3s = [mx.take(t["base"][p_]["s3"], gi, axis=0) for p_ in ("gate_proj", "up_proj", "down_proj")]
            l1s, bs = [[], [], []], [[], [], []]
            if len(A):
                si = mx.array(ss[A].astype(np.int32))
                for p, p_ in enumerate(("gate_proj", "up_proj", "down_proj")):
                    l1s[p].append(mx.take(t["sideP"][p_]["l1"], si, axis=0))
                    bs[p].append(mx.take(t["sideP"][p_]["b"], si, axis=0))
            if len(B):
                l1B, bB = lsb_pieces(t, lsb)
                for p in range(3):
                    l1s[p].append(l1B[p])
                    bs[p].append(bB[p])
            cat = lambda v: v[0] if len(v) == 1 else mx.concatenate(v, axis=0)
            reb = rebuild_parts(t, w3s, s3s, [cat(v) for v in l1s], [cat(v) for v in bs])
            if check:
                _pfd_check(tier, t, uniq[AB], reb, len(A))
            ix = mx.array(AB.astype(np.int32))
            for i in range(len(scr)):
                scr[i][ix] = reb[i]
            del reb, w3s, s3s, l1s, bs
            t4_ = time.perf_counter()
            mx.eval(*scr)                                # the rebuild + scatter, timed apart from the call
            PFD["tG"] += time.perf_counter() - t4_
        PFD["calls"] += 1
        PFD["A"] += len(A)
        PFD["B"] += len(B)
        PFD["C"] += len(C)
        PFD["ssd"] += (0 if FASTP else len(B) * lsz) + len(C) * (msb if FASTP else int(t["recsz"]))
        compact = np.searchsorted(uniq, ids).astype(np.int32)
        saved = [getattr(getattr(self, proj), field) for proj, field in PIECES]
        for (proj, field), arr in zip(PIECES, scr):
            setattr(getattr(self, proj), field, arr)
        try:
            t5_ = time.perf_counter()
            y = cls.__call__(self, x, mx.array(compact).astype(indices.dtype), *a, **kw)
            mx.eval(y)
            PFD["tM"] += time.perf_counter() - t5_
        finally:
            for (proj, field), arr in zip(PIECES, saved):
                setattr(getattr(self, proj), field, arr)
        del scr
        if CC:
            mx.clear_cache()
        PFD["tP"] += time.perf_counter() - tp_
        return y

    tier.prefill_call = prefill_call
    tier._pfd_installed = True
    tier._pfd_dbg = dict(stage_reads=stage_reads, gather=gather, grave=GRAVE)   # unit test (pfd_unit.py) of the staging


def _pfd_check_rows(tier, t, uniq, scr, kinds):
    """v3 R57_PFD_CHECK: every row of the assembled scratch vs tier._read_rebuilt of the same experts; differing rows by kind."""
    ref = [mx.zeros((len(uniq), *sh), dt) for sh, dt in t["shapes4"]]
    mx.eval(*ref)
    tier._read_rebuilt(t, uniq.astype(np.int32).tolist(), list(range(len(uniq))), ref)
    bad = np.zeros(len(uniq), bool)
    for g, r_ in zip(scr, ref):
        gv = np.array(g.view(mx.uint16) if g.dtype == mx.bfloat16 else g).reshape(len(uniq), -1)
        rv = np.array(r_.view(mx.uint16) if r_.dtype == mx.bfloat16 else r_).reshape(len(uniq), -1)
        bad |= (gv != rv).any(axis=1)
    PFD["checked"] += len(uniq)
    PFD["bad_A"] += int(bad[kinds == 0].sum())
    PFD["bad_B"] += int(bad[kinds == 1].sum())
    PFD["bad_C"] += int(bad[(kinds == 2) | (kinds == 3)].sum())


def _pfd_check(tier, t, ids, reb, nA):
    """R57_PFD_CHECK: the same rows from tier._read_rebuilt; counts rows of A (pool side) and B (read side) that differ."""
    ref = [mx.zeros((len(ids), *sh), dt) for sh, dt in t["shapes4"]]
    mx.eval(*ref)
    tier._read_rebuilt(t, ids.astype(np.int32).tolist(), list(range(len(ids))), ref)
    bad = np.zeros(len(ids), bool)
    for g, r_ in zip(reb, ref):
        gv = np.array(g.view(mx.uint16) if g.dtype == mx.bfloat16 else g).reshape(len(ids), -1)
        rv = np.array(r_.view(mx.uint16) if r_.dtype == mx.bfloat16 else r_).reshape(len(ids), -1)
        bad |= (gv != rv).any(axis=1)
    PFD["checked"] += len(ids)
    PFD["bad_A"] += int(bad[:nA].sum())
    PFD["bad_B"] += int(bad[nA:].sum())
