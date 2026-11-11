"""Exact SSD expert streaming for MLX MoE models (mlx-lm SwitchGLU: Qwen3.5/3.6 MoE, and
MTPLX's qwen4_exp for Qwen3.8-Flash-Next).

Each routed-expert layer keeps C slots in Metal-backed pools (one per quantized tensor:
gate/up/down x weight/scales/biases), managed by the native cache in ext/bigmoe_ext.cpp
(LRU, preadv with F_NOCACHE straight into the pools). Per layer, at decode/verify width:
  router ids -> GPU writes them to a shared row -> gate: the GPU waits on a CPU-signalled
  event -> native loader makes them resident, writes slot ids -> the original SwitchGLU
  runs on the pools with the slot ids. Same kernels, same rows: bit-identical output.
Prefill (more distinct experts than slots) takes the synchronous path in row groups.
A layer with C == E is fully resident and needs no gate at all.
"""
from __future__ import annotations
import fcntl, json, os, struct
import numpy as np
import mlx.core as mx
_SPLIT_M = int(__import__('os').environ.get('BIGMOE_SPLIT', '0'))
_SPLIT_MINN = int(__import__('os').environ.get('BIGMOE_SPLIT_MINN', '0'))
def _load_corr():                                            # [fork] BIGMOE_PF_CORR
    import os as _os, numpy as _np
    f = _os.environ.get('BIGMOE_PF_CORR')
    if not f: return None
    z = _np.load(_os.path.expanduser(f))
    import mlx.core as _mx
    return [(_mx.array(z['A'][i].astype(_np.float32)), _mx.array(z['B'][i].astype(_np.float32)),
             _mx.array(z['mu'][i]), _mx.array(z['delta'][i])) for i in range(z['A'].shape[0])]
_CORR = _load_corr()
_CFUSE = __import__('os').environ.get('BIGMOE_PF_CORR_FUSE') == '1'   # [fork] fold the correction into the router
_CW = {}
def _corr_fused(router, L, rows):                          # one addmm: rows @ (W^T + A B) + (delta - mu A B)
    w = _CW.get(L)
    if w is None:
        import mlx.core as _mx
        W = router.weight
        if hasattr(router, 'scales'):
            W = _mx.dequantize(W, router.scales, router.biases, router.group_size, router.bits)
        A, B, mu, dl = _CORR[L]; AB = A @ B
        bc = dl - mu @ AB
        if getattr(router, 'bias', None) is not None: bc = bc + router.bias.astype(_mx.float32)
        Wc = (W.astype(_mx.float32).T + AB).astype(rows.dtype); bc = bc.astype(rows.dtype)
        _mx.eval(Wc, bc); _CW[L] = w = (Wc, bc)
    return __import__('mlx.core', fromlist=['addmm']).addmm(w[1], rows, w[0])
from .gated import Gate, ext, MAXK

PIECES = [(p, f) for p in ("gate_proj", "up_proj", "down_proj") for f in ("weight", "scales", "biases")]
DT = {"U32": (np.uint32, mx.uint32), "BF16": (np.uint16, mx.bfloat16), "F16": (np.float16, mx.float16),
      "F32": (np.float32, mx.float32)}


def read_headers(model_dir):
    """tensor name -> (path, absolute byte offset, dtype, shape)"""
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for f in sorted(set(idx.values())):
        p = os.path.join(model_dir, f)
        if not os.path.exists(p):          # e.g. the vision tower, not downloaded
            continue
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                out[k] = (p, 8 + n + v["data_offsets"][0], v["dtype"], v["shape"])
    return out


_SUB = {}


_SORT_VERIFY = os.environ.get("BIGMOE_SORT_VERIFY") == "1"


def _switch_sorted(self, x, indices):
    """v2-8 (BIGMOE_SORT_VERIFY=1, v2 batch): mlx-lm SwitchGLU's expert-sorted path forced at verify widths (MLX sorts
    only at >= 64 ids), so rows sharing an expert are grouped (MoonShots: -5 % expert time at S=4). Same code as
    SwitchGLU.__call__ with do_sort=True; rows per expert stay <= 8 (no bm-64 gather, BUG1)."""
    from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort
    x = mx.expand_dims(x, (-2, -3))
    x, idx, inv_order = _gather_sort(x, indices)
    x_up = self.up_proj(x, idx, sorted_indices=True)
    x_gate = self.gate_proj(x, idx, sorted_indices=True)
    x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=True)
    return _scatter_unsort(x, inv_order, indices.shape).squeeze(-2)


def _streamed_class(cls):
    if cls in _SUB:
        return _SUB[cls]

    def __call__(self, x, indices, *a, **kw):
        bm = self._bm
        if bm["L"] == bm["first"]:
            bm["gate"].new_pass()
        ae = bm.get("async_every", 0)
        if ae and bm["L"] % ae == ae - 1:   # hand the graph built so far to the GPU while Python builds
            mx.async_eval(x)                # the remaining layers (MTPLX decodes synchronously otherwise)
        if bm["full"]:                                       # fully resident: expert id == slot id
            return cls.__call__(self, x, indices, *a, **kw)
        if indices.size <= MAXK:                             # decode / verify: sync-free exact gate
            if _STAGE["bufs"] is not None:
                _stage_drop()                                # prefill is over: free the staging buffers
            preds, counts = None, (0, 0)
            nx = bm.get("next")
            if nx is not None:                               # mid-token look-ahead: layer L+d's router on
                rows = x.reshape(-1, x.shape[-1])            # layer L's pre-MoE state (norm_{L+d}(h) =
                parts, cnt = [], []                          # norm_L(h) * w_{L+d}/w_L), top-B -> prefetch
                left = 30 + 31 * _PRED_LINES                 # prediction slots per routing row (lines 3+4, + more)
                for router, ratio, B in nx:                  # d = 1, 2, 3 as enabled
                    xin = rows * ratio if ratio is not None else rows
                    if _PF_RANKED:                           # MoonShots: one unique list per distance, ranked by
                        b = min(B, left)                     # the expert's best row log-prob (most likely first)
                        if b <= 0: break
                        lg = router(xin).astype(mx.float32)
                        if _CORR is not None and router is nx[0][0] and bm["L"] < len(_CORR):   # [fork] corrected d=1
                            _c = _CORR[bm["L"]]; lg = lg + ((rows.astype(mx.float32) - _c[2]) @ _c[0]) @ _c[1] + _c[3]
                        s = (lg - mx.logsumexp(lg, axis=-1, keepdims=True)).max(axis=0)
                        parts.append(mx.argsort(-s)[:b]); cnt.append(b)
                    else:
                        b = min(B, left // rows.shape[0])
                        if b <= 0: break
                        lg = router(xin) if not (_CFUSE and _CORR is not None and router is nx[0][0] and bm["L"] < len(_CORR)) else _corr_fused(router, bm["L"], xin)
                        if not _CFUSE and _CORR is not None and router is nx[0][0] and bm["L"] < len(_CORR):   # [fork] corrected d=1
                            _c = _CORR[bm["L"]]; lg = lg.astype(mx.float32) + ((rows.astype(mx.float32) - _c[2]) @ _c[0]) @ _c[1] + _c[3]
                        parts.append(mx.argpartition(-lg, kth=b - 1, axis=-1)[:, :b].reshape(-1))
                        cnt.append(b * rows.shape[0])
                    left -= cnt[-1]
                preds = mx.concatenate(parts) if len(parts) > 1 else parts[0]
                counts = tuple(cnt + [0] * (3 - len(cnt))) + (int(_PF_RANKED),)
            slots = bm["gate"].route_and_gate(bm["L"], indices, preds=preds, counts=counts)
            if _SORT_VERIFY and indices.size > indices.shape[-1] and hasattr(self, "activation"):
                return _switch_sorted(self, x, slots.astype(indices.dtype))
            M = _SPLIT_M
            if M <= 0 or indices.size <= _SPLIT_MINN:
                return cls.__call__(self, x, slots.astype(indices.dtype), *a, **kw)
            # [fork] split-phase MoE: phase A on (resident + dummy) slots, phase B on up to M late pairs
            yA = cls.__call__(self, x, slots.astype(indices.dtype), *a, **kw)
            dep, seq, n = bm["gate"]._last
            k, D = indices.shape[-1], yA.shape[-1]
            depB = mx.concatenate([dep[:1] + mx.isnan(yA.reshape(-1)[:1]).astype(mx.int32) * 0,   # order after y_A, NaN-safe
                                   mx.array([n, 0] * M, dtype=mx.int32)])
            lb = ext.spin_wait(depB, bm["L"], seq, 2 * M, 1)
            pos, slb = lb[0::2], lb[1::2]
            xf = x.reshape(-1, x.shape[-1])
            xpad = mx.concatenate([xf, mx.zeros((1, xf.shape[-1]), dtype=xf.dtype)])
            yB = cls.__call__(self, xpad[pos // k][None], slb[:, None][None].astype(indices.dtype), *a, **kw)[0, :, 0, :]
            yf = mx.concatenate([yA.reshape(n, D), mx.zeros((1, D), dtype=yA.dtype)])
            yf[pos] = yB.astype(yf.dtype)
            return yf[:n].reshape(yA.shape)
        return _sync_call(cls, self, x, indices, bm, *a, **kw)

    sub = type(cls.__name__ + "Streamed", (cls,), {"__call__": __call__})
    _SUB[cls] = sub
    return sub


SYNC = {"calls": 0, "rows": 0}
# PF1 (BIGMOE_PF_PREFILL=1, MoonShots): while layer L's prefill call computes, all of layer L+1's non-resident experts
# stream into one of two staging buffers; L's scratch pool is gathered from [pool; staging] in the same sorted order.
_PF_PREFILL = os.environ.get("BIGMOE_PF_PREFILL") == "1"
_PF_RANKED = os.environ.get("BIGMOE_PF_RANKED") == "1"   # MoonShots scheduler: ranked unique look-ahead lists
_PRED_LINES = max(0, min(4, int(os.environ.get("BIGMOE_PRED_LINES", "1"))))   # = gated.PRED_LINES
_STAGE = {"bufs": None, "pending": {}, "nl": 0, "t": {}}
import time as _time


def _tick(k, t0):
    _STAGE["t"][k] = _STAGE["t"].get(k, 0.0) + _time.perf_counter() - t0
    return _time.perf_counter()


def _stage_drop():
    for b in (0, 1):
        ext.stage_wait(b)
    _scr_adopt(_STAGE.get("scr"))                       # the last layer of the prefill
    _STAGE["scr"] = None; _STAGE["scrbuf"] = None
    if os.environ.get("BIGMOE_PF_PROF"):
        print("[bigmoe] PF1 prefill split (s):", {k: round(v, 3) for k, v in _STAGE["t"].items()}, flush=True)
        tl = _STAGE.pop("tl", {})
        print("[bigmoe] PF1 dense-before-router per layer (s):", " ".join(f"{L}:{tl[L]:.2f}" for L in sorted(tl)), flush=True)
    _STAGE["t"] = {}
    _STAGE["bufs"] = None; _STAGE["pending"] = {}
    mx.clear_cache()


def _stage_kick(L, pools):
    if L >= _STAGE["nl"] or L in _STAGE["pending"]:
        return
    if _STAGE["bufs"] is None:
        E = _STAGE["E"]
        _STAGE["bufs"] = [[mx.zeros((E,) + p.shape[1:], dtype=p.dtype) for p in pools] for _ in (0, 1)]
        for b in (0, 1):
            mx.eval(*_STAGE["bufs"][b]); ext.stage_bufs(b, _STAGE["bufs"][b], E)
    b = L % 2
    ext.stage_wait(b)                                   # the buffer's previous reads (L-2) are done and consumed
    _STAGE["pending"][L] = (b, np.array(ext.stage_start(b, L), dtype=np.int64))
    _STAGE["t"]["staged_experts"] = _STAGE["t"].get("staged_experts", 0) + len(_STAGE["pending"][L][1])


def _sync_call_staged(cls, self, x, indices, bm, *a, **kw):
    L = bm["L"]
    t = _time.perf_counter()
    ids = np.array(indices)                             # waits for the GPU: attention etc. up to this router
    t0r = t
    t = _tick("gpu_to_router", t)
    if os.environ.get("BIGMOE_PF_PROF"):                # per layer: dense part (attention/GDN/HC/PLE) before the router
        tl = _STAGE.setdefault("tl", {}); tl[L] = tl.get(L, 0.0) + t - t0r
    SYNC["calls"] += 1; SYNC["rows"] += ids.size // ids.shape[-1]
    pools = bm["pools"]
    _stage_kick(L, pools)                               # first layer of a chunk: no overlap
    nxt = _STAGE["layers"][L + 1] if L + 1 < _STAGE["nl"] else None
    if nxt is not None:
        _stage_kick(L + 1, nxt)                         # overlaps this layer's compute
    t = _tick("kick", t)
    b, sids = _STAGE["pending"].pop(L)
    ext.stage_wait(b)
    t = _tick("stage_wait", t)
    uniq, cnt = np.unique(ids, return_counts=True)
    if _FULLSCR:
        return _staged_full(cls, self, x, indices, ids, uniq, cnt, L, b, sids, pools, t, *a, **kw)
    compact = np.searchsorted(uniq, ids).astype(np.int32)
    slots = np.array(ext.lookup(L, uniq.astype(np.int32).tolist()), dtype=np.int64)
    res = slots >= 0
    sidx = np.searchsorted(sids, uniq[~res])
    assert np.all(sids[np.minimum(sidx, len(sids) - 1)] == uniq[~res]), "staged set misses a needed expert"
    order = np.concatenate([np.nonzero(res)[0], np.nonzero(~res)[0]])
    perm = mx.array(np.argsort(order).astype(np.int32))          # sorted position -> position in [resident; staged]
    rs, ss = mx.array(slots[res].astype(np.int32)), mx.array(sidx.astype(np.int32))
    scratch = []
    for p, sb in zip(pools, _STAGE["bufs"][b]):
        parts = ([mx.take(p, rs, axis=0)] if res.any() else []) + ([mx.take(sb, ss, axis=0)] if (~res).any() else [])
        scratch.append(mx.take(mx.concatenate(parts, axis=0), perm, axis=0) if len(parts) > 1 else parts[0])
    saved = [getattr(getattr(self, proj), field) for proj, field in PIECES]
    for (proj, field), arr in zip(PIECES, scratch):
        setattr(getattr(self, proj), field, arr)
    try:
        y = cls.__call__(self, x, mx.array(compact).astype(indices.dtype), *a, **kw)
        mx.eval(y)
    finally:
        for (proj, field), arr in zip(PIECES, saved):
            setattr(getattr(self, proj), field, arr)
    t = _tick("gather+moe", t)
    del scratch
    st = ~res                                            # staged experts used, most-routed first -> pool
    pri = np.argsort(-cnt[st], kind="stable")
    ext.adopt(L, b, uniq.astype(np.int32).tolist(), uniq[st][pri].astype(np.int32).tolist(), sidx[pri].astype(np.int32).tolist())
    mx.clear_cache()
    _tick("adopt+clear", t)
    return y


_FULLSCR = os.environ.get("BIGMOE_PF_FULLSCR") == "1"   # layer-major prefill only (MoonShots lmp / PF3): reuses one scratch per layer


def _scr_adopt(sc):
    """Layer-major prefill keeps adopt off while a layer's chunks run; once the layer is done, copy its most-used
    experts (counted over all chunks) from the full scratch into the pool, so decode does not start cold."""
    if sc is None or os.environ.get("BIGMOE_PF_WARM", "1") != "1":
        return
    order = np.argsort(-sc["cnt"], kind="stable")
    used = order[sc["cnt"][order] > 0]
    ext.adopt_full(sc["L"], sc["scratch"], used.astype(np.int32).tolist(), used.astype(np.int32).tolist())


def _staged_full(cls, self, x, indices, ids, uniq, cnt, L, b, sids, pools, t, *a, **kw):
    """BIGMOE_PF_FULLSCR=1: the scratch pool holds ALL E experts in id order ([pool; staging] gathered once), and is
    reused by every chunk of the same layer while its staging is unchanged (layer-major prefill calls a layer for all
    chunks in a row). The call indexes it with the expert ids themselves, which are order-preserving like the compact
    ids, so MLX's sorted gather sees the same grouping and order -> bit-identical, with one gather per layer instead of
    two per (layer, chunk)."""
    sc = _STAGE.get("scr")
    if sc is None or sc["L"] != L or sc["sids"] is not sids:
        _scr_adopt(sc)                                   # the previous layer is done: warm its pool from the scratch
        _STAGE["scr"] = None; mx.clear_cache()
        E = _STAGE["E"]
        if _STAGE.get("scrbuf") is None:                # one full-E buffer set, reused by every layer (CPU memcpy fill)
            _STAGE["scrbuf"] = [mx.zeros((E,) + p.shape[1:], dtype=p.dtype) for p in pools]
            mx.eval(*_STAGE["scrbuf"])
        scratch = _STAGE["scrbuf"]
        ext.scratch_full(L, b, scratch, [int(e) for e in sids])
        sc = _STAGE["scr"] = dict(L=L, sids=sids, scratch=scratch, cnt=np.zeros(E, np.int64))
        t = _tick("full_scratch", t)
    sc["cnt"][uniq] += cnt
    saved = [getattr(getattr(self, proj), field) for proj, field in PIECES]
    for (proj, field), arr in zip(PIECES, sc["scratch"]):
        setattr(getattr(self, proj), field, arr)
    try:
        y = cls.__call__(self, x, indices, *a, **kw)
        mx.eval(y)
    finally:
        for (proj, field), arr in zip(PIECES, saved):
            setattr(getattr(self, proj), field, arr)
    t = _tick("gather+moe", t)
    if ext.adopt is not None:                            # chunk-major: staged experts -> pool, scratch invalid after
        res_u = np.array(ext.lookup(L, uniq.astype(np.int32).tolist())) >= 0
        st = ~res_u
        if st.any():
            sidx_u = np.searchsorted(sids, uniq[st])
            pri = np.argsort(-cnt[st], kind="stable")
            if ext.adopt(L, b, uniq.astype(np.int32).tolist(), uniq[st][pri].astype(np.int32).tolist(), sidx_u[pri].astype(np.int32).tolist()):
                _STAGE["scr"] = None
    _tick("adopt+clear", t)
    return y


_PF_GROUP = int(os.environ.get("BIGMOE_PF_GROUP", "0") or 0)


def _sync_grouped(cls, self, x, indices, ids, uniq, bm, *a, **kw):
    """BIGMOE_PF_GROUP=G (24 GB mode): the prefill MoE runs in groups of <= G experts, each on its own (row, slot)
    pairs, with a scratch of only that group's experts (G x 2.8 MB instead of up to E x 2.8 MB). Exact: per-pair
    outputs do not depend on which other pairs share the gather (MoEPrefetch split_identity, incl. the sorted path;
    BIGMOE_SORT_VERIFY), every expert's pairs stay in one call (same rows per expert, same tiles), and the pairs are
    put back at their (row, slot) positions before the model's own combine."""
    L, C = bm["L"], bm["C"]
    k = ids.shape[-1]
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    flat = ids.reshape(-1)                                   # pair p = row * k + slot
    pools = bm["pools"]
    G = min(_PF_GROUP, int(C * 0.9))
    outs, poss = [], []
    for g0 in range(0, len(uniq), G):
        grp = uniq[g0:g0 + G]
        pos = np.nonzero(np.isin(flat, grp))[0]              # ascending pair positions of this group
        slots = mx.array(np.array(ext.ensure(L, mx.array(grp.astype(np.int32)))))
        scratch = [mx.take(p, slots, axis=0) for p in pools]
        saved = [getattr(getattr(self, proj), field) for proj, field in PIECES]
        for (proj, field), arr in zip(PIECES, scratch):
            setattr(getattr(self, proj), field, arr)
        try:
            xp = mx.take(x2, mx.array((pos // k).astype(np.int32)), axis=0)
            cp = mx.array(np.searchsorted(grp, flat[pos]).astype(np.int32)).reshape(-1, 1).astype(indices.dtype)
            y = cls.__call__(self, xp, cp, *a, **kw).reshape(len(pos), -1)
            mx.eval(y)                                       # before the next group can evict these experts
        finally:
            for (proj, field), arr in zip(PIECES, saved):
                setattr(getattr(self, proj), field, arr)
        outs.append(y); poss.append(pos)
        del scratch
    order = np.argsort(np.concatenate(poss), kind="stable").astype(np.int32)
    y = mx.take(mx.concatenate(outs, axis=0), mx.array(order), axis=0).reshape(*lead, k, -1)
    mx.eval(y)
    mx.clear_cache()
    return y


def _sync_call(cls, self, x, indices, bm, *a, **kw):
    if _PF_PREFILL and "layers" in _STAGE and bm["L"] < _STAGE["nl"]:   # the streamed MTP layer takes the plain path
        return _sync_call_staged(cls, self, x, indices, bm, *a, **kw)
    """Prefill (sync on the ids). Exactness: MLX's sorted gather groups rows by index value,
    so the indices must keep the reference's expert ORDER. The needed experts are made
    resident (in groups if their union exceeds the pool), gathered on the GPU into a scratch
    pool sorted by expert id, and the call uses compact, order-preserving ids into it."""
    L, C = bm["L"], bm["C"]
    ids = np.array(indices)
    SYNC["calls"] += 1; SYNC["rows"] += ids.size // ids.shape[-1]
    uniq = np.unique(ids)                                   # sorted expert ids
    if _PF_GROUP and len(uniq) > _PF_GROUP:
        return _sync_grouped(cls, self, x, indices, ids, uniq, bm, *a, **kw)
    compact = np.searchsorted(uniq, ids).astype(np.int32)   # order-preserving
    pools = bm["pools"]
    parts = [[] for _ in pools]
    cap = int(C * 0.9)
    for g0 in range(0, len(uniq), cap):                     # stage experts into a scratch pool, in order
        grp = uniq[g0:g0 + cap]
        slots = mx.array(np.array(ext.ensure(L, mx.array(grp.astype(np.int32)))))
        for i, p in enumerate(pools):
            parts[i].append(mx.take(p, slots, axis=0))
        if len(uniq) > cap:
            mx.eval(*[pp[-1] for pp in parts])              # before the next group can evict these
    scratch = [mx.concatenate(pp, axis=0) if len(pp) > 1 else pp[0] for pp in parts]
    saved = [getattr(getattr(self, proj), field) for proj, field in PIECES]
    for (proj, field), arr in zip(PIECES, scratch):
        setattr(getattr(self, proj), field, arr)
    try:
        y = cls.__call__(self, x, mx.array(compact).astype(indices.dtype), *a, **kw)
        mx.eval(y)
    finally:
        for (proj, field), arr in zip(PIECES, saved):
            setattr(getattr(self, proj), field, arr)
    del scratch, parts
    mx.clear_cache()            # the scratch pool (up to all experts of a layer) must not linger in MLX's cache
    return y


class ExpertStreamer:
    def __init__(self, model_dir, C, prefix_fmt="language_model.model.layers.{L}.mlp.switch_mlp"):
        """C: slots per layer (int, or a list per layer; C >= E means fully resident)."""
        self.hdr = read_headers(model_dir)
        self.fds = {}
        for p in sorted({v[0] for v in self.hdr.values()}):
            fd = os.open(p, os.O_RDONLY); fcntl.fcntl(fd, fcntl.F_NOCACHE, 1); self.fds[p] = fd
        self.C, self.prefix_fmt = C, prefix_fmt
        self.pools = {}
        self._rec_dir = os.path.join(model_dir, "records")
        self._records = (json.load(open(os.path.join(self._rec_dir, "index.json")))
                         if os.environ.get("BIGMOE_RECORDS") == "1" else None)

    def install(self, model, layers=None):
        # keep MLX's buffer cache small: on a machine near its memory limit, cached (freed) buffers
        # push the GPU working set over what stays resident and every layer slows ~50x
        mx.set_cache_limit(int(os.environ.get("BIGMOE_MLX_CACHE_MB", "1024")) << 20)
        layers = layers if layers is not None else model.language_model.model.layers
        nl = len(layers)
        # BIGMOE_MTP_STREAM=1: one more gate layer (index nl) for the MTP head's experts, attached later (install_mtp)
        self.gate = Gate(nl + (1 if os.environ.get("BIGMOE_MTP_STREAM") == "1" else 0))
        for L, layer in enumerate(layers):
            sw = layer.mlp.switch_mlp
            pre = self.prefix_fmt.format(L=L)
            E = self.hdr[f"{pre}.gate_proj.weight"][3][0]
            C = self.C[L] if isinstance(self.C, (list, tuple)) else self.C
            C = min(C, E)
            pools, fds, offs, pers = [], [], [], []
            for proj, field in PIECES:
                path, off, dt, shape = self.hdr[f"{pre}.{proj}.{field}"]
                pools.append(mx.zeros((C, *shape[1:]), DT[dt][1]))
                fds.append(self.fds[path]); offs.append(off)
                pers.append(int(np.prod(shape[1:])) * np.dtype(DT[dt][0]).itemsize)
            mx.eval(*pools)
            assert ext.add_layer(C, E, fds, offs, pers, pools) == L
            if self._records is not None:                   # BIGMOE_RECORDS=1: one preadv per expert (DESIGN_record.md)
                r = self._records["layers"][str(L)]
                assert r["piece_size"] == pers and r["E"] == E, f"record layout mismatch at layer {L}"
                ext.set_records(L, os.path.join(self._rec_dir, r["file"]), r["recsz"])
            _STAGE["nl"] = nl; _STAGE["E"] = E; _STAGE.setdefault("layers", {})[L] = pools
            for (proj, field), pool in zip(PIECES, pools):
                setattr(getattr(sw, proj), field, pool)       # drops the lazy full tensors unread
            full = C == E
            if full:                                            # load everything once: slot e = expert e
                for e0 in range(0, E, MAXK):
                    ext.ensure(L, mx.arange(e0, min(E, e0 + MAXK), dtype=mx.int32))
            spare = min(int(os.environ.get("BIGMOE_SPARE", "0")), C // 4) if not full else 0   # = native spare count
            object.__setattr__(sw, "_bm", dict(L=L, C=C - spare, full=full, gate=self.gate, first=0, pools=pools,
                                               async_every=int(os.environ.get("BIGMOE_ASYNC_EVERY", "0"))))
            sw.__class__ = _streamed_class(type(sw))
            self.pools[L] = pools
        B = int(os.environ.get("BIGMOE_PREFETCH", "0"))     # look-ahead width; 0 = off
        if B > 0:
            for L in range(nl - 1):
                sw = layers[L].mlp.switch_mlp
                if sw._bm["full"] or layers[L + 1].mlp.switch_mlp._bm["full"]:
                    continue
                nxt = []
                for d, Bd in ((1, B), (2, int(os.environ.get("BIGMOE_PREFETCH2", "0"))),
                              (3, int(os.environ.get("BIGMOE_PREFETCH3", "0")))):
                    if Bd <= 0 or L + d >= nl or layers[L + d].mlp.switch_mlp._bm["full"]:
                        break                                       # lists are positional: d=3 needs d=2
                    if hasattr(layers[L], "post_attention_layernorm"):      # Qwen3.5/3.6: exact norm swap
                        w0 = layers[L].post_attention_layernorm.weight.astype(mx.float32)
                        w1 = layers[L + d].post_attention_layernorm.weight.astype(mx.float32)
                        ratio = mx.where(mx.abs(w0) > 1e-6, w1 / w0, 0.0).astype(layers[L].post_attention_layernorm.weight.dtype)
                        mx.eval(ratio)
                    else:     # qwen4_exp: MoE input is a hyper-connection mix; use layer L's MoE input as is
                        ratio = None
                    nxt.append((layers[L + d].mlp.gate, ratio, Bd))
                if nxt:
                    sw._bm["next"] = nxt
        return self

    def install_mtp(self, model, model_dir, C):
        """BIGMOE_MTP_STREAM=1 (24 GB goal): stream the MTP head's 512 experts (1.57 GB, mtp.safetensors) through the gate
        as layer nl with a C-slot pool, instead of keeping them resident. Exact: same bytes, same kernels. Every MTP call
        starts its own gate pass (draft steps run between trunk passes)."""
        mtp = getattr(model.language_model, "mtp", None) or getattr(model, "mtp", None)
        if mtp is None:
            return None
        sw = mtp.layers[0].mlp.switch_mlp
        path = os.path.join(model_dir, "mtp.safetensors")
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
        if path not in self.fds:
            fd = os.open(path, os.O_RDONLY); fcntl.fcntl(fd, 48, 1); self.fds[path] = fd   # F_NOCACHE
        pre = "mtp.layers.0.mlp.switch_mlp"
        E = h[f"{pre}.gate_proj.weight"]["shape"][0]
        C = min(C, E)
        pools, fds, offs, pers = [], [], [], []
        for proj, field in PIECES:
            v = h[f"{pre}.{proj}.{field}"]
            pools.append(mx.zeros((C, *v["shape"][1:]), DT[v["dtype"]][1]))
            fds.append(self.fds[path]); offs.append(8 + n + v["data_offsets"][0])
            pers.append(int(np.prod(v["shape"][1:])) * np.dtype(DT[v["dtype"]][0]).itemsize)
        mx.eval(*pools)
        L = ext.add_layer(C, E, fds, offs, pers, pools)
        for (proj, field), pool in zip(PIECES, pools):
            setattr(getattr(sw, proj), field, pool)           # drops the resident full tensors
        object.__setattr__(sw, "_bm", dict(L=L, C=C, full=False, gate=self.gate, first=L, pools=pools, async_every=0))
        sw.__class__ = _streamed_class(type(sw))
        self.pools[L] = pools
        mx.clear_cache()
        return L

    def resident_gb(self):
        return sum(sum(p.nbytes for p in ps) for ps in self.pools.values()) / 1e9

    def stats(self):
        return ext.stats()
