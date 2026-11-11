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
                left = 61                                    # prediction slots per routing row (lines 3+4)
                for router, ratio, B in nx:                  # d = 1, then d = 2 if enabled
                    b = min(B, left // rows.shape[0])
                    if b <= 0: break
                    lg = router(rows * ratio if ratio is not None else rows)
                    parts.append(mx.argpartition(-lg, kth=b - 1, axis=-1)[:, :b].reshape(-1))
                    cnt.append(b * rows.shape[0]); left -= cnt[-1]
                preds = mx.concatenate(parts) if len(parts) > 1 else parts[0]
                counts = (cnt[0], cnt[1] if len(cnt) > 1 else 0)
            slots = bm["gate"].route_and_gate(bm["L"], indices, preds=preds, counts=counts)
            return cls.__call__(self, x, slots.astype(indices.dtype), *a, **kw)
        return _sync_call(cls, self, x, indices, bm, *a, **kw)

    sub = type(cls.__name__ + "Streamed", (cls,), {"__call__": __call__})
    _SUB[cls] = sub
    return sub


SYNC = {"calls": 0, "rows": 0}
# PF1 (BIGMOE_PF_PREFILL=1, MoonShots): while layer L's prefill call computes, all of layer L+1's non-resident experts
# stream into one of two staging buffers; L's scratch pool is gathered from [pool; staging] in the same sorted order.
_PF_PREFILL = os.environ.get("BIGMOE_PF_PREFILL") == "1"
_STAGE = {"bufs": None, "pending": {}, "nl": 0, "t": {}}
import time as _time


def _tick(k, t0):
    _STAGE["t"][k] = _STAGE["t"].get(k, 0.0) + _time.perf_counter() - t0
    return _time.perf_counter()


def _stage_drop():
    for b in (0, 1):
        ext.stage_wait(b)
    if os.environ.get("BIGMOE_PF_PROF"):
        print("[bigmoe] PF1 prefill split (s):", {k: round(v, 3) for k, v in _STAGE["t"].items()}, flush=True)
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
    t = _tick("gpu_to_router", t)
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


def _sync_call(cls, self, x, indices, bm, *a, **kw):
    if _PF_PREFILL and "layers" in _STAGE:
        return _sync_call_staged(cls, self, x, indices, bm, *a, **kw)
    """Prefill (sync on the ids). Exactness: MLX's sorted gather groups rows by index value,
    so the indices must keep the reference's expert ORDER. The needed experts are made
    resident (in groups if their union exceeds the pool), gathered on the GPU into a scratch
    pool sorted by expert id, and the call uses compact, order-preserving ids into it."""
    L, C = bm["L"], bm["C"]
    ids = np.array(indices)
    SYNC["calls"] += 1; SYNC["rows"] += ids.size // ids.shape[-1]
    uniq = np.unique(ids)                                   # sorted expert ids
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

    def install(self, model, layers=None):
        # keep MLX's buffer cache small: on a machine near its memory limit, cached (freed) buffers
        # push the GPU working set over what stays resident and every layer slows ~50x
        mx.set_cache_limit(int(os.environ.get("BIGMOE_MLX_CACHE_MB", "1024")) << 20)
        layers = layers if layers is not None else model.language_model.model.layers
        nl = len(layers)
        self.gate = Gate(nl)
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
                for d, Bd in ((1, B), (2, int(os.environ.get("BIGMOE_PREFETCH2", "0")))):
                    if Bd <= 0 or L + d >= nl or layers[L + d].mlp.switch_mlp._bm["full"]:
                        continue
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

    def resident_gb(self):
        return sum(sum(p.nbytes for p in ps) for ps in self.pools.values()) / 1e9

    def stats(self):
        return ext.stats()
