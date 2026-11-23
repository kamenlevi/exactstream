"""TrainLab (Sun 25 Oct, NewIdeas B10 v2): chosen MoE layers computed densely from per-expert weight SOURCES, on the
Eval suite's engine (read-only imports), for layer-sliced quality screens (budget125_tf.py arms d<L> / x<L> / t<L>).
In a chosen layer, on verify passes only (STATE.tiers_prefill, as v3's tiers), the SparseMoeBlock computes each row's
10 experts from dense weights: rank 0-1 uses take the source's "hi" level, ranks 2-9 its "lo" level. The streamed
switch still runs on the same ids (BigMoE's pool, loader and spin gate stay in step) and its output is tied in with
mx.depends but unused. Sources (the same in-RAM layout: [512, out, in] per projection and level):
  V3Src     the engine's own 4-bit codes (st.hdr, read-only memmap): hi = 4-bit, lo = MSB-3 (the suite's tier_weights)
            -> arm d<L> must reproduce v3; all4=True (arm x<L>): hi = lo = 4-bit, layer L's share of v3's KLD
  TrSrc     B10 v2 trellis experts (Fit12): hi = base + residual (W_full), lo = base (W_base), from Fit12's decode(L, e)
            (~/Fit12/scripts/fit_b10.py, agreed 07:11), held as int8 with a scale per 64 inputs
The block math is the suite's (evalsuite.model.install's block: softmax, top-k, norm, rank sort, shared expert)."""
import importlib.util, os, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import mlx.core as mx

DENSE = {"on": False, "src": {}, "calls": 0}
PROJ = ("gate_proj", "up_proj", "down_proj")
_NP = {"U32": np.uint32, "F16": np.float16, "F32": np.float32, "BF16": np.uint16, "U8": np.uint8, "I8": np.int8}


def read_full(st, name):
    """one tensor of the bare model, whole (all 512 experts), from the streamer's header (path, offset, dtype, shape)"""
    p, off, dt, shape = st.hdr[name]
    a = np.array(np.memmap(p, dtype=_NP[dt], mode="r", offset=off, shape=tuple(shape)))
    return mx.array(a).view(mx.bfloat16) if dt == "BF16" else mx.array(a)


class V3Src:
    """the engine's own expert weights at layer L: hi = exact 4-bit, lo = MSB-3 (or 4-bit too when all4)"""

    def __init__(self, st, L, M, all4=False, gs=64):
        pre = st.prefix_fmt.format(L=L)
        self.q = {p: tuple(read_full(st, f"{pre}.{p}.{f}") for f in ("weight", "scales", "biases")) for p in PROJ}
        self.M, self.all4, self.gs = M, all4, gs
        mx.eval(*[a for t in self.q.values() for a in t])

    def gather(self, flat, hi):
        """flat: [n] expert ids; hi: True = the 'hi' level (ranks 0-1) for all of them -> {proj: [n, out, in] float32}.
        The codes are dequantized in float32 (q * scale + bias exact), the tier as the suite's tier_weights."""
        out = {}
        for p, (w, s, b) in self.q.items():
            w_, s_, b_ = w[flat], s[flat], b[flat]
            if not (hi or self.all4):
                w_, b_ = self.M.tier_weights(w_, s_, b_, self.gs, 3)
            out[p] = mx.dequantize(w_, s_.astype(mx.float32), b_.astype(mx.float32), group_size=self.gs, bits=4)
        return out


class TrSrc:
    """B10 v2 trellis experts at layer L from Fit12's decoder: hi = W_full (base + residual), lo = W_base. Every expert
    is decoded once on the CPU (workers threads) and held in RAM as int8 with a scale per gs inputs (~5.4 GB per layer
    for both levels; err: the int8 step's relative squared error per level, against sum W^2, for the log)"""

    def __init__(self, L, decode, n_experts=512, gs=64, workers=4):
        """decode(L, e) -> {proj: (W_base, W_full)}, proj 'gate' / 'up' / 'down' (or with '_proj'), each [out, in]"""
        self.gs, self.err, acc = gs, {}, {lv: [0.0, 0.0] for lv in ("base", "full")}
        lock, buf = threading.Lock(), {}

        def put(e, d=None):
            d = decode(L, e) if d is None else d
            for p in PROJ:
                for lv, W in zip(("base", "full"), d[p] if p in d else d[p.split("_")[0]]):
                    W = np.asarray(W, np.float32)
                    q, s = int8_groups(W, gs)
                    w8, sc = buf[p, lv]
                    w8[e], sc[e] = q, s
                    de = (q.reshape(s.shape + (gs,)) * s[..., None]).reshape(W.shape) - W
                    with lock:
                        acc[lv][0] += float(np.dot(de.ravel(), de.ravel()))
                        acc[lv][1] += float(np.dot(W.ravel(), W.ravel()))

        d0 = decode(L, 0)                                        # shapes from expert 0, then every expert
        for p in PROJ:
            for lv, W in zip(("base", "full"), d0[p] if p in d0 else d0[p.split("_")[0]]):
                sh = np.shape(W)
                buf[p, lv] = (np.empty((n_experts,) + sh, np.int8), np.empty((n_experts, sh[0], sh[1] // gs), np.float32))
        put(0, d0)
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(put, range(1, n_experts)))
        self.lv = {}
        for k in list(buf):                                      # to MLX one tensor at a time (peak + one tensor)
            w8, sc = buf.pop(k)
            self.lv[k] = (mx.array(w8), mx.array(sc))
            mx.eval(*self.lv[k])
            del w8, sc
        self.err = {lv: a[0] / a[1] for lv, a in acc.items()}

    def gather(self, flat, hi):
        out = {}
        for p in PROJ:
            w8, sc = self.lv[p, "full" if hi else "base"]
            w, s = w8[flat], sc[flat]                            # [n, out, in] int8, [n, out, in // gs]
            out[p] = (w.astype(mx.float32).reshape(*s.shape, self.gs) * s[..., None]).reshape(w.shape)
        return out


def int8_groups(W, gs=64):
    """W [out, in] float32 -> (int8 [out, in], float32 scales [out, in // gs]): symmetric, a scale per gs inputs"""
    g = W.reshape(W.shape[0], -1, gs)
    s = np.maximum(np.abs(g).max(-1), 1e-30) / 127.0
    return np.clip(np.rint(g / s[..., None]), -127, 127).astype(np.int8).reshape(W.shape), s.astype(np.float32)


def load_decoder(path="~/Fit12/scripts/fit_b10.py"):
    """Fit12's B10 v2 decoder, imported read-only from its script: decode(L, e) -> {proj: (W_base, W_full)}"""
    spec = importlib.util.spec_from_file_location("fit_b10", os.path.expanduser(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.decode


def _apply(W, x, k):
    """W: {proj: [rows * k, out, in] float32}, x [rows, H] -> [rows, k, H]: down(silu(gate x) * up x), float32"""
    xr = mx.repeat(x.astype(mx.float32), k, axis=0)[:, :, None]                          # [rows * k, H, 1]
    g = (W["gate_proj"] @ xr)[..., 0]
    u = (W["up_proj"] @ xr)[..., 0]
    h = (g * mx.sigmoid(g)) * u
    return (W["down_proj"] @ h[..., None])[..., 0].reshape(x.shape[0], k, -1)


def dense_experts(x, inds, src, n_hi=2):
    """x [..., rows, H], inds [..., rows, k] rank-ordered expert ids -> [..., rows, k, H] in x's dtype: SwitchGLU's
    SwiGLU expert per (row, rank), the source's hi level for ranks < n_hi, lo for the rest"""
    k, H = inds.shape[-1], x.shape[-1]
    xf, ii = x.reshape(-1, H), inds.reshape(-1, k)
    hi = _apply(src.gather(ii[:, :n_hi].reshape(-1), True), xf, n_hi)
    lo = _apply(src.gather(ii[:, n_hi:].reshape(-1), False), xf, k - n_hi)
    return mx.concatenate([hi, lo], axis=1).astype(x.dtype).reshape(*inds.shape, H)


def install_block(M):
    """wrap the suite's SparseMoeBlock call: chosen layers (DENSE['src'][L]) take dense experts on verify passes"""
    from mtplx.models import qwen4_exp
    blk = qwen4_exp.SparseMoeBlock
    if getattr(blk, "_trainlab_dense", False):
        return
    suite_block = blk.__call__

    def block(self, x):
        sw = self.switch_mlp
        bm = getattr(sw, "_bm", None)
        src = DENSE["src"].get(bm["L"]) if (DENSE["on"] and bm is not None and M.STATE.tiers_prefill) else None
        if src is None:
            return suite_block(self, x)
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        order = mx.argsort(-scores, axis=-1)
        inds, scores = mx.take_along_axis(inds, order, axis=-1), mx.take_along_axis(scores, order, axis=-1)
        y_sw = sw(x, inds)                                       # keeps the streamed pool and its loader gate in step
        y_e = mx.depends([dense_experts(x, inds, src)], [y_sw])[0]
        DENSE["calls"] += 1
        y = (y_e * scores[..., None]).sum(axis=-2)
        return y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)

    blk.__call__ = block
    blk._trainlab_dense = True
