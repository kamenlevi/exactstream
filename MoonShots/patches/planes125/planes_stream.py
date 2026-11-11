"""Plane-granular expert streaming for the 125B on BigMoE's runtime (prototype; DESIGN.md). Nothing in ~/BigMoE changes:
its C++ ext streams the plane files written by make_planes.py into per-plane slot pools.
  - Each real layer L is two virtual layers: MSB planes at L (Cm slots), LSB planes at nl+L (Cl slots), so BigMoE's
    L+1..L+3 look-ahead encoding stays meaningful within each half. Gate(2*nl) with a 256-entry release ring (BigMoE: the
    default 128 recycles after ~1.3 passes at nl=96). JIT table publication off (BigMoE's advice).
  - Decode, per layer: new_pass at layer 0; MSB slots for the top-K ids (gate-rank order), LSB slots for the top-R ids;
    all K ranks run the MSB plane (2-bit, scale 4s, bias b+s*off), the top R add the LSB term (2-bit, scale s, bias
    b-bias_msb) inside gate/up/down before the activation, so ranks < R use the exact 4-bit weights (bf16 rounding).
  - Prefill: as BigMoE's _sync_call: unique ids in groups of <= 0.9*C, ext.ensure, gather into a scratch in sorted-id
    order, compact order-preserving ids.
Quality = planes125.py's mixR-msb2m (125B GSM8K-150: 145 vs 147 at R=4).   API: load_planes(model_dir, plane_dir, Cm, Cl, R)"""
import atexit, fcntl, json, os, sys
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import bigmoe.gated as G
from bigmoe.gated import ext

DTYPES = {"uint32": mx.uint32, "uint16": mx.bfloat16}          # make_planes stores bf16 as raw uint16
STATS = {"decode_calls": 0, "prefill_calls": 0}


class PlaneGate(G.Gate):
    """bigmoe.gated.Gate.__init__ with the release-region ring raised to nreg entries."""
    def __init__(self, nl, nreg=256):
        self.nl = nl
        self.route = mx.zeros((max(4096, nl * G.ROW),), mx.int32)
        self.ring = mx.zeros((4 << 20,), mx.int32)
        self.bump = mx.zeros((64 << 20,), mx.uint8) if G.MODE == "event" else mx.zeros((4096,), mx.uint8)
        self.nreg = nreg
        self.region = mx.zeros((self.nreg * G.PAIRS * (96 if G.XLINES else 64),), mx.int32)
        self.status = mx.zeros((4096,), mx.int32)
        self.nopred = mx.zeros((1,), mx.int32)
        self.ntab = 1024
        self.tables = mx.zeros((self.ntab * G.TLINES * 32,), mx.int32)
        mx.eval(self.route, self.bump, self.ring, self.region, self.status, self.tables)
        ext.start(self.route, self.ring, nl, G.MAXK)
        if G.MODE == "spin":
            ext.set_regions(self.region, G.PAIRS)
            ext.set_status(self.status)
            if G.FASTPATH:
                ext.set_tables(self.tables, G.TLINES)
        atexit.register(ext.stop)
        self.seq = 0


def _proj(xr, cids, codes, scales, bias, gs=64):
    """rows xr [N, 1, Din] through experts cids [N] (indices into codes/scales/bias), sorted for gather_qmm, unsorted back."""
    order = mx.argsort(cids)
    inv = mx.argsort(order)
    y = mx.gather_qmm(xr[order], codes, scales, bias, rhs_indices=cids[order], transpose=True, group_size=gs, bits=2,
                      sorted_indices=True)
    return y[inv]


def _glu(act, xr, cm, M, cl, Lp, sl_scales, R, K):
    """xr [T, 1, D]; cm [T, K] MSB indices into M; cl [T, R] LSB indices into Lp; sl_scales: per projection, MSB scales4
    arranged in Lp's index space. Returns [T, K, D]."""
    T = xr.shape[0]
    xm = mx.repeat(xr, K, axis=0)                                  # rows (t, k)
    xl = mx.repeat(xr, R, axis=0)                                  # rows (t, r), r < R
    fm, fl = cm.reshape(-1), cl.reshape(-1)

    def both(p, hm, hl):
        ym = _proj(hm, fm, M[p + ".codes"], M[p + ".scales4"], M[p + ".bias"])          # [T*K, 1, out]
        yl = _proj(hl, fl, Lp[p + ".codes"], sl_scales[p] / 4, Lp[p + ".bias"])        # [T*R, 1, out]
        ym = ym.reshape(T, K, *ym.shape[1:])
        yl = yl.reshape(T, R, *yl.shape[1:])
        return mx.concatenate([ym[:, :R] + yl, ym[:, R:]], axis=1)                     # [T, K, 1, out]

    up = both("up_proj", xm, xl)
    gt = both("gate_proj", xm, xl)
    h = act(up, gt)                                                                    # [T, K, 1, I]
    hm = h.reshape(T * K, *h.shape[2:])
    hl = h[:, :R].reshape(T * R, *h.shape[2:])
    return both("down_proj", hm, hl).squeeze(-2)                                       # [T, K, D]


class PlaneStreamer:
    def __init__(self, plane_dir, Cm, Cl, R):
        self.dir, self.Cm, self.Cl, self.R = plane_dir, Cm, Cl, R
        self.lay = json.load(open(os.path.join(plane_dir, "layout.json")))

    def install(self, model):
        mx.set_cache_limit(int(os.environ.get("BIGMOE_MLX_CACHE_MB", "1024")) << 20)
        layers = model.language_model.model.layers
        nl = self.nl = len(layers)
        self.gate = PlaneGate(2 * nl)
        self.pools, self.fds = {}, []
        for kind, C, base in (("msb", self.Cm, 0), ("lsb", self.Cl, nl)):
            pieces = self.lay[kind]["pieces"]
            for L in range(nl):
                fd = os.open(os.path.join(self.dir, f"L{L}.{kind}"), os.O_RDONLY)
                fcntl.fcntl(fd, fcntl.F_NOCACHE, 1); self.fds.append(fd)
                pools = [mx.zeros((C, *pc["shape"]), DTYPES[pc["dtype"]]) for pc in pieces]
                mx.eval(*pools)
                E = 512
                vl = ext.add_layer(C, E, [fd] * len(pieces), [pc["offset"] for pc in pieces], [pc["bytes"] for pc in pieces],
                                   pools)
                assert vl == base + L, (vl, base, L)
                self.pools[(kind, L)] = {pc["name"]: p for pc, p in zip(pieces, pools)}
        for L, layer in enumerate(layers):                       # the lazy 4-bit expert tensors are never read
            sw = layer.mlp.switch_mlp
            for proj in ("gate_proj", "up_proj", "down_proj"):
                lin = getattr(sw, proj)
                for f in ("weight", "scales", "biases"):
                    setattr(lin, f, mx.zeros((1,), mx.uint32 if f == "weight" else mx.bfloat16))
            object.__setattr__(layer.mlp, "_pl", (self, L))
        _patch_block()


def _decode(st, L, x, inds):
    """x [1, 1, D]; inds [1, 1, K] gate-rank order -> [1, 1, K, D]."""
    K, R, nl = inds.shape[-1], st.R, st.nl
    if L == 0:
        st.gate.new_pass()
    sm = st.gate.route_and_gate(L, inds).reshape(1, K).astype(mx.uint32)
    sl = st.gate.route_and_gate(nl + L, inds[..., :R]).reshape(1, R).astype(mx.uint32)
    M, Lp = st.pools[("msb", L)], st.pools[("lsb", L)]
    cl = mx.arange(R).astype(mx.uint32).reshape(1, R)
    # the LSB term's tensors gathered into a small R-expert set (its scales live in the MSB pool, in another slot space)
    Lr = {k: mx.take(v, sl[0], axis=0) for k, v in Lp.items()}
    ss = {p: mx.take(M[p + ".scales4"], sm[0, :R], axis=0) for p in ("gate_proj", "up_proj", "down_proj")}
    STATS["decode_calls"] += 1
    return _glu(st.act, x.reshape(1, 1, -1), sm, M, cl, Lr, ss, R, K)[None]


def _prefill(st, L, x, inds):
    """BigMoE _sync_call pattern for both planes. x [1, T, D]; inds [1, T, K] -> [1, T, K, D]."""
    T, K, R, nl = inds.shape[1], inds.shape[2], st.R, st.nl
    ids = np.array(inds)[0]                                       # [T, K]
    scr = {}
    for kind, vl, use, C in (("msb", L, ids, st.Cm), ("lsb", nl + L, ids[:, :R], st.Cl)):
        uniq = np.unique(use)
        pool = st.pools[(kind, L)]
        parts = {k: [] for k in pool}
        cap = int(C * 0.9)
        for g0 in range(0, len(uniq), cap):
            grp = uniq[g0:g0 + cap]
            slots = mx.array(np.array(ext.ensure(vl, mx.array(grp.astype(np.int32)))))
            for k, p in pool.items():
                parts[k].append(mx.take(p, slots, axis=0))
            if len(uniq) > cap:
                mx.eval(*[v[-1] for v in parts.values()])
        scr[kind] = (uniq, {k: (mx.concatenate(v, axis=0) if len(v) > 1 else v[0]) for k, v in parts.items()})
    um, M = scr["msb"]
    ul, Lp = scr["lsb"]
    cm = mx.array(np.searchsorted(um, ids).astype(np.uint32))                 # [T, K]
    cl = mx.array(np.searchsorted(ul, ids[:, :R]).astype(np.uint32))         # [T, R]
    pos = mx.array(np.searchsorted(um, ul).astype(np.int32))                  # LSB experts' rows in the MSB scratch
    ss = {p: mx.take(M[p + ".scales4"], pos, axis=0) for p in ("gate_proj", "up_proj", "down_proj")}
    STATS["prefill_calls"] += 1
    y = _glu(st.act, x.reshape(T, 1, -1), cm, M, cl, Lp, ss, R, K)[None]
    mx.eval(y)
    del scr, M, Lp
    mx.clear_cache()
    return y


def _patch_block():
    from mtplx.models import qwen4_exp
    blk = qwen4_exp.SparseMoeBlock
    if getattr(blk, "_planes_patched", False):
        return

    def block(self, x):
        st, L = self._pl
        if not hasattr(st, "act"):
            st.act = self.switch_mlp.activation
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        order = mx.argsort(-scores, axis=-1)                       # gate-rank order: the top R get the LSB plane
        inds, scores = mx.take_along_axis(inds, order, axis=-1), mx.take_along_axis(scores, order, axis=-1)
        if x.size == x.shape[-1]:
            y = _decode(st, L, x, inds)
        else:
            y = _prefill(st, L, x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(x.dtype)
        return y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)

    blk.__call__ = block
    blk._planes_patched = True


def load_planes(model_dir, plane_dir, Cm, Cl, R, mtp=False):
    """BigMoE's load_streamed with a PlaneStreamer: the 4-bit expert tensors are replaced before anything is evaluated."""
    import mlx_lm.utils as mu
    st = PlaneStreamer(plane_dir, Cm, Cl, R)
    orig = mu.load_model

    def lazy_load_model(path, *a, **kw):
        a = list(a)
        if a: a[0] = True
        else: kw["lazy"] = True
        model, cfg = orig(path, *a, **kw)
        if hasattr(model, "language_model"):
            st.install(model)
            mx.eval(model.parameters())
        return model, cfg

    mu.load_model = lazy_load_model
    try:
        from mtplx.runtime import load
        rt = load(model_dir, mtp=mtp)
    finally:
        mu.load_model = orig
    from bigmoe import fuse
    applied = fuse.apply(rt.model)          # BigMoE's env-switched dense fusions, for parity with its streamer
    if applied:
        print(f"[planes] fusions: {applied}", flush=True)
    return rt, st
