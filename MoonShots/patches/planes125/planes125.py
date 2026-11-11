"""MoonShots patch over BigMoE's streamed 125B MoE (decode only): gate-rank-aware nested bit-planes, mix<R>-msb2m.
Each token's top-R of its K=10 routed experts (by gate) run on the exact 4-bit slot weights; ranks R..K-1 run on the MSB
plane of the same slots (2-bit codes q>>2, scale 4s, bias b + s*mean(q&3) per group), sliced on the fly from the 4-bit
slot pool. Streaming, residency and the gate are untouched; prefill stays exact. Quality test only: the slicing costs time
here, a real runtime would store the MSB and LSB planes separately and fetch the LSB plane only for the top-R ranks.
On the 35B (scripts/gsm8k_planes35.py) mix4-msb2m of 8 ranks: ppl +1.0 %, GSM8K-150 144 vs 145 (147/150 answers equal).
install(R): call after bigmoe.loader.load_streamed (it wraps the streamed switch's base class and the MoE block)."""
import mlx.core as mx
from mtplx.models import qwen4_exp

STATS = {"mixed": 0}


@mx.compile
def _msb_pack(w):
    """4-bit packed words [..., K/8] -> (MSB 2-bit packed words [..., K/16], per-word sum of the 8 low 2-bit parts).
    Bit ops on the packed words (no unpacking): 2-bit value v of a word sits at bits 2v, so 4-bit words 2k and 2k+1 (values
    0-7 and 8-15) become the low and high halves of 2-bit word k."""
    m16 = lsum = None
    for j in range(8):
        nib = (w >> (4 * j)) & 15
        m = (nib >> 2) << (2 * j)
        m16 = m if m16 is None else m16 | m
        lsum = (nib & 3) if lsum is None else lsum + (nib & 3)
    m16 = m16.reshape(*w.shape[:-1], -1, 2)
    return m16[..., 0] | (m16[..., 1] << 16), lsum


def _msb_proj(lin, h, sl, idx):
    """lin's MSB plane for the slots `sl` (bias offset = per-group mean of the low 2 bits), applied to h."""
    w, s, b = lin.weight[sl], lin.scales[sl], lin.biases[sl]
    gs = lin.group_size
    m2, lsum = _msb_pack(w)
    off = lsum.reshape(*s.shape, gs // 8).sum(-1).astype(mx.float32) / gs
    bias = (b.astype(mx.float32) + s.astype(mx.float32) * off).astype(s.dtype)
    return mx.gather_qmm(h, m2, 4 * s, bias, rhs_indices=idx, transpose=True, group_size=gs, bits=2)


KEEP = {3: 0xEEEEEEEE, 2: 0xCCCCCCCC}            # keep the top n bits of every nibble


def _make_mask(n):
    low, keep = (1 << (4 - n)) - 1, mx.array(KEEP[n], dtype=mx.uint32)

    @mx.compile
    def f(w):                                     # (codes with the low 4-n bits cleared, per-word sum of those bits)
        lsum = None
        for j in range(8):
            part = (w >> (4 * j)) & low
            lsum = part if lsum is None else lsum + part
        return w & keep, lsum
    return f


_MASK = {n: _make_mask(n) for n in KEEP}


def _tier_proj(lin, h, sl, idx, n):
    """MSB-n tier of lin for the slots `sl` in 4-bit format (low bits cleared; bias += s * mean(cleared bits) per group):
    the exact MSB-n values, without repacking."""
    w, s, b = lin.weight[sl], lin.scales[sl], lin.biases[sl]
    gs = lin.group_size
    wm, lsum = _MASK[n](w)
    off = lsum.reshape(*s.shape, gs // 8).sum(-1).astype(mx.float32) / gs
    bias = (b.astype(mx.float32) + s.astype(mx.float32) * off).astype(s.dtype)
    return mx.gather_qmm(h, wm, s, bias, rhs_indices=idx, transpose=True, group_size=gs, bits=4)


def _selftest():
    """_msb_pack must equal the plain unpack -> (q >> 2) -> repack, and its low-bit sums must match."""
    w = mx.random.randint(0, 2 ** 31, (3, 8, 32), key=mx.random.key(7)).astype(mx.uint32) * 2 + 1
    q = ((w[..., None] >> mx.arange(0, 32, 4).astype(mx.uint32)) & 15).reshape(3, 8, -1)
    ref = ((q >> 2).reshape(3, 8, -1, 16) << mx.arange(0, 32, 2).astype(mx.uint32)).sum(-1).astype(mx.uint32)
    m2, lsum = _msb_pack(w)
    ok = mx.array_equal(m2, ref).item() and mx.array_equal(lsum, (q & 3).reshape(3, 8, -1, 8).sum(-1).astype(mx.uint32)).item()
    assert ok, "planes125: _msb_pack disagrees with the reference unpack/repack"
    for n, low in ((3, 1), (2, 3)):
        wm, ls = _MASK[n](w)
        qm = ((wm[..., None] >> mx.arange(0, 32, 4).astype(mx.uint32)) & 15).reshape(3, 8, -1)
        ok = mx.array_equal(qm, q & (15 - low)).item() and mx.array_equal(ls, (q & low).reshape(3, 8, -1, 8).sum(-1).astype(mx.uint32)).item()
        assert ok, f"planes125: masked MSB-{n} tier disagrees with the reference"


def install(R, model, plan=None):
    """R: top-R ranks exact, the rest MSB-2 (packed). plan: bits per gate rank instead, e.g. "4333322222" (tiers 3 and 2
    as masked 4-bit codes)."""
    _selftest()
    segs = []
    if plan is not None:
        for r, c in enumerate(plan):
            n = int(c)
            if segs and segs[-1][2] == n:
                segs[-1][1] = r + 1
            else:
                segs.append([r, r + 1, n])
    layers = model.language_model.model.layers if hasattr(model, "language_model") else model.model.layers
    sws = [l.mlp.switch_mlp for l in layers if getattr(getattr(l, "mlp", None), "switch_mlp", None) is not None]
    sw0 = next(sw for sw in sws if getattr(sw, "_bm", None) is not None)
    base = type(sw0).__mro__[1]                   # the class bigmoe.stream._streamed_class wrapped
    for p in ("gate_proj", "up_proj", "down_proj"):
        lin = getattr(sw0, p)
        assert lin.bits == 4 and lin.group_size % 16 == 0, (p, lin.bits, lin.group_size)
    orig_sw = base.__call__

    def mixed(self, x, indices, *a, **kw):
        # only single-token decode on streamed layers; indices are slots (or expert ids on full layers), in gate-rank order
        if getattr(self, "_bm", None) is None or indices.size != indices.shape[-1]:
            return orig_sw(self, x, indices, *a, **kw)
        if segs:
            assert segs[-1][1] == indices.shape[-1], (plan, indices.shape)
            outs, xe = [], mx.expand_dims(x, (-2, -3))
            for a0, b0, n in segs:
                ids = indices[..., a0:b0]
                if n == 4:
                    outs.append(orig_sw(self, x, ids, *a, **kw))
                    continue
                sl = ids.reshape(-1)
                idx = mx.arange(sl.size).astype(mx.uint32).reshape(ids.shape)
                h = self.activation(_tier_proj(self.up_proj, xe, sl, idx, n), _tier_proj(self.gate_proj, xe, sl, idx, n))
                outs.append(_tier_proj(self.down_proj, h, sl, idx, n).squeeze(-2))
            STATS["mixed"] += 1
            return mx.concatenate([o.astype(outs[0].dtype) for o in outs], axis=-2)
        if indices.shape[-1] <= R:
            return orig_sw(self, x, indices, *a, **kw)
        hi = orig_sw(self, x, indices[..., :R], *a, **kw)
        lo_ids = indices[..., R:]
        sl = lo_ids.reshape(-1)
        idx = mx.arange(sl.size).astype(mx.uint32).reshape(lo_ids.shape)
        xe = mx.expand_dims(x, (-2, -3))
        h = self.activation(_msb_proj(self.up_proj, xe, sl, idx), _msb_proj(self.gate_proj, xe, sl, idx))
        lo = _msb_proj(self.down_proj, h, sl, idx).squeeze(-2)
        STATS["mixed"] += 1
        return mx.concatenate([hi, lo.astype(hi.dtype)], axis=-2)

    base.__call__ = mixed
    blk = qwen4_exp.SparseMoeBlock
    orig_block = blk.__call__

    def block(self, x):
        # decode: mlx-lm Qwen3NextSparseMoeBlock's math (BigMoE's eager path), with the top-k sorted by gate so the
        # streamed switch receives rank-ordered ids (route_and_gate keeps the order); everything else: the original
        if x.shape[-2] != 1 or x.size != x.shape[-1] or getattr(self.switch_mlp, "_bm", None) is None:
            return orig_block(self, x)
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        order = mx.argsort(-scores, axis=-1)
        inds, scores = mx.take_along_axis(inds, order, axis=-1), mx.take_along_axis(scores, order, axis=-1)
        y = (self.switch_mlp(x, inds) * scores[..., None]).sum(axis=-2)
        return y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)

    blk.__call__ = block
    return base
