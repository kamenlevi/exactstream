"""Round 18 PoC-1 (35B quality, NOT bit-exact): deeper nested bit-planes and gate-rank tiers. Every MSB-n tier is a prefix of
the existing 4-bit codes q: MSB-3 = q>>1 (3.5 bits/weight), MSB-2 = q>>2 (2.5), MSB-1 = q>>3 (1.5), each with scale
2^(4-n)*s and bias b + s*mean(low bits) per group of 64. Ranks are gate ranks (0 = largest gate) of a token's top-8.
  T(a,b)      : ranks [0,a) exact 4-bit, [a,b) MSB-2, [b,8) MSB-1. Memory-halving route: MSB-1 planes of every expert are
                1.5 bits/weight (125B: ~23 GB), upper planes stream for the higher ranks
  mixR-msb3   : ranks [0,R) exact, the rest MSB-3
  fallback p  : mix4-msb2m, but each top-4 use runs MSB-2 with probability p (its LSB plane arrived late; never waited for)
  dense-msb3  : every non-expert 4-bit QuantizedLinear in the layers (attention, GDN, shared expert; not the routers, not
                lm_head) at MSB-3 (4-bit codes with the low bit cleared = exactly the MSB-3 values), alone and with mix4-msb2m
Each tier's weights run all 8 ranks through gather_qmm; per (token, rank) the tier map picks one. MSB-2/MSB-1 are packed
2-bit (layout validated in expert_planes35.py); MSB-3 is 4-bit codes with bit 0 cleared.
usage: expert_tiers35.py   (GPU, ~40 GB peak: exclusive window)"""
import math, os, time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

MODEL = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit")
model, tok = load(MODEL)
tm = model.language_model.model
text = ""
for p in ("~/SpecDecode/FINDINGS.md", "~/BigMoE/FINDINGS.md", "~/MoEPrefetch/FINDINGS.md"):
    text += open(os.path.expanduser(p)).read() + "\n\n"
ids = mx.array(tok.encode(text)[:2049])[None]
inp, tgt = ids[:, :-1], ids[:, 1:]
P3 = ("gate_proj", "up_proj", "down_proj")


def score(ref_top=None):
    out = model(inp)
    lg = (out[0] if isinstance(out, tuple) else out)[0].astype(mx.float32)
    nll = -mx.take_along_axis(lg - mx.logsumexp(lg, axis=-1, keepdims=True), tgt[0][:, None], axis=-1)[:, 0]
    top = mx.argmax(lg, axis=-1)
    mx.eval(nll, top)
    agree = float((top == ref_top).astype(mx.float32).mean().item()) if ref_top is not None else 1.0
    return float(nll.mean().item()), agree, top


base_nll, _, base_top = score()
print(f"4-bit reference: ppl {math.exp(base_nll):.4f} over {inp.shape[1]} tokens", flush=True)
sws = [l.mlp.switch_mlp for l in tm.layers]
orig = [{p: (getattr(sw, p).weight, getattr(sw, p).scales, getattr(sw, p).biases, getattr(sw, p).bits, getattr(sw, p).group_size)
         for p in P3} for sw in sws]
assert all(v[3] == 4 and v[4] == 64 for o in orig for v in o.values())
SH4 = mx.arange(0, 32, 4).astype(mx.uint32)
SH2 = mx.arange(0, 32, 2).astype(mx.uint32)
CLR0 = mx.array(0xEEEEEEEE, dtype=mx.uint32)                       # clears bit 0 of every nibble


def unpack4(w):
    return ((w[..., None] >> SH4) & 15).reshape(*w.shape[:-1], -1)


def pack2(q):
    return (q.reshape(*q.shape[:-1], -1, 16) << SH2).sum(-1).astype(mx.uint32)


def msb_term(w, s, b, n):
    """the MSB-n prefix of 4-bit codes w (scales s, biases b, groups of 64) as a gather_qmm term."""
    q = unpack4(w)
    off = (q & ((1 << (4 - n)) - 1)).reshape(*s.shape, 64).astype(mx.float32).mean(-1)
    bias = (b.astype(mx.float32) + s.astype(mx.float32) * off).astype(s.dtype)
    if n == 3:
        return (w & CLR0, s, bias, 4, 64)                            # s*(q & ~1) + bias = 2s*(q>>1) + b + s*off
    return (pack2(q >> (4 - n)), (1 << (4 - n)) * s, bias, 2, 64)


def build(n):
    out = []
    for o in orig:
        d = {p: (v if n is None else msb_term(v[0], v[1], v[2], n)) for p, v in o.items()}
        mx.eval([t[:3] for t in d.values()])
        out.append(d)
    return out


def glu(sw, x, idx, T):
    x, sidx, inv = _gather_sort(mx.expand_dims(x, (-2, -3)), idx)

    def proj(p, h):
        w, s, b, bits, gs = T[p]
        return mx.gather_qmm(h, w, s, b, rhs_indices=sidx, transpose=True, group_size=gs, bits=bits, sorted_indices=True)

    h = sw.activation(proj("up_proj", x), proj("gate_proj", x))
    return _scatter_unsort(proj("down_proj", h), inv, idx.shape).squeeze(-2)


CUR = {}


def moe_call(self, x):
    gates = mx.softmax(self.gate(x), axis=-1, precise=True)
    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    order = mx.argsort(-scores, axis=-1)
    inds, scores = mx.take_along_axis(inds, order, axis=-1), mx.take_along_axis(scores, order, axis=-1)
    tiers = CUR["tiers"]
    if len(tiers) == 1:
        y = glu(self.switch_mlp, x, inds, tiers[0][self._li])
    else:
        tmap = CUR["tmap"](self._li, inds.shape)
        y = None
        for t, T in enumerate(tiers):
            Yt = mx.where((tmap == t)[..., None], glu(self.switch_mlp, x, inds, T[self._li]), 0)
            y = Yt if y is None else y + Yt
    y = (y * scores[..., None]).sum(axis=-2)
    return y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)


for li, l in enumerate(tm.layers):
    object.__setattr__(l.mlp, "_li", li)
type(tm.layers[0].mlp).__call__ = moe_call


def by_rank(*t):
    a = mx.array(t, dtype=mx.int32)
    return lambda li, shape: mx.broadcast_to(a, shape)


def fallback(p):
    return lambda li, shape: mx.where((mx.arange(shape[-1]) < 4) & (mx.random.uniform(shape=shape, key=mx.random.key(1000 + li)) >= p), 0, 1).astype(mx.int32)


def run(name, tiers, tmap=None, note=""):
    CUR.update(tiers=tiers, tmap=tmap)
    t0 = time.perf_counter()
    nll, ag, _ = score(base_top)
    print(f"{name:22s}: ppl {math.exp(nll):.4f} ({100 * (math.exp(nll - base_nll) - 1):+.2f} %, dNLL {nll - base_nll:+.4f}), "
          f"agreement {ag:.4f}  {note} ({time.perf_counter() - t0:.0f}s)", flush=True)


W4 = build(None)
run("path4", [W4], note="(harness check)")
M3 = build(3)                                                     # MSB-3 first (4-bit-sized copy), freed before the rest
run("msb3 all", [M3], note="(3.5 bits/weight)")
run("mix4-msb3", [W4, M3], by_rank(0, 0, 0, 0, 1, 1, 1, 1))
run("mix2-msb3", [W4, M3], by_rank(0, 0, 1, 1, 1, 1, 1, 1))
del M3; CUR.clear(); mx.clear_cache()
M2 = build(2)
run("mix4-msb2m", [W4, M2], by_rank(0, 0, 0, 0, 1, 1, 1, 1), "(reference from expert_planes35: +1.02 %)")
for p in (0.10, 0.25):
    run(f"mix4 fallback p={p}", [W4, M2], fallback(p), f"({int(p * 100)} % of top-4 uses MSB-2)")
M1 = build(1)
run("msb1 all", [M1], note="(1.5 bits/weight)")
run("T(4,6)", [W4, M2, M1], by_rank(0, 0, 0, 0, 1, 1, 2, 2))
run("T(4,4) mix4-msb1", [W4, M2, M1], by_rank(0, 0, 0, 0, 2, 2, 2, 2))
run("T(3,6)", [W4, M2, M1], by_rank(0, 0, 0, 1, 1, 1, 2, 2))
run("T(2,4)", [W4, M2, M1], by_rank(0, 0, 1, 1, 2, 2, 2, 2))
run("T(2,6)", [W4, M2, M1], by_rank(0, 0, 1, 1, 1, 1, 2, 2))
del M1; CUR.clear(); mx.clear_cache()
# dense part (non-expert 4-bit linears in the layers, not the routers) at MSB-3, in place
dense = [(n, m) for n, m in tm.named_modules() if isinstance(m, nn.QuantizedLinear) and "switch_mlp" not in n
         and not n.endswith("mlp.gate") and not n.endswith("shared_expert_gate") and m.bits == 4 and m.group_size == 64]
keep = [(m, m.weight, m.biases) for _, m in dense]
nb = 0
for _, m in dense:
    w, s, b, _, _ = msb_term(m.weight, m.scales, m.biases, 3)
    m.weight, m.biases = w, b
    nb += m.weight.size * 32 // 4
mx.eval([m.weight for _, m in dense] + [m.biases for _, m in dense])
print(f"dense-msb3: {len(dense)} linears, {nb / 1e9:.2f} G weights at 3.5 instead of 4.5 bits", flush=True)
run("dense-msb3", [W4], note="(experts exact)")
run("dense-msb3 + mix4-msb2m", [W4, M2], by_rank(0, 0, 0, 0, 1, 1, 1, 1))
for m, w, b in keep:
    m.weight, m.biases = w, b
