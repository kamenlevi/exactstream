"""Memory-halving hunt, round 2 (NOT bit-exact: quality is the measure). Qwen3.6-35B-A3B 4-bit, routed experts only, 2048
teacher-forced tokens of real text; per variant: perplexity, mean NLL delta and greedy top-1 agreement vs the 4-bit model.

1. Nested bit-planes, training-free: slice each existing 4-bit code q = 4m + l into an MSB plane m (2 bits, with its own
   scale 4s and bias b + off*s) and an LSB plane l (2 bits, scale s, bias 0). MSB + LSB reproduces the 4-bit weights exactly,
   so a runtime can keep the MSB plane of every expert resident and stream the LSB plane only for each token's top-R gate
   ranks: those ranks stay exactly 4-bit, the rest run on 2.5 bits/weight (vs 4.5). Offsets: msb2 = bin midpoint 1.5,
   msb2m = per-group mean of the LSB codes (same storage).
   mixR: each token's top-R of 8 ranks on the full 4-bit weights, the other ranks on msb2m (or another 2-bit set).
   Prior art: HOBBIT (2024) loads separate low-precision copies of less important experts on cache misses; MatQuant (2025)
   nests int2 in int4/int8 by MSB slicing but co-trains for it. Not found: training-free MSB/LSB planes of an existing 4-bit
   MoE with gate-rank-aware LSB streaming.
2. Cheap training-free fixes for plain RTN at 2-3 bits: clip (per-group MSE-optimal symmetric range shrink, a grid search
   around the group midpoint), rot (randomized Hadamard rotation of each projection's input dim, QuIP#-style incoherence,
   then RTN; the runtime rotates the activations: x @ Q for gate/up, h @ Q for down).
Checks: a packing unit test (MSB + LSB planes == the 4-bit dequant) and `path4`, the 4-bit weights through this script's own
MoE path (must match the reference up to summation order).
usage: expert_planes35.py   (GPU, ~35 GB peak: exclusive window)"""
import math, os, time
import numpy as np
import mlx.core as mx
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


def unpack4(w):                       # uint32 [..., K/8] -> codes [..., K] (value j of a word sits at bits 4j)
    return ((w[..., None] >> SH4) & 15).reshape(*w.shape[:-1], -1)


def pack2(q):                         # codes 0..3 [..., K] -> uint32 [..., K/16]
    return (q.reshape(*q.shape[:-1], -1, 16) << SH2).sum(-1).astype(mx.uint32)


def msb_term(w, s, b, off):           # off: None = per-group mean of the LSB codes, else a constant
    q = unpack4(w)
    if off is None:
        o = (q & 3).reshape(*s.shape, 64).astype(mx.float32).mean(-1)
        bias = (b.astype(mx.float32) + s.astype(mx.float32) * o).astype(s.dtype)
    else:
        bias = (b.astype(mx.float32) + off * s.astype(mx.float32)).astype(s.dtype)
    return (pack2(q >> 2), 4 * s, bias, 2, 64)


# packing unit test on random data: MSB plane (bias b) + LSB plane (bias 0) must give the 4-bit dequant back
t = mx.random.normal((4, 64, 256), key=mx.random.key(0)).astype(mx.bfloat16)
w_, s_, b_ = mx.quantize(t, group_size=64, bits=4)
d4 = mx.dequantize(w_, s_, b_, group_size=64, bits=4).astype(mx.float32)
q_ = unpack4(w_)
e_unpack = mx.abs(s_.astype(mx.float32)[..., None] * q_.reshape(*s_.shape, 64).astype(mx.float32)
                  + b_.astype(mx.float32)[..., None] - d4.reshape(*s_.shape, 64)).max().item()
dm = mx.dequantize(pack2(q_ >> 2), 4 * s_, b_, group_size=64, bits=2).astype(mx.float32)
dl = mx.dequantize(pack2(q_ & 3), s_, mx.zeros_like(b_), group_size=64, bits=2).astype(mx.float32)
e_planes = mx.abs(dm + dl - d4).max().item()
print(f"packing test: unpack4 max err {e_unpack:.2e}, MSB+LSB planes max err {e_planes:.2e} (values ~N(0,1))", flush=True)
assert e_unpack < 0.1 and e_planes < 0.1                        # bf16 rounding ~0.01; a packing bug gives errors ~1


def hadamard_q(n, seed):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    sg = np.random.default_rng(seed).choice([-1.0, 1.0], n)
    return mx.array(sg[:, None] * H / math.sqrt(n), dtype=mx.float32)   # Q = diag(s) Had / sqrt(n), orthogonal


QF = {2048: hadamard_q(2048, 1), 512: hadamard_q(512, 2)}
QB = {n: q.astype(mx.bfloat16) for n, q in QF.items()}


def deq32(w, s, b, bits, gs):
    return mx.dequantize(w, s, b, group_size=gs, bits=bits).astype(mx.float32)


def clip_quant(x, dt, bits, gs, grid):
    """per-group MSE-optimal symmetric shrink of [min, max] around the midpoint, then MLX affine quantization."""
    sh = x.shape
    xg = x.reshape(*sh[:-1], -1, gs)
    lo, hi = xg.min(-1, keepdims=True), xg.max(-1, keepdims=True)
    c, r = (lo + hi) / 2, (hi - lo) / 2
    best = best_a = None
    for a in grid:
        xc = mx.clip(xg, c - a * r, c + a * r).reshape(sh).astype(dt)
        qw, qs, qb = mx.quantize(xc, group_size=gs, bits=bits)
        e = ((mx.dequantize(qw, qs, qb, group_size=gs, bits=bits).astype(mx.float32).reshape(xg.shape) - xg) ** 2).sum(-1, keepdims=True)
        if best is None:
            best, best_a = e, mx.full(e.shape, a)
        else:
            m = e < best
            best, best_a = mx.where(m, e, best), mx.where(m, a, best_a)
        mx.eval(best, best_a)
    xc = mx.clip(xg, c - best_a * r, c + best_a * r).reshape(sh).astype(dt)
    return tuple(mx.quantize(xc, group_size=gs, bits=bits)) + (bits, gs)


GRID = {2: (1.0, 0.9, 0.8, 0.7, 0.6, 0.5), 3: (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)}


def build(kind):
    """per layer: {proj: [terms]}; the gather_qmm outputs of a projection's terms are summed."""
    out = []
    for o in orig:
        d = {}
        for p, (w, s, b, bits, gs) in o.items():
            if kind == "4bit":
                d[p] = [(w, s, b, bits, gs)]
            elif kind == "planes":
                d[p] = [msb_term(w, s, b, 0.0), (pack2(unpack4(w) & 3), s, mx.zeros_like(b), 2, 64)]
            elif kind in ("msb2", "msb2m"):
                d[p] = [msb_term(w, s, b, 1.5 if kind == "msb2" else None)]
            else:                                   # clip2 / clip2g32 / clip3 / rot2 / rotc2 / rot3
                x = deq32(w, s, b, bits, gs)
                nb, g = int(kind[-1]) if kind[-1].isdigit() else 2, 64
                if kind == "clip2g32":
                    nb, g = 2, 32
                if kind.startswith("rot"):
                    x = x @ QF[x.shape[-1]]
                grid = GRID[nb] if (kind.startswith("clip") or kind.startswith("rotc")) else (1.0,)
                d[p] = [clip_quant(x, s.dtype, nb, g, grid)]
            mx.eval([t[:3] for t in d[p]])
        out.append(d)
    return out


def glu(sw, x, idx, terms, rot):
    if rot:
        x = x @ QB[x.shape[-1]]
    x, sidx, inv = _gather_sort(mx.expand_dims(x, (-2, -3)), idx)

    def proj(p, h):
        y = None
        for (w, s, b, bits, gs) in terms[p]:
            o = mx.gather_qmm(h, w, s, b, rhs_indices=sidx, transpose=True, group_size=gs, bits=bits, sorted_indices=True)
            y = o if y is None else y + o
        return y

    h = sw.activation(proj("up_proj", x), proj("gate_proj", x))
    if rot:
        h = h @ QB[h.shape[-1]]
    return _scatter_unsort(proj("down_proj", h), inv, idx.shape).squeeze(-2)


CUR = {}


def moe_call(self, x):
    gates = mx.softmax(self.gate(x), axis=-1, precise=True)
    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    order = mx.argsort(-scores, axis=-1)                           # gate rank 0 = largest gate
    inds, scores = mx.take_along_axis(inds, order, axis=-1), mx.take_along_axis(scores, order, axis=-1)
    R = CUR["R"]
    y = glu(self.switch_mlp, x, inds[..., :R], CUR["hi"][self._li], CUR["hi_rot"])
    if R < k:
        y = mx.concatenate([y, glu(self.switch_mlp, x, inds[..., R:], CUR["lo"][self._li], CUR["lo_rot"])], axis=-2)
    y = (y * scores[..., None]).sum(axis=-2)
    return y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)


for li, l in enumerate(tm.layers):
    object.__setattr__(l.mlp, "_li", li)
type(tm.layers[0].mlp).__call__ = moe_call


def run(name, hi, R=8, lo=None, hi_rot=False, lo_rot=False, note=""):
    CUR.update(hi=hi, lo=lo, R=R, hi_rot=hi_rot, lo_rot=lo_rot)
    nll, ag, _ = score(base_top)
    print(f"{name:10s}: ppl {math.exp(nll):.4f} ({100 * (math.exp(nll - base_nll) - 1):+.2f} %, dNLL {nll - base_nll:+.4f}), "
          f"agreement {ag:.4f}  {note}", flush=True)


W4 = build("4bit")
run("path4", W4, note="(harness check: 4-bit through this MoE path)")
for kind, bpw in (("msb2", 2.5), ("msb2m", 2.5)):
    t0 = time.perf_counter(); S = build(kind)
    run(kind, S, note=f"(MSB plane only, {bpw} bits/weight; build {time.perf_counter() - t0:.0f}s)")
    for R in ((2, 4, 6) if kind == "msb2m" else (4,)):
        run(f"mix{R}-{kind}", W4, R=R, lo=S, note=f"(top-{R} ranks exact 4-bit = MSB+LSB, rest {kind})")
    del S; CUR.clear(); mx.clear_cache()
for kind, bpw in (("clip2", 2.5), ("clip2g32", 3.0), ("clip3", 3.5), ("rot2", 2.5), ("rotc2", 2.5), ("rot3", 3.5)):
    t0 = time.perf_counter(); S = build(kind); rot = kind.startswith("rot")
    run(kind, S, hi_rot=rot, note=f"({bpw} bits/weight; build {time.perf_counter() - t0:.0f}s)")
    if kind in ("clip2", "rotc2"):
        run(f"mix4-{kind}", W4, R=4, lo=S, lo_rot=rot, note=f"(top-4 ranks 4-bit, rest {kind})")
    del S; CUR.clear(); mx.clear_cache()
