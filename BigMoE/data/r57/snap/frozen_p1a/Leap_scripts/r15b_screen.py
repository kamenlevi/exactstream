"""R-15 step 2 (NewIdeas Tue ~22:30; Leap, pre-registered Tue 2026-10-27 22:28 EET in ~/Leap/TABLE_350.md 'R-15 step 2'):
the drafter's dense class (4.66 of 10.93 ms at b16, 4-bit; R-15 step 1) on faster kernels. Any numerics are allowed
(the drafter only proposes); tokens must stay within 1 %. At M = 16 the drafter's MLP runs MLX's slow non-NAX split-K
(gate / up 7680 x 2560 at split 2, down 2560 x 7680 at split 6), R-13's family 2.
Arms on the 4-bit drafter, eager path (R-10's E4: two-stage exact top-16 head):
  E4    as is
  D2    split-K calls -> R-13d's D2 (MLX's own sums at 16-row tiles; bit-identical)
  GU    gate | up merged into one 15,360-wide quantized matmul (480 tiles: MLX's NAX qmm, no split)
  GUD2  GU + the down projection via D2
Draft ms: R-10's timed walk on one held-out sample (median of 30 after 5), blocks 16 and 32; tokens per check:
refine19's teacher-forced replay on the 24 held-out samples.
Lines (QUEUE R-15): PASS draft <= 6.5 ms at b16 with tokens within 1 % at b16 and b32; KILL > 8.5 ms (the head is
unchanged here, so this is the 'top class swapped' half of R-15).
Usage: [BigMoE venv] r15b_screen.py -> ~/Leap/data/r15b_screen.txt"""
import glob, os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/Leap_third_party"))
import dflash_model_mlx as DF

D = os.path.expanduser("~/Leap/data")
V, K = 248320, 16
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
med = lambda v: float(np.median(v))

class Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(V, 2560, group_size=64, bits=4)
        self.lm_head = nn.QuantizedLinear(2560, V, bias=False, group_size=64, bits=4)
w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
tgt = Target()
tgt.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales", "biases")]
                 + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
del w
mx.eval(tgt.parameters())

def load(bits):
    d = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
    d.load_weights(os.path.join(D, "l120_ft.safetensors"), strict=False)
    if bits:
        nn.quantize(d, group_size=64, bits=bits,
                    class_predicate=lambda p, m: isinstance(m, nn.Linear) and (p.startswith("layers") or p == "fc"))
    own = sum(v.nbytes for _, v in tree_flatten(d.parameters()))
    d.bind(tgt)
    mx.eval(d.parameters())
    return d, own

def head2s(d, hid):
    """the exact two-stage top-16 (R-06): candidates [1, bs - 1, K] as vocabulary ids, their logits"""
    lg = d.compute_logits(hid)
    B, s = 128, lg.shape
    blk = lg.reshape(*s[:-1], s[-1] // B, B)
    tb = mx.argpartition(blk.max(axis=-1), -K, axis=-1)[..., -K:]
    sel = mx.take_along_axis(blk, mx.broadcast_to(tb[..., None], (*s[:-1], K, B)), axis=-2).reshape(*s[:-1], K * B)
    j = mx.argpartition(sel, -K, axis=-1)[..., -K:]
    c = mx.take_along_axis(tb, j // B, axis=-1) * B + j % B
    return c, mx.take_along_axis(lg, c, axis=-1)

def chain(d, blk, unary, cand, hp):
    sel = d.candidate_selector
    pred, path = blk[:, 0], []
    for p in range(hp.shape[1]):
        sc = unary[:, p] + mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None]
                                  * sel.successor_codebook(cand[:, p]), axis=-1)
        pred = mx.take_along_axis(cand[:, p], mx.argmax(sc, axis=-1)[:, None], axis=-1)[:, 0]
        path.append(pred)
    return mx.stack(path, axis=1)

def tail(d, hid, blk):
    """hid = the normed hidden of rows 1.. (hidden_states with logits_start 1)"""
    cand, unary = head2s(d, hid)
    return chain(d, blk, unary, cand, d.candidate_selector.hidden_projection(hid))

def eager_fn(d):
    return lambda blk, ctx, cache: tail(d, d.hidden_states(blk, ctx, cache, 1), blk)

ORIG_CONV = DF._grouped_dynamic_convolve
def conv_pad(hidden, dynamic, base, group_size):
    """DF._grouped_dynamic_convolve's ops in its order, on shifted views of one zero-padded copy"""
    b, L, H = hidden.shape
    g, KS = H // group_size, base.shape[0]
    blocks = hidden.reshape(b, L, g, group_size)
    dynamic = dynamic.reshape(b, L, KS, g, 1)
    pad = mx.concatenate((mx.zeros((b, KS - 1, g, group_size), hidden.dtype), blocks), axis=1)
    out = mx.zeros_like(blocks)
    for o in range(KS):
        v = pad[:, KS - 1 - o:KS - 1 - o + L]
        out = out + base[o].reshape(1, 1, g, group_size).astype(hidden.dtype) * v
        out = out + dynamic[:, :, o] * v
    return out.reshape(hidden.shape)

def attn_mid(a, q, pk, pv, x_ctx, rope, cache):
    """DFlashAttention.__call__ from the context rows on; queries and the proposal keys / values come in projected"""
    B, S = x_ctx.shape[:2]
    if a.is_sliding and S > a.sliding_window - 1:
        skip = S - (a.sliding_window - 1)
        x_ctx = x_ctx[:, skip:]
        S = x_ctx.shape[1]
        cache.offset += skip
    L = q.shape[2]
    ctx_keys = a.k_norm(a.k_proj(x_ctx).reshape(B, S, a.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    ctx_values = a.v_proj(x_ctx).reshape(B, S, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
    q = rope(q, offset=cache.offset + S)
    ctx_keys = rope(ctx_keys, offset=cache.offset)
    pk = rope(pk, offset=cache.offset + S)
    keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
    ctx_len = keys.shape[2]
    keys = mx.concatenate([keys, pk], axis=2)
    values = mx.concatenate([values, pv], axis=2)
    mask = DF.create_causal_mask(L, offset=ctx_len) if a.is_causal else None
    if a.is_sliding:
        query = ctx_len + mx.arange(L)[:, None]
        key = mx.arange(ctx_len + L)[None]
        context = (key < ctx_len) & (query - key < a.sliding_window)
        block = key >= ctx_len
        if a.is_causal:
            block = block & (key <= query)
        mask = context | block
    return mx.fast.scaled_dot_product_attention(q, keys, values, scale=a.scale, mask=mask)

def compiled_fn(d, conv):
    DF._grouped_dynamic_convolve = conv                    # looked up at trace time by prepare / finish
    fns = []
    for layer in d.layers:
        a = layer.self_attn
        def pre(x, layer=layer, a=a):
            xa, k1 = layer.attention_conv.prepare(layer.input_layernorm(x))
            B, L, _ = xa.shape
            q = a.q_norm(a.q_proj(xa).reshape(B, L, a.n_heads, -1)).transpose(0, 2, 1, 3)
            pk = a.k_norm(a.k_proj(xa).reshape(B, L, a.n_kv_heads, -1)).transpose(0, 2, 1, 3)
            pv = a.v_proj(xa).reshape(B, L, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
            return q, pk, pv, k1
        def post(x, o, k1, layer=layer, a=a):
            B, L, _ = x.shape
            x = x + layer.attention_conv.finish(a.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1)), k1)
            h, k2 = layer.mlp_conv.prepare(layer.post_attention_layernorm(x))
            return x + layer.mlp_conv.finish(layer.mlp(h), k2)
        fns.append((mx.compile(pre), mx.compile(post)))
    ctail = mx.compile(lambda h, blk: tail(d, d.norm(h[:, 1:]), blk))
    def fn(blk, ctx, cache):
        h = d.embed_tokens(blk) * d.embed_scale
        h_ctx = d.hidden_norm(d.fc(ctx))
        for layer, c, (pre, post) in zip(d.layers, cache, fns):
            q, pk, pv, k1 = pre(h)
            h = post(h, attn_mid(layer.self_attn, q, pk, pv, h_ctx, d.rope, c), k1)
        return ctail(h, blk)
    return fn

samples = []
for p in sorted(glob.glob(os.path.join(D, "dfl2d_code", "*.npz"))):
    z = np.load(p)
    samples.append((z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])))

def walk(d, bs, sample, fn, limit=None):
    """refine19's teacher-forced official loop over one sample; fn(blk, ctx, cache) -> draft list"""
    f, ids, off, gen = sample
    cache = d.make_cache()
    MASK = int(d.config.mask_token_id)
    start = max(off, gen - (d.config.sliding_window - 1))
    for c in cache:
        c.offset = start
    fed, a, tok, blocks, n = start, gen, 0, 0, len(ids)
    while a < off + n - 1 and (limit is None or blocks < limit):
        ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
        blk = mx.array([[int(ids[a - off])] + [MASK] * (bs - 1)])
        dr = fn(blk, ctx, cache)
        fed = a
        true = ids[a - off + 1:a - off + bs]
        k = 0
        while k < len(true) and dr[k] == true[k]:
            k += 1
        tok += min(k + 1, off + n - 1 - a); blocks += 1
        if (trim := cache[0].offset - a) > 0:
            DF._trim_recent_cache(cache, trim)
        a = a + k + 1
    return tok, blocks

def timed(fn, ts, tb, drafts):
    def f(blk, ctx, cache):
        mx.eval(blk, ctx); mx.synchronize()
        t0 = time.perf_counter()
        out = fn(blk, ctx, cache)
        t1 = time.perf_counter()
        dr = np.array(out[0]).tolist()
        ts.append(1e3 * (time.perf_counter() - t0)); tb.append(1e3 * (t1 - t0)); drafts.append(dr)
        return dr
    return f


sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/Leap_scripts"))
import leapk as LK
from mlx_lm.models.qwen3 import MLP
D2_ON = [False]
ORIG_QMM = mx.quantized_matmul
def qmm(x, w, *args, **kw):
    scales = args[0] if len(args) > 0 else kw.get("scales")
    biases = args[1] if len(args) > 1 else kw.get("biases")
    transpose = args[2] if len(args) > 2 else kw.get("transpose", True)
    group_size = args[3] if len(args) > 3 else kw.get("group_size")
    bits = args[4] if len(args) > 4 else kw.get("bits")
    mode = args[5] if len(args) > 5 else kw.get("mode", "affine")
    if (D2_ON[0] and transpose and bits == 4 and group_size == 64 and mode == "affine" and biases is not None
            and x.dtype == mx.bfloat16 and x.ndim in (2, 3) and (x.ndim == 2 or x.shape[0] == 1)):
        M, Kd, N = x.shape[-2], x.shape[-1], w.shape[0]
        if M == 16 and LK.routes_splitk(M, N, Kd):
            return LK.splitk_d2(x.reshape(M, Kd), (w, scales, biases)).reshape(*x.shape[:-1], N)
    return ORIG_QMM(x, w, *args, **kw)
mx.quantized_matmul = qmm

GU_ON = [False]
ORIG_MLP = MLP.__call__
def mlp_call(self, x):
    if GU_ON[0] and isinstance(self.gate_proj, nn.QuantizedLinear):
        if not hasattr(self, "_gu"):
            g, u = self.gate_proj, self.up_proj
            self._gu = tuple(mx.concatenate([g[k], u[k]], axis=0) for k in ("weight", "scales", "biases"))
            mx.eval(self._gu)
        n = self.gate_proj["weight"].shape[0]
        gu = mx.quantized_matmul(x, *self._gu, transpose=True, group_size=64, bits=4)
        return self.down_proj(swiglu(gu[..., :n], gu[..., n:]))
    return ORIG_MLP(self, x)
MLP.__call__ = mlp_call
from mlx_lm.models.activations import swiglu

ARMS = {"E4": (False, False), "D2": (True, False), "GU": (False, True), "GUD2": (True, True)}
d, own = load(4)
RES = {}
for name, (d2, gu) in ARMS.items():
    D2_ON[0], GU_ON[0] = d2, gu
    fn = eager_fn(d)
    for bs in (16, 32):
        ts, tb, drafts = [], [], []
        walk(d, bs, samples[0], timed(fn, ts, tb, drafts), limit=35)
        tok = blk = 0
        for s_ in samples:
            x_, y_ = walk(d, bs, s_, lambda b_, c_, ca: np.array(fn(b_, c_, ca)[0]).tolist())
            tok += x_; blk += y_
        RES[(name, bs)] = (med(ts[5:]), tok / blk)
        say(f"    {name:4s} block {bs}: draft {med(ts[5:]):6.2f} ms (min {min(ts[5:]):.2f}); {tok / blk:.3f} tokens per check")
    D2_ON[0] = GU_ON[0] = False
worst = lambda n: min(RES[(n, bs)][1] / RES[("E4", bs)][1] for bs in (16, 32))
ok = sorted((RES[(n, 16)][0], n) for n in ARMS if worst(n) >= 0.99)
for n in ARMS:
    say(f"  {n:4s}: b16 {RES[(n, 16)][0]:6.2f} ms, tokens x{RES[(n, 16)][1] / RES[('E4', 16)][1]:.4f}; b32 {RES[(n, 32)][0]:6.2f} ms, "
        f"tokens x{RES[(n, 32)][1] / RES[('E4', 32)][1]:.4f}")
best = ok[0] if ok else None
v = ("PASS" if best[0] <= 6.5 else "KILL" if best[0] > 8.5 else "BETWEEN") if best else "KILL (no arm within 1 %)"
say(f"R-15 step 2 verdict (PASS <= 6.5 ms at b16, tokens within 1 %; KILL > 8.5 ms): {v}"
    + (f" (best {best[1]} {best[0]:.2f} ms; E4 {RES[('E4', 16)][0]:.2f})" if best else ""))
open(os.path.join(D, "r15b_screen.txt"), "w").write("\n".join(lines) + "\n")
