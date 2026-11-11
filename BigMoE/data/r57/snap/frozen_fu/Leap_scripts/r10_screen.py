"""R-10 DCOMP (NewIdeas' rapid screen, Mon 2026-10-26; Leap, pre-registered 20:51 EET in ~/Leap/TABLE_350.md 'R-10'):
the drafter compiled. R-06: DFlash2's draft (L1-20 weights, two-stage exact top-16) costs 10.92 ms at block 16 in bf16
and 9.64 in 4-bit, so its bytes are not its cost; ~125 small kernels per layer (the grouped dynamic convolution alone
is ~23 per call, 4 calls per layer) plus Python graph building are.
Arms (block 16 and 32; the attention core stays eager: its shapes and RoPE offsets change every check):
  E   eager (R-06's 2s arm)
  C   mx.compile of each layer's fixed-shape parts (norm + conv prepare + q / proposal k, v; o_proj + conv finish +
      MLP part) and of the tail (final norm, head, two-stage top-16, selector chain) as one function
  P   C with the convolution as shifted views of one zero-padded copy (the same ops in the same order, no concatenate
      per offset)
  E4 / P4  the same on the 4-bit drafter (R-06's fastest within 1 %)
Draft ms: the whole draft as one graph, one sync, median of 30 after 5 on one held-out sample; build ms = until the
lazy graph is built. Tokens per check: refine19's teacher-forced replay on the 24 held-out HumanEval samples.
Usage: [BigMoE venv] r10_screen.py -> ~/Leap/data/r10_screen.txt"""
import glob, os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_fu/Leap_third_party"))
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

RES, DRAFTS = {}, {}
for bits, arms in ((0, ("E", "C", "P")), (4, ("E", "P"))):
    d, own = load(bits)
    tag = "4-bit" if bits else "bf16"
    say(f"  drafter {tag}: {own / 1e9:.2f} GB of its own parameters")
    for arm in arms:
        DF._grouped_dynamic_convolve = ORIG_CONV
        fn = eager_fn(d) if arm == "E" else compiled_fn(d, conv_pad if arm == "P" else ORIG_CONV)
        name = arm + ("4" if bits else "")
        for bs in (16, 32):
            ts, tb, drafts = [], [], []
            walk(d, bs, samples[0], timed(fn, ts, tb, drafts), limit=35)
            t0 = time.time()
            tok = blk = 0
            for s in samples:
                x, y = walk(d, bs, s, lambda b_, c_, ca: np.array(fn(b_, c_, ca)[0]).tolist())
                tok += x; blk += y
            RES[(name, bs)] = (med(ts[5:]), tok / blk, med(tb[5:]))
            DRAFTS[(name, bs)] = drafts
            same = "" if name in ("E", "E4") else \
                f"; drafts on sample 0 {'bitwise = ' if drafts == DRAFTS[('E4' if bits else 'E', bs)] else 'DIFFER from '}eager"
            say(f"    {name:2s} block {bs}: draft {med(ts[5:]):6.2f} ms (min {min(ts[5:]):.2f}, graph build {med(tb[5:]):.2f}); "
                f"{tok / blk:.3f} tokens per check ({blk} checks, {time.time() - t0:.0f} s){same}")
    DF._grouped_dynamic_convolve = ORIG_CONV
    del d, fn
    mx.clear_cache()

for name in ("E", "C", "P", "E4", "P4"):
    r16, r32 = RES[(name, 16)], RES[(name, 32)]
    say(f"  {name:2s}: b16 {r16[0]:6.2f} ms, tokens x{r16[1] / RES[('E', 16)][1]:.4f}; "
        f"b32 {r32[0]:6.2f} ms, tokens x{r32[1] / RES[('E', 32)][1]:.4f}")
worst = lambda n: min(RES[(n, bs)][1] / RES[("E", bs)][1] for bs in (16, 32))
ok = sorted((RES[(n, 16)][0], n) for n in ("C", "P", "P4") if worst(n) >= 0.99)
if ok and ok[0][0] <= 6.5:
    v = f"PASS ({ok[0][1]} {ok[0][0]:.2f} ms at b16, tokens within 1 %)"
elif not ok or ok[0][0] > 8.5:
    v = "KILL" + (f" (best within 1 %: {ok[0][1]} {ok[0][0]:.2f} ms)" if ok else " (no compiled arm within 1 %)")
else:
    v = f"BETWEEN (best within 1 %: {ok[0][1]} {ok[0][0]:.2f} ms)"
say(f"R-10 verdict (PASS draft <= 6.5 ms at b16 with tokens within 1 % at b16 and b32; KILL > 8.5 ms): {v}")
open(os.path.join(D, "r10_screen.txt"), "w").write("\n".join(lines) + "\n")
