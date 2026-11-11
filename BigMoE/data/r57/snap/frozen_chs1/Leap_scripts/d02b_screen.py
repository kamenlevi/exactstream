"""D-02b DOMAINS (NewIdeas' DRAFTER TRACK D98, Kamen Wed 15:33; split by Leap 15:44; pre-registered Wed 2026-10-28 15:44 EET
in ~/Leap/TABLE_350.md 'D-02'): how far DFlash2 (L1-20 weights, bf16, block 16, the official 1-step loop, R-06's
exact two-stage top-16 head) is from the near-copy on the domains where features exist: code (dfl2d_code, 24
HumanEval+ continuations of the 125B) and math (dfl2d_math, 60 GSM8K CoTs). Teacher-forced along the 125B's own
tokens (refine19's loop). Per domain: tokens kept per 16-row check (accepted + 1), the per-token rate (accepted /
attempted drafts), the rate at each depth (P(depth d right | depths < d right)), and P(first 15 right).
A measurement for the track (no pass / kill). The >= 0.99-sure share of first misses comes with D-02a, whose extract
stores the 125B's probabilities.
Usage: [BigMoE venv] d02b_screen.py -> ~/Leap/data/d02b_screen.txt"""
import glob, os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs1/Leap_third_party"))
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


def walk_stats(d, bs, sample, fn):
    """refine19's teacher-forced loop over one sample; returns the accepted-draft count of every check"""
    f, ids, off, gen = sample
    cache = d.make_cache()
    MASK = int(d.config.mask_token_id)
    start = max(off, gen - (d.config.sliding_window - 1))
    for c in cache:
        c.offset = start
    fed, a, out, n = start, gen, [], len(ids)
    while a < off + n - 1:
        ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
        blk = mx.array([[int(ids[a - off])] + [MASK] * (bs - 1)])
        dr = np.array(fn(blk, ctx, cache)[0]).tolist()
        fed = a
        true = ids[a - off + 1:a - off + bs]
        k = 0
        while k < len(true) and dr[k] == true[k]:
            k += 1
        out.append((k, len(true)))                       # accepted drafts, drafts that could be checked here
        if (trim := cache[0].offset - a) > 0:
            DF._trim_recent_cache(cache, trim)
        a = a + k + 1
    return out

def domain(name, folder):
    smp = []
    for p in sorted(glob.glob(os.path.join(D, folder, "*.npz"))):
        z = np.load(p)
        smp.append((z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])))
    fn = eager_fn(d)
    recs = [r for s in smp for r in walk_stats(d, 16, s, fn)]
    full = [k for k, m in recs if m == 15]                # checks with all 15 drafts checkable
    kept = np.mean([k + 1 for k, m in recs])
    attempted = sum(min(k + 1, m) for k, m in recs)
    rate = sum(k for k, m in recs) / max(attempted, 1)
    reach = np.zeros(16); right = np.zeros(16)
    for k, m in recs:
        for dd in range(1, min(k + 1, m) + 1):            # depths attempted: 1 .. k + 1 (the first miss), capped at m
            reach[dd] += 1
            right[dd] += dd <= k
    per = [right[x] / reach[x] if reach[x] else float("nan") for x in range(1, 16)]
    p15 = float(np.mean([k == 15 for k in full])) if full else float("nan")
    say(f"  {name}: {len(smp)} samples, {len(recs)} checks; kept per check {kept:.2f}; per-token rate {rate:.4f}; "
        f"P(first 15 right) {p15:.3f} (of {len(full)} full checks)")
    say("    rate per depth 1-15: " + " ".join(f"{x:.3f}" for x in per))
    return kept, rate, per

d, own = load(0)
say(f"D-02b: DFlash2 L1-20 (bf16, {own / 1e9:.2f} GB), block 16, official 1-step loop, exact two-stage top-16 head")
domain("code (HumanEval+)", "dfl2d_code")
domain("math (GSM8K CoT)", "dfl2d_math")
open(os.path.join(D, "d02b_screen.txt"), "w").write("\n".join(lines) + "\n")
