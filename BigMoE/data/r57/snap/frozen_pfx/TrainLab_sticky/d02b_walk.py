"""MTP Head copy of /Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_scripts/d02b_screen.py for D-08 CALC (NewIdeas Wed 16:3x): the same walk, plus every check's
anchor saved per sample (data/d02b_walk_<domain>.npz: per sample ids, off, gen and checks [anchor, accepted, checkable]),
so tokens anchor+1..anchor+k are DFlash2 hits and anchor+k+1 its first miss. Output data/d02b_walk.txt.
D-02b DOMAINS (NewIdeas' DRAFTER TRACK D98, Kamen Wed 15:33; split by Leap 15:44; pre-registered Wed 2026-10-28 15:44 EET
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
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_third_party"))
import dflash_model_mlx as DF

D = os.path.expanduser("~/Leap/data")
TD = os.path.expanduser("~/TrainLab/data")
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

HC = os.environ.get("TL_HC")                         # D-16: folder of widened rows (data/d16w); the drafter reads h + proj(rmsnorm(wd))
class HCExtra(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(10240, eps=1e-6)
        self.proj = nn.Linear(10240, 2560, bias=False)
        self.proj.weight = mx.zeros_like(self.proj.weight)
    def __call__(self, x):
        return self.proj(self.norm(x))
class Ident(nn.Module):
    def __call__(self, x):
        return x
class HnPlus(nn.Module):                             # hidden_norm(fc(taps)) + hc_extra(wd) on [taps | wd] rows
    def __init__(self, fc, hn, ex):
        super().__init__()
        self.fc_, self.hn_, self.ex_ = fc, hn, ex
    def __call__(self, x):
        return self.hn_(self.fc_(x[..., :12800])) + self.ex_(x[..., 12800:]).astype(x.dtype)

def load(bits):
    d = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
    if os.environ.get("TL_EXPAND"):                  # P-CAP: the 10-layer drafter (sticky/pcap_expand.py) before its weights
        sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/TrainLab_sticky"))
        import pcap_expand
        pcap_expand.expand_interleave(d)
    if HC:
        d.hc_extra = HCExtra()
    d.load_weights(os.environ.get("TL_DRAFT", os.path.join(D, "l120_ft.safetensors")), strict=False)   # D-09: TL_DRAFT
    if bits:
        nn.quantize(d, group_size=64, bits=bits,
                    class_predicate=lambda p, m: isinstance(m, nn.Linear) and (p.startswith("layers") or p == "fc"))
    own = sum(v.nbytes for _, v in tree_flatten(d.parameters()))
    d.bind(tgt)
    mx.eval(d.parameters())
    if HC:                                           # D-16: route the context rows through HnPlus
        fc, hn = d.fc, d.hidden_norm
        d.hidden_norm = HnPlus(fc, hn, d.hc_extra)
        d.fc = Ident()
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

def chain_cf(d, blk, unary, cand, hp, prev=None, stats=None):
    """HPD (11-02): chain() plus each pick's cf (the selector's softmax over its 16 candidates at the pick, as the live
    engine's d23_switchdef.py chain); with prev (1, bs - 1) = the true tokens, the predecessor at depth p + 1 is the true
    token of depth p instead of the own pick (teacher-forced: each depth's pick no longer depends on earlier misses)"""
    sel = d.candidate_selector
    pred, path, conf = blk[:, 0], [], []
    for p in range(hp.shape[1]):
        sc = unary[:, p] + mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None]
                                  * sel.successor_codebook(cand[:, p]), axis=-1)
        s_ = mx.argmax(sc, axis=-1)
        pick = mx.take_along_axis(cand[:, p], s_[:, None], axis=-1)[:, 0]
        path.append(pick)
        sm = mx.softmax(sc.astype(mx.float32), axis=-1)
        conf.append(mx.take_along_axis(sm, s_[:, None], axis=-1)[:, 0])
        if stats is not None:                            # k-head (11-02): selector p1, p2, entropy at this pick
            ss = mx.sort(sm, axis=-1)
            stats.append(mx.stack([ss[:, -1], ss[:, -2], -(sm * mx.log(sm + 1e-12)).sum(axis=-1)], axis=-1))
        pred = pick if prev is None else prev[:, p]
    return mx.stack(path, axis=1), mx.stack(conf, axis=1)

def cf_fn(d):
    """HPD: one drafter call -> own picks, their cf, teacher-forced picks, their cf (each (1, bs - 1))"""
    def f(blk, ctx, cache, prev):
        hid = d.hidden_states(blk, ctx, cache, 1)
        cand, unary = head2s(d, hid)
        hp = d.candidate_selector.hidden_projection(hid)
        if KH:                                           # k-head (11-02): + the drafter's hidden per draft position and
            stt = []                                     # per pick [selector p1, p2, entropy, unary p1, p2, entropy]
            own = chain_cf(d, blk, unary, cand, hp, None, stt)
            us = mx.softmax(unary.astype(mx.float32), axis=-1)
            uss = mx.sort(us, axis=-1)
            ust = mx.stack([uss[..., -1], uss[..., -2], -(us * mx.log(us + 1e-12)).sum(axis=-1)], axis=-1)
            st = mx.concatenate([mx.stack(stt, axis=1), ust], axis=-1)
            return own + chain_cf(d, blk, unary, cand, hp, prev) + (hid.astype(mx.float16), st)
        return chain_cf(d, blk, unary, cand, hp) + chain_cf(d, blk, unary, cand, hp, prev)
    return f

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


CF = bool(os.environ.get("TL_WALK_CF"))             # HPD: also save every check's picks and cf (own and teacher-forced)
KH = CF and bool(os.environ.get("TL_WALK_KH"))      # k-head (11-02): + the drafter's hidden and pick stats per check

def walk_stats(d, bs, sample, fn, anchors=None, ext=None):
    """refine19's teacher-forced loop over one sample; returns the accepted-draft count of every check.
    HPD: anchors = a fixed list of check positions (another drafter's walk) instead of following acceptance; with CF,
    ext collects per check (own picks, their cf, teacher-forced picks, their cf), bs - 1 each (MASK / NaN past the end)"""
    f, ids, off, gen = sample
    cache = d.make_cache()
    MASK = int(d.config.mask_token_id)
    start = max(off, gen - (d.config.sliding_window - 1))
    for c in cache:
        c.offset = start
    fed, a, out, n = start, gen, [], len(ids)
    todo = None if anchors is None else [int(x) for x in anchors]
    if todo is not None:
        a = todo.pop(0)
    while a < off + n - 1:
        ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
        blk = mx.array([[int(ids[a - off])] + [MASK] * (bs - 1)])
        true = ids[a - off + 1:a - off + bs]
        if CF:
            prev = mx.array([[int(x) for x in true] + [MASK] * (bs - 1 - len(true))])
            r_ = fn(blk, ctx, cache, prev)
            p0, c0, p1, c1 = r_[:4]
            dr = np.array(p0[0]).tolist()
            ext.append((np.array(p0[0]), np.array(c0[0]), np.array(p1[0]), np.array(c1[0]))
                       + ((np.array(r_[4][0]), np.array(r_[5][0])) if KH else ()))
        else:
            dr = np.array(fn(blk, ctx, cache)[0]).tolist()
        fed = a
        k = 0
        while k < len(true) and dr[k] == true[k]:
            k += 1
        out.append((k, len(true), a))                       # accepted drafts, drafts that could be checked here
        if (trim := cache[0].offset - a) > 0:
            DF._trim_recent_cache(cache, trim)
        if todo is not None:
            if not todo:
                break
            a = todo.pop(0)
            continue
        a = a + 1 if os.environ.get("TL_WALK_EVERY") else a + k + 1   # D-17: a draft at every position
    return out

def domain(name, folder, tag=None, pattern=None):
    smp, names = [], []
    for p in sorted(glob.glob(pattern or os.path.join(D, folder, "*.npz"))):
        names.append(os.path.basename(p))
        z = np.load(p)
        f = z["f"].reshape(z["f"].shape[0], -1)
        if HC:                                       # D-16: append the widened rows (same rows as f)
            y = np.load(os.path.join(HC, os.path.basename(p)))
            assert int(y["off"]) == int(z["off"]) and y["wd"].shape[0] >= f.shape[0], p
            f = np.concatenate([f, y["wd"][:f.shape[0]].astype(f.dtype)], axis=1)
        smp.append((f, z["ids"], int(z["off"]), int(z["gen"])))
    fn = cf_fn(d) if CF else eager_fn(d)
    st = tag or folder.split('_')[-1]
    anc = [None] * len(smp)
    if os.environ.get("TL_WALK_ANCHORS"):              # HPD: check at another walk's anchors (its tag), same samples
        Z = np.load(os.path.join(TD, f"d02b_walk{os.environ['TL_WALK_ANCHORS']}_{st}.npz"))
        assert [str(x) for x in Z["names"]] == names, "TL_WALK_ANCHORS: different samples"
        anc = [Z[f"{i}_checks"][:, 2] for i in range(len(smp))]
    ext = [[] for _ in smp]
    per_s = [walk_stats(d, 16, s, fn, anc[i], ext[i]) for i, s in enumerate(smp)]
    extra = {}
    if CF:
        for i, e in enumerate(ext):
            for j, k in enumerate(("dtok", "dcf", "ttok", "tcf")):
                extra[f"{i}_{k}"] = np.stack([x[j] for x in e]).astype(np.int32 if k.endswith("tok") else np.float32)
            if KH:                                       # k-head: [checks, 15, 2560] f16 and [checks, 15, 6] f32
                extra[f"{i}_khid"] = np.stack([x[4] for x in e]).astype(np.float16)
                extra[f"{i}_kst"] = np.stack([x[5] for x in e]).astype(np.float32)
    np.savez(os.path.join(TD, f"d02b_walk{os.environ.get('TL_WALK_TAG', '')}_{st}.npz"), names=np.array(names),
             **{f"{i}_{k}": v for i, (s, r) in enumerate(zip(smp, per_s)) for k, v in
                (("ids", s[1]), ("off", np.int64(s[2])), ("gen", np.int64(s[3])), ("checks", np.array(r, np.int64)))},
             **extra)
    recs = [(k, m) for r in per_s for k, m, _ in r]
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
if os.environ.get("TL_E5"):                           # E5 (11-01): DFlash-X, the trained expert branch on the official path
    sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/TrainLab_sticky"))
    import e5x
    e5x.install_official(d, e5x.load_branch(os.environ["TL_E5"]))
    say(f"E5: expert branch {os.environ['TL_E5']} installed (top-1 of {e5x.NRES} experts at taps {e5x.TAPL})")
say(f"D-02b: DFlash2 L1-20 (bf16, {own / 1e9:.2f} GB), block 16, official 1-step loop, exact two-stage top-16 head")
if os.environ.get("TL_WALK_SETS"):                  # D-02a: "tag=glob,..." (e.g. ifeval=~/TrainLab/data/d02a/tf_ifeval_*.npz)
    for s in os.environ["TL_WALK_SETS"].split(","):
        t, g = s.split("=", 1)
        domain(t, None, t, os.path.expanduser(g))
else:
    domain("code (HumanEval+)", "dfl2d_code")
    domain("math (GSM8K CoT)", "dfl2d_math")
open(os.path.join(TD, f"d02b_walk{os.environ.get('TL_WALK_TAG', '')}.txt"), "w").write("\n".join(lines) + "\n")
