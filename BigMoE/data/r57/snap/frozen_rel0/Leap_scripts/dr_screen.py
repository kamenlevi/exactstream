"""R-06 + R-07 (NewIdeas' rapid screens, Mon 2026-10-26; Leap, pre-registered 19:15 EET in ~/Leap/TABLE_350.md 'R-06 /
R-07'): drafting at the speed of its bytes. DFlash2's 1-step draft (L1-20 weights) costs 12.5 ms at block 16 (CA:
forward 7.46, head 4.46, selector 0.83).
Arms: the drafter's own layers + fc in bf16 / 8-bit / 4-bit (group 64; the selector, the embedding and the target's
4-bit head unchanged) x the head's top-16:
  argp  argpartition over all 248,320 logits (today)
  2s    exact two-stage: max per 128-token block, top-16 blocks, top-16 of their 2,048 logits (the same candidate set)
  v32k  FR-Spec (2502.14856): the head over 32,768 tokens only, ranked by counts on L1-20's training tokens (ids x10 +
        the target's top-32 lists), the rest of the 32K filled in id order (BPE merge rank)
Draft ms: the whole draft as one graph, one sync, median of 30 after 5, on one held-out sample. Pieces (head and top-k
alone) timed the same way. Tokens per check: refine19's teacher-forced 1-step replay on the 24 held-out HumanEval
samples (data/dfl2d_code), blocks 16 and 32.
Usage: [BigMoE venv] dr_screen.py -> ~/Leap/data/dr_screen.txt"""
import glob, os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel0/Leap_third_party"))
import dflash_model_mlx as DF

D = os.path.expanduser("~/Leap/data")
V, K, NSUB = 248320, 16, 32768
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

# FR-Spec's sub-vocabulary: counts from L1-20's training set (not the held-out samples), then id order
cnt = np.zeros(V, np.int64)
for p in glob.glob(os.path.join(D, "l120", "*.npz")):
    z = np.load(p)
    cnt += 10 * np.bincount(z["ids"], minlength=V)[:V] + np.bincount(z["tk"].reshape(-1), minlength=V)[:V]
order = np.lexsort((np.arange(V), -cnt))                  # by count, ties by id
SUB = np.sort(order[:NSUB])
say(f"R-06 / R-07: FR-Spec set = {int((cnt > 0).sum())} tokens seen in L1-20's data + {NSUB - int((cnt > 0).sum())} "
    f"filled in id order (largest id {int(SUB[-1])})")
sub_head = nn.QuantizedLinear(2560, NSUB, bias=False, group_size=64, bits=4)
si = mx.array(SUB.astype(np.int32))
sub_head.load_weights([(s, mx.take(getattr(tgt.lm_head, s), si, axis=0)) for s in ("weight", "scales", "biases")])
mx.eval(sub_head.parameters())

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

def cap(d, lg):
    lg = lg * d.config.output_multiplier
    c = d.config.final_logit_softcapping
    return mx.tanh(lg / c) * c if c is not None and c > 0 else lg

def head(d, hid, kind):
    """(candidates [1, bs, K] as vocabulary ids, their logits)"""
    if kind == "v32k":
        lg = cap(d, sub_head(hid))
        c = mx.argpartition(lg, -K, axis=-1)[..., -K:]
        return mx.take(si, c), mx.take_along_axis(lg, c, axis=-1)
    lg = d.compute_logits(hid)
    if kind == "argp":
        c = mx.argpartition(lg, -K, axis=-1)[..., -K:]
    else:
        B = 128
        s = lg.shape
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

def draft_fn(d, kind, blk, ctx, cache):
    hid = d.hidden_states(blk, ctx, cache, 1)
    cand, unary = head(d, hid, kind)
    return chain(d, blk, unary, cand, d.candidate_selector.hidden_projection(hid))

samples = []
for p in sorted(glob.glob(os.path.join(D, "dfl2d_code", "*.npz"))):
    z = np.load(p)
    samples.append((z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])))
gen_ids = np.concatenate([ids[gen - off + 1:] for _, ids, off, gen in samples])
say(f"  held-out answer tokens inside the FR-Spec set: {np.isin(gen_ids, SUB).mean():.4f} of {len(gen_ids)}")

def walk(d, kind, bs, sample, fn, limit=None):
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

def timed_fn(d, kind, ts):
    def fn(blk, ctx, cache):
        mx.eval(blk, ctx); mx.synchronize()
        t0 = time.perf_counter()
        dr = np.array(draft_fn(d, kind, blk, ctx, cache)[0]).tolist()
        ts.append(1e3 * (time.perf_counter() - t0))
        return dr
    return fn

def pieces(d, bs):
    """head pieces alone at this block size, on a random hidden (ms, median of 30 after 5)"""
    hid = mx.random.normal((1, bs - 1, 2560)).astype(mx.bfloat16)
    lg = d.compute_logits(hid); mx.eval(hid, lg)
    out = {}
    for name, fn in (("lm_head", lambda: d.compute_logits(hid)), ("argp alone", lambda: mx.argpartition(lg, -K, axis=-1)),
                     ("lm_head + 2s", lambda: head(d, hid, "2s")), ("lm_head + argp", lambda: head(d, hid, "argp")),
                     ("v32k head + top-16", lambda: head(d, hid, "v32k"))):
        ts = []
        for i in range(35):
            mx.synchronize(); t0 = time.perf_counter(); mx.eval(fn()); mx.synchronize()
            ts.append(1e3 * (time.perf_counter() - t0))
        out[name] = med(ts[5:])
    return out

RES, PCS = {}, {}
for bits in (0, 8, 4):
    d, own = load(bits)
    tag = f"{bits}-bit" if bits else "bf16"
    say(f"  drafter {tag}: {own / 1e9:.2f} GB of its own parameters")
    for bs in (16, 32):
        if bits == 0:
            PCS[bs] = pieces(d, bs)
            say(f"    block {bs} head pieces: " + ", ".join(f"{k} {v:.2f} ms" for k, v in PCS[bs].items()))
        for kind in ("argp", "2s", "v32k"):
            ts = []
            walk(d, kind, bs, samples[0], timed_fn(d, kind, ts), limit=35)
            if kind == "2s" and bits != 0:                 # 2s = argp's candidate set: timed only, tokens from argp
                RES[(bits, bs, kind)] = (med(ts[5:]), None)
                continue
            t0 = time.time()
            tok = blk = 0
            for s in samples:
                a, b = walk(d, kind, bs, s, lambda x, c, ca: np.array(draft_fn(d, kind, x, c, ca)[0]).tolist())
                tok += a; blk += b
            RES[(bits, bs, kind)] = (med(ts[5:]), tok / blk)
            say(f"    block {bs} {kind:4s}: draft {med(ts[5:]):6.2f} ms (min {min(ts[5:]):.2f}); "
                f"{tok / blk:.3f} tokens per check ({blk} checks, {time.time() - t0:.0f} s)")
        if bits != 0:
            say(f"    block {bs} 2s  : draft {RES[(bits, bs, '2s')][0]:6.2f} ms (tokens = argp's)")
    del d
    mx.clear_cache()

base = {bs: RES[(0, bs, "argp")][1] for bs in (16, 32)}
say(f"  2s exact: bf16 tokens 2s / argp = {RES[(0, 16, '2s')][1] / base[16]:.4f} (b16), {RES[(0, 32, '2s')][1] / base[32]:.4f} (b32)")
for bits in (0, 8, 4):
    for kind in ("argp", "2s", "v32k"):
        r16, r32 = RES[(bits, 16, kind)], RES[(bits, 32, kind)]
        t16 = (r16[1] if r16[1] is not None else RES[(bits, 16, "argp")][1]) / base[16]
        t32 = (r32[1] if r32[1] is not None else RES[(bits, 32, "argp")][1]) / base[32]
        say(f"  {bits or 'bf16':>4} {kind:4s}: b16 {r16[0]:6.2f} ms, tokens x{t16:.4f}; b32 {r32[0]:6.2f} ms, tokens x{t32:.4f}")
tok = lambda b, k, bs: (RES[(b, bs, k)][1] or RES[(b, bs, "argp")][1]) / base[bs]
worst = lambda b, k: min(tok(b, k, 16), tok(b, k, 32))
arms = [(RES[(b, 16, k)][0], b, k) for b in (0, 8, 4) for k in ("argp", "2s")]
ok6 = sorted(a for a in arms if worst(a[1], a[2]) >= 0.99)
fast = [a for a in arms if a[0] <= 5.0]
name = lambda a: f"{a[1] or 'bf16'}{'-bit' if a[1] else ''} {a[2]} {a[0]:.2f} ms"
if ok6 and ok6[0][0] <= 5.0:
    v6 = f"PASS ({name(ok6[0])} at b16, tokens within 1 %)"
elif not ok6 or ok6[0][0] > 8.0 or (fast and all(worst(a[1], a[2]) < 0.97 for a in fast)):
    v6 = "KILL" + (f" (best within 1 %: {name(ok6[0])})" if ok6 else " (no arm within 1 %)")
else:
    v6 = f"BETWEEN (best within 1 %: {name(ok6[0])}; fastest: {name(min(arms))}, tokens x{worst(min(arms)[1], min(arms)[2]):.4f})"
say(f"R-06 verdict (PASS <= 5 ms at b16 with tokens within 1 % at b16 and b32; KILL best within 1 % > 8 ms, or every arm "
    f"<= 5 ms loses > 3 %): {v6}")
r7, h7 = worst(0, "v32k"), PCS[16]["v32k head + top-16"]
v7 = "KILL" if r7 < 0.98 else "PASS" if r7 >= 0.99 and h7 <= 1.5 else "BETWEEN"
say(f"R-07 verdict (PASS v32k head <= 1.5 ms at b16 and tokens >= 99 % of the full head; KILL < 98 %): head {h7:.2f} ms, "
    f"tokens x{r7:.4f} (worse of b16 / b32): {v7}")
open(os.path.join(D, "dr_screen.txt"), "w").write("\n".join(lines) + "\n")
