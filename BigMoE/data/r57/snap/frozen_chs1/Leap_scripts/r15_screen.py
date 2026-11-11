"""R-15 DSK step 1 (NewIdeas Tue ~19:35; Leap, pre-registered Tue 2026-10-27 19:44 EET in ~/Leap/TABLE_350.md 'R-15'):
where one block-16 draft spends its GPU time, by kernel class. R-10: the drafter is GPU-bound (graph build 0.46 ms of
11.2), and compiling its elementwise ops saved only 0.7 ms. The drafter only proposes tokens, so any faster kernel
is allowed (tokens within 1 %).
Knockouts (CA's method): a knocked-out call still builds its graph but returns the same call's output from an
earlier draft, so its GPU work is pruned; its inputs are evaluated with the draft so upstream work stays. Classes:
  dense   every Linear / QuantizedLinear in the drafter (q, k, v, o, MLP, conv kernel projections, fc, selector
          projection), not the bound head
  conv    the grouped dynamic convolution's elementwise part (4 calls per layer)
  attn    the attention core (scaled_dot_product_attention)
  norm    every RMSNorm
  head    compute_logits (the bound 4-bit head, 248,320 x 2,560)
  topk    the exact two-stage top-16 after the head
  select  the 15-step selector chain
  all     every class above (what remains: embedding, residual adds, reshapes, RoPE, cache writes, masks)
Each draft: block 16, 10 new context rows (sample 0's real features, advancing 10 per draft; shapes stay fixed),
one sync. Variants interleaved per round, 30 rounds after 3; class time = paired median of none - variant.
bf16 and 4-bit drafters (R-10's E and E4 paths).
Usage: [BigMoE venv] r15_screen.py -> ~/Leap/data/r15_screen.txt"""
import glob, os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs1/Leap_third_party"))
import dflash_model_mlx as DF

D = os.path.expanduser("~/Leap/data")
V, TK, BS, CTX = 248320, 16, 16, 10
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
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

# ---------------------------------------------------------------- knockouts
DIFF, DEAD, KO_ON = [False], [0], set()
STORE, ORD, KA = {}, {}, []

def leaves(x, out=None):
    out = [] if out is None else out
    if isinstance(x, mx.array):
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for y in x:
            leaves(y, out)
    return out

def KO(cls, thunk, inputs):
    """as ca_anat.KO: the call's output from an earlier draft when its class is knocked out"""
    if not DIFF[0] or cls is None:
        return thunk()
    n = ORD.get(cls, 0); ORD[cls] = n + 1
    if DEAD[0]:
        return thunk()
    key = (cls, n)
    if cls in KO_ON and key in STORE:
        DEAD[0] += 1
        try:
            thunk()
        finally:
            DEAD[0] -= 1
        KA.extend(leaves(list(inputs)))
        return STORE[key]
    y = thunk()
    STORE[key] = y
    return y

HEAD = [None]
def wrap_method(klass, cls_of):
    orig = klass.__call__
    def call(self, *a, **kw):
        return KO(cls_of(self), lambda: orig(self, *a, **kw), [x for x in a if isinstance(x, mx.array)])
    klass.__call__ = call
wrap_method(nn.Linear, lambda m: "dense")
wrap_method(nn.QuantizedLinear, lambda m: None if m is HEAD[0] else "dense")
wrap_method(nn.RMSNorm, lambda m: "norm")
ORIG_CONV, ORIG_SDPA = DF._grouped_dynamic_convolve, mx.fast.scaled_dot_product_attention
DF._grouped_dynamic_convolve = lambda h, dyn, base, gs: KO("conv", lambda: ORIG_CONV(h, dyn, base, gs), [h, dyn])
mx.fast.scaled_dot_product_attention = lambda q, k, v, **kw: KO("attn", lambda: ORIG_SDPA(q, k, v, **kw), [q, k, v])

def topk(lg):
    """R-06's exact two-stage top-16: candidates [1, bs - 1, K] as vocabulary ids, their logits"""
    B, s = 128, lg.shape
    blk = lg.reshape(*s[:-1], s[-1] // B, B)
    tb = mx.argpartition(blk.max(axis=-1), -TK, axis=-1)[..., -TK:]
    sel = mx.take_along_axis(blk, mx.broadcast_to(tb[..., None], (*s[:-1], TK, B)), axis=-2).reshape(*s[:-1], TK * B)
    j = mx.argpartition(sel, -TK, axis=-1)[..., -TK:]
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

def draft_fn(d, blk, ctx, cache):
    hid = d.hidden_states(blk, ctx, cache, 1)
    lg = KO("head", lambda: d.compute_logits(hid), [hid])
    cand, unary = KO("topk", lambda: topk(lg), [lg])
    hp = d.candidate_selector.hidden_projection(hid)
    return KO("select", lambda: chain(d, blk, unary, cand, hp), [blk, unary, cand, hp])

# ---------------------------------------------------------------- the walk
z = np.load(sorted(glob.glob(os.path.join(D, "dfl2d_code", "*.npz")))[0])
FEAT, IDS = z["f"].reshape(z["f"].shape[0], -1), z["ids"]
CLASSES = ["dense", "conv", "attn", "norm", "head", "topk", "select"]
VARS = ["none"] + CLASSES + ["all"]

CPU = {}                                                 # per variant: (Python build, eval wall, eval main-thread CPU) ms

def run(d, rounds=30, warm=3):
    MASK = int(d.config.mask_token_id)
    cache = d.make_cache()
    a, T = 0, {v: [] for v in VARS}
    CPU.clear(); CPU.update({v: [] for v in VARS})
    STORE.clear()
    for r in range(warm + rounds):
        while os.path.exists(PAUSE):
            time.sleep(5)
        for v in VARS:
            ORD.clear(); KA.clear(); KO_ON.clear()
            KO_ON.update(CLASSES if v == "all" else [v] if v in CLASSES else [])
            if a + CTX > len(FEAT):
                a = 0
            ctx = mx.array(FEAT[a:a + CTX][None]).astype(mx.bfloat16)
            blk = mx.array([[int(IDS[a % len(IDS)])] + [MASK] * (BS - 1)])
            mx.eval(ctx, blk); mx.synchronize()
            DIFF[0] = True
            t0 = time.perf_counter()
            out = draft_fn(d, blk, ctx, cache)
            tb, cb = time.perf_counter(), time.thread_time()
            mx.eval(out, *KA)
            np.array(out).tolist()
            t1, c1 = time.perf_counter(), time.thread_time()
            DIFF[0] = False
            KA.clear()
            a += CTX                                     # the cache keeps only context rows (the block's are not cached)
            if r >= warm:
                T[v].append(1e3 * (t1 - t0))
                CPU[v].append((1e3 * (tb - t0), 1e3 * (t1 - tb), 1e3 * (c1 - cb)))
    mx.synchronize()
    return T

RES = {}
for bits in (0, 4):
    d, own = load(bits)
    HEAD[0] = d.lm_head
    tag = "4-bit" if bits else "bf16"
    T = run(d)
    none = med(T["none"])
    dif = {v: med([x - y for x, y in zip(T["none"], T[v])]) for v in CLASSES + ["all"]}
    RES[tag] = (none, dif)
    say(f"  {tag} drafter ({own / 1e9:.2f} GB own parameters): draft {none:.2f} ms at block {BS} (knockout hooks on, "
        f"nothing knocked out); all classes out {med(T['all']):.2f} ms")
    say("    class time (none - without it, ms): " + ", ".join(f"{c} {dif[c]:.2f}" for c in CLASSES)
        + f"; sum {sum(dif[c] for c in CLASSES):.2f} vs all {dif['all']:.2f}")
    for v in ("none", "all"):
        b_, w_, c_ = (med([x[i] for x in CPU[v]]) for i in range(3))
        say(f"    {v:4s}: Python build {b_:.2f} ms, eval {w_:.2f} ms wall with {c_:.2f} ms of main-thread CPU (MLX's "
            f"encoding; {c_ / w_:.0%} of the eval: near 100 % = CPU-bound, not GPU-bound)")
    top = sorted(CLASSES, key=lambda c: -dif[c])[:2]
    say(f"    top two: {top[0]} {dif[top[0]]:.2f}, {top[1]} {dif[top[1]]:.2f} ms; if both went to zero: "
        f"~{none - dif[top[0]] - dif[top[1]]:.2f} ms (PASS <= 6.5)")
    del d
    mx.clear_cache()
say("R-15 step 1: breakdown only (no verdict); step 2 swaps the top class or two")
open(os.path.join(D, "r15_screen.txt"), "w").write("\n".join(lines) + "\n")
