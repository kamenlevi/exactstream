"""L1-19 (Leap, Sat 2026-10-24, pre-registered in ~/Leap/TABLE_350.md): a 2-step refining DFlash2 drafter, no training.
DFlash2 drafts a block in one denoising step: [anchor, MASK x 15] -> 15 tokens. L1-18 showed its agreement on tokens the
125B is sure of decays with depth (0.57 at depth 10), so try a second step (MaskGIT-style): feed the block again with the
step-1 tokens kept where the selector is confident and MASK elsewhere, and predict again. The drafter's cache holds only
the target-context rows (block tokens are never cached), so step 2 runs with an empty context at the same positions.
Variants (teacher-forced replay of the official loop on the 24 public HumanEval samples, data/dfl2d_code):
  1-step (as shipped); 2-step re-mask conf < t (t = 0.5 / 0.9), kept tokens forced; 2-step re-mask everything after the
  first position with conf < 0.9; 2-step with no masks (pure re-prediction).
Reports tokens per pass at block 16 (and 32 for the best), plus agreement by depth on target-confident tokens.
Usage: [BigMoE venv] refine19.py  -> ~/Leap/data/refine19.txt"""
import os, sys, glob, time, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, mlx.core as mx, mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/Leap_third_party"))
import dflash_model_mlx as D

class Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(248320, 2560, group_size=64, bits=4)
        self.lm_head = nn.QuantizedLinear(2560, 248320, bias=False, group_size=64, bits=4)
w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
tgt = Target()
tgt.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales", "biases")]
                 + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
draft = D.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.bind(tgt)
WEIGHTS = os.environ.get("R19_WEIGHTS", "")          # L1-20: score a fine-tuned drafter with the same replay
if WEIGHTS:
    draft.load_weights(os.path.expanduser(WEIGHTS), strict=False)
mx.eval(draft.parameters(), tgt.parameters())
MASK = int(draft.config.mask_token_id)

# Step 2 must see exactly step 1's context keys / values at the same positions, without touching the cache. Step 1
# stashes each attention layer's (keys, values) as update_and_fetch returns them; step 2 (REUSE on) rebuilds the
# attention from the stash with the original's rope offsets and masks.
_orig_attn = D.DFlashAttention.__call__
STASH, REUSE, CHECKED = {}, {"on": False}, {"done": False}
def _attn(self, x, x_ctx, rope, cache):
    if not REUSE["on"]:
        orig_uf = cache.update_and_fetch
        def uf(k, v):
            kk, vv = orig_uf(k, v)
            STASH[id(self)] = (kk, vv)
            return kk, vv
        cache.update_and_fetch = uf
        try:
            return _orig_attn(self, x, x_ctx, rope, cache)
        finally:
            del cache.update_and_fetch
    B, L, _ = x.shape
    kk, vv = STASH[id(self)]
    ctx_len = kk.shape[2]
    queries = self.q_norm(self.q_proj(x).reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
    prop_keys = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    prop_values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
    queries = rope(queries, offset=cache.offset)            # = step 1's offset + S (the cache has advanced by S)
    prop_keys = rope(prop_keys, offset=cache.offset)
    keys = mx.concatenate([kk, prop_keys], axis=2)
    values = mx.concatenate([vv, prop_values], axis=2)
    mask = D.create_causal_mask(L, offset=ctx_len) if self.is_causal else None
    if self.is_sliding:
        query = ctx_len + mx.arange(L)[:, None]
        key = mx.arange(ctx_len + L)[None]
        context = (key < ctx_len) & (query - key < self.sliding_window)
        block = key >= ctx_len
        if self.is_causal:
            block = block & (key <= query)
        mask = context | block
    output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
    return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))
D.DFlashAttention.__call__ = _attn
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def select_conf(block, hidden):
    """the candidate selector at temperature 0 (as propose), plus the selected candidate's softmax at each depth"""
    logits = draft.compute_logits(hidden)
    sel = draft.candidate_selector
    cand = mx.argpartition(logits, -sel.top_k, axis=-1)[..., -sel.top_k:]
    unary = mx.take_along_axis(logits, cand, axis=-1)
    hp = sel.hidden_projection(hidden)
    pred, path, conf = block[:, 0], [], []
    for p in range(hp.shape[1]):
        sc = unary[:, p] + mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None] * sel.successor_codebook(cand[:, p]), axis=-1)
        s = mx.argmax(sc, axis=-1)
        conf.append(mx.take_along_axis(mx.softmax(sc.astype(mx.float32), axis=-1), s[:, None], axis=-1)[:, 0])
        pred = mx.take_along_axis(cand[:, p], s[:, None], axis=-1)[:, 0]
        path.append(pred)
    return mx.stack(path, axis=1), mx.stack(conf, axis=1)

def draft_block(anchor, ctx, cache, bs, mode):
    blk = mx.array([[anchor] + [MASK] * (bs - 1)])
    h1 = draft.hidden_states(blk, ctx, cache, 1)
    p1, c1 = select_conf(blk, h1)
    mx.eval(p1, c1)
    p1 = np.array(p1[0]).tolist(); c1 = np.array(c1[0])
    if not CHECKED["done"]:                                   # the reuse path must reproduce step 1 on the same block
        REUSE["on"] = True
        try:
            pc, _ = select_conf(blk, draft.hidden_states(blk, ctx[:, -1:], cache, 1)); mx.eval(pc)
        finally:
            REUSE["on"] = False
        same = np.array(pc[0]).tolist() == p1
        say(f"  reuse-path check (step 2 on the unchanged block == step 1): {'OK' if same else 'MISMATCH'}")
        CHECKED["done"] = True
        if not same:
            raise SystemExit("reuse path differs from step 1")
    if mode == "1step":
        return p1
    if mode.startswith("remask"):
        t = float(mode.split("_")[1]); keep = c1 >= t
    elif mode == "tail_0.9":
        bad = np.nonzero(c1 < 0.9)[0]; first = bad[0] if bad.size else len(c1); keep = np.arange(len(c1)) < first
    else:                                                   # "nomask": everything filled with the step-1 tokens
        keep = np.ones(len(c1), bool)
    if mode != "nomask" and keep.all():
        return p1
    blk2 = mx.array([[anchor] + [p1[j] if keep[j] else MASK for j in range(len(p1))]])
    REUSE["on"] = True
    try:
        h2 = draft.hidden_states(blk2, ctx[:, -1:], cache, 1)          # the context rows are ignored: step 1's stash
        p2, _ = select_conf(blk2, h2)
        mx.eval(p2)
    finally:
        REUSE["on"] = False
    p2 = np.array(p2[0]).tolist()
    if mode == "nomask":
        return p2
    return [p1[j] if keep[j] else p2[j] for j in range(len(p1))]

def replay(f, ids, off, gen, bs, mode, p1t=None, stats=None):
    n = len(ids); cache = draft.make_cache()
    start = max(off, gen - (draft.config.sliding_window - 1))
    for c in cache:
        c.offset = start
    fed, a, tok, blocks = start, gen, 0, 0
    while a < off + n - 1:
        ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
        d = draft_block(int(ids[a - off]), ctx, cache, bs, mode)
        fed = a
        true = ids[a - off + 1:a - off + bs]
        k = 0
        while k < len(true) and d[k] == true[k]:
            k += 1
        if stats is not None and p1t is not None:
            for j in range(len(true)):
                if p1t[a - off + j] >= 0.99:
                    stats[0][j] += int(d[j] == true[j]); stats[1][j] += 1
        tok += min(k + 1, off + n - 1 - a); blocks += 1
        if (trim := cache[0].offset - a) > 0:
            D._trim_recent_cache(cache, trim)
        a = a + k + 1
    return tok, blocks

samples = []
for path in sorted(glob.glob(os.path.expanduser("~/Leap/data/dfl2d_code/*.npz"))):
    key = os.path.basename(path)[:-4]
    z = np.load(path); tz = np.load(os.path.expanduser(f"~/Leap/data/l118_tgt/{key}.npz"))
    samples.append((z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"]),
                    np.exp(tz["lp"][:, 0].astype(np.float64))))
say(f"L1-19 / L1-20: 1- and 2-step DFlash2 drafting, weights {WEIGHTS or 'incoai (original)'}, 24 public HumanEval "
    f"samples, teacher-forced official loop")
res = {}
MODES = [m for m in os.environ.get("R19_MODES", "").split(",") if m]   # DS: e.g. 32:1step,32:tail_0.9 (default: all)
PER = {}                                                                  # per-sample (tokens, passes), for R19_JSON
for bs in (16, 32):
    for mode in ("1step", "remask_0.5", "remask_0.9", "tail_0.9", "nomask"):
        if bs == 32 and mode not in ("1step", "remask_0.5", "remask_0.9", "tail_0.9"):
            continue
        if MODES and f"{bs}:{mode}" not in MODES:
            continue
        t0 = time.time(); st = [np.zeros(bs - 1), np.zeros(bs - 1)]; tok = blk = 0; per = []
        for f, ids, off, gen, p1t in samples:
            a, b = replay(f, ids, off, gen, bs, mode, p1t, st); tok += a; blk += b; per.append((a, b))
        res[(bs, mode)] = tok / blk
        PER[f"{bs}:{mode}"] = per
        dep = " ".join(f"{st[0][j] / max(1, st[1][j]):.2f}" for j in range(0, bs - 1, 1 if bs == 16 else 2))
        say(f"  block {bs:2d} {mode:11s}: {tok / blk:5.2f} tokens per pass ({blk} blocks, {time.time() - t0:.0f} s); "
            f"agreement on target-confident tokens by depth: {dep}")
if (16, "1step") in res:
    b1 = res[(16, "1step")]
    best = max((v, k) for k, v in res.items() if k[0] == 16)
    say(f"best block-16 variant {best[1][1]}: {best[0]:.2f} vs 1-step {b1:.2f} ({best[0] / b1 - 1:+.1%}); "
        f"kill < +10 %, pass >= +15 %")
top = max((v, k) for k, v in res.items())
say(f"best overall: block {top[1][0]} {top[1][1]}: {top[0]:.2f} tokens per pass (L1-20: kill < 10.0, pass >= 13.0)")
open(os.path.expanduser(f"~/Leap/data/refine19{os.environ.get('R19_TAG', '')}.txt"), "w").write("\n".join(lines) + "\n")
if os.environ.get("R19_JSON"):
    json.dump({"weights": WEIGHTS, "samples": [os.path.basename(p)[:-4] for p in sorted(glob.glob(os.path.expanduser(
        "~/Leap/data/dfl2d_code/*.npz")))], "res": {f"{k[0]}:{k[1]}": v for k, v in res.items()}, "per": PER},
              open(os.path.expanduser(os.environ["R19_JSON"]), "w"))
