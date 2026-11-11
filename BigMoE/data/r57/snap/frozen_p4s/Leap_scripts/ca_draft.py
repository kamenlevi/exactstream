"""CA's drafting part (Leap, Mon 2026-10-26): where DFlash2's 1-step draft time goes (ca_anat.py time measured 12.7 ms
at block 16, 15.6 at block 32, in refine19's path). Same drafter (L1-20 weights, bf16) and real features (one
dfl2d_code sample), timed in pieces, each piece evaluated on its own (sync), medians of 30 calls after 5 warm-up:
  fwd     draft.hidden_states over the block (5 layers, the new context rows + the block)
  head    compute_logits (the 125B's 4-bit head over the block's positions) + top-16 candidates
  select  the first-order selector's per-position chain (Python loop of small ops)
  all     the three in one graph, one sync (= ca_anat.py's figure)
Usage: [BigMoE venv] ca_draft.py -> ~/Leap/data/ca_draft.txt"""
import glob, os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p4s/Leap_third_party"))
import dflash_model_mlx as DF

D = os.path.expanduser("~/Leap/data")
class Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(248320, 2560, group_size=64, bits=4)
        self.lm_head = nn.QuantizedLinear(2560, 248320, bias=False, group_size=64, bits=4)
w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
tgt = Target()
tgt.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales", "biases")]
                 + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
del w
draft = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.bind(tgt)
draft.load_weights(os.path.join(D, "l120_ft.safetensors"), strict=False)
mx.eval(draft.parameters(), tgt.parameters())
nbytes = sum(v.nbytes for _, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(draft.parameters()))
MASK = int(draft.config.mask_token_id)
sel = draft.candidate_selector
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
say(f"CA drafting: DFlash2 (L1-20 weights), {nbytes / 1e9:.2f} GB of drafter parameters incl. the bound head / embedding")

def chain(block, unary, cand, hp):
    pred, path = block[:, 0], []
    for p in range(hp.shape[1]):
        sc = unary[:, p] + mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None]
                                  * sel.successor_codebook(cand[:, p]), axis=-1)
        pred = mx.take_along_axis(cand[:, p], mx.argmax(sc, axis=-1)[:, None], axis=-1)[:, 0]
        path.append(pred)
    return mx.stack(path, axis=1)

def timed(fn):
    mx.synchronize(); t0 = time.perf_counter(); out = fn(); mx.eval(out); mx.synchronize()
    return out, 1e3 * (time.perf_counter() - t0)

z = np.load(sorted(glob.glob(os.path.join(D, "dfl2d_code", "*.npz")))[0])
f, sid, off, gen = z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])
for bs in (16, 32):
    cache = draft.make_cache()
    start = max(off, gen - (draft.config.sliding_window - 1))
    for c in cache:
        c.offset = start
    fed, a, T = start, gen, {k: [] for k in ("fwd", "head", "select", "all")}
    n = 0
    while a < off + len(sid) - 1 and n < 35:
        ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
        blk = mx.array([[int(sid[a - off])] + [MASK] * (bs - 1)])
        mx.eval(ctx, blk)
        if n % 2 == 0:                                   # pieces, each synced
            hid, t1 = timed(lambda: draft.hidden_states(blk, ctx, cache, 1))
            def head():
                lg = draft.compute_logits(hid)
                cand = mx.argpartition(lg, -sel.top_k, axis=-1)[..., -sel.top_k:]
                return lg, cand, mx.take_along_axis(lg, cand, axis=-1), sel.hidden_projection(hid)
            (lg, cand, unary, hp), t2 = timed(head)
            d, t3 = timed(lambda: chain(blk, unary, cand, hp))
            if n >= 5:
                T["fwd"].append(t1); T["head"].append(t2); T["select"].append(t3)
        else:                                            # the whole draft, one graph, one sync
            def whole():
                hid = draft.hidden_states(blk, ctx, cache, 1)
                lg = draft.compute_logits(hid)
                cand = mx.argpartition(lg, -sel.top_k, axis=-1)[..., -sel.top_k:]
                return chain(blk, mx.take_along_axis(lg, cand, axis=-1), cand, sel.hidden_projection(hid))
            d, t4 = timed(whole)
            if n >= 5:
                T["all"].append(t4)
        d = np.array(d[0]).tolist()
        fed = a
        true = sid[a - off + 1:a - off + bs]
        k = 0
        while k < len(true) and d[k] == true[k]:
            k += 1
        if (trim := cache[0].offset - a) > 0:
            DF._trim_recent_cache(cache, trim)
        a = a + k + 1
        n += 1
    say(f"  block {bs}: " + ", ".join(f"{k} {np.median(v):.2f} ms" for k, v in T.items()) +
        f" (medians of {len(T['fwd'])} / {len(T['all'])})")
open(os.path.join(D, "ca_draft.txt"), "w").write("\n".join(lines) + "\n")
