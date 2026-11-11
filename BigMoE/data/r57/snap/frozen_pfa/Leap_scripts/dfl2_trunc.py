"""Leap L5 side analysis (Sat 2026-10-24): verify fewer rows when the block drafter is unsure (dynamic draft length;
published: SpecDec++, AdaEDL, DISCO). Replays the official loop at block BS exactly as dfl2_eval.py and records, per
block, the accepted length k and the selector's confidence at each depth (softmax of the selected candidate over its
16 scores, as select() at temperature 0 computes them). Then, per policy, rows r per pass and tokens min(k, r - 1) + 1:
  full: r = BS; fixed r; threshold t: r = 1 + the depths whose running product of confidences is >= t;
  oracle: r = k + 1 (the bound).
Costs per pass: in RAM, 15 + 2.88 x rows ms (today's exact 3c expert kernel, ~6 us per (row, rank) pair; Chat1000
REPORT); SSD-streamed C192, 8.8 + 3.15 x 10 rows^0.7185 ms (Fit12's 4- and 8-row passes; copyspan_width.py).
Usage: [BigMoE venv] dfl2_trunc.py --dir DIR [--weights W] [--bs 16] [--cpu] [--tag T]"""
import os, sys, glob, time, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, mlx.core as mx, mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_third_party"))
import dflash_model_mlx as D

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--weights", default=""); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--cpu", action="store_true"); ap.add_argument("--tag", default="")
A = ap.parse_args()
if A.cpu:
    mx.set_default_device(mx.cpu)

class Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(248320, 2560, group_size=64, bits=4)
        self.lm_head = nn.QuantizedLinear(2560, 248320, bias=False, group_size=64, bits=4)

w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
t = Target()
t.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales", "biases")]
               + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
draft = D.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.bind(t)
if A.weights:
    draft.load_weights(os.path.expanduser(A.weights), strict=False)
mx.eval(draft.parameters())
MASK = int(draft.config.mask_token_id)

def propose_conf(block, ctx, cache):
    """draft.propose(..., temperature 0, logits_start=1) plus the selected candidate's softmax at each depth"""
    hidden = draft.hidden_states(block, ctx, cache, 1)
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

def replay(f, ids, off, gen, bs):
    n = len(ids); cache = draft.make_cache()
    start = max(off, gen - (draft.config.sliding_window - 1))
    for c in cache:
        c.offset = start
    fed, a, out = start, gen, []
    while a < off + n - 1:
        ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
        path, conf = propose_conf(mx.array([[int(ids[a - off])] + [MASK] * (bs - 1)]), ctx, cache)
        mx.eval(path, conf)
        fed = a
        d = np.array(path[0]).tolist(); cf = np.array(conf[0]); true = ids[a - off + 1:a - off + bs]
        k = 0
        while k < len(true) and d[k] == true[k]:
            k += 1
        out.append((k, len(true), off + n - 1 - a, cf))
        if (trim := cache[0].offset - a) > 0:
            D._trim_recent_cache(cache, trim)
        a = a + k + 1
    return out

UNION = lambda r: 10.0 * r ** 0.7185
COST = {"RAM": lambda r: 15.0 + 2.88 * r, "SSD": lambda r: 8.8 + 3.15 * UNION(r)}

def policy(blocks, rule):
    """total (tokens, rows, ms per model) when each block verifies r rows chosen by rule(k, m, conf)"""
    tok, rows, ms = 0, 0, {c: 0.0 for c in COST}
    for k, m, r_left, cf in blocks:
        r = min(rule(k, m, cf), m + 1)
        tok += min(min(k, r - 1) + 1, r_left); rows += r
        for c, T in COST.items():
            ms[c] += T(r)
    return tok, rows, ms

blocks = []
t0 = time.time()
for p in sorted(glob.glob(os.path.join(os.path.expanduser(A.dir), "*.npz"))):
    z = np.load(p)
    blocks += replay(z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"]), A.bs)
print(f"[trunc] {A.dir} weights {A.weights or 'original'} bs {A.bs}: {len(blocks)} blocks in {time.time() - t0:.0f}s", flush=True)
cum = lambda cf: np.cumprod(cf)
rules = [("full", lambda k, m, cf: A.bs)] + [(f"fixed {r}", (lambda r_: lambda k, m, cf: r_)(r)) for r in (6, 8, 10, 12)]
rules += [(f"conf >= {t:.2f}", (lambda t_: lambda k, m, cf: 1 + int((cum(cf[:m]) >= t_).sum()))(t)) for t in (0.1, 0.2, 0.3, 0.4, 0.5)]
rules += [("oracle", lambda k, m, cf: k + 1)]
lines = [f"{A.dir} | weights {A.weights or 'original'} | bs {A.bs} | {len(blocks)} blocks"]
for name, rule in rules:
    tok, rows, ms = policy(blocks, rule)
    lines.append(f"{name:12s} tok/pass {tok / len(blocks):5.2f} rows/pass {rows / len(blocks):5.2f} | ms/token RAM "
                 f"{ms['RAM'] / tok:5.2f} SSD {ms['SSD'] / tok:5.1f}")
for l in lines:
    print(l, flush=True)
open(os.path.expanduser(f"~/Leap/data/trunc{A.tag}.txt"), "w").write("\n".join(lines) + "\n")
