"""NT-1 (NewIdeas, Mon 2026-10-26): how much could near-tie (lossy) acceptance add to tokens per pass at block 32?
Published: Judge Decoding (ICLR 2025), AutoJudge (2504.20039, NeurIPS 2025: up to 1.5x accepted tokens per cycle at
< 1 % GSM8K loss, 1B draft / 8B target). Untested here (Leap TABLE_350 row (d) 'near-tie acceptance', empty).
Leap's headroom18 (incoai, block 16): 63 % of first misses fall where the 125B is >= 99 % sure, which no acceptance rule
can forgive, so the gain is bounded well below AutoJudge's.
Replays L1-20 (data/l120_ft.safetensors) exactly as Leap's cand21.py 'sel' policy at t 0.0 (teacher-forced official
loop, 24 public HumanEval samples, data/dfl2d_code), and reads the 125B's own top-32 at every first miss
(data/l118_tgt; row i predicts token i + 1). All inputs read-only; MLX on the CPU by default (NT_DEV=gpu to change).
Output: ~/NewIdeas/data/nt/nt_accept_b{BS}.txt + .jsonl (one record per block)."""
import os, sys, glob, time, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, mlx.core as mx, mlx.nn as nn
if os.environ.get("NT_DEV", "cpu") == "cpu":
    mx.set_default_device(mx.cpu)
sys.path.insert(0, os.path.expanduser("~/Leap/third_party"))
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
del w
draft = D.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.bind(tgt)
draft.load_weights(os.path.expanduser("~/Leap/data/l120_ft.safetensors"), strict=False)
mx.eval(draft.parameters(), tgt.parameters())
MASK = int(draft.config.mask_token_id)
BS = int(os.environ.get("NT_BS", "32"))
LIMIT = int(os.environ.get("NT_LIMIT", "0"))
OUT = os.path.expanduser("~/NewIdeas/data/nt"); os.makedirs(OUT, exist_ok=True)

def select_conf(block, hidden, logits):
    """cand21.py's selector at temperature 0, verbatim"""
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

recs, t0 = [], time.time()
paths = sorted(glob.glob(os.path.expanduser("~/Leap/data/dfl2d_code/*.npz")))
if LIMIT:
    paths = paths[:LIMIT]
for path in paths:
    key = os.path.basename(path)[:-4]
    z = np.load(path); tz = np.load(os.path.expanduser(f"~/Leap/data/l118_tgt/{key}.npz"))
    f = z["f"].reshape(z["f"].shape[0], -1); ids = z["ids"]; off, gen = int(z["off"]), int(z["gen"]); n = len(ids)
    tk, lp = tz["tk"], tz["lp"].astype(np.float64)
    cache = draft.make_cache()
    start = max(off, gen - (draft.config.sliding_window - 1))
    for c in cache:
        c.offset = start
    fed, a = start, gen
    while a < off + n - 1:
        ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
        blk = mx.array([[int(ids[a - off])] + [MASK] * (BS - 1)])
        h = draft.hidden_states(blk, ctx, cache, 1)
        logits = draft.compute_logits(h)
        fed = a
        true = np.asarray(ids[a - off + 1:a - off + BS])
        p, c = select_conf(blk, h, logits); mx.eval(p, c)
        d = np.array(p[0])[:len(true)]
        hit = d == true
        k = int(np.argmin(hit)) if not hit.all() else len(true)
        rem = off + n - 1 - a
        acc = min(k + 1, rem)
        r = {"key": key, "a": int(a), "k": k, "acc": int(acc), "nb": int(len(true)), "rem": int(rem),
             "hit": hit.astype(int).tolist(), "p1": [float(np.exp(lp[a - off + j, 0])) for j in range(len(true))]}
        if k < len(true):
            i = a - off + k
            wh = np.where(tk[i] == d[k])[0]
            r.update(t1=int(tk[i, 0]), dt=int(d[k]), rank=int(wh[0]) if len(wh) else -1,
                     pd=float(np.exp(lp[i, wh[0]])) if len(wh) else 0.0, p32=float(np.exp(lp[i, -1])))
        recs.append(r)
        if (trim := cache[0].offset - a) > 0:
            D._trim_recent_cache(cache, trim)
        a = a + acc
    print(f"{key}: {len(recs)} blocks so far, {time.time() - t0:.0f} s", flush=True)
with open(f"{OUT}/nt_accept_b{BS}.jsonl", "w") as fh:
    for r in recs:
        fh.write(json.dumps(r) + "\n")
print(f"replay done: {len(recs)} blocks, {sum(r['acc'] for r in recs) / len(recs):.2f} tokens per pass, {time.time() - t0:.0f} s")
