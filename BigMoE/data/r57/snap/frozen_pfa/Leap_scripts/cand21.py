"""L1-20 verdict aid + L1-21 screen (Leap, Sat 2026-10-24), no training.
(1) Price the rows. L1-20's 13.0 line assumed a 16-row pass; a block-32 verify costs more because the expert union grows
    ~ rows^0.72. So score each drafter as tok/s at the ~280 stack (TABLE_350 "One number"): pass time =
    (dense diet 2.1 GB + 3.0-bit experts 5.06 GB x (rows / 16)^0.72) / 224 GB/s + drafting (3.5 ms block 16, 4.0 ms
    block 32). rows = 1 + drafts verified; confidence truncation (L1-14) keeps the drafts while the running product of the
    selector's confidence stays >= t.
(2) Lattice headroom. DFlash2's selector picks one path through its top-16 candidates per depth with a first-order
    (predecessor x successor) score. An oracle over the same top-K candidates accepts the longest prefix whose true token
    is in each depth's top-K: the upper bound for any reranker over the lattice (e.g. a small autoregressive one).
Teacher-forced official loop on the 24 public HumanEval samples (data/dfl2d_code), as refine19.py.
Usage: [BigMoE venv] [R21_WEIGHTS=data/l120_ft.safetensors R21_TAG=_l120] cand21.py -> ~/Leap/data/cand21{tag}.txt"""
import os, sys, glob, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, mlx.core as mx, mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_third_party"))
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
WEIGHTS = os.environ.get("R21_WEIGHTS", "")
if WEIGHTS:
    draft.load_weights(os.path.expanduser(WEIGHTS), strict=False)
mx.eval(draft.parameters(), tgt.parameters())
MASK = int(draft.config.mask_token_id)
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def pass_ms(rows, bs, experts=5.06):
    """experts = GB of 3.0-bit experts per 16-row pass: 5.06 with adaptive k p 0.8 (the 15:40 stack), 5.52 with top-8
    (Fit12 TK1: p 0.8 collapses acceptance on code; top-8 passed the Eval suite 5/5)"""
    return (2.1 + experts * (rows / 16) ** 0.72) / 224 * 1000 + (3.5 if bs <= 16 else 4.0)

def select_conf(block, hidden, logits):
    """the candidate selector at temperature 0 (as propose), plus the selected candidate's softmax at each depth"""
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

def replay(f, ids, off, gen, bs, policy, arg, rec):
    """policy 'sel': the selector path, truncated at running confidence arg; 'orc': oracle over the top-arg lattice"""
    n = len(ids); cache = draft.make_cache()
    start = max(off, gen - (draft.config.sliding_window - 1))
    for c in cache:
        c.offset = start
    fed, a, tok, blocks, ms, ms8, rows = start, gen, 0, 0, 0.0, 0.0, 0
    while a < off + n - 1:
        ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
        blk = mx.array([[int(ids[a - off])] + [MASK] * (bs - 1)])
        h = draft.hidden_states(blk, ctx, cache, 1)
        logits = draft.compute_logits(h)
        fed = a
        true = np.asarray(ids[a - off + 1:a - off + bs])
        if policy == "sel":
            p, c = select_conf(blk, h, logits); mx.eval(p, c)
            d = np.array(p[0]); keep = int(np.sum(np.cumprod(np.array(c[0], np.float64)) >= arg))
            hit = d[:len(true)] == true
        else:
            cand = mx.argpartition(logits, -arg, axis=-1)[..., -arg:]; mx.eval(cand)
            cand = np.array(cand[0])[:len(true)]
            hit = (cand == true[:, None]).any(-1)
            keep = bs - 1
            rec[0][:len(true)] += hit; rec[1][:len(true)] += 1
        k = int(np.argmin(hit)) if not hit.all() else len(true)
        acc = min(min(k, keep) + 1, off + n - 1 - a)
        tok += acc; blocks += 1; rows += keep + 1; ms += pass_ms(keep + 1, bs); ms8 += pass_ms(keep + 1, bs, 5.52)
        if (trim := cache[0].offset - a) > 0:
            D._trim_recent_cache(cache, trim)
        a = a + acc
    return tok, blocks, rows, ms, ms8

samples = []
for path in sorted(glob.glob(os.path.expanduser("~/Leap/data/dfl2d_code/*.npz"))):
    z = np.load(path)
    samples.append((z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])))
say(f"L1-20 verdict aid + L1-21 lattice screen: weights {WEIGHTS or 'incoai (original)'}, 24 public HumanEval samples, "
    f"teacher-forced official loop; tok/s at the ~280 stack (pass = (2.1 + 5.06 x (rows/16)^0.72) GB / 224 GB/s + drafting)")
for bs in (16, 32):
    for policy, arg in (("sel", 0.0), ("sel", 0.1), ("sel", 0.3), ("sel", 0.5), ("orc", 16), ("orc", 64)):
        t0 = time.time(); rec = [np.zeros(bs - 1), np.zeros(bs - 1)]; tok = blk = rows = 0; ms = ms8 = 0.0
        for f, ids, off, gen in samples:
            a, b, r, m, m8 = replay(f, ids, off, gen, bs, policy, arg, rec); tok += a; blk += b; rows += r; ms += m; ms8 += m8
        name = f"selector, truncate t {arg}" if policy == "sel" else f"ORACLE over top-{arg} lattice"
        say(f"  block {bs:2d} {name:27s}: {tok / blk:5.2f} tokens per pass at {rows / blk:4.1f} rows, "
            f"{tok / ms * 1000:4.0f} tok/s at the stack (top-8: {tok / ms8 * 1000:4.0f}) ({blk} blocks, {time.time() - t0:.0f} s)")
        if policy == "orc":
            say(f"      top-{arg} recall by depth: " + " ".join(f"{rec[0][j] / max(1, rec[1][j]):.2f}"
                                                         for j in range(0, bs - 1, 1 if bs == 16 else 2)))
say("350 at the stack needs: 12.4 tokens per 16-row pass; at 32 rows ~17.7 (pass ~50.5 ms); top-8 stack: 13.1 / 18.9")
open(os.path.expanduser(f"~/Leap/data/cand21{os.environ.get('R21_TAG', '')}.txt"), "w").write("\n".join(lines) + "\n")
