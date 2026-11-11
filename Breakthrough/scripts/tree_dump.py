"""TREES (NewIdeas Sun 06:1x; RANKED #10, Leap's option B; chat added 08:3x): dump DFlash2's per-block lattices for a CPU price
of exact draft trees. A copy of Leap's scripts/cand21.py (L1-20 verdict aid, read-only source) with its replay kept on the
selector's own trajectory (policy 'sel', no truncation: the 9.54-tokens-per-pass run on code) and, per block, a record of:
  sel_cand [15, 16]  the selector's top-16 candidate ids per depth (argpartition of the drafter logits, as select_conf)
  sel_prob [15, 16]  the selector's softmax over those 16 at each depth (its first-order scores along its own path)
  pick     [15]      the selector's pick per depth (= its argmax)
  cand     [15, 16]  the same candidates in the drafter's order (argpartition), which 'trans' indexes
  trans    [15, 16, 16]  the selector's first-order lattice: P(candidate j at depth d | candidate i at depth d - 1)
  true     [<=15]    the answer's next tokens (teacher-forced; -1 padded)
  vtop     [<=15]    the capture's own top-1 at those positions (the exact verifier's token; differs from 'true' at 0.3-5 %)
  dom, sample        the capture's domain (file name without its index) and file index
Inputs: TD_SRC, comma-separated globs (default Leap's 24 HumanEval captures; MTP Head's data/d02a has chat / agent / mgsm / gsm
in the same format). CPU only (mx.set_default_device(mx.cpu)), run under taskpolicy -b. TD_F32LM=1 (default): the 4-bit lm_head
dequantized once to an f32 Linear (MLX's CPU quantized matmul took ~115 s per block on the background cores; Accelerate's f32
GEMM is the fast path); logits are rounded to bf16 as the GPU path's are. TD_F32=1: the drafter's weights in f32 as well.
L1-20 weights by default (R21_WEIGHTS).
TD_PROF=N: time the stages of the first N blocks, then stop.
Usage: [BigMoE venv] taskpolicy -b python tree_dump.py -> TD_OUT (default ~/Breakthrough/data/trees/lattice_l120.npz)"""
import os, sys, glob, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, mlx.core as mx, mlx.nn as nn
mx.set_default_device(mx.cpu)
sys.path.insert(0, os.path.expanduser("~/Leap/third_party"))
import dflash_model_mlx as D


class Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(248320, 2560, group_size=64, bits=4)
        self.lm_head = nn.QuantizedLinear(2560, 248320, bias=False, group_size=64, bits=4)


t0 = time.time()
w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
tgt = Target()
tgt.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales", "biases")]
                 + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
del w
if os.environ.get("TD_F32LM", "1") == "1":
    q, lin = tgt.lm_head, nn.Linear(2560, 248320, bias=False)
    lin.weight = mx.dequantize(q.weight, q.scales, q.biases, group_size=64, bits=4).astype(mx.float32)
    tgt.lm_head = lin
draft = D.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.bind(tgt)
WEIGHTS = os.path.expanduser(os.environ.get("R21_WEIGHTS", "~/Leap/data/l120_ft.safetensors"))
if WEIGHTS:
    draft.load_weights(WEIGHTS, strict=False)
if os.environ.get("TD_F32") == "1":                                   # the drafter in f32 too (Accelerate's GEMM path)
    draft.set_dtype(mx.float32)
mx.eval(draft.parameters(), tgt.parameters())
MASK = int(draft.config.mask_token_id)
BS = 16
PROF = int(os.environ.get("TD_PROF", "0"))
OUT = os.path.expanduser(os.environ.get("TD_OUT", "~/Breakthrough/data/trees/lattice_l120.npz"))
print(f"[tree_dump] loaded in {time.time() - t0:.0f} s (lm_head {tgt.lm_head.weight.dtype})", flush=True)


def select_full(block, hidden, logits):
    """cand21.select_conf at temperature 0 (its pick path, unchanged), keeping the selector's softmax along that path, plus its
    whole first-order lattice: the softmax over each depth's 16 candidates given each of the previous depth's 16 (depth 0: given
    the anchor token, the same row 16 times). The candidate sets don't depend on the path (DFlash2 drafts every depth at once);
    only the selector's edge term does, so this is everything the drafter + selector know about any branch."""
    sel = draft.candidate_selector
    cand = mx.argpartition(logits, -sel.top_k, axis=-1)[..., -sel.top_k:]
    unary = mx.take_along_axis(logits, cand, axis=-1)
    hp = sel.hidden_projection(hidden)
    pred, path, probs = block[:, 0], [], []
    for p in range(hp.shape[1]):
        sc = unary[:, p] + mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None] * sel.successor_codebook(cand[:, p]), axis=-1)
        s = mx.argmax(sc, axis=-1)
        probs.append(mx.softmax(sc.astype(mx.float32), axis=-1))
        pred = mx.take_along_axis(cand[:, p], s[:, None], axis=-1)[:, 0]
        path.append(pred)
    succ = sel.successor_codebook(cand)                                             # [1, 15, 16, r]; the loop's own op order
    e0 = mx.sum(sel.predecessor_codebook(block[:, 0])[:, None] * hp[:, 0, None] * succ[:, 0], axis=-1)
    e = mx.sum(sel.predecessor_codebook(cand[:, :-1])[:, :, :, None] * hp[:, 1:, None, None] * succ[:, 1:, None], axis=-1)
    t0 = mx.broadcast_to((unary[:, 0] + e0)[:, None], (1, sel.top_k, sel.top_k))[:, None]
    trans = mx.softmax(mx.concatenate([t0, unary[:, 1:, None, :] + e], axis=1).astype(mx.float32), axis=-1)
    return cand, mx.stack(probs, axis=1), mx.stack(path, axis=1), trans


def tick(x, st, name):
    if PROF:
        mx.eval(x); t = time.time(); st[name] = st.get(name, 0) + t - st["t"]; st["t"] = t
    return x


R = {k: [] for k in ("sel_cand", "sel_prob", "pick", "cand", "trans", "true", "vtop", "ntrue", "sample", "dom")}
paths = [p for g in os.environ.get("TD_SRC", "~/Leap/data/dfl2d_code/*.npz").split(",") for p in sorted(glob.glob(os.path.expanduser(g)))]
t0, nb = time.time(), 0
for si, path in enumerate(paths):
    z = np.load(path)
    f, ids, off, gen = z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])
    top1, dom = z["top1"], os.path.basename(path)[:-4].rsplit("_", 1)[0]
    n = len(ids); cache = draft.make_cache()
    start = max(off, gen - (draft.config.sliding_window - 1))
    for c in cache:
        c.offset = start
    fed, a = start, gen
    while a < off + n - 1:
        st = {"t": time.time()}
        ctx = mx.array(f[fed - off:a - off][None]).astype(mx.float32 if os.environ.get("TD_F32") == "1" else mx.bfloat16)
        blk = mx.array([[int(ids[a - off])] + [MASK] * (BS - 1)])
        h = tick(draft.hidden_states(blk, ctx, cache, 1), st, "draft")
        logits = draft.compute_logits(h)
        if os.environ.get("TD_F32LM", "1") == "1":
            logits = logits.astype(mx.bfloat16)
        logits = tick(logits, st, "lm_head")
        fed = a
        true = np.asarray(ids[a - off + 1:a - off + BS])
        cand, prob, pick, trans = select_full(blk, h, logits)
        mx.eval(cand, prob, pick, trans)
        tick(trans, st, "select")
        cand, prob, pick, trans = np.array(cand[0]), np.array(prob[0]), np.array(pick[0]), np.array(trans[0])
        o = np.argsort(-prob, axis=-1)                                    # each depth's 16 candidates, most probable first
        R["sel_cand"].append(np.take_along_axis(cand, o, -1).astype(np.int32))
        R["sel_prob"].append(np.take_along_axis(prob, o, -1).astype(np.float16))
        R["pick"].append(pick.astype(np.int32))
        R["cand"].append(cand.astype(np.int32)); R["trans"].append(trans.astype(np.float16))
        tt = np.full(BS - 1, -1, np.int32); tt[:len(true)] = true
        vt = np.full(BS - 1, -1, np.int32); vt[:len(true)] = top1[a - off:a - off + len(true)]
        R["true"].append(tt); R["vtop"].append(vt); R["ntrue"].append(len(true)); R["sample"].append(si); R["dom"].append(dom)
        hit = pick[:len(true)] == true                                    # advance as the selector's own run does
        k = int(np.argmin(hit)) if not hit.all() else len(true)
        acc = min(k + 1, off + n - 1 - a)
        if (trim := cache[0].offset - a) > 0:
            D._trim_recent_cache(cache, trim)
        a += acc
        nb += 1
        if PROF:
            print(f"[tree_dump] block {nb}: " + ", ".join(f"{k} {v * 1e3:.0f} ms" for k, v in st.items() if k != "t"), flush=True)
            if nb >= PROF:
                sys.exit(0)
    print(f"[tree_dump] {si + 1}/{len(paths)} {dom} {os.path.basename(path)}: {len(R['pick'])} blocks so far, {time.time() - t0:.0f} s", flush=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(OUT, **{k: np.array(v) for k, v in R.items()}, weights=os.path.basename(WEIGHTS))
print(f"[tree_dump] done: {len(R['pick'])} blocks in {time.time() - t0:.0f} s -> {OUT}", flush=True)
