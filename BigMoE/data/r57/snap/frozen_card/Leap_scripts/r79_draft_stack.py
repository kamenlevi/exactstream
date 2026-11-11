"""R-79 DRAFT-STACK (Leap, Sun 2026-11-01; shared GPU, rough, day-sized ~3 GB): the drafter's exact pieces together vs
DFlash2 as shipped, driven as r57_screen drives it: R-76 (conv as one kernel, mask memo), R-77 (the selector walk in 8
kernels), R-78 (the 16-row block projections in their faster bitwise spelling).
1. Bits: 60 cycles on two identically built caches (prefill context 1,000 rows, 8 new rows per cycle): path and confidences
   bitwise equal every cycle, so the drafts (and kept per check) cannot move.
2. ms per propose (one eval + read-back), interleaved arms, and kernels per propose.
Usage: [BigMoE venv] r79_draft_stack.py -> ~/Leap/data/r79_draft_stack.txt"""
import os, sys, json, re, tempfile, collections, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_card/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_card/Leap_third_party"))
import dflash_model_mlx as DF
import draft_fused as DFU
from chain_fused import chain_fused

lines = []
def say(s):
    print(s, flush=True); lines.append(s)
D = os.path.expanduser("~/BigMoE/models/bare")
wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
def get(k):
    return mx.load(f"{D}/{wmap[k]}")[k]
emb = nn.QuantizedEmbedding(248320, 2560, group_size=64, bits=4)
head = nn.QuantizedLinear(2560, 248320, bias=False, group_size=64, bits=4)
for m, p in ((emb, "language_model.model.embed_tokens"), (head, "language_model.lm_head")):
    m.weight, m.scales, m.biases = get(p + ".weight"), get(p + ".scales"), get(p + ".biases")
class _T:
    pass
tgt = _T(); tgt.embed_tokens = emb; tgt.lm_head = head
draft = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.load_weights(os.path.expanduser("~/Leap/data/l120_ft.safetensors"), strict=False)
draft.bind(tgt)
mx.eval(draft.parameters(), emb.parameters(), head.parameters())
MASK, R, K = int(draft.config.mask_token_id), 16, 16
sel = draft.candidate_selector

def head2s(hid):
    lg = draft.compute_logits(hid)
    B, s = 128, lg.shape
    blk = lg.reshape(*s[:-1], s[-1] // B, B)
    tb = mx.argpartition(blk.max(axis=-1), -K, axis=-1)[..., -K:]
    sel_ = mx.take_along_axis(blk, mx.broadcast_to(tb[..., None], (*s[:-1], K, B)), axis=-2).reshape(*s[:-1], K * B)
    j = mx.argpartition(sel_, -K, axis=-1)[..., -K:]
    c = mx.take_along_axis(tb, j // B, axis=-1) * B + j % B
    return c, mx.take_along_axis(lg, c, axis=-1)
def chain(blk, unary, cand, hp):
    pred, path, conf = blk[:, 0], [], []
    for p in range(hp.shape[1]):
        sc = unary[:, p] + mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None]
                                  * sel.successor_codebook(cand[:, p]), axis=-1)
        s_ = mx.argmax(sc, axis=-1)
        pred = mx.take_along_axis(cand[:, p], s_[:, None], axis=-1)[:, 0]
        path.append(pred)
        conf.append(mx.take_along_axis(mx.softmax(sc.astype(mx.float32), axis=-1), s_[:, None], axis=-1)[:, 0])
    return mx.stack(path, axis=1), mx.stack(conf, axis=1)
def propose(blk, ctx, dc, stack):
    hid = draft.hidden_states(blk, ctx, dc, 1)
    cand, unary = head2s(hid)
    hp = sel.hidden_projection(hid)
    return chain_fused(blk, unary, cand, hp, sel) if stack else chain(blk, unary, cand, hp)
def arm(stack):
    DFU.uninstall()
    if stack:
        DFU.install(forms=True, draft=draft)

def seq(stack, seed=21, cycles=60, timing=False):
    arm(stack)
    rng = np.random.default_rng(seed)
    dc = draft.make_cache()
    ctx0 = mx.array((rng.standard_normal((1000, 5 * 2560)) * 0.5).astype(np.float16))
    mx.eval(draft.hidden_states(mx.array([[1] + [MASK] * (R - 1)]), ctx0[None].astype(mx.bfloat16), dc, 1))
    out, ts = [], []
    for c in range(cycles):
        a = dc[0].offset + 8
        ctx = mx.array((rng.standard_normal((8, 5 * 2560)) * 0.5).astype(np.float16)); mx.eval(ctx)
        blk = mx.array([[int(rng.integers(0, 248000))] + [MASK] * (R - 1)])
        t0 = time.perf_counter()
        pth, cf = propose(blk, ctx[None].astype(mx.bfloat16), dc, stack)
        out.append((np.array(pth), np.array(cf.view(mx.uint32))))
        ts.append(time.perf_counter() - t0)
        if (trim := dc[0].offset - a) > 0:
            DF._trim_recent_cache(dc, trim)
    return out, ts
A, _ = seq(False)
B, _ = seq(True)
bad = [c for c, (x, y) in enumerate(zip(A, B)) if not all((p == q).all() for p, q in zip(x, y))]
say(f"bits over 60 cycles (path + confidences): {'PASS' if not bad else f'FAIL at cycles {bad[:6]}'}")
res = {False: [], True: []}
for rep in range(4):
    for stack in ((False, True) if rep % 2 == 0 else (True, False)):
        _, ts = seq(stack, seed=100 + rep, cycles=40, timing=True)
        res[stack].append(float(np.median(ts[5:])))
say(f"ms per propose (one eval + read-back; 4 interleaved rounds, medians): shipped {[f'{1e3 * v:.2f}' for v in res[False]]}, "
    f"stack {[f'{1e3 * v:.2f}' for v in res[True]]}; best {1e3 * min(res[False]):.2f} -> {1e3 * min(res[True]):.2f} ms "
    f"(shared GPU, rough)")
DFU.uninstall()
open(os.path.expanduser("~/Leap/data/r79_draft_stack.txt"), "w").write("\n".join(lines) + "\n")
