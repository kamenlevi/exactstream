"""R-77 CHAIN screen (Leap, Sun 2026-11-01; shared GPU, day-sized ~3 GB): chain_fused vs r57_screen's chain on the real
DFlash2 drafter (L1-20 bf16, real 4-bit embed / lm_head), with draft_fused installed (R-76) as in tonight's K arm.
1. Bits: 60 cycles (prefill context 1,000 rows, 8 new context rows per cycle, random features and anchors): path and
   confidences bitwise equal, and E (every predecessor x successor edge) == the shipped per-step sums on the path.
2. Kernels of the chain (census) and ms of the chain piece (synced) and of the whole propose (one eval + read-back).
Usage: [BigMoE venv] r77_chain_test.py -> ~/Leap/data/r77_chain.txt"""
import os, sys, json, re, tempfile, collections, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_fu/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_fu/Leap_third_party"))
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
DFU.install()
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
    pred, path, conf, edges = blk[:, 0], [], [], []
    for p in range(hp.shape[1]):
        e = mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None] * sel.successor_codebook(cand[:, p]), axis=-1)
        sc = unary[:, p] + e
        s_ = mx.argmax(sc, axis=-1)
        pred = mx.take_along_axis(cand[:, p], s_[:, None], axis=-1)[:, 0]
        path.append(pred)
        conf.append(mx.take_along_axis(mx.softmax(sc.astype(mx.float32), axis=-1), s_[:, None], axis=-1)[:, 0])
        edges.append(e)
    return mx.stack(path, axis=1), mx.stack(conf, axis=1), mx.stack(edges, axis=1)

rng = np.random.default_rng(11)
dc = draft.make_cache()
ctx0 = mx.array((rng.standard_normal((1000, 5 * 2560)) * 0.5).astype(np.float16))
mx.eval(draft.hidden_states(mx.array([[1] + [MASK] * (R - 1)]), ctx0[None].astype(mx.bfloat16), dc, 1))
bad, ebad, ties = [], 0, 0
for c in range(60):
    a = dc[0].offset + 8
    ctx = mx.array((rng.standard_normal((8, 5 * 2560)) * 0.5).astype(np.float16))
    blk = mx.array([[int(rng.integers(0, 248000))] + [MASK] * (R - 1)])
    hid = draft.hidden_states(blk, ctx[None].astype(mx.bfloat16), dc, 1)
    cand, unary = head2s(hid)
    hp = sel.hidden_projection(hid)
    p0, c0, e0 = chain(blk, unary, cand, hp)
    p1, c1 = chain_fused(blk, unary, cand, hp, sel)
    mx.eval(p0, c0, e0, p1, c1)
    same = (np.array(p0) == np.array(p1)).all() and (np.array(c0.view(mx.uint32)) == np.array(c1.view(mx.uint32))).all()
    if not same:
        bad.append(c)
    sc = np.array((unary + e0).astype(mx.float32))[0]
    ties += int(sum((np.sort(r)[-1] == np.sort(r)[-2]) for r in sc))
    if (trim := dc[0].offset - a) > 0:
        DF._trim_recent_cache(dc, trim)
say(f"bits over 60 real cycles (path + confidences): {'PASS' if not bad else f'FAIL at {bad[:6]}'}; top-2 ties in the "
    f"walk's scores (first-index rule exercised): {ties}")

# synthetic stress: many ties (scores on a coarse grid) and random candidates
fails = 0
for t in range(200):
    P = 15
    cand = mx.array(rng.integers(0, 248320, (1, P, K)).astype(np.uint32))
    hp = mx.array((rng.standard_normal((1, P, 256)) * (0.05 if t % 2 else 1.0)).astype(np.float32)).astype(mx.bfloat16)
    unary = mx.array(np.round(rng.standard_normal((1, P, K)) * (2 if t % 3 else 8)) / 4).astype(mx.bfloat16)
    blk = mx.array([[int(rng.integers(0, 248000))] + [MASK] * (R - 1)])
    p0, c0, _ = chain(blk, unary, cand, hp)
    p1, c1 = chain_fused(blk, unary, cand, hp, sel)
    mx.eval(p0, c0, p1, c1)
    fails += not ((np.array(p0) == np.array(p1)).all() and (np.array(c0.view(mx.uint32)) == np.array(c1.view(mx.uint32))).all())
say(f"synthetic stress (200 walks, coarse-grid logits for ties): {'PASS' if fails == 0 else f'FAIL ({fails})'}")

VIEWS = {"Reshape", "Broadcast", "BroadcastAxes", "Transpose", "ExpandDims", "Squeeze", "Slice", "AsStrided",
         "Flatten", "Unflatten", "StopGradient", "Depends", "Split"}
def census(*arrs):
    cnt = collections.Counter()
    fd, p = tempfile.mkstemp(suffix=".dot"); os.close(fd)
    try:
        mx.export_to_dot(p, *arrs)
        for line in open(p):
            m = re.search(r'label\s*=\s*"([A-Z][A-Za-z0-9_]*)', line)
            if m:
                cnt[m.group(1)] += 1
    finally:
        os.remove(p)
    return cnt
nk = lambda c: sum(v for k_, v in c.items() if k_ not in VIEWS)
cand, unary = head2s(hid); hp = sel.hidden_projection(hid); mx.eval(cand, unary, hp)
k0 = nk(census(*chain(blk, unary, cand, hp)[:2]))
k1 = nk(census(*chain_fused(blk, unary, cand, hp, sel)))
def tchain(f, n=60):
    ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter()
        pth, cf = f()[:2]
        np.array(pth[0]); np.array(cf[0]); ts.append(time.perf_counter() - t0)
    return float(np.median(ts[5:]))
t0 = min(tchain(lambda: chain(blk, unary, cand, hp)) for _ in range(3))
t1 = min(tchain(lambda: chain_fused(blk, unary, cand, hp, sel)) for _ in range(3))
say(f"chain: kernels {k0} -> {k1}; ms (inputs ready, + read-back) {1e3 * t0:.2f} -> {1e3 * t1:.2f} (shared GPU, rough)")
DFU.uninstall()
open(os.path.expanduser("~/Leap/data/r77_chain.txt"), "w").write("\n".join(lines) + "\n")
