"""R-76 DRAFT-FUSE screen (Leap, Sun 2026-11-01; shared GPU, day-sized ~3 GB): draft_fused.install() (one kernel per
grouped dynamic conv call, the attention mask built once per key) vs DFlash2 as shipped, driven as r57_screen drives it.
1. Bits: the same cycle sequence (prefill context 1,000 rows, then 30 cycles of 8 new context rows with random features,
   the block's anchor token varied) on two identically built draft caches: hidden_states, the candidates, the path and
   the confidences bitwise equal every cycle.
2. Kernels per propose (census) and ms per propose (one eval + read-back; synced pieces too), shared GPU, rough.
Usage: [BigMoE venv] r76_draft_fuse.py -> ~/Leap/data/r76_draft_fuse.txt"""
import os, sys, json, re, tempfile, collections, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p4s/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p4s/Leap_third_party"))
import dflash_model_mlx as DF
import draft_fused as DFU

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
cfg = draft.config
MASK, R, K = int(cfg.mask_token_id), 16, 16
say(f"base_kernel dtype {draft.layers[0].attention_conv.base_kernel.dtype}; conv kernel {cfg.conv_kernel_size}, group "
    f"{cfg.conv_group_size}")

def head2s(hid):
    lg = draft.compute_logits(hid)
    B, s = 128, lg.shape
    blk = lg.reshape(*s[:-1], s[-1] // B, B)
    tb = mx.argpartition(blk.max(axis=-1), -K, axis=-1)[..., -K:]
    sel = mx.take_along_axis(blk, mx.broadcast_to(tb[..., None], (*s[:-1], K, B)), axis=-2).reshape(*s[:-1], K * B)
    j = mx.argpartition(sel, -K, axis=-1)[..., -K:]
    c = mx.take_along_axis(tb, j // B, axis=-1) * B + j % B
    return c, mx.take_along_axis(lg, c, axis=-1)
def chain(blk, unary, cand, hp):
    sel = draft.candidate_selector
    pred, path, conf = blk[:, 0], [], []
    for p in range(hp.shape[1]):
        sc = unary[:, p] + mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None]
                                  * sel.successor_codebook(cand[:, p]), axis=-1)
        s_ = mx.argmax(sc, axis=-1)
        pred = mx.take_along_axis(cand[:, p], s_[:, None], axis=-1)[:, 0]
        path.append(pred)
        conf.append(mx.take_along_axis(mx.softmax(sc.astype(mx.float32), axis=-1), s_[:, None], axis=-1)[:, 0])
    return mx.stack(path, axis=1), mx.stack(conf, axis=1)
def propose(blk, ctx, dc):
    hid = draft.hidden_states(blk, ctx, dc, 1)
    cand, unary = head2s(hid)
    pth, cf = chain(blk, unary, cand, draft.candidate_selector.hidden_projection(hid))
    return hid, cand, pth, cf

def run_seq(fused, n0=1000, cycles=30, seed=5):
    (DFU.install if fused else DFU.uninstall)()
    rng = np.random.default_rng(seed)
    dc = draft.make_cache()
    ctx0 = mx.array((rng.standard_normal((n0, 5 * 2560)) * 0.5).astype(np.float16))
    hid = draft.hidden_states(mx.array([[1] + [MASK] * (R - 1)]), ctx0[None].astype(mx.bfloat16), dc, 1)
    mx.eval(hid)
    out = []
    for c in range(cycles):
        a = dc[0].offset + 8
        ctx = mx.array((rng.standard_normal((8, 5 * 2560)) * 0.5).astype(np.float16))
        blk = mx.array([[int(rng.integers(0, 248000))] + [MASK] * (R - 1)])
        hid, cand, pth, cf = propose(blk, ctx[None].astype(mx.bfloat16), dc)
        mx.eval(hid, cand, pth, cf)
        out.append((np.array(hid.view(mx.uint16)), np.array(cand), np.array(pth), np.array(cf.view(mx.uint32))))
        if (trim := dc[0].offset - a) > 0:
            DF._trim_recent_cache(dc, trim)
    return out
A = run_seq(False)
B = run_seq(True)
bad = [c for c, (x, y) in enumerate(zip(A, B)) if not all((p == q).all() for p, q in zip(x, y))]
say(f"bits over 30 cycles (hidden, candidates, path, confidences): {'PASS' if not bad else f'FAIL at cycles {bad[:5]}'}")

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

def timed(fused, n=40):
    (DFU.install if fused else DFU.uninstall)()
    rng = np.random.default_rng(9)
    dc = draft.make_cache()
    ctx0 = mx.array((rng.standard_normal((1000, 5 * 2560)) * 0.5).astype(np.float16))
    mx.eval(draft.hidden_states(mx.array([[1] + [MASK] * (R - 1)]), ctx0[None].astype(mx.bfloat16), dc, 1))
    ts, th, cen = [], [], None
    for c in range(n + 5):
        a = dc[0].offset + 8
        ctx = mx.array((rng.standard_normal((8, 5 * 2560)) * 0.5).astype(np.float16)); mx.eval(ctx)
        blk = mx.array([[int(rng.integers(0, 248000))] + [MASK] * (R - 1)])
        t0 = time.perf_counter()
        if c == 2:
            hid = draft.hidden_states(blk, ctx[None].astype(mx.bfloat16), dc, 1)
            ch = census(hid)
            cand, unary = head2s(hid)
            pth, cf = chain(blk, unary, cand, draft.candidate_selector.hidden_projection(hid))
            cen = (nk(ch), nk(census(pth, cf)))
            mx.eval(pth, cf)
        elif c % 2:
            hid = draft.hidden_states(blk, ctx[None].astype(mx.bfloat16), dc, 1); mx.eval(hid); t1 = time.perf_counter()
            cand, unary = head2s(hid)
            pth, cf = chain(blk, unary, cand, draft.candidate_selector.hidden_projection(hid))
            np.array(pth[0]); np.array(cf[0])
            if c >= 5:
                th.append(t1 - t0)
        else:
            _, _, pth, cf = propose(blk, ctx[None].astype(mx.bfloat16), dc)
            np.array(pth[0]); np.array(cf[0])
            if c >= 5:
                ts.append(time.perf_counter() - t0)
        if (trim := dc[0].offset - a) > 0:
            DF._trim_recent_cache(dc, trim)
    return float(np.median(ts)), float(np.median(th)), cen
res = {}
for rep in range(2):
    for fused in (False, True):
        t, h, cen = timed(fused)
        res.setdefault(fused, []).append((t, h, cen))
for fused in (False, True):
    t = min(r[0] for r in res[fused]); h = min(r[1] for r in res[fused]); cen = res[fused][0][2]
    say(f"{'fused' if fused else 'shipped'}: kernels hidden_states {cen[0]}, whole propose {cen[1]}; ms per propose (one eval + "
        f"read-back) {1e3 * t:.2f}, hidden_states synced {1e3 * h:.2f} (shared GPU, rough)")
DFU.uninstall()
open(os.path.expanduser("~/Leap/data/r76_draft_fuse.txt"), "w").write("\n".join(lines) + "\n")
