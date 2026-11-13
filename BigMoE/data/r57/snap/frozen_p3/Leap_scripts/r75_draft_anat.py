"""R-75 DRAFT-ANAT (Leap, Sun 2026-11-01; shared GPU, day-sized ~2.5 GB): where the drafter's share of the 45 GB cycle goes
(cycle 131.9 - check + commit 121.1 = ~10.8 ms in REF2-T's C 224 run, s45t_B). DFlash2 (incoai/Qwen3.8-Flash-Next-DFlash2 +
Leap's L1-20 fine-tune data/l120_ft.safetensors) bound to the 125B's real 4-bit embed_tokens / lm_head, driven exactly as
BigMoE's r57_screen does (propose = hidden_states -> head2s -> chain; the chain's path is read back to the CPU).
Per piece: ms (synced per piece, best of medians) and kernel census; the whole propose as one eval + the read-back; at
steady state (8 new context rows per cycle) and after a prefill (1,000 context rows). Random context features of the
right shape (5 taps x 2560, fp16 -> bf16); timings, not drafts, are the point.
Usage: [BigMoE venv] r75_draft_anat.py -> ~/Leap/data/r75_draft_anat.txt"""
import os, sys, json, re, tempfile, collections, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_third_party"))
import dflash_model_mlx as DF

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
MASK, SWIN, R, K = int(cfg.mask_token_id), int(cfg.sliding_window), 16, 16
say(f"DFlash2: {cfg.num_hidden_layers} layers {cfg.layer_types}, hidden {cfg.hidden_size}, heads {cfg.num_attention_heads}/"
    f"{cfg.num_key_value_heads} x {cfg.head_dim}, ffn {cfg.intermediate_size}, window {SWIN}, selector rank {cfg.selector_rank}"
    f" top-k {cfg.selector_top_k}; taps {cfg.target_layer_ids}")

def head2s(hid):                                         # r57_screen's head2s / chain / propose, verbatim
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

mx.random.seed(0)
def fresh(n0):
    dc = draft.make_cache()
    ctx0 = (mx.random.normal((n0, 5 * 2560)) * 0.5).astype(mx.float16)
    # the first propose after a prefill: the context's rows enter the draft cache
    blk = mx.array([[1] + [MASK] * (R - 1)])
    hid = draft.hidden_states(blk, ctx0[None].astype(mx.bfloat16), dc, 1)
    mx.eval(hid)
    return dc

def one_cycle(dc, n_new, parts):
    """one steady-state propose: n_new context rows, then trim the block rows back out (r57_screen's bookkeeping)"""
    a = dc[0].offset + n_new
    ctx = (mx.random.normal((n_new, 5 * 2560)) * 0.5).astype(mx.float16)
    mx.eval(ctx)
    blk = mx.array([[7] + [MASK] * (R - 1)])
    t0 = time.perf_counter()
    if parts:
        hid = draft.hidden_states(blk, ctx[None].astype(mx.bfloat16), dc, 1); mx.eval(hid); t1 = time.perf_counter()
        cand, unary = head2s(hid); mx.eval(cand, unary); t2 = time.perf_counter()
        hp = draft.candidate_selector.hidden_projection(hid)
        pth, cf = chain(blk, unary, cand, hp); dr = np.array(pth[0]).tolist(); np.array(cf[0]); t3 = time.perf_counter()
        out = (t1 - t0, t2 - t1, t3 - t2, t3 - t0)
    else:
        hid = draft.hidden_states(blk, ctx[None].astype(mx.bfloat16), dc, 1)
        cand, unary = head2s(hid)
        pth, cf = chain(blk, unary, cand, draft.candidate_selector.hidden_projection(hid))
        dr = np.array(pth[0]).tolist(); np.array(cf[0])
        out = (time.perf_counter() - t0,)
    if (trim := dc[0].offset - a) > 0:
        DF._trim_recent_cache(dc, trim)
    return out

# census of one propose (steady state)
dc = fresh(1000)
ctx = (mx.random.normal((8, 5 * 2560)) * 0.5).astype(mx.float16); mx.eval(ctx)
blk = mx.array([[7] + [MASK] * (R - 1)])
a = dc[0].offset + 8
hid = draft.hidden_states(blk, ctx[None].astype(mx.bfloat16), dc, 1)
c_h = census(hid)
cand, unary = head2s(hid)
c_hs = census(cand, unary)
pth, cf = chain(blk, unary, cand, draft.candidate_selector.hidden_projection(hid))
c_all = census(pth, cf)
mx.eval(pth, cf)
if (trim := dc[0].offset - a) > 0:
    DF._trim_recent_cache(dc, trim)
say(f"kernels per propose: hidden_states {nk(c_h)}, + head2s {nk(c_hs) - nk(c_h)}, + chain {nk(c_all) - nk(c_hs)}; total {nk(c_all)}")
say("  hidden_states: " + ", ".join(f"{k_} {v}" for k_, v in c_h.most_common() if k_ not in VIEWS))
say("  whole: " + ", ".join(f"{k_} {v}" for k_, v in c_all.most_common(14) if k_ not in VIEWS))

for n0, n_new in ((1000, 8), (1000, 16), (300, 8)):
    dc = fresh(n0)
    for _ in range(5):
        one_cycle(dc, n_new, True)
    P, W = [], []
    for _ in range(40):
        P.append(one_cycle(dc, n_new, True))
        W.append(one_cycle(dc, n_new, False)[0])
    P = np.array(P)
    say(f"context {n0} rows, {n_new} new per cycle: synced pieces hidden_states {1e3 * np.median(P[:, 0]):.2f} ms, head2s "
        f"(lm_head 248,320 x 16 rows + top-k) {1e3 * np.median(P[:, 1]):.2f}, chain (15 steps + read-back) "
        f"{1e3 * np.median(P[:, 2]):.2f}, sum {1e3 * np.median(P[:, 3]):.2f}; one eval + read-back {1e3 * np.median(W):.2f} ms "
        f"(shared GPU, rough)")
open(os.path.expanduser("~/Leap/data/r75_draft_anat.txt"), "w").write("\n".join(lines) + "\n")
