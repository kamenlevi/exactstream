"""BP-10 (Leap, Wed 2026-11-04 17:0x; TABLE_350 'BP-10'): where the best exact-safe propose (D8-NAX drafter + the
screened draft head on qmm16c, BP-9: ~5.8 ms) spends its time, by ablation (timing only; ablated arms compute garbage).
BP-0's cycles (1,000 context rows; 8 new + a 16-row block per cycle; one eval + read-back per propose).
Arms: full; no-proj (every NaxQL8 call a same-shape fill: layers' Linears, conv kernel projections, fc); no-sdpa (the SDPA
returns q, but still consumes K and V so their concat / projections stay live); no-conv (the grouped dynamic convs,
with their kernel projections, are identities); no-norm (RMSNorm identities); no-head (screened head2s -> constants);
no-chain (chain_fused -> constants); skeleton (all of them).
usage: [BigMoE venv] bp_d8_ablate.py -> ~/Leap/data/bp_d8_ablate.txt (GPU, MEM ~2.5 GB; share lock)"""
import os, sys, json, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs1/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs1/Leap_third_party"))
import dflash_model_mlx as DF
import draft_fused as DFU
import chain_fused as CF
import d7b_nax

OUT = os.path.expanduser("~/Leap/data/bp_d8_ablate.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nBP-10 {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}")
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
n8 = d7b_nax.apply8(draft)
draft.bind(tgt)
mx.eval(draft.parameters(), emb.parameters(), head.parameters())
DFU.install(forms=False, draft=draft)
SUB = d7b_nax.DraftSubset(head)
sel = draft.candidate_selector
MASK, R = int(draft.config.mask_token_id), 16

def propose(blk, ctx, dc):
    hid = draft.hidden_states(blk, ctx, dc, 1)
    cand, unary = d7b_nax.screened_head2s(draft, hid, SUB)
    hp = sel.hidden_projection(hid)
    return CF.chain_fused(blk, unary, cand, hp, sel)

ORIG = {"q8": d7b_nax.NaxQL8.__call__, "sdpa": mx.fast.scaled_dot_product_attention,
        "prep": DF.GroupedDynamicCausalConv.prepare, "fin": DF.GroupedDynamicCausalConv.finish,
        "norm": nn.RMSNorm.__call__, "head": d7b_nax.screened_head2s, "chain": CF.chain_fused}
C = {}
def fill_q8(self, x):
    return mx.broadcast_to(x[..., :1], (*x.shape[:-1], self["weight"].shape[0])) + 0
def fake_sdpa(q, k, v, scale=1.0, mask=None, **kw):
    return q + (k[..., :1, :1].sum() + v[..., :1, :1].sum()).astype(q.dtype) * 0
def fake_prep(self, hidden):
    groups = hidden.shape[-1] // self.group_size
    return hidden + 0, mx.zeros((*hidden.shape[:-1], self.kernel_size, groups), hidden.dtype)
def fake_fin(self, hidden, dynamic):
    return hidden + 0
def fake_norm(self, x):
    return x + 0
def fake_head(draft_, hid, sub, K=16):                 # consumes hid (one element) so hidden_states stays live
    return C["cand"], C["unary"] + (hid[..., :1, :1].sum() * 0).astype(C["unary"].dtype)
def fake_chain(blk, unary, cand, hp, sel_):           # consumes its inputs (one element each) so the head stays live
    z = (unary[..., :1, :1].sum().astype(mx.float32) + hp[..., :1, :1].sum().astype(mx.float32)
         + cand[..., :1, :1].sum().astype(mx.float32)) * 0
    return C["path"] + z.astype(C["path"].dtype), C["conf"] + z.astype(C["conf"].dtype)
PATCH = {"proj": lambda on: setattr(d7b_nax.NaxQL8, "__call__", fill_q8 if on else ORIG["q8"]),
         "sdpa": lambda on: setattr(mx.fast, "scaled_dot_product_attention", fake_sdpa if on else ORIG["sdpa"]),
         "conv": lambda on: (setattr(DF.GroupedDynamicCausalConv, "prepare", fake_prep if on else ORIG["prep"]),
                             setattr(DF.GroupedDynamicCausalConv, "finish", fake_fin if on else ORIG["fin"])),
         "norm": lambda on: setattr(nn.RMSNorm, "__call__", fake_norm if on else ORIG["norm"]),
         "head": lambda on: setattr(d7b_nax, "screened_head2s", fake_head if on else ORIG["head"]),
         "chain": lambda on: setattr(CF, "chain_fused", fake_chain if on else ORIG["chain"])}
ARMS = {"full": [], "no-proj": ["proj"], "no-sdpa": ["sdpa"], "no-conv": ["conv"], "no-norm": ["norm"], "no-head": ["head"],
        "no-chain": ["chain"], "skeleton": ["proj", "sdpa", "conv", "norm", "head", "chain"]}

def run(arm, n=26, seed=3):
    for k in PATCH:
        PATCH[k](k in ARMS[arm])
    try:
        rng = np.random.default_rng(seed)
        dc = draft.make_cache()
        ctx0 = mx.array((rng.standard_normal((1000, 5 * 2560)) * 0.5).astype(np.float16))
        mx.eval(draft.hidden_states(mx.array([[1] + [MASK] * (R - 1)]), ctx0[None].astype(mx.bfloat16), dc, 1))
        ts = []
        for c in range(n):
            a = dc[0].offset + 8
            ctx = mx.array((rng.standard_normal((8, 5 * 2560)) * 0.5).astype(np.float16)).astype(mx.bfloat16)[None]
            blk = mx.array([[int(rng.integers(0, 248000))] + [MASK] * (R - 1)])
            mx.eval(ctx, blk)
            t0 = time.perf_counter()
            pth, cf = propose(blk, ctx, dc)
            np.array(pth); np.array(cf)
            ts.append(time.perf_counter() - t0)
            if (trim := dc[0].offset - a) > 0:
                DF._trim_recent_cache(dc, trim)
        return float(np.median(ts[4:]))
    finally:
        for k in PATCH:
            PATCH[k](False)

# constants for the head / chain stubs from one real propose
rng0 = np.random.default_rng(1)
dc0 = draft.make_cache()
ctx0 = mx.array((rng0.standard_normal((1000, 5 * 2560)) * 0.5).astype(np.float16))
mx.eval(draft.hidden_states(mx.array([[1] + [MASK] * (R - 1)]), ctx0[None].astype(mx.bfloat16), dc0, 1))
blk0 = mx.array([[5] + [MASK] * (R - 1)])
ctx1 = mx.array((rng0.standard_normal((8, 5 * 2560)) * 0.5).astype(np.float16)).astype(mx.bfloat16)[None]
hid0 = draft.hidden_states(blk0, ctx1, dc0, 1)
C["cand"], C["unary"] = d7b_nax.screened_head2s(draft, hid0, SUB)
C["path"], C["conf"] = CF.chain_fused(blk0, C["unary"], C["cand"], sel.hidden_projection(hid0), sel)
mx.eval(C["cand"], C["unary"], C["path"], C["conf"])
res = {k: [] for k in ARMS}
for rnd in range(3):
    for arm in (list(ARMS) if rnd % 2 == 0 else list(ARMS)[::-1]):
        res[arm].append(run(arm, seed=3 + rnd))
base = np.median(res["full"])
say(f"D8-NAX ({n8} NaxQL8) + screened head on qmm16c; ms per propose (median of 3 interleaved rounds): " + ", ".join(
    f"{k} {1e3 * np.median(v):.2f} ({1e3 * (np.median(v) - base):+.2f})" for k, v in res.items()))

# BP-10b: the propose's CPU graph build vs eval (full arm): t_build = propose() returning lazy arrays; t_eval = the
# read-back after it. The engine runs them in series (the draft gates the check).
rng = np.random.default_rng(9)
dc = draft.make_cache()
ctx0 = mx.array((rng.standard_normal((1000, 5 * 2560)) * 0.5).astype(np.float16))
mx.eval(draft.hidden_states(mx.array([[1] + [MASK] * (R - 1)]), ctx0[None].astype(mx.bfloat16), dc, 1))
tb, te = [], []
for c in range(30):
    a = dc[0].offset + 8
    ctx = mx.array((rng.standard_normal((8, 5 * 2560)) * 0.5).astype(np.float16)).astype(mx.bfloat16)[None]
    blk = mx.array([[int(rng.integers(0, 248000))] + [MASK] * (R - 1)])
    mx.eval(ctx, blk)
    t0 = time.perf_counter()
    pth, cf = propose(blk, ctx, dc)
    t1 = time.perf_counter()
    np.array(pth); np.array(cf)
    t2 = time.perf_counter()
    tb.append(t1 - t0); te.append(t2 - t1)
    if (trim := dc[0].offset - a) > 0:
        DF._trim_recent_cache(dc, trim)
say(f"BP-10b full propose: CPU graph build {1e3 * np.median(tb[4:]):.2f} ms + eval / read-back {1e3 * np.median(te[4:]):.2f} ms "
    f"(GPU work runs only after the build: serial)")

# BP-11b A/B: qmm16c8's in-kernel row bound vs the old padding pass (forced back in), interleaved in this process
import qmm16c8 as Q8
_q8 = Q8.qmm16c8
def q8_padded(x, w, s, b, ks=1, nt=1):
    M, K = x.shape
    if M < 16:
        return _q8(mx.concatenate([x, mx.zeros((16 - M, K), x.dtype)], 0), w, s, b, ks=ks, nt=nt)[:M]
    return _q8(x, w, s, b, ks=ks, nt=nt)
ab = {"row bound (new)": [], "padded (old)": []}
for rnd in range(6):
    for arm in (list(ab) if rnd % 2 == 0 else list(ab)[::-1]):
        Q8.qmm16c8 = _q8 if arm.startswith("row") else q8_padded
        ab[arm].append(run("full", seed=40 + rnd))
Q8.qmm16c8 = _q8
say("BP-11b A/B (ms per propose, median of 6 interleaved rounds): " + ", ".join(f"{k} {1e3 * np.median(v):.2f}" for k, v in ab.items()))
