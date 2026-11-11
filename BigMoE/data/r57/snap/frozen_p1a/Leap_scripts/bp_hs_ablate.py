"""BP-3 (Leap, Wed 2026-11-04 16:1x; TABLE_350 'BP-3'): where DFlash2's hidden_states (7.8 ms of a 10.2 ms propose, BP-0)
spends its time, by ablation (timing only; ablated arms compute garbage). bf16 drafter, plain modules (no forms), as in
BP-0's cycles (1,000 context rows, then 8 new context rows + a 16-row block per cycle).
Arms: full; no-proj (every nn.Linear returns a same-shape fill); no-sdpa (attention returns the queries); no-conv (the
grouped dynamic convs are identities); no-norm (RMSNorm identities); skeleton (all four).
usage: [BigMoE venv] bp_hs_ablate.py -> ~/Leap/data/bp_hs_ablate.txt (GPU, MEM ~3 GB; share lock)"""
import os, sys, json, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/Leap_third_party"))
import dflash_model_mlx as DF

OUT = os.path.expanduser("~/Leap/data/bp_hs_ablate.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nBP-3 {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}")
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
MASK, R = int(draft.config.mask_token_id), 16

ORIG = {"lin": nn.Linear.__call__, "sdpa": mx.fast.scaled_dot_product_attention, "prep": DF.GroupedDynamicCausalConv.prepare,
        "fin": DF.GroupedDynamicCausalConv.finish, "norm": nn.RMSNorm.__call__}
def fill_lin(self, x):
    n = self.weight.shape[0]
    return mx.broadcast_to(x[..., :1], (*x.shape[:-1], n)) + 0
def fake_sdpa(q, k, v, scale=1.0, mask=None, **kw):
    return q + 0
def fake_prep(self, hidden):
    groups = hidden.shape[-1] // self.group_size
    return hidden + 0, mx.zeros((*hidden.shape[:-1], self.kernel_size, groups), hidden.dtype)
def fake_fin(self, hidden, dynamic):
    return hidden + 0
def fake_norm(self, x):
    return x + 0
PATCH = {"proj": lambda on: setattr(nn.Linear, "__call__", fill_lin if on else ORIG["lin"]),
         "sdpa": lambda on: setattr(mx.fast, "scaled_dot_product_attention", fake_sdpa if on else ORIG["sdpa"]),
         "conv": lambda on: (setattr(DF.GroupedDynamicCausalConv, "prepare", fake_prep if on else ORIG["prep"]),
                             setattr(DF.GroupedDynamicCausalConv, "finish", fake_fin if on else ORIG["fin"])),
         "norm": lambda on: setattr(nn.RMSNorm, "__call__", fake_norm if on else ORIG["norm"])}
ARMS = {"full": [], "no-proj": ["proj"], "no-sdpa": ["sdpa"], "no-conv": ["conv"], "no-norm": ["norm"],
        "skeleton": ["proj", "sdpa", "conv", "norm"]}

def run(arm, n=24, seed=3):
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
            mx.synchronize(); t0 = time.perf_counter()
            hid = draft.hidden_states(blk, ctx, dc, 1)
            mx.eval(hid); mx.synchronize()
            ts.append(time.perf_counter() - t0)
            if (trim := dc[0].offset - a) > 0:
                DF._trim_recent_cache(dc, trim)
        return float(np.median(ts[4:]))
    finally:
        for k in PATCH:
            PATCH[k](False)

res = {k: [] for k in ARMS}
for rnd in range(3):
    for arm in (list(ARMS) if rnd % 2 == 0 else list(ARMS)[::-1]):
        res[arm].append(run(arm, seed=3 + rnd))
base = np.median(res["full"])
say("hidden_states ms (median of 3 interleaved rounds): " + ", ".join(
    f"{k} {1e3 * np.median(v):.2f} ({100 * (np.median(v) / base - 1):+.0f} %)" for k, v in res.items()))
