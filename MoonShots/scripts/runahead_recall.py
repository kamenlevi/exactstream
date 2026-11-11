"""Runahead execution for SSD-streamed MoE: how well does it predict the next layers' experts?
Qwen3.6-35B-A3B 4-bit (40 layers x 256 experts, top-8), real weights, ~900 tokens of real text (teacher-forced prefill;
routing matches token-by-token decode on ~98 % of top-8 sets, MoEPrefetch). Per-layer LRU residency at 55 % (141 of 256).
For layer L and each token, three predictors of layer L+1's true top-8:
  look-ahead  : L+1's router on L's pre-MLP residual (BigMoE prefetch v1 / MoEPrefetch pmid1)
  ra-noattn   : L's MoE computed from resident experts only (missing ones substituted by the router's next-best resident
                experts, as MoEPrefetch does), then L+1's router on that residual (cheap runahead: no attention)
  runahead    : as ra-noattn, then layer L+1's attention/GDN on that scratch state, then L+1's router (CPU runahead
                execution, Mutlu et al. 2003, applied to expert streaming). Pessimistic: the scratch run also perturbs
                earlier tokens' K/V, whereas real runahead would only perturb the current token.
Depth 2 (every 4th L): L+2's true experts vs look-ahead-2 (L+2's router on L's pre-MLP residual) and 2-layer runahead.
Recall@B = share of the true top-8 found in the predictor's top-B, averaged over tokens that miss at layer L (runahead only
runs during stalls) and over all tokens. Usage: runahead_recall.py   (GPU, ~20 GB: needs an exclusive window)"""
import os, sys
from collections import OrderedDict
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask

MODEL = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit")
model, tok = load(MODEL)
tm = model.language_model.model
layers = tm.layers
NL = len(layers)
E = layers[0].mlp.num_experts
K = layers[0].mlp.top_k
C = int(0.55 * E)
WARM = 300

text = open(os.path.expanduser("~/SpecDecode/FINDINGS.md")).read()[:6000] + "\n\n" + \
       open(os.path.expanduser("~/SpecDecode/mtp.py")).read()[:4000]
ids = mx.array(tok.encode(text)[:900])[None]
T = ids.shape[1]
h = tm.embed_tokens(ids)
fa = create_attention_mask(h, None)
ssm = create_ssm_mask(h, None)


def mixer(layer, x):
    m = ssm if layer.is_linear else fa
    f = layer.linear_attn if layer.is_linear else layer.self_attn
    return x + f(layer.input_layernorm(x), m, None)


def gates_of(layer, hmid):
    return mx.softmax(layer.mlp.gate(layer.post_attention_layernorm(hmid)), axis=-1, precise=True)


def moe_with(layer, hmid, inds_np, gates):
    """layer's MoE output with the given expert ids per token (scores = router probs, renormalised over the chosen ids)."""
    mlp = layer.mlp
    xn = layer.post_attention_layernorm(hmid)
    inds = mx.array(inds_np[None].astype(np.int32))
    sc = mx.take_along_axis(gates, inds, axis=-1)
    sc = sc / sc.sum(axis=-1, keepdims=True)
    y = (mlp.switch_mlp(xn, inds) * sc[..., None]).sum(axis=-2)
    return y + mx.sigmoid(mlp.shared_expert_gate(xn)) * mlp.shared_expert(xn)


# exact pass: per-layer pre-MLP residual and true routing
x = h
hmid, top16, gts = [], [], []
for L, layer in enumerate(layers):
    hm = mixer(layer, x)
    g = gates_of(layer, hm)
    x = hm + layer.mlp(layer.post_attention_layernorm(hm))
    mx.eval(hm, g, x)
    hmid.append(hm); gts.append(g)
    top16.append(np.argsort(-np.array(g[0].astype(mx.float32)), axis=-1)[:, :16])

# LRU residency per layer (resident set before each token's access)
resident = []
for L in range(NL):
    lru, res = OrderedDict(), []
    for t in range(T):
        res.append(set(lru))
        for e in top16[L][t, :K]:
            lru[int(e)] = 1; lru.move_to_end(int(e))
        while len(lru) > C:
            lru.popitem(last=False)
    resident.append(res)


def substitute(L):
    """per token: the true top-8 with missing experts replaced by the next-best resident ones from the top-16."""
    out = np.zeros((T, K), np.int64); miss = np.zeros(T, bool)
    for t in range(T):
        r = resident[L][t]
        tr = [int(e) for e in top16[L][t, :K]]
        miss[t] = any(e not in r for e in tr)
        ch = [e for e in tr if e in r] + [int(e) for e in top16[L][t, K:] if int(e) in r]
        ch = (ch + [e for e in tr if e not in r])[:K]          # pad (rare) with true ids if too few are resident
        out[t] = ch
    return out, miss


def recall(pred_gates, true_top8, rows):
    p = np.argsort(-np.array(pred_gates[0].astype(mx.float32)), axis=-1)
    r = {}
    for B in (8, 12, 16):
        hit = [len(set(p[t, :B]) & set(true_top8[t])) / K for t in rows]
        r[B] = float(np.mean(hit))
    return r


acc = {k: {B: [] for B in (8, 12, 16)} for k in ("look-ahead", "ra-noattn", "runahead", "ra-attn-only", "ra-attn+shared", "look-ahead-2", "runahead-2")}
acc_all = {k: {B: [] for B in (8, 12, 16)} for k in ("look-ahead", "runahead")}
stall_share = []
for L in range(NL - 1):
    sub, miss = substitute(L)
    rows = [t for t in range(WARM, T) if miss[t]]
    allr = list(range(WARM, T))
    stall_share.append(len(rows) / len(allr))
    nxt = layers[L + 1]
    x1 = hmid[L] + moe_with(layers[L], hmid[L], sub, gts[L])          # layer L output from resident experts
    hm1 = mixer(nxt, x1)                                               # runahead through L+1's attention/GDN
    g_la, g_na, g_ra = gates_of(nxt, hmid[L]), gates_of(nxt, x1), gates_of(nxt, hm1)
    # cheaper slices: skip L's routed experts entirely (attention only), or add only the always-resident shared expert
    xn = layers[L].post_attention_layernorm(hmid[L])
    shared = mx.sigmoid(layers[L].mlp.shared_expert_gate(xn)) * layers[L].mlp.shared_expert(xn)
    g_ao = gates_of(nxt, mixer(nxt, hmid[L]))
    g_as = gates_of(nxt, mixer(nxt, hmid[L] + shared))
    mx.eval(g_la, g_na, g_ra, g_ao, g_as)
    true1 = top16[L + 1][:, :K]
    for name, g in (("look-ahead", g_la), ("ra-noattn", g_na), ("runahead", g_ra), ("ra-attn-only", g_ao),
                    ("ra-attn+shared", g_as)):
        for B, v in recall(g, true1, rows).items():
            acc[name][B].append(v)
    for name, g in (("look-ahead", g_la), ("runahead", g_ra)):
        for B, v in recall(g, true1, allr).items():
            acc_all[name][B].append(v)
    if L % 4 == 0 and L + 2 < NL:                                      # depth 2
        sub1, _ = substitute(L + 1)
        # runahead continues: L+1's MoE from resident experts, routed by the runahead's own L+1 prediction
        pred1 = np.argsort(-np.array(g_ra[0].astype(mx.float32)), axis=-1)
        sub1r = np.zeros((T, K), np.int64)
        for t in range(T):
            r = resident[L + 1][t]
            ch = [int(e) for e in pred1[t, :16] if int(e) in r][:K]
            sub1r[t] = (ch + [int(e) for e in pred1[t, :K] if int(e) not in ch])[:K]
        x2 = hm1 + moe_with(nxt, hm1, sub1r, g_ra)
        hm2 = mixer(layers[L + 2], x2)
        g_la2, g_ra2 = gates_of(layers[L + 2], hmid[L]), gates_of(layers[L + 2], hm2)
        mx.eval(g_la2, g_ra2)
        for name, g in (("look-ahead-2", g_la2), ("runahead-2", g_ra2)):
            for B, v in recall(g, top16[L + 2][:, :K], rows).items():
                acc[name][B].append(v)
    print(f"L={L:>2} stall share {stall_share[-1]:.0%}  recall@16 look-ahead {acc['look-ahead'][16][-1]:.3f} "
          f"ra-noattn {acc['ra-noattn'][16][-1]:.3f} runahead {acc['runahead'][16][-1]:.3f}", flush=True)

print(f"\n{T} tokens, 55 % resident (LRU, C={C}); tokens that miss at layer L: {np.mean(stall_share):.0%} on average")
print("recall of the next layer's true top-8 (mean over layers), on tokens that stall at L:")
for name in ("look-ahead", "ra-noattn", "runahead", "ra-attn-only", "ra-attn+shared", "look-ahead-2", "runahead-2"):
    print(f"  {name:13s} " + "  ".join(f"@{B}: {np.mean(v):.3f}" for B, v in acc[name].items()))
print("on all tokens:")
for name in ("look-ahead", "runahead"):
    print(f"  {name:13s} " + "  ".join(f"@{B}: {np.mean(v):.3f}" for B, v in acc_all[name].items()))
