"""Cheap continuous runahead slice: recall of layer T's true top-8 at distance d = 2, 3 (Qwen3.6-35B-A3B 4-bit, real weights,
~900 tokens of real text, teacher-forced; same harness as runahead_recall.py). Seed = layer s = T-d's pre-MoE residual (what
router reuse at distance d uses). The slice runs, for j = s .. T-1: x = hm + MoEapprox_j(hm), hm = attn_{j+1}(x); then T's router.
Variants (attention weights x MoE approximation):
  attention: exact (the model's 4-bit) | 2bit (projections requantized to 2-bit g64) | lr<r> (rank-r SVD, factors 4-bit g64)
  MoE approx: shared (shared expert only) | dflt (shared + per-layer mean of the routed output, calibrated on tokens < WARM)
              | resid (routed MoE from the experts resident under LRU at 55 %, the 0.999 variant)
Baseline: router reuse (T's router on hmid[s]). Pessimistic like round 8: the slice perturbs all tokens' GDN/KV history.
Prints recall@8/12/16 averaged over targets and tokens >= WARM, and the slice's bytes per predicted layer.
usage: slice_recall35.py [variants]   (GPU, ~22 GB: exclusive window)"""
import os, sys, time
from collections import OrderedDict
import numpy as np
import mlx.core as mx
import mlx.nn as nn
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


def attn_mod(layer):
    return layer.linear_attn if layer.is_linear else layer.self_attn


def mixer(layer, x, mod=None):
    m = ssm if layer.is_linear else fa
    f = mod if mod is not None else attn_mod(layer)
    return x + f(layer.input_layernorm(x), m, None)


def gates_of(layer, hmid):
    return mx.softmax(layer.mlp.gate(layer.post_attention_layernorm(hmid)), axis=-1, precise=True)


def shared_of(layer, hm):
    xn = layer.post_attention_layernorm(hm)
    return mx.sigmoid(layer.mlp.shared_expert_gate(xn)) * layer.mlp.shared_expert(xn)


def routed_with(layer, hm, inds_np, gates):
    xn = layer.post_attention_layernorm(hm)
    inds = mx.array(inds_np[None].astype(np.int32))
    sc = mx.take_along_axis(gates, inds, axis=-1)
    sc = sc / sc.sum(axis=-1, keepdims=True)
    return (layer.mlp.switch_mlp(xn, inds) * sc[..., None]).sum(axis=-2)


HEADW = {}                                         # layer -> per value-head contribution weight (calibrated, tokens < WARM)
# exact pass
x = h
hmid, top16, gts, dflt = [], [], [], []
for L, layer in enumerate(layers):
    hm = mixer(layer, x)
    if layer.is_linear:
        mod = layer.linear_attn
        cap = {}
        orig_out = mod.out_proj
        def _cap(u, _o=orig_out, _c=cap): _c["u"] = u; return _o(u)
        mod.out_proj = _cap
        mod(layer.input_layernorm(x[:, :WARM]), ssm[..., :WARM] if ssm is not None else None, None)
        mod.out_proj = orig_out
        u = cap["u"][0].astype(mx.float32)                                            # [WARM, Hv*Dv]
        Hv = mod.num_v_heads; Dv = u.shape[-1] // Hv
        wo = mx.dequantize(orig_out.weight, orig_out.scales, orig_out.biases, group_size=orig_out.group_size, bits=orig_out.bits).astype(mx.float32)
        HEADW[L] = np.array([float(mx.abs(u[:, i * Dv:(i + 1) * Dv] @ wo[:, i * Dv:(i + 1) * Dv].T).mean()) for i in range(Hv)])
    g = gates_of(layer, hm)
    sh = shared_of(layer, hm)
    out = layer.mlp(layer.post_attention_layernorm(hm))
    dflt.append((out - sh)[:, :WARM].astype(mx.float32).mean(axis=1, keepdims=True).astype(hm.dtype))   # mean routed part
    x = hm + out
    mx.eval(hm, g, x, dflt[-1])
    hmid.append(hm); gts.append(g)
    top16.append(np.argsort(-np.array(g[0].astype(mx.float32)), axis=-1)[:, :16])
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


def topres(L, hm, pred_gates, m):
    """routed output from only the m best resident experts by predicted gate weight (not renormalised: their own weights)."""
    p = np.argsort(-np.array(pred_gates[0].astype(mx.float32)), axis=-1)
    inds = np.zeros((T, m), np.int64)
    for t in range(T):
        r = resident[L][t]
        ch = [int(e) for e in p[t, :24] if int(e) in r][:m]
        inds[t] = (ch + [int(e) for e in p[t, :m] if int(e) not in ch])[:m]
    xn = layers[L].post_attention_layernorm(hm)
    ii = mx.array(inds[None].astype(np.int32))
    sc = mx.take_along_axis(pred_gates, ii, axis=-1)
    return (layers[L].mlp.switch_mlp(xn, ii) * sc[..., None]).sum(axis=-2)


def resident_ids(L, pred_gates):
    p = np.argsort(-np.array(pred_gates[0].astype(mx.float32)), axis=-1)
    out = np.zeros((T, K), np.int64)
    for t in range(T):
        r = resident[L][t]
        ch = [int(e) for e in p[t, :24] if int(e) in r][:K]
        out[t] = (ch + [int(e) for e in p[t, :K] if int(e) not in ch])[:K]
    return out


# ---- shadow attention modules
def requant(ql, bits):
    w = mx.dequantize(ql.weight, ql.scales, ql.biases, group_size=ql.group_size, bits=ql.bits)
    q = nn.QuantizedLinear(w.shape[1], w.shape[0], bias=False, group_size=64, bits=bits)
    q.weight, q.scales, q.biases = mx.quantize(w, group_size=64, bits=bits)
    return q, q.weight.nbytes + q.scales.nbytes + q.biases.nbytes


class LowRank(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a, self.b = a, b                              # x @ A^T -> [r], then @ B^T -> [out]

    def __call__(self, x):
        return self.b(self.a(x))


def lowrank(ql, r):
    w = np.array(mx.dequantize(ql.weight, ql.scales, ql.biases, group_size=ql.group_size, bits=ql.bits).astype(mx.float32))
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    A = mx.array((vt[:r] * np.sqrt(s[:r])[:, None]).astype(np.float32)).astype(mx.bfloat16)        # [r, in]
    B = mx.array((u[:, :r] * np.sqrt(s[:r])[None]).astype(np.float32)).astype(mx.bfloat16)        # [out, r]
    qa = nn.QuantizedLinear(A.shape[1], r, bias=False, group_size=64, bits=4); qa.weight, qa.scales, qa.biases = mx.quantize(A, 64, 4)
    qb = nn.QuantizedLinear(r, B.shape[0], bias=False, group_size=64, bits=4); qb.weight, qb.scales, qb.biases = mx.quantize(B, 64, 4)
    nb = sum(a.nbytes for a in (qa.weight, qa.scales, qa.biases, qb.weight, qb.scales, qb.biases))
    return LowRank(qa, qb), nb


PROJ = ("in_proj_qkvz", "in_proj_ba", "out_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
        "q_proj", "k_proj", "v_proj", "o_proj")




def prune_heads(mod, L, frac):
    """zero the out_proj input columns of the weakest value heads (by mean |contribution| on calibration tokens)."""
    ql = mod.out_proj
    w = mx.dequantize(ql.weight, ql.scales, ql.biases, group_size=ql.group_size, bits=ql.bits)     # [d, Hv*Dv]
    Hv = mod.num_v_heads; Dv = w.shape[1] // Hv
    score = HEADW[L]
    drop = np.argsort(score)[: int(frac * Hv)]
    mask = np.ones(w.shape[1], np.float32)
    for hh in drop: mask[hh * Dv:(hh + 1) * Dv] = 0
    return w * mx.array(mask)[None].astype(w.dtype)


def shadow(mod, kind, L=None):
    """a shallow clone of an attention module with its quantized projections replaced; returns (module, bytes of projections)."""
    import copy
    m2 = copy.copy(mod)
    nb = 0
    for name in PROJ:
        ql = getattr(mod, name, None)
        if not isinstance(ql, nn.QuantizedLinear):
            continue
        if kind == "exact":
            nb += ql.weight.nbytes + ql.scales.nbytes + ql.biases.nbytes
            continue
        if kind.startswith("gdn2bit") and not hasattr(mod, "num_v_heads"):
            nb += ql.weight.nbytes + ql.scales.nbytes + ql.biases.nbytes      # full-attention layer: exact
            continue
        if kind == "gdn2bit_h50" and name == "out_proj":
            w = prune_heads(mod, L, 0.5)
            q = nn.QuantizedLinear(w.shape[1], w.shape[0], bias=False, group_size=64, bits=2)
            q.weight, q.scales, q.biases = mx.quantize(w, group_size=64, bits=2)
            b = (q.weight.nbytes + q.scales.nbytes + q.biases.nbytes) // 2           # pruned heads' columns skipped
        elif kind == "gdn2bit_h50" and name in ("in_proj_qkvz", "in_proj_qkv", "in_proj_z"):
            q, b = requant(ql, 2); b = int(b * 0.67)                                  # v and z rows of pruned heads skipped
        elif kind in ("2bit", "gdn2bit", "gdn2bit_h50"):
            q, b = requant(ql, 2)
        else:
            r = int(kind[2:])
            q, b = (lowrank(ql, r) if min(ql.weight.shape[0], ql.weight.shape[1] * 32 // ql.bits) > 2 * r else requant(ql, 4))
        object.__setattr__(m2, name, q) if False else setattr(m2, name, q)
        nb += b
    return m2, nb


variants = sys.argv[1].split(",") if len(sys.argv) > 1 else ["exact:shared", "exact:dflt", "exact:resid", "2bit:dflt", "lr256:dflt", "lr128:dflt"]
kinds = sorted({v.split(":")[0] for v in variants})
shadows, sbytes = {}, {}
for kd in kinds:
    t0 = time.perf_counter()
    shadows[kd] = []; tot = 0
    for Li, layer in enumerate(layers):
        m2, nb = shadow(attn_mod(layer), kd, Li)
        mx.eval(m2.parameters())
        shadows[kd].append(m2); tot += nb
    sbytes[kd] = tot / NL
    print(f"shadow {kd}: attention projections {sbytes[kd]/1e6:.2f} MB/layer ({time.perf_counter()-t0:.0f}s)", flush=True)


def recall(pg, true8, rows):
    p = np.argsort(-np.array(pg[0].astype(mx.float32)), axis=-1)
    return {B: float(np.mean([len(set(p[t, :B]) & set(true8[t])) / K for t in rows])) for B in (8, 12, 16)}


rows = list(range(WARM, T))
acc = {}
for d in (2, 3):
    for Tg in range(d, NL):
        s = Tg - d
        true8 = top16[Tg][:, :K]
        g_rr = gates_of(layers[Tg], hmid[s]); mx.eval(g_rr)
        for B, v in recall(g_rr, true8, rows).items(): acc.setdefault(("router-reuse", d), {}).setdefault(B, []).append(v)
        for var in variants:
            kd, ma = var.split(":")
            hm, g_prev = hmid[s], gts[s]
            for j in range(s, Tg):
                if ma == "shared":
                    xo = hm + shared_of(layers[j], hm)
                elif ma == "dflt":
                    xo = hm + shared_of(layers[j], hm) + dflt[j]
                elif ma in ("res1", "res2"):
                    xo = hm + shared_of(layers[j], hm) + topres(j, hm, g_prev, int(ma[3]))
                else:
                    xo = hm + shared_of(layers[j], hm) + routed_with(layers[j], hm, resident_ids(j, g_prev), g_prev)
                hm = mixer(layers[j + 1], xo, shadows[kd][j + 1])
                g_prev = gates_of(layers[j + 1], hm)
                mx.eval(hm, g_prev)
            for B, v in recall(g_prev, true8, rows).items(): acc.setdefault((var, d), {}).setdefault(B, []).append(v)
        if Tg % 8 == 0:
            print(f"d={d} T={Tg}: router-reuse@12 {acc[('router-reuse', d)][12][-1]:.3f} " +
                  " ".join(f"{v}@12 {acc[(v, d)][12][-1]:.3f}" for v in variants), flush=True)
print(f"\n{T} tokens (recall over tokens >= {WARM}), mean over target layers; slice attention bytes per predicted layer:")
for d in (2, 3):
    for name in ["router-reuse"] + variants:
        r = acc[(name, d)]
        kd = name.split(":")[0]
        ma = name.split(":")[1] if ":" in name else ""
        ew = sum(a.nbytes for a in (layers[0].mlp.switch_mlp.gate_proj.weight, layers[0].mlp.switch_mlp.gate_proj.scales,
                                    layers[0].mlp.switch_mlp.gate_proj.biases)) * 3 / E
        moe_b = {"shared": 0, "dflt": 0, "res1": ew, "res2": 2 * ew, "resid": K * ew}.get(ma, 0)
        full = sbytes.get("exact", 18e6) + K * ew
        extra = (f"  per step: attn {sbytes[kd]/1e6:5.2f} + routed {moe_b/1e6:5.2f} MB = {(sbytes[kd] + moe_b)/full:.2f} of a layer (x {d} steps)"
                 if name != "router-reuse" else "")
        print(f"  d={d} {name:14s} " + "  ".join(f"@{B}: {np.mean(v):.3f}" for B, v in r.items()) + extra)
