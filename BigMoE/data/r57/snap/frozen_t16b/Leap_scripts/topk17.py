"""L1-17 (Leap, Sat 2026-10-24, pre-registered in ~/Leap/TABLE_350.md): fewer experts per token on the 125B. Fit12's
captured MoE inputs (~/Fit12/data/q2/x_L*.npy, read only; 8,336 tokens, layers 2 / 12 / 24 / 36 / 46) through the real
router (softmax over 512, top-10, renormalized: qwen4_exp norm_topk_prob) and the real 4-bit experts + gated shared
expert (weights from ~/BigMoE/models/bare, read only).
Output error of the MoE block, err = sum ||y - y10||^2 / sum ||y10||^2 over tokens, for:
  top-k (k = 8, 6, 4, renormalized over the kept experts); adaptive (smallest set holding >= p of the top-10 weight,
  at least 2); MSB-3 on all 10 routed experts (the reference scale).
Plus the mean union of experts over windows of 16 consecutive tokens (what a 16-row verify pass reads).
Usage: [SpecDecode venv] topk17.py  -> ~/Leap/data/topk17.txt"""
import json, os, time
import numpy as np
import mlx.core as mx
MD = os.path.expanduser("~/BigMoE/models/bare")
XD = os.path.expanduser("~/Fit12/data/q2")
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
cfg = json.load(open(os.path.join(MD, "config.json")))
QC = cfg.get("quantization", {})
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
shards = {}
def tensor(name):
    f = idx[name]
    if f not in shards:
        shards[f] = mx.load(os.path.join(MD, f))
    return shards[f][name]
def qparams(mod):
    o = QC.get(mod, {})
    return (o.get("group_size", QC.get("group_size", 64)), o.get("bits", QC.get("bits", 4))) if isinstance(o, dict) else (64, 4)
def qlinear(mod, x):
    w, s, b = (tensor(mod + "." + k) for k in ("weight", "scales", "biases"))
    gs, bits = qparams(mod)
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=gs, bits=bits)
def msb3(w, s, b):
    """the high 3 bits of each 4-bit code, bias at the midpoint, as a dense bf16 matrix"""
    W4 = mx.dequantize(w, s, b, group_size=64, bits=4)
    codes = mx.stack([(w >> (4 * i)) & 0xF for i in range(8)], -1).reshape(W4.shape)
    s4 = mx.repeat(s.astype(mx.float32), 64, -1); b4 = mx.repeat(b.astype(mx.float32), 64, -1)
    return (2 * s4 * (codes >> 1).astype(mx.float32) + b4 + s4 / 2).astype(mx.bfloat16)

LAYERS = [L for L in (2, 12, 24, 36, 46) if os.path.exists(os.path.join(XD, f"x_L{L}.npy"))]
say(f"L1-17: fewer experts per token, layers {LAYERS}, 8,336 tokens each (Fit12 Q2 capture); MoE-block output error vs top-10")
VAR = ["MSB-3 all 10", "top-8", "top-6", "top-4", "adaptive p0.9", "adaptive p0.8"]
tot = {v: [0.0, 0.0] for v in VAR}; uni = {v: [] for v in ["top-10"] + VAR[1:]}
t0 = time.time()
for L in LAYERS:
    pre = f"language_model.model.layers.{L}.mlp."
    X = mx.array(np.load(os.path.join(XD, f"x_L{L}.npy"))).astype(mx.bfloat16)          # [T, 2560]
    T = X.shape[0]
    gates = mx.softmax(qlinear(pre + "gate", X).astype(mx.float32), axis=-1, precise=True)  # [T, 512]
    order = mx.argsort(-gates, axis=-1)[:, :10]                                         # top-10, by weight
    w10 = mx.take_along_axis(gates, order, axis=-1)
    shared = mx.sigmoid(qlinear(pre + "shared_expert_gate", X).astype(mx.float32)) * \
        qlinear(pre + "shared_expert.down_proj", (lambda g, u: g * mx.sigmoid(g) * u)(
            qlinear(pre + "shared_expert.gate_proj", X), qlinear(pre + "shared_expert.up_proj", X))).astype(mx.float32)
    mx.eval(order, w10, shared)
    order_np = np.array(order); w_np = np.array(w10)
    # each routed expert's output (4-bit and MSB-3) for its tokens: E4 / E3 [T, 10, 2560]
    E4 = np.zeros((T, 10, 2560), np.float32); E3 = np.zeros((T, 10, 2560), np.float32)
    sw = pre + "switch_mlp."
    G = [tensor(sw + p + "." + k) for p in ("gate_proj", "up_proj", "down_proj") for k in ("weight", "scales", "biases")]
    for e in range(512):
        tok, slot = np.nonzero(order_np == e)
        if tok.size == 0:
            continue
        xe = X[mx.array(tok)]
        ws = [G[i][e] for i in range(9)]
        g4 = mx.quantized_matmul(xe, ws[0], ws[1], ws[2], transpose=True, group_size=64, bits=4)
        u4 = mx.quantized_matmul(xe, ws[3], ws[4], ws[5], transpose=True, group_size=64, bits=4)
        y4 = mx.quantized_matmul(g4 * mx.sigmoid(g4) * u4, ws[6], ws[7], ws[8], transpose=True, group_size=64, bits=4)
        Wg3, Wu3, Wd3 = msb3(*ws[0:3]), msb3(*ws[3:6]), msb3(*ws[6:9])
        g3 = xe @ Wg3.T; u3 = xe @ Wu3.T
        y3 = (g3 * mx.sigmoid(g3) * u3) @ Wd3.T
        E4[tok, slot] = np.array(y4.astype(mx.float32)); E3[tok, slot] = np.array(y3.astype(mx.float32))
    sh = np.array(shared)
    wn = w_np / w_np.sum(-1, keepdims=True)                                            # renormalized top-10 weights
    y10 = (wn[..., None] * E4).sum(1) + sh
    den = float((y10 ** 2).sum())
    def keep_mask(var):
        if var.startswith("top-"):
            k = int(var.split("-")[1]); m = np.zeros_like(wn, bool); m[:, :k] = True; return m
        p = float(var.split(" p")[1]); c = np.cumsum(wn, -1)
        kk = np.maximum(2, (c < p).sum(-1) + 1)
        return np.arange(10)[None, :] < kk[:, None]
    for var in VAR:
        if var == "MSB-3 all 10":
            y = (wn[..., None] * E3).sum(1) + sh
        else:
            m = keep_mask(var); wk = np.where(m, w_np, 0.0); wk = wk / wk.sum(-1, keepdims=True)
            y = (wk[..., None] * E4).sum(1) + sh
        tot[var][0] += float(((y - y10) ** 2).sum()); tot[var][1] += den
    # union of experts per window of 16 consecutive tokens
    for var in ["top-10"] + VAR[1:]:
        m = np.ones_like(wn, bool) if var == "top-10" else keep_mask(var)
        us = [len(np.unique(order_np[i:i + 16][m[i:i + 16]])) for i in range(0, T - 15, 16)]
        uni[var].append(float(np.mean(us)))
    say(f"  layer {L:2d} ({time.time() - t0:.0f} s): top-10 weight held by top-1 {wn[:, 0].mean():.3f}, top-4 "
        f"{wn[:, :4].sum(-1).mean():.3f}, top-6 {wn[:, :6].sum(-1).mean():.3f}, top-8 {wn[:, :8].sum(-1).mean():.3f}; "
        f"shared share of ||y10||^2 {float((sh ** 2).sum()) / den:.3f}")
base = tot["MSB-3 all 10"][0] / tot["MSB-3 all 10"][1]
say("variant | MoE-block output error | vs MSB-3 all 10 | 16-token union (mean over layers) | vs top-10")
u10 = np.mean(uni["top-10"])
for var in VAR:
    e = tot[var][0] / tot[var][1]
    u = np.mean(uni[var]) if var in uni else u10
    say(f"  {var:15s} | {e:.4f} | {e / base:5.2f}x | {u:5.1f} | {u / u10:5.2f}")
say(f"  top-10 union {u10:.1f} of 512 per layer")
open(os.path.expanduser("~/Leap/data/topk17.txt"), "w").write("\n".join(lines) + "\n")
