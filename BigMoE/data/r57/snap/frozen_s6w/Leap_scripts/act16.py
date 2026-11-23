"""L1-16 step 2 (Leap, Sat 2026-10-24): OUTPUT error of representations that share structure across a layer's experts, at
equal bits per parameter, on the 125B. Same metric, layers and experts as Fit12's Q2 (scripts/fit_q2.py) so the numbers
compare directly: err(W) = tr((W - W4) H (W - W4)^T) / tr(W4 H W4^T); H = X^T X for gate / up (X = the layer's MoE inputs,
Fit12's capture ~/Fit12/data/q2/x_L*.npy, read only), H = h^T h for down (h = silu(X Wg4^T) * (X Wu4^T)); 8 random experts
per layer (seed 0). Bits per parameter include bf16 scale + bias per 64 values; per-layer shared parts are amortized over
the 512 experts.
Formats:
  MSB-3 (3.5 b): the high 3 bits of each 4-bit code, bias at the midpoint (the B12 / v3 base tier);
  RTN-2 g64 (2.5 b);
  SUB-r: shared input subspace (gate / up: top-r eigenvectors of H, the same H for every expert of the layer) or shared
    output subspace (down: top-r eigenvectors of sum_e W4_e Hh_e W4_e^T over the 512 experts), per-expert coefficients
    at 4 bits g64 (4.5 b each) -> 4.5 r / 2560 b;
  SUB2-r: the same with 2-bit coefficients (2.5 b each) -> 2.5 r / 2560 b;
  LR-r: per-expert activation-aware low rank (SVD of W H^1/2), both factors at 4 bits g64 -> 4.5 r (640 + 2560) / (640 * 2560) b.
Usage: [SpecDecode venv] act16.py  -> ~/Leap/data/act16.txt"""
import json, os, time
import numpy as np
import mlx.core as mx
MD = os.path.expanduser("~/BigMoE/models/bare")
XD = os.path.expanduser("~/Fit12/data/q2")
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
LAYERS = [L for L in (2, 12, 24, 36, 46) if os.path.exists(os.path.join(XD, f"x_L{L}.npy"))]
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
shards = {}
def tensor(name):
    f = idx[name]
    if f not in shards:
        shards[f] = mx.load(os.path.join(MD, f))
    return shards[f][name]

def rbf16(a):
    return np.array(mx.array(a.astype(np.float32)).astype(mx.bfloat16).astype(mx.float32))

def rtn(W, bits, g=64):
    o, K = W.shape
    G = W.reshape(o, K // g, g)
    lo, hi = G.min(-1, keepdims=True), G.max(-1, keepdims=True)
    n = (1 << bits) - 1
    s = rbf16(np.maximum((hi - lo) / n, 1e-12)); b = rbf16(lo)
    return (np.clip(np.round((G - b) / s), 0, n) * s + b).reshape(o, K)

def half(H):
    """H^1/2 and H^-1/2 (a small ridge keeps the inverse finite)"""
    lh, Vh = np.linalg.eigh(H + 1e-6 * np.trace(H) / H.shape[0] * np.eye(H.shape[0]))
    lh = np.maximum(lh, 1e-12)
    return (Vh * np.sqrt(lh)) @ Vh.T, (Vh / np.sqrt(lh)) @ Vh.T

def err(W, W4, H):
    """tr(D H D^T) / tr(W4 H W4^T) with BLAS matmuls (np.einsum on three operands runs single-threaded loops)"""
    d = W - W4
    return float(np.sum((d @ H) * d) / np.sum((W4 @ H) * W4))

def experts(L, mat, es):
    pre = f"language_model.model.layers.{L}.mlp.switch_mlp.{mat}."
    w, s, b = (tensor(pre + k) for k in ("weight", "scales", "biases"))
    out = {}
    for e in es:
        W4 = np.array(mx.dequantize(w[e:e + 1], s[e:e + 1], b[e:e + 1], group_size=64, bits=4).astype(mx.float32)[0], np.float64)
        q = np.array(w[e:e + 1])[0]                                       # packed codes, for MSB-3
        codes = np.stack([(q >> (4 * i)) & 0xF for i in range(8)], -1).reshape(W4.shape)
        s4 = np.repeat(np.array(s[e].astype(mx.float32), np.float64), 64, -1)
        b4 = np.repeat(np.array(b[e].astype(mx.float32), np.float64), 64, -1)
        W3 = 2 * s4 * (codes >> 1) + b4 + s4 / 2                          # high 3 bits, midpoint bias
        out[e] = (W4, W3)
    return out

def all_experts(L, mat):
    pre = f"language_model.model.layers.{L}.mlp.switch_mlp.{mat}."
    w, s, b = (tensor(pre + k) for k in ("weight", "scales", "biases"))
    return w, s, b

t0 = time.time()
say(f"L1-16 step 2: layers {LAYERS} (Fit12 Q2 inputs), 8 random experts per layer (seed 0); err relative to the 4-bit expert")
rng = np.random.default_rng(0)
RS_IN, RS_LR = (512, 768, 1024, 1536), (128, 192, 256)
acc = {}
def add(fmt, mat, v):
    acc.setdefault(fmt, {}).setdefault(mat, []).append(v)
for L in LAYERS:
    X = np.load(os.path.join(XD, f"x_L{L}.npy")).astype(np.float64).reshape(-1, 2560)
    Hx = X.T @ X
    lam, V = np.linalg.eigh(Hx); V = V[:, ::-1]; lam = lam[::-1]
    SX = half(Hx)
    es = sorted(rng.choice(512, 8, replace=False).tolist())
    G4 = experts(L, "gate_proj", es); U4 = experts(L, "up_proj", es); D4 = experts(L, "down_proj", es)
    # shared output subspace for down: sum over all 512 experts of W4_e Hh_e W4_e^T (Hh_e from this layer's X)
    wg, sg, bg = all_experts(L, "gate_proj"); wu, su, bu = all_experts(L, "up_proj"); wd, sd, bd = all_experts(L, "down_proj")
    Xm = mx.array(X.astype(np.float32))
    Gout = mx.zeros((2560, 2560), mx.float32)
    for e0 in range(0, 512, 16):
        Wg = mx.dequantize(wg[e0:e0 + 16], sg[e0:e0 + 16], bg[e0:e0 + 16], group_size=64, bits=4).astype(mx.float32)
        Wu = mx.dequantize(wu[e0:e0 + 16], su[e0:e0 + 16], bu[e0:e0 + 16], group_size=64, bits=4).astype(mx.float32)
        Wd = mx.dequantize(wd[e0:e0 + 16], sd[e0:e0 + 16], bd[e0:e0 + 16], group_size=64, bits=4).astype(mx.float32)
        a = mx.matmul(Xm[None], Wg.transpose(0, 2, 1)); hh = (a * mx.sigmoid(a)) * mx.matmul(Xm[None], Wu.transpose(0, 2, 1))
        yo = mx.matmul(hh, Wd.transpose(0, 2, 1))                         # [16, T, 2560]: each expert's output on X
        Gout = Gout + mx.sum(mx.matmul(yo.transpose(0, 2, 1), yo), axis=0)
        mx.eval(Gout)
    mu, Uo = np.linalg.eigh(np.array(Gout, np.float64)); Uo = Uo[:, ::-1]
    for e in es:
        Wg4, Wg3 = G4[e]; Wu4, Wu3 = U4[e]; Wd4, Wd3 = D4[e]
        a = X @ Wg4.T; h = (a / (1 + np.exp(-a))) * (X @ Wu4.T); Hh = h.T @ h
        sq = None
        for mat, W4, W3, H in (("gate", Wg4, Wg3, Hx), ("up", Wu4, Wu3, Hx), ("down", Wd4, Wd3, Hh)):
            add("MSB-3 (3.5 b)", mat, err(W3, W4, H))
            add("RTN-2 g64 (2.5 b)", mat, err(rtn(W4, 2), W4, H))
            for r in RS_IN:
                if mat != "down":
                    Vr = V[:, :r]; C = W4 @ Vr                                # [640, r] coefficients
                    for bits, tag in ((4, "SUB"), (2, "SUB2")):
                        add(f"{tag}-{r} ({(bits + 0.5) * r / 2560:.2f} b)", mat, err(rtn(C, bits) @ Vr.T, W4, H))
                else:
                    Ur = Uo[:, :r]; C = Ur.T @ W4                             # [r, 640]
                    for bits, tag in ((4, "SUB"), (2, "SUB2")):
                        add(f"{tag}-{r} ({(bits + 0.5) * r / 2560:.2f} b)", mat, err(Ur @ rtn(C, bits), W4, H))
            # per-expert activation-aware low rank: W ~ (W S)_r S^-1, S = H^1/2 (eigen form)
            S, Si = SX if mat != "down" else half(H)
            Uw, sw, Vw = np.linalg.svd(W4 @ S, full_matrices=False)
            for r in RS_LR:
                A = Uw[:, :r] * sw[:r]; B = Vw[:r] @ Si
                o, K = W4.shape
                add(f"LR-{r} ({4.5 * r * (o + K) / (o * K):.2f} b)", mat, err(rtn(A, 4) @ rtn(B, 4), W4, H))
    say(f"  layer {L} done ({X.shape[0]} tokens, {time.time() - t0:.0f} s); H's top-r energy: " +
        "  ".join(f"r{r} {lam[:r].sum() / lam.sum():.4f}" for r in (128, 256, 512, 1024)))
base = {m: np.mean(acc["MSB-3 (3.5 b)"][m]) for m in ("gate", "up", "down")}
say("format | gate | up | down | mean vs MSB-3")
for fmt, v in acc.items():
    m = {k: np.mean(x) for k, x in v.items()}
    say(f"  {fmt:22s} | {m['gate']:.4f} | {m['up']:.4f} | {m['down']:.4f} | {np.mean([m[k] / base[k] for k in m]):.2f}x")
open(os.path.expanduser("~/Leap/data/act16.txt"), "w").write("\n".join(lines) + "\n")
