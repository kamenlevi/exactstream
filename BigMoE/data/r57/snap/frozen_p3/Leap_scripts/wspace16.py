"""L1-16 step 1 (Leap, Sat 2026-10-24): weight-space structure SHARED across a layer's 512 experts in the 125B (4-bit MLX
weights, ~/BigMoE/models/bare, read only). For layers 12 / 24 / 36, per matrix:
- mean expert: share of the total energy sum_i ||W_i||^2 explained by the mean expert (large only if experts are
  aligned, e.g. upcycled from one dense FFN; ~1/512 if unrelated);
- shared input subspace (gate / up: they read the hidden state, 2560): the share of total energy captured by the top-r
  eigenvectors of G = sum_i W_i^T W_i, against r / 2560 for an isotropic spread;
- shared output subspace (down: it writes the hidden state): the same for sum_i W_i W_i^T;
- per-expert own top-r (one expert's SVD) for scale.
Energy in weight space only; the activation-weighted (output-error) version is step 2.
Usage: [SpecDecode venv] wspace16.py  -> ~/Leap/data/wspace16.txt"""
import json, os
import numpy as np
import mlx.core as mx
MD = os.path.expanduser("~/BigMoE/models/bare")
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
cfg = json.load(open(os.path.join(MD, "config.json")))
q = cfg.get("quantization", {})
GS, BITS = q.get("group_size", 64), q.get("bits", 4)
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
shards = {}
def tensor(name):
    f = idx[name]
    if f not in shards:
        shards[f] = mx.load(os.path.join(MD, f))
    return shards[f][name]
RS = (64, 128, 256, 512, 1024, 1536)
say(f"125B experts, {GS}-group {BITS}-bit; energy shares (1.0 = all of sum_i ||W_i||^2)")
for L in (12, 24, 36):
    pre = f"language_model.model.layers.{L}.mlp.switch_mlp."
    for mat in ("gate_proj", "up_proj", "down_proj"):
        w, s, b = (tensor(pre + mat + "." + k) for k in ("weight", "scales", "biases"))
        E = w.shape[0]
        G = None; msum = None; tot = 0.0
        for e0 in range(0, E, 32):
            W = mx.dequantize(w[e0:e0 + 32], s[e0:e0 + 32], b[e0:e0 + 32], group_size=GS, bits=BITS).astype(mx.float32)
            # gate / up: [32, 640, 2560] (rows = expert neurons, columns = hidden); down: [32, 2560, 640]
            Wh = W if mat != "down_proj" else W.transpose(0, 2, 1)          # hidden dimension last: [32, n, 2560]
            g = mx.sum(mx.matmul(Wh.transpose(0, 2, 1), Wh), axis=0)        # sum_i Wh_i^T Wh_i  [2560, 2560]
            G = g if G is None else G + g
            ms = mx.sum(W, axis=0)
            msum = ms if msum is None else msum + ms
            tot += float(mx.sum(W * W))
            mx.eval(G, msum)
        mean_share = float(mx.sum((msum / E) ** 2)) * E / tot
        ev = np.sort(np.linalg.eigvalsh(np.array(G, dtype=np.float64)))[::-1]
        cum = np.cumsum(ev) / ev.sum()
        W0 = np.array(mx.dequantize(w[:1], s[:1], b[:1], group_size=GS, bits=BITS).astype(mx.float32)[0], dtype=np.float64)
        sv = np.linalg.svd(W0 if mat != "down_proj" else W0.T, compute_uv=False) ** 2
        own = np.cumsum(sv) / sv.sum()
        shared = "  ".join(f"r{r}: {cum[r - 1]:.3f}" for r in RS)
        ownr = "  ".join(f"r{r}: {own[min(r, len(own)) - 1]:.3f}" for r in RS[:4])
        say(f"layer {L:2d} {mat:9s} ({E} experts): mean expert {mean_share:.4f};  shared {'input' if mat != 'down_proj' else 'output'} "
            f"subspace (isotropic r/2560 -> r512 0.200, r1024 0.400): {shared}")
        say(f"           one expert's own top-r: {ownr}")
open(os.path.expanduser("~/Leap/data/wspace16.txt"), "w").write("\n".join(lines) + "\n")
