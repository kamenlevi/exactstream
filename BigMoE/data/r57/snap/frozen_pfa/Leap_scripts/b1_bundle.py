"""B1 step 3 (Leap, Sun 2026-10-25): can ANY predictor rescue neuron-sparse reads on this SSD? The SSD caps small reads
at ~265k/s (b1_ssd: 16 KB 4.3 GB/s, 64 KB 10.2, 128 KB 12.7, whole expert 13.8), so a neuron read must come in bundles
of B neurons stored together (gate row + up row + down column, re-laid; 4.32 KB per neuron at 4-bit g64). This gives
the predictor perfect knowledge (oracle scores |h_j| ||d_j||, per token) and bundles built IN-SAMPLE (optimistic):
  stored  neurons in their stored order (the layout as is)
  static  neurons sorted by mean oracle score over the expert's tokens
  peak    neurons grouped by the token they matter most for (co-activation seriation)
Per (token, expert) pair, keep the best bundles (by summed squared score) and measure exact retained output energy.
Same data as b1_energy.py (Fit12 x_L*.npy, layers 2/12/24/36/46, first 2048 tokens, real router top-10).
Usage: [BigMoE venv] b1_bundle.py -> ~/Leap/data/b1_bundle.txt"""
import json, os, time
import numpy as np
import mlx.core as mx

MD = os.path.expanduser("~/BigMoE/models/bare")
XD = os.path.expanduser("~/Fit12/data/q2")
N_TOK = 2048
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
shards = {}
def tensor(name):
    f = idx[name]
    if f not in shards:
        shards[f] = mx.load(os.path.join(MD, f))
    return shards[f][name]
def qlinear(mod, x):
    w, s, b = (tensor(mod + "." + k) for k in ("weight", "scales", "biases"))
    bits = w.shape[-1] * 32 // (s.shape[-1] * 64)
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=bits)
def deq(w, s, b):
    return mx.dequantize(w, s, b, group_size=64, bits=4).astype(mx.float32)

FR = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
BS = (1, 16, 32, 64)                                   # neurons per bundle: 4 KB / 69 KB / 138 KB / 276 KB reads
ORD = ("stored", "static", "peak")
LAYERS = (2, 12, 24, 36, 46)
say(f"B1 step 3: oracle selection in bundles of B neurons; layers {LAYERS}, {N_TOK} tokens, top-10")
t0 = time.time()
ret = {(o, B): [] for o in ORD for B in BS}
for L in LAYERS:
    pre = f"language_model.model.layers.{L}.mlp."
    X = mx.array(np.load(os.path.join(XD, f"x_L{L}.npy"))[:N_TOK]).astype(mx.bfloat16)
    order = mx.argsort(-qlinear(pre + "gate", X).astype(mx.float32), axis=-1)[:, :10]
    order_np = np.array(order)
    sw = pre + "switch_mlp."
    G = [tensor(sw + p + "." + k) for p in ("gate_proj", "up_proj", "down_proj") for k in ("weight", "scales", "biases")]
    for e in range(512):
        tok = np.nonzero((order_np == e).any(-1))[0]
        if tok.size == 0:
            continue
        xe = X[mx.array(tok)].astype(mx.float32)
        Wg, Wu, Wd = deq(*[G[i][e] for i in range(0, 3)]), deq(*[G[i][e] for i in range(3, 6)]), deq(*[G[i][e] for i in range(6, 9)])
        g = xe @ Wg.T
        h = g * mx.sigmoid(g) * (xe @ Wu.T)
        y = h @ Wd.T
        yy = mx.maximum((y * y).sum(-1), 1e-30)
        s = mx.abs(h) * mx.sqrt((Wd * Wd).sum(0))       # oracle neuron score, [n, 640]
        s_np = np.array(s)
        perms = {"stored": np.arange(640), "static": np.argsort(-s_np.mean(0))}
        pk = s_np / np.maximum(s_np.sum(0, keepdims=True), 1e-30)
        perms["peak"] = np.lexsort((-pk.max(0), pk.argmax(0)))
        for o in ORD:
            p = mx.array(perms[o])
            inv = mx.array(np.argsort(perms[o]))
            sp = (s * s)[:, p]                           # scores in bundle order
            for B in BS:
                if B == 1 and o != "stored":
                    continue                             # B 1 is the per-neuron oracle, order does not matter
                nb = 640 // B
                bsc = sp.reshape(-1, nb, B).sum(-1)
                rank = mx.argsort(mx.argsort(-bsc, axis=-1), axis=-1)
                rs = []
                for f in FR:
                    kb = (rank < int(round(f * nb))).astype(mx.float32)
                    keep = mx.repeat(kb, B, axis=-1)[:, inv]   # back to the stored neuron order
                    yS = (h * keep) @ Wd.T
                    rs.append(1 - ((y - yS) ** 2).sum(-1) / yy)
                R = mx.stack(rs, -1)
                mx.eval(R)
                ret[(o, B)].append(np.array(R))
    say(f"  layer {L:2d} done ({time.time() - t0:.0f} s)")
    shards.clear()

def need(r, target):
    out = np.full(r.shape[0], 1.0)
    for i in range(len(FR) - 1, -1, -1):
        out[r[:, i] >= target] = FR[i]
    for i in range(1, len(FR)):
        sel = (out == FR[i]) & (r[:, i - 1] < target)
        a, b = r[sel, i - 1], r[sel, i]
        out[sel] = FR[i - 1] + (FR[i] - FR[i - 1]) * (target - a) / np.maximum(b - a, 1e-9)
    return out

# measured random-read speed (b1_ssd, 64 threads) by read size, GB/s
SSD = [(4096, 1.10), (8192, 2.18), (16384, 4.28), (32768, 7.10), (65536, 10.15), (131072, 12.72), (262144, 13.55),
       (1048576, 13.68), (2899968, 13.78)]
def speed(nbytes):
    xs, ys = np.log([a for a, _ in SSD]), [b for _, b in SSD]
    return float(np.interp(np.log(nbytes), xs, ys))
say("fraction of bundles for 90 / 95 % energy (median) and cold-expert read time vs a whole-expert read (1.00):")
for o in ORD:
    for B in BS:
        if (o, B) not in ret or not ret[(o, B)]:
            continue
        R = np.concatenate(ret[(o, B)])
        f90, f95 = np.median(need(R, 0.9)), np.median(need(R, 0.95))
        rd = B * 4320
        t90 = f90 * 13.78 / speed(rd)
        say(f"  {o:6s} B {B:3d} ({rd / 1024:5.0f} KB reads, {speed(rd):5.2f} GB/s): {f90:.2f} / {f95:.2f}; "
            f"time at 90 % {t90:.2f}  (median retained at 30 %: {np.median(R[:, 2]):.3f})")
open(os.path.expanduser("~/Leap/data/b1_bundle.txt"), "w").write("\n".join(lines) + "\n")
