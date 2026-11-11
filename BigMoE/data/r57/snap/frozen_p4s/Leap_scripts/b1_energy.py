"""B1 step 1 (Leap, Sun 2026-10-25; NewIdeas' bet order 04:12): how concentrated is a 125B expert's output in its 640
intermediate neurons? Fit12's captured MoE inputs (~/Fit12/data/q2/x_L*.npy, read only; first N_TOK tokens of 8,336;
layers 2 / 12 / 24 / 36 / 46), the real router (top-10) and real 4-bit experts.
Per (token, expert) pair: g = x Wg^T, u = x Wu^T, h = silu(g) * u, y = Wd h = sum_j h_j d_j. Keep the top fraction f of
neurons under three rankings and measure retained output energy 1 - ||y - y_S||^2 / ||y||^2 (exact, cross terms included):
  oracle  |h_j| ||d_j||                      (needs up: not usable before the second read; the ceiling)
  gate-w  |silu(g_j)| ||Wu_j|| ||d_j||        (gate-only, weight-informed: usable after reading only the gate rows)
  gate    |silu(g_j)|                         (CATS-style)
Report: median / p10 retained energy at each f; the f needed for 90 / 95 / 99 % (median, by linear interpolation on the
grid); and the MoE-block output error (renormalized top-10 + shared expert, as L1-17) with every routed expert cut to
f under gate-w, vs MSB-3 on all 10 (the v3 tier's base code; L1-17's yardstick).
Bytes per cold expert read gate-first: 1/3 (gate rows) + f x 2/3 (bundled up rows + down columns).
Usage: [BigMoE venv] b1_energy.py -> ~/Leap/data/b1_energy.txt"""
import json, os, time
import numpy as np
import mlx.core as mx

MD = os.path.expanduser("~/BigMoE/models/bare")
XD = os.path.expanduser("~/Fit12/data/q2")
N_TOK = int(os.environ.get("B1_TOK", "2048"))
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
    bits = w.shape[-1] * 32 // (s.shape[-1] * 64)                       # the router and shared gate are 8-bit
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=bits)
def deq(w, s, b):
    return mx.dequantize(w, s, b, group_size=64, bits=4).astype(mx.float32)
def msb3(w, s, b):
    W4 = mx.dequantize(w, s, b, group_size=64, bits=4)
    codes = mx.stack([(w >> (4 * i)) & 0xF for i in range(8)], -1).reshape(W4.shape)
    s4 = mx.repeat(s.astype(mx.float32), 64, -1); b4 = mx.repeat(b.astype(mx.float32), 64, -1)
    return 2 * s4 * (codes >> 1).astype(mx.float32) + b4 + s4 / 2

FR = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
RANK = ("oracle", "gate-w", "gate")
BLOCK_F = (0.2, 0.3, 0.5)
LAYERS = [L for L in (2, 12, 24, 36, 46) if os.path.exists(os.path.join(XD, f"x_L{L}.npy"))]
say(f"B1 step 1: neuron concentration of 125B expert outputs; layers {LAYERS}, {N_TOK} tokens each, top-10 routing")
t0 = time.time()
allret = {r: [] for r in RANK}                       # per pair: retained energy at each FR
blk = {f: [0.0, 0.0] for f in BLOCK_F}; blk3 = [0.0, 0.0]
for L in LAYERS:
    pre = f"language_model.model.layers.{L}.mlp."
    X = mx.array(np.load(os.path.join(XD, f"x_L{L}.npy"))[:N_TOK]).astype(mx.bfloat16)
    T = X.shape[0]
    gates = mx.softmax(qlinear(pre + "gate", X).astype(mx.float32), axis=-1, precise=True)
    order = mx.argsort(-gates, axis=-1)[:, :10]
    w10 = mx.take_along_axis(gates, order, axis=-1)
    gsh = qlinear(pre + "shared_expert.gate_proj", X); ush = qlinear(pre + "shared_expert.up_proj", X)
    shared = mx.sigmoid(qlinear(pre + "shared_expert_gate", X).astype(mx.float32)) * \
        qlinear(pre + "shared_expert.down_proj", gsh * mx.sigmoid(gsh) * ush).astype(mx.float32)
    mx.eval(order, w10, shared)
    order_np = np.array(order); wn = np.array(w10); wn = wn / wn.sum(-1, keepdims=True)
    E = np.zeros((T, 10, 2560), np.float32); E3 = np.zeros((T, 10, 2560), np.float32)
    ES = {f: np.zeros((T, 10, 2560), np.float32) for f in BLOCK_F}
    sw = pre + "switch_mlp."
    G = [tensor(sw + p + "." + k) for p in ("gate_proj", "up_proj", "down_proj") for k in ("weight", "scales", "biases")]
    for e in range(512):
        tok, slot = np.nonzero(order_np == e)
        if tok.size == 0:
            continue
        xe = X[mx.array(tok)].astype(mx.float32)
        Wg, Wu, Wd = deq(*[G[i][e] for i in range(0, 3)]), deq(*[G[i][e] for i in range(3, 6)]), deq(*[G[i][e] for i in range(6, 9)])
        g = xe @ Wg.T; u = xe @ Wu.T
        sg = g * mx.sigmoid(g)
        h = sg * u                                                           # [n, 640]
        y = h @ Wd.T                                                         # [n, 2560]
        dn = mx.sqrt((Wd * Wd).sum(0)); un = mx.sqrt((Wu * Wu).sum(1))       # ||d_j||, ||Wu_j||
        score = {"oracle": mx.abs(h) * dn, "gate-w": mx.abs(sg) * un * dn, "gate": mx.abs(sg)}
        yy = (y * y).sum(-1)
        for r in RANK:
            rank = mx.argsort(mx.argsort(-score[r], axis=-1), axis=-1)      # 0 = best neuron of the row
            rets = []
            for f in FR:
                keep = (rank < int(round(f * 640))).astype(mx.float32)
                yS = (h * keep) @ Wd.T
                rets.append(1 - ((y - yS) ** 2).sum(-1) / mx.maximum(yy, 1e-30))
                if r == "gate-w" and float(f) in BLOCK_F:
                    ES[float(f)][tok, slot] = np.array(yS)
            R = mx.stack(rets, -1)
            mx.eval(R)
            allret[r].append(np.array(R))
        g3 = xe @ msb3(*[G[i][e] for i in range(0, 3)]).T; u3 = xe @ msb3(*[G[i][e] for i in range(3, 6)]).T
        y3 = (g3 * mx.sigmoid(g3) * u3) @ msb3(*[G[i][e] for i in range(6, 9)]).T
        E[tok, slot] = np.array(y); E3[tok, slot] = np.array(y3)
    sh = np.array(shared)
    y10 = (wn[..., None] * E).sum(1) + sh
    den = float((y10 ** 2).sum())
    for f in BLOCK_F:
        yb = (wn[..., None] * ES[f]).sum(1) + sh
        blk[f][0] += float(((yb - y10) ** 2).sum()); blk[f][1] += den
    yb3 = (wn[..., None] * E3).sum(1) + sh
    blk3[0] += float(((yb3 - y10) ** 2).sum()); blk3[1] += den
    say(f"  layer {L:2d} done ({time.time() - t0:.0f} s)")
    shards.clear()

def need(ret, target):
    """smallest f on the grid (linear interpolation) with retained >= target, per pair"""
    out = np.full(ret.shape[0], 1.0)
    for i in range(len(FR) - 1, -1, -1):
        ok = ret[:, i] >= target
        out[ok] = FR[i]
    # interpolate between the first grid point that passes and the previous one
    for i in range(1, len(FR)):
        sel = (out == FR[i]) & (ret[:, i - 1] < target)
        a, b = ret[sel, i - 1], ret[sel, i]
        out[sel] = FR[i - 1] + (FR[i] - FR[i - 1]) * (target - a) / np.maximum(b - a, 1e-9)
    return out

say("retained output energy per (token, expert) pair, median [p10], by fraction of the 640 neurons kept:")
say("  f:        " + " ".join(f"{f:>11.2f}" for f in FR))
for r in RANK:
    R = np.concatenate(allret[r])
    say(f"  {r:8s}  " + " ".join(f"{np.median(R[:, i]):5.3f}[{np.percentile(R[:, i], 10):4.2f}]" for i in range(len(FR))))
for r in RANK:
    R = np.concatenate(allret[r])
    say(f"  {r:8s} fraction of neurons for 90 / 95 / 99 % energy (median): " +
        " / ".join(f"{np.median(need(R, t)):.2f}" for t in (0.9, 0.95, 0.99)) + f"   ({R.shape[0]} pairs)")
e3 = blk3[0] / blk3[1]
say(f"MoE-block output error (rel. energy, renormalized top-10 + shared), every routed expert at f under gate-w; MSB-3 all 10 = {e3:.4f}:")
for f in BLOCK_F:
    e = blk[f][0] / blk[f][1]
    say(f"  f {f:.1f}: {e:.4f} = {e / e3:.2f}x MSB-3's; gate-first read bytes {(1 + 2 * f) / 3:.2f} of the expert")
open(os.path.expanduser("~/Leap/data/b1_energy.txt"), "w").write("\n".join(lines) + "\n")
