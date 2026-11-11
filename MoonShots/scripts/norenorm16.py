"""M-09c NORENORM (MoonShots, Fri 2026-10-30; from M-09b's side finding): per-row top-k WITHOUT renormalizing the kept
weights (dropped experts' contributions removed; the model renormalizes over its top-10 only) vs top-k renormalized over
the k kept (L1-17, EvalSuite's b12llk8). Same inputs, router, experts, shared expert and block-error metric as M-09 / M-09b.
Scored in sample on tokens 1024-2047 and OUT of sample on 6144-8191 (16-token windows for the union).
Lines (QUEUE M-09c, written first): PASS = out of sample, top-7 no-renorm block error <= top-8-renorm's; KILL = top-8
no-renorm >= 0.9x top-8-renorm's error; BETWEEN otherwise.
usage: [BigMoE venv] python norenorm16.py (shared GPU, ~1 min, rough) -> stdout + data/norenorm16.txt"""
import json, os, time
import numpy as np
import mlx.core as mx

MD = os.path.expanduser("~/BigMoE/models/bare")
XD = os.path.expanduser("~/Fit12/data/q2")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
CAL = (0, 2048)
EVAL = {"in": (1024, 2048), "out": (6144, 8192)}
W = 16
BUD = (16, 24, 32, 40, 48, 56, 64, 72)
KS = (4, 5, 6, 7, 8)
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


LAYERS = [L for L in (2, 12, 24, 36, 46) if os.path.exists(os.path.join(XD, f"x_L{L}.npy"))]
say(f"M-09c NORENORM: per-row top-k renormalized vs not; layers {LAYERS}; in sample {EVAL['in']}, out of sample "
    f"{EVAL['out']}; shared GPU, rough")
t0 = time.time()
acc = {}


def add(key, err, den, u):
    a = acc.setdefault(key, [0.0, 0.0, 0.0, 0])
    a[0] += err; a[1] += den; a[2] += u; a[3] += 1


for L in LAYERS:
    while os.path.exists(PAUSE):
        time.sleep(20)
    pre = f"language_model.model.layers.{L}.mlp."
    raw = np.load(os.path.join(XD, f"x_L{L}.npy"))
    sel = np.r_[EVAL["in"][0]:EVAL["in"][1], EVAL["out"][0]:EVAL["out"][1]]
    X = mx.array(raw[sel]).astype(mx.bfloat16)
    T = X.shape[0]
    gates = mx.softmax(qlinear(pre + "gate", X).astype(mx.float32), axis=-1, precise=True)
    order = mx.argsort(-gates, axis=-1)[:, :10]
    w10 = mx.take_along_axis(gates, order, axis=-1)
    gsh = qlinear(pre + "shared_expert.gate_proj", X); ush = qlinear(pre + "shared_expert.up_proj", X)
    shared = mx.sigmoid(qlinear(pre + "shared_expert_gate", X).astype(mx.float32)) * \
        qlinear(pre + "shared_expert.down_proj", gsh * mx.sigmoid(gsh) * ush).astype(mx.float32)
    mx.eval(order, w10, shared)
    ids = np.array(order); wn = np.array(w10); wn = wn / wn.sum(-1, keepdims=True)
    sh = np.array(shared)
    E = np.zeros((T, 10, 2560), np.float32)
    sw = pre + "switch_mlp."
    G = [tensor(sw + p + "." + k) for p in ("gate_proj", "up_proj", "down_proj") for k in ("weight", "scales", "biases")]
    for e in range(512):
        tok, slot = np.nonzero(ids == e)
        if tok.size == 0:
            continue
        xe = X[mx.array(tok)].astype(mx.float32)
        Wg, Wu, Wd = (deq(*[G[i][e] for i in range(j, j + 3)]) for j in (0, 3, 6))
        g = xe @ Wg.T
        y = ((g * mx.sigmoid(g)) * (xe @ Wu.T)) @ Wd.T
        mx.eval(y)
        E[tok, slot] = np.array(y)
    shards.clear()
    ni = EVAL["in"][1] - EVAL["in"][0]
    rows = {"in": np.arange(0, ni), "out": np.arange(ni, T)}
    for split, rr in rows.items():
        for w0 in range(0, rr.size, W):
            r = rr[w0:w0 + W]
            I, Wt, Ev, S = ids[r], wn[r], E[r], sh[r]
            y10 = (Wt[..., None] * Ev).sum(1) + S
            den = float((y10 ** 2).sum())
            for k in KS:
                uk = np.unique(I[:, :k]).size
                wk = Wt.copy(); wk[:, k:] = 0
                yr = ((wk / wk.sum(-1, keepdims=True))[..., None] * Ev).sum(1) + S
                yd = (wk[..., None] * Ev).sum(1) + S
                add((split, "renorm", k), float(((yr - y10) ** 2).sum()), den, uk)
                add((split, "norenorm", k), float(((yd - y10) ** 2).sum()), den, uk)
    say(f"  layer {L:2d} done ({time.time() - t0:.0f} s)")


def val(split, arm, k):
    a = acc[(split, arm, k)]
    return a[0] / a[1], a[2] / a[3]


for split in ("in", "out"):
    e8 = val(split, "renorm", 8)[0]
    say(f"== {split} of sample: per-row top-8 renormalized block error {e8:.4f}")
    for k in KS:
        (er, uk), (ed, _) = val(split, "renorm", k), val(split, "norenorm", k)
        say(f"  k {k}: union {uk:5.1f}; renorm {er:.4f} ({er / e8:.2f}x top-8-renorm's); no-renorm {ed:.4f} "
            f"({ed / e8:.2f}x)")
o8, o7 = val("out", "norenorm", 8)[0], val("out", "norenorm", 7)[0]
r8 = val("out", "renorm", 8)[0]
call = "PASS" if o7 <= r8 else ("KILL" if o8 >= 0.9 * r8 else "BETWEEN")
say(f"Lines (out of sample): top-7 no-renorm {o7 / r8:.2f}x top-8-renorm's (PASS <= 1.00); top-8 no-renorm {o8 / r8:.2f}x "
    f"(KILL >= 0.90) -> {call}")
say(f"({time.time() - t0:.0f} s)")
open(os.path.expanduser("~/MoonShots/data/norenorm16.txt"), "w").write("\n".join(lines) + "\n")
