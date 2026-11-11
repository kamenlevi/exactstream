"""M-09b MUCOMP (MoonShots, Fri 2026-10-30; follow-up of M-09's descriptive arm): replace a dropped expert's contribution
w E_e(x) by w mu_e, where mu_e is the expert's mean output (2,560 floats). Same inputs, router, experts, shared expert and
block-error metric as M-09 (scripts/contrib16.py; Leap's L1-17 / B1 code).
mu_e from tokens 0-2047 of each layer's capture. Scored in sample on tokens 1024-2047 and OUT of sample on 6144-8191.
Window budgets (16 consecutive tokens, union cut to B by router mass), dropped experts handled three ways:
  renorm  kept weights renormalized (MoE-Spec style)
  drop    dropped contributions removed, no renormalization
  mu      dropped contributions replaced by w mu_e, no renormalization
Per-row form (a served model defined per token, so spec decoding stays exact vs its own AR): each row keeps its top-k
  topk    renormalized over the k kept (L1-17's top-k)
  topk+mu kept weights unchanged, ranks k+1..10 replaced by w mu_e
Reference: per-row top-8 renormalized (passed the Eval suite 5/5).
Lines (QUEUE M-09b, written first): PASS = out of sample, mu needs <= 0.8x renorm's experts per window at top-8's error AND
per-row top-7 or top-6 + mu has block error <= top-8's; KILL = drop within 10 % of mu's experts at that error, or out of
sample mu needs >= 0.9x renorm's; BETWEEN otherwise.
usage: [BigMoE venv] python mucomp16.py (shared GPU, ~1 min, rough) -> stdout + data/mucomp16.txt"""
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
say(f"M-09b MUCOMP: dropped experts -> w x mean output; layers {LAYERS}; mu from tokens {CAL[0]}-{CAL[1] - 1}; windows of "
    f"{W}, in sample {EVAL['in']}, out of sample {EVAL['out']}; shared GPU, rough")
t0 = time.time()
acc = {}                                  # (split, arm, param) -> [err, den, union sum, windows]


def add(key, err, den, u):
    a = acc.setdefault(key, [0.0, 0.0, 0.0, 0])
    a[0] += err; a[1] += den; a[2] += u; a[3] += 1


for L in LAYERS:
    while os.path.exists(PAUSE):
        time.sleep(20)
    pre = f"language_model.model.layers.{L}.mlp."
    raw = np.load(os.path.join(XD, f"x_L{L}.npy"))
    sel = np.r_[CAL[0]:CAL[1], EVAL["out"][0]:EVAL["out"][1]]            # in-sample eval lies inside CAL
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
    nc = CAL[1] - CAL[0]                                                  # rows 0..nc-1 of X are the calibration tokens
    mu = np.zeros((512, 2560), np.float32); cnt = np.zeros(512)
    np.add.at(mu, ids[:nc].ravel(), E[:nc].reshape(-1, 2560)); np.add.at(cnt, ids[:nc].ravel(), 1)
    mu /= np.maximum(cnt, 1)[:, None]
    rows = {"in": np.arange(EVAL["in"][0] - CAL[0], EVAL["in"][1] - CAL[0]), "out": np.arange(nc, T)}
    for split, rr in rows.items():
        for w0 in range(0, rr.size, W):
            r = rr[w0:w0 + W]
            I, Wt, Ev, S = ids[r], wn[r], E[r], sh[r]
            MU = mu[I]                                                    # [W, 10, 2560]
            y10 = (Wt[..., None] * Ev).sum(1) + S
            den = float((y10 ** 2).sum())
            U = np.unique(I)
            add((split, "natural", 0), 0.0, den, U.size)
            sc = np.zeros(512); np.add.at(sc, I.ravel(), Wt.ravel())
            rank = U[np.argsort(-sc[U])]
            for B in BUD:
                keep = np.zeros(512, bool); keep[rank[:B]] = True
                m = keep[I]
                wk = np.where(m, Wt, 0.0)
                sk = wk.sum(-1, keepdims=True)
                wr = np.where(sk > 0, wk / np.maximum(sk, 1e-30), 0.0)
                ub = min(B, U.size)
                for arm, y in (("renorm", (wr[..., None] * Ev).sum(1) + S),
                               ("drop", (wk[..., None] * Ev).sum(1) + S),
                               ("mu", (wk[..., None] * Ev).sum(1) + (np.where(m, 0.0, Wt)[..., None] * MU).sum(1) + S)):
                    add((split, arm, B), float(((y - y10) ** 2).sum()), den, ub)
            for k in KS:
                uk = np.unique(I[:, :k]).size
                wk = Wt.copy(); wk[:, k:] = 0
                yk = ((wk / wk.sum(-1, keepdims=True))[..., None] * Ev).sum(1) + S
                ym = (wk[..., None] * Ev).sum(1) + ((Wt - wk)[..., None] * MU).sum(1) + S
                add((split, "topk", k), float(((yk - y10) ** 2).sum()), den, uk)
                add((split, "topk+mu", k), float(((ym - y10) ** 2).sum()), den, uk)
    say(f"  layer {L:2d} done ({time.time() - t0:.0f} s)")


def val(split, arm, p):
    a = acc[(split, arm, p)]
    return a[0] / a[1], a[2] / a[3]


def kept_at(split, arm, target):
    pts = [val(split, arm, B) for B in BUD]
    e = np.array([p[0] for p in pts]); k = np.array([p[1] for p in pts])
    if e[-1] > target:
        return float("nan")
    for i in range(len(e)):
        if e[i] <= target:
            return float(k[0]) if i == 0 else float(k[i - 1] + (k[i] - k[i - 1]) * (e[i - 1] - target) /
                                                    max(e[i - 1] - e[i], 1e-12))
    return float("nan")


res = {}
for split in ("in", "out"):
    un = val(split, "natural", 0)[1]
    e8, u8 = val(split, "topk", 8)
    say(f"== {split} of sample: natural window union {un:.1f}; per-row top-8: union {u8:.1f}, block error {e8:.4f}")
    say("  window budget by router mass, block error at B:   " + " ".join(f"{B:>7d}" for B in BUD))
    for arm in ("renorm", "drop", "mu"):
        say(f"    {arm:7s}                                        " +
            " ".join(f"{val(split, arm, B)[0]:7.4f}" for B in BUD))
    ka, kb, kc = (kept_at(split, a, e8) for a in ("renorm", "drop", "mu"))
    say(f"  experts per window at top-8's error: renorm {ka:.1f}, drop {kb:.1f}, mu {kc:.1f} "
        f"(mu = {kc / ka:.2f}x renorm, {100 * (kc / un - 1):+.1f} % vs natural; drop = {kb / kc:.2f}x mu)")
    say("  per-row top-k (union per window / block error), renormalized vs + mu for ranks k+1..10:")
    for k in KS:
        (ek, uk), (em, _) = val(split, "topk", k), val(split, "topk+mu", k)
        say(f"    k {k}: union {uk:5.1f} ({100 * (uk / un - 1):+5.1f} %); topk {ek:.4f} ({ek / e8:.2f}x top-8's); "
            f"topk+mu {em:.4f} ({em / e8:.2f}x top-8's)")
    res[split] = dict(ka=ka, kb=kb, kc=kc, e8=e8, em6=val(split, "topk+mu", 6)[0], em7=val(split, "topk+mu", 7)[0])
o = res["out"]
kill = (o["kb"] <= 1.10 * o["kc"]) or (o["kc"] >= 0.9 * o["ka"])
pas = (o["kc"] <= 0.8 * o["ka"]) and (min(o["em6"], o["em7"]) <= o["e8"])
call = "KILL" if kill else ("PASS" if pas else "BETWEEN")
say(f"Lines (out of sample): mu/renorm {o['kc'] / o['ka']:.2f} (PASS <= 0.80, KILL >= 0.90); drop/mu {o['kb'] / o['kc']:.2f} "
    f"(KILL <= 1.10); per-row top-7 / top-6 + mu vs top-8's error {o['em7'] / o['e8']:.2f}x / {o['em6'] / o['e8']:.2f}x "
    f"(PASS needs <= 1.00x for one) -> {call}")
say(f"({time.time() - t0:.0f} s)")
open(os.path.expanduser("~/MoonShots/data/mucomp16.txt"), "w").write("\n".join(lines) + "\n")
