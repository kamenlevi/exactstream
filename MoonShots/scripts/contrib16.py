"""M-09 CONTRIB16 (MoonShots, Fri 2026-10-30; Kamen 18:45 FAST INTEGRATION item 4: a new bet on the bytes wall): does a
16-token verify window's expert budget hold quality better when the kept experts are chosen by their actual output
contribution instead of router mass?
Inputs (read only): Fit12's captured MoE inputs (~/Fit12/data/q2/x_L*.npy; layers 2 / 12 / 24 / 36 / 46), the real
router (softmax over 512, top-10, renormalized as the model does), the real 4-bit experts and the gated shared expert;
the loading and block arithmetic follow Leap's L1-17 / B1 scripts (~/Leap/scripts/b1_energy.py).
Per layer: tokens 0-1023 give each expert's static output scale (mean ||E_e(x)|| over its routed tokens); windows of 16
consecutive tokens over tokens 1024-2047 (64 windows) are scored. Per window, the union U of the rows' top-10 is cut to B
experts by:
  MASS     sum over the window's rows of the renormalized router weight (MoE-Spec / AcceptMoE style)
  CONTRIB  sum over rows of ||w E_e(x_r)||, the actual contribution (oracle: needs the expert computed)
  MSCALE   MASS-like sum of w x the expert's static scale (free at run time)
Each row keeps its top-10 experts that are in the kept set, weights renormalized over them; the shared expert stays.
Error = L1-17's MoE-block output error: sum ||y_B - y_10||^2 / sum ||y_10||^2 over the windows' rows (all 5 layers pooled).
Reference: per-row top-8 (renormalized; passed the Eval suite 5/5), computed here with its own window union.
Lines (QUEUE M-09, written first): PASS = at top-8's block error, CONTRIB needs >= 25 % fewer experts per window than MASS
AND MSCALE captures >= half of that saving; KILL = CONTRIB needs < 10 % fewer than MASS; BETWEEN otherwise.
usage: [BigMoE venv] python contrib16.py (shared GPU, ~3 min, rough) -> stdout + data/contrib16.txt"""
import json, os, sys, time
import numpy as np
import mlx.core as mx

MD = os.path.expanduser("~/BigMoE/models/bare")
XD = os.path.expanduser("~/Fit12/data/q2")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
CAL, EV = (0, 1024), (1024, 2048)
W = 16
BUD = (16, 24, 32, 40, 48, 56, 64, 72)
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


def wait_pause():
    while os.path.exists(PAUSE):
        time.sleep(20)


LAYERS = [L for L in (2, 12, 24, 36, 46) if os.path.exists(os.path.join(XD, f"x_L{L}.npy"))]
say(f"M-09 CONTRIB16: verify-window expert budgets by MASS / CONTRIB / MSCALE; layers {LAYERS}; windows of {W} over "
    f"tokens {EV[0]}-{EV[1] - 1}, static scales from tokens {CAL[0]}-{CAL[1] - 1}; shared GPU, rough")
t0 = time.time()
acc = {}                                     # (crit, B) -> [err, den, union sum, windows]
ref8 = [0.0, 0.0, 0.0, 0]
nat = [0.0, 0]
for L in LAYERS:
    wait_pause()
    pre = f"language_model.model.layers.{L}.mlp."
    T = EV[1]
    X = mx.array(np.load(os.path.join(XD, f"x_L{L}.npy"))[:T]).astype(mx.bfloat16)
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
    nrm = np.linalg.norm(E, axis=-1)                                      # [T, 10] ||E_e(x)||
    scale = np.zeros(512)
    cnt = np.zeros(512)
    np.add.at(scale, ids[CAL[0]:CAL[1]].ravel(), nrm[CAL[0]:CAL[1]].ravel())
    np.add.at(cnt, ids[CAL[0]:CAL[1]].ravel(), 1)
    scale = np.where(cnt > 0, scale / np.maximum(cnt, 1), np.median(scale[cnt > 0] / cnt[cnt > 0]))
    mu = np.zeros((512, 2560), np.float32)                                # descriptive arm: each expert's mean output
    np.add.at(mu, ids[CAL[0]:CAL[1]].ravel(), E[CAL[0]:CAL[1]].reshape(-1, 2560))
    mu /= np.maximum(cnt, 1)[:, None]
    for w0 in range(EV[0], EV[1], W):
        r = slice(w0, w0 + W)
        I, Wt, Ev, Nr, S = ids[r], wn[r], E[r], nrm[r], sh[r]
        y10 = (Wt[..., None] * Ev).sum(1) + S
        den = float((y10 ** 2).sum())
        U = np.unique(I)
        nat[0] += U.size; nat[1] += 1
        # per-row top-8, renormalized
        w8 = Wt.copy(); w8[:, 8:] = 0; w8 /= w8.sum(-1, keepdims=True)
        y8 = (w8[..., None] * Ev).sum(1) + S
        ref8[0] += float(((y8 - y10) ** 2).sum()); ref8[1] += den
        ref8[2] += np.unique(I[:, :8]).size; ref8[3] += 1
        score = {}
        for crit, val in (("MASS", Wt), ("CONTRIB", Wt * Nr), ("MSCALE", Wt * scale[I])):
            sc = np.zeros(512)
            np.add.at(sc, I.ravel(), val.ravel())
            score[crit] = sc
        for crit, sc in score.items():
            rank = U[np.argsort(-sc[U])]
            for B in BUD:
                keep = np.zeros(512, bool); keep[rank[:B]] = True
                m = keep[I]                                               # [W, 10]
                wk = np.where(m, Wt, 0.0)
                sk = wk.sum(-1, keepdims=True)
                wk = np.where(sk > 0, wk / np.maximum(sk, 1e-30), 0.0)
                yB = (wk[..., None] * Ev).sum(1) + S
                a = acc.setdefault((crit, B), [0.0, 0.0, 0.0, 0])
                a[0] += float(((yB - y10) ** 2).sum()); a[1] += den; a[2] += min(B, U.size); a[3] += 1
                if crit != "CONTRIB":                                     # descriptive: dropped -> w x mean output
                    yC = (np.where(m, Wt, 0.0)[..., None] * Ev).sum(1) + (np.where(m, 0.0, Wt)[..., None] * mu[I]).sum(1) + S
                    a = acc.setdefault((crit + "+mu", B), [0.0, 0.0, 0.0, 0])
                    a[0] += float(((yC - y10) ** 2).sum()); a[1] += den; a[2] += min(B, U.size); a[3] += 1
    say(f"  layer {L:2d} done ({time.time() - t0:.0f} s)")

e8 = ref8[0] / ref8[1]
u8 = ref8[2] / ref8[3]
un = nat[0] / nat[1]
say(f"natural top-10 window union {un:.1f} experts; per-row top-8: union {u8:.1f} ({100 * (u8 / un - 1):+.1f} %), block "
    f"error {e8:.4f}")
say("block error by kept experts per window (union cut to B; mean kept = min(B, U)):")
say("  B:        " + " ".join(f"{B:>8d}" for B in BUD))
curve = {}
for crit in ("MASS", "CONTRIB", "MSCALE"):
    errs = [acc[(crit, B)][0] / acc[(crit, B)][1] for B in BUD]
    kept = [acc[(crit, B)][2] / acc[(crit, B)][3] for B in BUD]
    curve[crit] = (np.array(kept), np.array(errs))
    say(f"  {crit:8s}  " + " ".join(f"{e:8.4f}" for e in errs))
say("  mean kept " + " ".join(f"{k:8.1f}" for k in curve["MASS"][0]))
for crit in ("MASS+mu", "MSCALE+mu"):
    errs = [acc[(crit, B)][0] / acc[(crit, B)][1] for B in BUD]
    curve[crit] = (curve["MASS"][0], np.array(errs))
    say(f"  {crit:9s} " + " ".join(f"{e:8.4f}" for e in errs) + "   (descriptive, not in the lines: dropped experts "
        "replaced by w x their mean output from tokens 0-1023, no renormalization)")


def kept_at(crit, target):
    """experts per window needed to reach the target error (linear interpolation on the B grid; errors fall with B)"""
    k, e = curve[crit]
    if e[-1] > target:
        return float("nan")
    for i in range(len(e)):
        if e[i] <= target:
            if i == 0:
                return float(k[0])
            return float(k[i - 1] + (k[i] - k[i - 1]) * (e[i - 1] - target) / max(e[i - 1] - e[i], 1e-12))
    return float("nan")


km, kc, ks = (kept_at(c, e8) for c in ("MASS", "CONTRIB", "MSCALE"))
save_c = 1 - kc / km
cap = (km - ks) / max(km - kc, 1e-9)
say(f"experts per window at top-8's error {e8:.4f}: MASS {km:.1f}, CONTRIB {kc:.1f} ({100 * save_c:+.1f} % fewer than "
    f"MASS), MSCALE {ks:.1f} (captures {100 * cap:.0f} % of CONTRIB's saving); top-8 itself {u8:.1f}")
say("  descriptive: " + ", ".join(f"{c} {kept_at(c, e8):.1f}" for c in ("MASS+mu", "MSCALE+mu")) + " experts per window "
    "at that error")
call = "PASS" if (save_c >= 0.25 and cap >= 0.5) else ("KILL" if save_c < 0.10 else "BETWEEN")
say(f"Lines: PASS = CONTRIB >= 25 % fewer than MASS AND MSCALE >= half of it; KILL = CONTRIB < 10 % fewer -> {call}")
say(f"({time.time() - t0:.0f} s)")
open(os.path.expanduser("~/MoonShots/data/contrib16.txt"), "w").write("\n".join(lines) + "\n")
