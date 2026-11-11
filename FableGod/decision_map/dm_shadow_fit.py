"""Sketch-on-miss, activation-aware shadows: the CPU function-space check (Breakthrough, Tue 2026-11-03 02:1x; pre-registered
SCOREBOARD a9d81e8, the orchestrator Mon 22:3x). Data: out/xcap/<window>.npz (the MoE input x of every answer row at
every layer, the rank-sorted router ids and normalised weights; run_map.py xcapture, Tue 02:0x). Split per domain:
even-indexed windows (sorted by name) fit, odd-indexed are held out.
Shadows of the 4-bit model as served (~/BigMoE/models/bare, dequantized), rank 32 and 64, float factors:
- plain: SVD of each matrix;
- whitened: gate / up = SVD of W C^1/2 (C = the layer's input second moment over the fit rows + a ridge of 1e-3 x its mean
  eigenvalue) -> A B' -> B = B' C^-1/2; down = the same with the expert's own intermediate h = silu(W_g x) * (W_u x)
  over its routed fit rows (ridge 0.1 x the mean eigenvalue; plain SVD when it has < 16 fit rows).
Metric on held-out (row, expert) pairs as routed: e = ||E(x) - S(x)|| / ||E(x)|| for the whole SwiGLU expert; the router-
weighted mean over ranks 0-9 and over ranks 6-9 (a drop = 1.0). Sample: layers 0 / 8 / 16 / 24 / 32 / 40 / 47, the 48
experts per layer with the most held-out pairs (they carry most of the held-out weight; the share is printed).
Line (a9d81e8): GO to the map arm if the whitened error is <= 0.5 at rank 32, or <= 0.5 at rank 64; STOP (KILL for
whitened shadows too) if > 0.7 at rank 64; else BETWEEN.
Usage (BigMoE venv): VECLIB_MAXIMUM_THREADS=4 python dm_shadow_fit.py -> out/SHADOW_read.txt"""
import glob, json, os, time
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.cpu)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
M = os.path.expanduser("~/BigMoE/models/bare")
WM = json.load(open(os.path.join(M, "model.safetensors.index.json")))["weight_map"]
LAYERS, NE, RANKS = (0, 8, 16, 24, 32, 40, 47), 48, (32, 64)
lines = []
def say(s=""): print(s, flush=True); lines.append(s)
silu = lambda z: z / (1 + np.exp(-z))

files = sorted(glob.glob(os.path.join(OUT, "xcap", "*.npz")))
dom = lambda f: os.path.basename(f).split("_")[1]
fit, held = [], []
for d in sorted({dom(f) for f in files}):
    fs = [f for f in files if dom(f) == d]; fit += fs[0::2]; held += fs[1::2]
def gather(fs):
    X = {l: [] for l in LAYERS}; I = {l: [] for l in LAYERS}; W = {l: [] for l in LAYERS}
    for f in fs:
        z = np.load(f)
        for l in LAYERS:
            X[l].append(z["x"][:, l, :].astype(np.float32)); I[l].append(z["ids"][:, l, :]); W[l].append(z["w"][:, l, :].astype(np.float32))
    return {l: (np.concatenate(X[l]), np.concatenate(I[l]), np.concatenate(W[l])) for l in LAYERS}
t0 = time.time(); F, H = gather(fit), gather(held)
say(f"Activation-aware shadows: {len(fit)} fit / {len(held)} held-out windows, layers {LAYERS}, {NE} experts per layer "
    f"(loaded in {time.time() - t0:.0f} s)")

def psd_pow(C, p):
    v, U = np.linalg.eigh(C); v = np.maximum(v, 1e-12); return (U * v ** p) @ U.T

def lowrank(W, Ch=None, Cih=None):
    """{r: rank-r approximation of W (out x in)} for r in RANKS, one SVD; whitened by C^1/2 / C^-1/2 when given"""
    U, S, Vt = np.linalg.svd(W if Ch is None else W @ Ch, full_matrices=False)
    return {r: (U[:, :r] * S[:r]) @ (Vt[:r] if Ch is None else Vt[:r] @ Cih) for r in RANKS}

def ridge(C, a):
    return C + a * np.trace(C) / C.shape[0] * np.eye(C.shape[0], dtype=C.dtype)

say("| layer | held-out weight share sampled | plain r32: all / ranks 6-9 | plain r64 | whitened r32 | whitened r64 |")
say("|---|---|---|---|---|---|")
tot = {k: [0.0, 0.0, 0.0, 0.0] for k in ("p32", "p64", "w32", "w64")}       # sum w*e all, sum w all, sum w*e 6-9, sum w 6-9
for l in LAYERS:
    xf, idf, _ = F[l]; xh, idh, wh = H[l]
    C = ridge((xf.T @ xf) / len(xf), 1e-3); Ch, Cih = psd_pow(C, 0.5), psd_pow(C, -0.5)
    cnt = np.bincount(idh.reshape(-1).astype(np.int64), weights=wh.reshape(-1), minlength=512)
    experts = np.argsort(-cnt)[:NE]; share = cnt[experts].sum() / cnt.sum()
    pre = f"language_model.model.layers.{l}.mlp.switch_mlp."
    Wd = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        w = mx.load(os.path.join(M, WM[pre + proj + ".weight"])); e = mx.array(experts.astype(np.int32))
        Wd[proj] = np.array(mx.dequantize(w[pre + proj + ".weight"][e], w[pre + proj + ".scales"][e], w[pre + proj + ".biases"][e],
                                          group_size=64, bits=4).astype(mx.float32))
        del w
    acc = {k: [0.0, 0.0, 0.0, 0.0] for k in tot}
    for j, ex in enumerate(experts):
        Wg, Wu, Wdn = Wd["gate_proj"][j], Wd["up_proj"][j], Wd["down_proj"][j]
        rf = np.nonzero((idf == ex).any(axis=1))[0]
        hpair = np.nonzero(idh == ex); rows, ranks = hpair; ww = wh[rows, ranks]
        if not len(rows): continue
        xr = xh[rows]; E = (silu(xr @ Wg.T) * (xr @ Wu.T)) @ Wdn.T; En = np.linalg.norm(E, axis=1) + 1e-12
        hf = silu(xf[rf] @ Wg.T) * (xf[rf] @ Wu.T) if len(rf) else None
        Chd = ridge((hf.T @ hf) / len(hf), 0.1) if (hf is not None and len(hf) >= 16) else None
        LR = {"p": (lowrank(Wg), lowrank(Wu), lowrank(Wdn)),
              "w": (lowrank(Wg, Ch, Cih), lowrank(Wu, Ch, Cih),
                    lowrank(Wdn) if Chd is None else lowrank(Wdn, psd_pow(Chd, 0.5), psd_pow(Chd, -0.5)))}
        for r in RANKS:
            for kind in ("p", "w"):
                g, u, dn = (LR[kind][i][r] for i in range(3))
                S = (silu(xr @ g.T) * (xr @ u.T)) @ dn.T
                err = np.linalg.norm(E - S, axis=1) / En; k = f"{kind}{r}"; lo = ranks >= 6
                acc[k][0] += float((ww * err).sum()); acc[k][1] += float(ww.sum())
                acc[k][2] += float((ww[lo] * err[lo]).sum()); acc[k][3] += float(ww[lo].sum())
    for k in tot:
        for i in range(4): tot[k][i] += acc[k][i]
    f = lambda k: f"{acc[k][0] / acc[k][1]:.3f} / {acc[k][2] / max(acc[k][3], 1e-12):.3f}"
    say(f"| {l} | {100 * share:.0f} % | {f('p32')} | {f('p64')} | {f('w32')} | {f('w64')} | ({time.time() - t0:.0f} s)")
g = lambda k: (tot[k][0] / tot[k][1], tot[k][2] / max(tot[k][3], 1e-12))
say("\n## All sampled layers (router-weighted mean relative output error, ranks 0-9 / ranks 6-9; a drop = 1.0)")
for k, name in (("p32", "plain rank 32"), ("p64", "plain rank 64"), ("w32", "whitened rank 32"), ("w64", "whitened rank 64")):
    a, b = g(k); say(f"- {name}: {a:.3f} / {b:.3f}")
w32, w64 = g("w32")[0], g("w64")[0]
v = "GO" if (w32 <= 0.5 or w64 <= 0.5) else "STOP" if w64 > 0.7 else "BETWEEN"
say(f"\nVerdict (a9d81e8): whitened rank 32 {w32:.3f}, rank 64 {w64:.3f} -> {v} (GO <= 0.5 at 32 or 64; STOP > 0.7 at 64)")
open(os.path.join(OUT, "SHADOW_read.txt"), "w").write("\n".join(lines) + "\n")
