"""Sketch-on-miss shadows (Breakthrough, Mon 2026-11-02 22:2x; Fable A's NEWIDEAS_SCREEN.md section 2, the orchestrator
22:2x; pre-registered SCOREBOARD ad1370c). Rank-r SVD of every routed expert's three matrices (gate / up 640 x 2560,
down 2560 x 640) of the 4-bit model as served (~/BigMoE/models/bare, mx.dequantize, group 64), W ~ A B with A = U_r S_r
(m x r) and B = V_r^T (r x n), each stored int8 with a per-row fp16 scale (symmetric, max-abs / 127).
Output per layer: SHADOW_DIR/L<ll>.npz with {proj}_A int8 (512, m, r), {proj}_As f16 (512, m), {proj}_B int8 (512, r, n),
{proj}_Bs f16 (512, r) and {proj}_err f32 (512,) = ||W - A_q B_q||_F / ||W||_F after the int8 rounding.
CPU only (MLX's CPU stream for the dequantize, numpy / Accelerate for the SVD; VECLIB_MAXIMUM_THREADS caps the BLAS
threads); memory: one chunk of experts at a time (~0.2 GB). Skips layers already written.
Usage (BigMoE venv): VECLIB_MAXIMUM_THREADS=4 python dm_shadow_build.py [first_layer last_layer] [--test]"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.cpu)
M = os.path.expanduser("~/BigMoE/models/bare")
SHADOW = os.environ.get("SHADOW_DIR", os.path.expanduser("~/FableGod/decision_map/shadows_r32"))
R, CH = int(os.environ.get("SHADOW_RANK", "32")), 16
WM = json.load(open(os.path.join(M, "model.safetensors.index.json")))["weight_map"]

def q8(X):
    """int8 per row along the last axis; returns (int8, fp16 scale)"""
    s = np.abs(X).max(axis=-1, keepdims=True) / 127.0; s[s == 0] = 1.0
    return np.clip(np.rint(X / s), -127, 127).astype(np.int8), s[..., 0].astype(np.float16)

def layer(li, test=False):
    out = os.path.join(SHADOW, f"L{li:02d}.npz")
    if os.path.exists(out) and not test: return None
    res, t0 = {}, time.time()
    for proj in ("gate_proj", "up_proj", "down_proj"):
        pre = f"language_model.model.layers.{li}.mlp.switch_mlp.{proj}."
        w = mx.load(os.path.join(M, WM[pre + "weight"]))
        Wq, Sc, Bi = w[pre + "weight"], w[pre + "scales"], w[pre + "biases"]
        E = 8 if test else Wq.shape[0]; A = Bm = None; err = np.zeros(E, np.float32)
        for e0 in range(0, E, CH):
            e1 = min(E, e0 + CH)
            W = np.array(mx.dequantize(Wq[e0:e1], Sc[e0:e1], Bi[e0:e1], group_size=64, bits=4).astype(mx.float32))
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            a, b = U[..., :R] * S[..., None, :R], Vt[..., :R, :]
            aq, as_ = q8(a); bq, bs = q8(b)
            if A is None:
                m, n = W.shape[1:]
                A, As = np.zeros((E, m, R), np.int8), np.zeros((E, m), np.float16)
                Bm, Bs = np.zeros((E, R, n), np.int8), np.zeros((E, R), np.float16)
            A[e0:e1], As[e0:e1], Bm[e0:e1], Bs[e0:e1] = aq, as_, bq, bs
            Wr = (aq * as_[..., None].astype(np.float32)) @ (bq * bs[..., None].astype(np.float32))
            err[e0:e1] = np.linalg.norm(W - Wr, axis=(1, 2)) / np.linalg.norm(W, axis=(1, 2))
        res.update({f"{proj}_A": A, f"{proj}_As": As, f"{proj}_B": Bm, f"{proj}_Bs": Bs, f"{proj}_err": err})
        print(f"[shadow] L{li:02d} {proj}: {E} experts, rel. error mean {err.mean():.3f} max {err.max():.3f}, "
              f"{time.time() - t0:.0f} s", flush=True)
        del w, Wq, Sc, Bi
    if not test:
        os.makedirs(SHADOW, exist_ok=True); tmp = out + ".tmp.npz"; np.savez(tmp, **res); os.replace(tmp, out)
    return res

if __name__ == "__main__":
    test = "--test" in sys.argv; a = [int(x) for x in sys.argv[1:] if x.lstrip("-").isdigit()]
    lo, hi = (a[0], a[1]) if len(a) == 2 else (0, 47)
    for li in range(lo, hi + 1): layer(li, test)
