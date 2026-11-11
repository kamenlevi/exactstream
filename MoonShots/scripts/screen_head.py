"""Screened lm_head: how many vocab rows must be read exactly so the argmax (greedy) or the
Gumbel-max sample (temperature T) is provably the same as with the full head?
Proxy: Qwen3.5-2B (same 248320 x 2048 head shape as Qwen3.6-35B-A3B, tied embedding).
Screen = a low-bit requantization of the head + a per-row error bound. Rows whose upper bound
falls below the best lower bound can't win; the rest are computed exactly.
Step 1 (GPU, small): hidden states -> data/screen_h.npz. Step 2 (CPU, numpy): the analysis."""
import os, sys
import numpy as np
import mlx.core as mx

MODEL = "/Users/kamen/.lmstudio/models/lmstudio-community/Qwen3.5-2B-MLX-4bit"
NTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 900
CACHE = "/Users/kamen/MoonShots/data/screen_h.npz"

if not os.path.exists(CACHE):
    from mlx_lm import load
    m, tok = load(MODEL)
    tm = m.language_model.model
    texts = [open(p).read() for p in ["/Users/kamen/SpecDecode/FINDINGS.md", "/Users/kamen/SpecDecode/mtp.py",
                                      "/Users/kamen/MoEPrefetch/FINDINGS.md"]]
    H = []
    for t in texts:
        ids = tok.encode(t)[: NTOK // len(texts)]
        h = tm(mx.array(ids)[None])[0]                  # after the final norm
        mx.eval(h); H.append(np.array(h.astype(mx.float32)))
    emb = tm.embed_tokens
    W = np.array(mx.dequantize(emb.weight, emb.scales, emb.biases, group_size=emb.group_size,
                               bits=emb.bits).astype(mx.float32))
    np.savez(CACHE, H=np.concatenate(H), W=W)
    print("saved hidden states", flush=True)
    sys.exit()

z = np.load(CACHE)
H, W = z["H"], z["W"]
V, D = W.shape
N = len(H)
Z = H @ W.T
top2 = np.partition(Z, -2, axis=1)[:, -2:]
gap = np.abs(top2[:, 1] - top2[:, 0])
print(f"N={N} hidden states, head {V}x{D}; top1-top2 logit gap median {np.median(gap):.2f}, "
      f"p10 {np.percentile(gap, 10):.2f}", flush=True)
HEAD_BYTES = V * D * 8.5 / 8                           # 35B's head at Q8 (g64): 0.54 GB
rng = np.random.default_rng(0)
hn = np.linalg.norm(H, axis=1)


def requant(Wf, bits, g=64):
    q, s, b = mx.quantize(mx.array(Wf), group_size=g, bits=bits)
    return np.array(mx.dequantize(q, s, b, group_size=g, bits=bits))


def evaluate(name, Zs, Bd, screen_bytes):
    """Zs: screened logits (N, V); Bd: bound on |exact - screened| (N, V)."""
    lo = (Zs - Bd).max(1, keepdims=True)
    cands = (Zs + Bd >= lo).sum(1)
    tot = screen_bytes + cands.mean() * D * 8.5 / 8
    print(f"{name:44s} candidates median {int(np.median(cands)):>6} p90 {int(np.percentile(cands, 90)):>6} "
          f"max {cands.max():>6}  bytes {tot/1e6:5.0f} MB = {tot/HEAD_BYTES:6.1%} of the Q8 head", flush=True)


def gumbel(T):
    u = rng.random((N, V), dtype=np.float32)
    return -np.log(-np.log(np.clip(u, 1e-12, 1 - 1e-7))) * T   # argmax(z + T*g) == sample at temperature T


for bits in (2, 3, 4):
    Ws = requant(W, bits)
    E = W - Ws
    Zs = H @ Ws.T
    rown = np.linalg.norm(E, axis=1)
    Bcs = hn[:, None] * rown[None, :]                    # Cauchy-Schwarz
    Eg = np.abs(E).reshape(V, D // 64, 64).max(-1)       # per-group max |err|
    Bg = np.abs(H).reshape(N, D // 64, 64).sum(-1) @ Eg.T
    sb = V * D * (bits + 0.5) / 8
    evaluate(f"{bits}-bit screen, Cauchy-Schwarz row bound", Zs, Bcs, sb + V * 4)
    evaluate(f"{bits}-bit screen, group-L1 bound", Zs, Bg, sb + V * D // 64)
    if bits >= 3:
        for T in (0.7, 1.0):
            evaluate(f"{bits}-bit screen, group-L1, sample T={T}", Zs + gumbel(T), Bg, sb + V * D // 64)
    del Zs, Bcs, Bg

P = np.linalg.svd(W[rng.choice(V, 20000, replace=False)], full_matrices=False)[2]
for r in (256, 512):
    Pr = P[:r]
    Wr = (W @ Pr.T) @ Pr
    rn = np.linalg.norm(W - Wr, axis=1)
    hperp = np.linalg.norm(H - (H @ Pr.T) @ Pr, axis=1)
    evaluate(f"rank-{r} screen (bf16), projected CS bound", H @ Wr.T, hperp[:, None] * rn[None, :], V * r * 2)
