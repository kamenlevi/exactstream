"""Fake-quant with a QTIP-style bitshift trellis (Breakthrough D2, 2026-10-20). Published method: QTIP (Tseng et al.,
NeurIPS 2024, arXiv 2406.11235): random Hadamard transform (incoherence) + bitshift trellis + Viterbi encoding. Here:
- RHT: random signs + mx.hadamard_transform along the input dimension, per-row RMS normalisation.
- Trellis: L-bit state; each weight shifts in k new bits (state = ((state << k) | b) & (2^L - 1)); the weight is
  reconstructed as LUT[state]. LUT = 2^L Gaussian quantiles in a fixed pseudo-random order (a "random Gaussian code",
  which QTIP shows behaves like its computed 1MAD/3INST codes). Free initial state per tile (L bits of overhead).
- Viterbi (exact MSE minimisation over all state paths) on the GPU, vectorised over tiles; predecessors of state s
  differ only in the k shifted-out bits, so the min over predecessors is shared by the 2^k states with the same s >> k.
- No Hessian-aware rounding (BlockLDLQ) here: data-free, like HIGGS. Rate = k + L/T + 16/rowlen bits/weight.
L = 12 (not QTIP's 16) keeps encoding ~16x cheaper on the M5 Pro at a small MSE cost.
usage: trellis.py test   -> NMSE on iid Gaussian data for k = 2, 3, 4 (sanity vs theory)
"""
import math, sys
import numpy as np
import mlx.core as mx

T = 256          # weights per trellis tile


def make_lut(L, seed=0):
    from statistics import NormalDist
    nd = NormalDist()
    q = np.array([nd.inv_cdf((i + 0.5) / (1 << L)) for i in range(1 << L)], np.float32)
    rng = np.random.default_rng(seed)
    return mx.array(q[rng.permutation(1 << L)])


def viterbi(x, lut, L, k):
    """x: [B, T] float32 (unit-variance-ish). Returns the reconstruction [B, T] that minimises the squared error."""
    S, K = 1 << L, 1 << k
    B, n = x.shape
    cost = (x[:, :1] - lut[None, :]) ** 2                       # [B, S], any initial state
    bps = []
    for t in range(1, n):
        prev = cost.reshape(B, K, S // K)                        # p = h * (S/K) + j, h = shifted-out bits
        bh = mx.argmin(prev, axis=1).astype(mx.uint8)            # [B, S/K]
        best = mx.min(prev, axis=1)
        cost = mx.repeat(best, K, axis=1) + (x[:, t:t + 1] - lut[None, :]) ** 2
        bps.append(bh)
        if t % 32 == 0:
            mx.eval(cost, *bps[-32:])
    s = mx.argmin(cost, axis=1)
    mx.eval(s, *bps)
    s = np.array(s).astype(np.int64)
    bp = np.stack([np.array(b) for b in bps])                    # [n-1, B, S/K]
    states = np.empty((n, B), np.int64)
    states[-1] = s
    rows = np.arange(B)
    for t in range(n - 1, 0, -1):
        j = states[t] >> k
        h = bp[t - 1, rows, j].astype(np.int64)
        states[t - 1] = (h << (L - k)) | j
    return mx.array(np.array(lut)[states.T])


_lib = None


def viterbi_cpu(x, lut, L, k):
    """C encoder (scripts/libtrellis.dylib, GCD threads): x float32 numpy [n, T] -> reconstruction [n, T]."""
    import ctypes, os
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libtrellis.dylib"))
        _lib.viterbi_tiles.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    x = np.ascontiguousarray(x, np.float32)
    lut = np.ascontiguousarray(lut, np.float32)
    out = np.empty_like(x)
    _lib.viterbi_tiles(x.ctypes.data, out.ctypes.data, None, x.shape[0], x.shape[1], L, k, lut.ctypes.data)
    return out


def block_hadamard(x, d):
    """Orthonormal Hadamard on blocks of the largest power of two dividing d (Sylvester: symmetric, self-inverse)."""
    b = d & -d
    y = mx.hadamard_transform(x.reshape(-1, d // b, b), scale=1.0 / math.sqrt(b))
    return y.reshape(x.shape)


def quantize(w, k, L=12, seed=0, chunk_tiles=8192, cpu=True):
    """Fake-quantize w [..., d]: random signs + block Hadamard, per-row RMS, trellis tiles of T, inverse."""
    shape, d = w.shape, w.shape[-1]
    lut = make_lut(L, seed)
    signs = mx.array(np.random.default_rng(seed + 1).choice([-1.0, 1.0], d).astype(np.float32))
    x = block_hadamard(w.reshape(-1, d).astype(mx.float32) * signs, d)
    rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True)) + 1e-12
    tiles = (x / rms).reshape(-1, T)
    if cpu:
        xh = mx.array(viterbi_cpu(np.array(tiles), np.array(lut), L, k))
    else:
        xh = mx.concatenate([viterbi(tiles[a:a + chunk_tiles], lut, L, k) for a in range(0, tiles.shape[0], chunk_tiles)])
    wh = block_hadamard(xh.reshape(-1, d) * rms, d) * signs
    return wh.reshape(shape)


if __name__ == "__main__" and sys.argv[1:] == ["test"]:
    import time
    g = mx.random.normal((64, 4096), key=mx.random.key(0))
    for k, cpu in ((3, False), (2, True), (3, True), (4, True)):
        t0 = time.time()
        wh = quantize(g, k, cpu=cpu)
        nmse = float(mx.mean((wh - g) ** 2).item() / mx.mean(g * g).item())
        print(f"k={k} {'cpu' if cpu else 'mlx'}: NMSE {nmse:.4f} (rate-distortion bound 2^-2k = {2 ** (-2 * k):.4f}; Lloyd-Max scalar "
              f"{ {2: 0.1175, 3: 0.03454, 4: 0.009497}[k]:.4f}), {time.time() - t0:.1f}s for {g.size} weights", flush=True)
