"""Lever 'noisehead' (orchestrator 20:2x, item 3, stage A: a NOISE-ROBUST HEAD). The head tolerates unbiased noise around
the true state (true W + noise at R2 ~0.5 drafts 82.7 % at MBPP 23-30 vs the chain's 57.3). Training only: at every
depth j >= 2, a random half of the rows reads true W[s + j - 1] + F x (per-dim std) x N(0, 1) instead of the carried
stand-in (F = 0.7: R2 ~0.5), the other half its own stand-in, all with the usual TTT loss. Read-outs use the plain chain.
Noise and the row choice are keyed on (file, slice start, depth), so both passes of the exact step see the same draw.
Stage B (a stand-in predictor whose errors look like that noise) is not in this screen."""
import os, zlib
import mlx.core as mx

F = float(os.environ.get("NOISEHEAD_F", "0.7"))
_SD = {}


def standin(j, h, anc):
    from sb import core as C
    if not C.CUR["train"] or C.CUR["x"] is None:
        return h
    x = C.CUR["x"]
    W = x["W"].astype(mx.float32)
    if _SD.get("f") != (x["f"], x["a0"]):
        _SD.update(f=(x["f"], x["a0"]), sd=mx.sqrt(((W - W.mean(1, keepdims=True)) ** 2).mean(1, keepdims=True)))
    k = mx.random.key(zlib.crc32(f"{os.path.basename(x['f'])}:{x['a0']}:{j}".encode()))
    k1, k2 = mx.random.split(k)
    n = h.shape[1]
    noisy = W[:, anc + j - 1] + F * _SD["sd"] * mx.random.normal((1, n, W.shape[2]), key=k1)
    pick = (mx.random.uniform(shape=(n,), key=k2) < 0.5)[None, :, None]
    return mx.where(pick, noisy.astype(h.dtype), h)
