"""Lever 'tapsq' (transfer check 2, anchor-only fused taps; the 125B's answer: TAPSQ2, BigMoE 068dc9b, KILL with a small
loss). A zero-init Linear(5 x 2048 -> 2048) of the target's true residual taps after layers 2, 7, 12, 17, 22 (each
RMS-normalized, then concatenated), added to the head's fused input at every TRUE pair row (depth 1: the context and the
anchors), never at drafted rows, as TAPSQ added DFlash2's fused taps to every target row. The 125B used DFlash2's trained
fusion of its taps; this uses the raw taps (a format difference, noted in TRANSFER.md). Own Adam at the head's lr.
Hooked through a wrapper of core.depth1 that acts only on drafters carrying 'tq_proj'."""
import os
import numpy as np
import mlx.core as mx
import mlx.nn as nn

PREFIX, MODULE_LR = "tq_", float(os.environ.get("TAPSQ_LR", "1e-5"))
_CACHE = {}


def _taps(n1):
    from sb import core as C
    f = C.CUR["f"]
    if _CACHE.get("f") != f:
        tz = np.load(f[:-4] + ".taps.npz")["taps"].astype(np.float32)          # [5, S, 2048]
        tz = tz / np.sqrt((tz ** 2).mean(-1, keepdims=True) + 1e-6)
        _CACHE.update(f=f, t=mx.array(np.concatenate(list(tz), -1)))         # [S, 10240]
    return _CACHE["t"][:n1][None]


def _install():
    from sb import core as C
    if getattr(C.depth1, "_tapsq", False):
        return
    orig = C.depth1

    def depth1(D, W, ida, a0):
        if "tq_proj" not in D:
            return orig(D, W, ida, a0)
        S = W.shape[1]
        n1 = S - 1
        x1 = C.fuse(D, W[:, :n1], C.embed(ida[None, 1:n1 + 1]))
        x1 = x1 + D.tq_proj(_taps(n1))
        q, gate, K1, V1 = C.qkv(D, D.input_layernorm(x1), offset=0)
        o = mx.fast.scaled_dot_product_attention(q[:, :, a0:], K1, V1, scale=D.scale, mask="causal")
        n = n1 - a0
        blk = D.o_proj(o.transpose(0, 2, 1, 3).reshape(1, n, -1) * mx.sigmoid(gate[:, a0:]))
        return C.block(D, x1[:, a0:], blk), K1, V1
    depth1._tapsq = True
    C.depth1 = depth1


def setup(D, args):
    D.tq_proj = nn.Linear(5 * D.d, D.d, bias=False)
    D.tq_proj.weight = mx.zeros((D.d, 5 * D.d), mx.float32)
    _install()
