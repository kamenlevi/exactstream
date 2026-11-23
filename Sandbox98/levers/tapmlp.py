"""Lever 'tapmlp' (WEEK98 open item 1: a nonlinear, trained map from an early tap to the deep state): at depth j >= 2 the
carried stand-in h gets a learned correction from the TARGET's true residual after layer TAP at the drafted row
(anc + j - 1), i.e. a partial trunk pass over the drafted tokens (cost: TAP + 1 of the target's layers per drafted
token; price it before adopting). The MLP's output layer starts at zero, so step 0 is the base drafter exactly.
On the 125B, LINEAR maps from early taps were worse than the stale chain (LAYERS: {5} 41.66 vs 72.13); this asks whether
a trained nonlinear map, used as a correction to the stand-in rather than a replacement, does better.
Settings (env): TAPMLP_TAP (a core.TAPS layer, default 12), TAPMLP_WIDTH (2048), TAPMLP_LR (the module's own constant lr,
default 1e-5 = the 125B TAPSQ2 tap's lr; the head keeps the screen's lr)."""
import os
import numpy as np
import mlx.core as mx
import mlx.nn as nn

TAP, WIDTH = int(os.environ.get("TAPMLP_TAP", "12")), int(os.environ.get("TAPMLP_WIDTH", "2048"))
PREFIX, MODULE_LR = "tap_", float(os.environ.get("TAPMLP_LR", "1e-5"))
_D, _CACHE = [None], {}


def setup(D, args):
    _D[0] = D
    D.tap_in = nn.RMSNorm(2048)
    D.tap_up = nn.Linear(2048, WIDTH, bias=False)
    D.tap_down = nn.Linear(WIDTH, 2048, bias=False)
    D.tap_down.weight = mx.zeros((2048, WIDTH), mx.float32)


def _taps():
    from sb import core as C
    f = C.CUR["f"]
    if _CACHE.get("f") != f:
        tz = np.load(f[:-4] + ".taps.npz")["taps"]
        _CACHE.update(f=f, t=mx.array(tz[list(C.TAPS).index(TAP)].astype(np.float32)))
    return _CACHE["t"]


def standin(j, h, anc):
    D = _D[0]
    t = _taps()[anc + j - 1][None]                     # [1, n, 2048]: the true layer-TAP residual of the drafted row
    return h + D.tap_down(nn.silu(D.tap_up(D.tap_in(t))))
