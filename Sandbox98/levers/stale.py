"""Stand-in hook 'stale' (ROLLOUT's baseline, eval only): at every depth j the stand-in is the TRUE state at the anchor,
W_s: fresh where the draft starts, never advanced. s30 unchanged."""
import numpy as np
import mlx.core as mx
_C = {}


def standin(j, h, anc):
    from sb import core as C
    f = C.CUR["f"]
    if _C.get("f") != f:
        _C.update(f=f, W=C.load_W(np.load(f))[0].astype(mx.float32))
    return _C["W"][anc][None]
