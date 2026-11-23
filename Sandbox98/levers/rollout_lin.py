"""Stand-in hook 'rollout_lin' (ROLLOUT, eval only): at depth j the stand-in is f rolled j - 1 steps from the TRUE anchor
state W_s along the true tokens (sb/rollout.py, transition model 'lin'): the predicted W_{s+j-1}. s30 unchanged."""
import numpy as np
import mlx.core as mx
_C = {}


def standin(j, h, anc):
    from sb import core as C
    from sb import rollout as RO
    f = C.CUR["f"]
    if _C.get("f") != f:
        z = np.load(f)
        W = C.load_W(z)[0].astype(mx.float32)
        E = C.embed(mx.array(np.asarray(z["ids"], np.int32)))
        _C.update(f=f, R=RO.rollout_table("lin", W, E))
    return _C["R"][j - 1][anc][None]
