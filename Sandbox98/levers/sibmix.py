"""SIB fine-tune lever (results/LEVERS_A.md, section SIB): at the training window's post-miss walk starts (s30's walk,
data/sib/pool/KEY.npz from sb/sibstate.py build --sets pool), depths 2-9 read the 2B's sibling rows (post-norm) instead
of the carried stand-in, with the real tokens. Every other anchor trains as usual. The replaced rows are constants, so
no gradient flows through them into the previous depth (ttt_step calls this hook in both passes)."""
import os
import numpy as np
import mlx.core as mx

NSIB = 8
_C = {}


def _load(f):
    from sb import core as C
    key = os.path.basename(f)[:-4]
    p = os.path.join(C.DATA, "sib", "pool", key + ".npz")
    S = len(np.load(f)["ids"])
    msk = np.zeros((S, NSIB), bool)
    rep = mx.zeros((1, NSIB, 2048), mx.float32)            # placeholder when the window has no sibling rows
    if os.path.exists(p):
        o = np.load(p)
        st, L = o["starts"], o["L"]
        if len(st):
            d = o["sib"].shape[-1]
            idx_s = np.repeat(st, NSIB); idx_d = np.tile(np.arange(NSIB), len(st))
            vi = np.flatnonzero(idx_d < np.repeat(L, NSIB))
            msk[idx_s[vi], idx_d[vi]] = True
            rep = mx.zeros((S, NSIB, d), mx.float32)
            rep[mx.array(idx_s[vi]), mx.array(idx_d[vi])] = C.final_norm(
                mx.array(o["sib"].reshape(-1, d)[vi])).astype(mx.float32)
    else:
        raise FileNotFoundError(f"sibmix: no sibling rows for {key} (run sb/sibstate.py build --sets pool)")
    _C.update(f=f, rep=rep, msk=mx.array(msk), any=bool(msk.any()))


def standin(j, h, anc):
    from sb import core as C
    if j < 2 or j > NSIB + 1:
        return h
    if _C.get("f") != C.CUR["f"]:
        _load(C.CUR["f"])
    if not _C["any"]:
        return h
    m = _C["msk"][anc, j - 2]
    return mx.where(m[:, None], _C["rep"][anc, j - 2], h[0])[None]
