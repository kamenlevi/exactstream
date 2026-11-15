"""MTP Head lever 'marginw' (WEEK98 S5 WALKW, arm 2, orchestrator 06:2x; REPORT 'WEEK98 S5 WALKW' amendment 1): close-call
weighted TTT. walkw weighs check-start anchors, a proxy for the close calls where kept/32 is lost (NewIdeas' ceilings).
This arm weighs the close calls themselves: at every depth, the rows whose TARGET margin (the target's top-1 minus top-2
log-prob at that row, from the window's top-32) is under MARGIN nats get LAM x the same per-depth KL added (core._kl_hit:
the same rows, targets and top-32), so they weigh 3x. accept-until-fail is unchanged. Training-only."""
import os
import numpy as np
import mlx.core as mx

LAM = float(os.environ.get("MARGINW_LAM", "2.0"))
MARGIN = float(os.environ.get("MARGINW_NATS", "1.0"))
_D = [None]


def setup(D, args):
    _D[0] = D


def extra(j, h, anc, m, x):
    from sb import core as C
    anc_np = np.array(anc); m_np = np.array(m)
    ids = np.asarray(x["ids"]); S = len(ids)
    lpn = np.asarray(x["lp"]); T = lpn.shape[0]
    row_i = np.minimum(anc_np + j, T - 1); tgt_i = np.minimum(anc_np + j + 1, S - 1)
    marg = lpn[row_i, 0].astype(np.float32) - lpn[row_i, 1].astype(np.float32)
    sel = np.flatnonzero((marg < MARGIN) & (m_np > 0))
    if len(sel) == 0:
        return mx.array(0.0)
    hn = _D[0].norm(h[:, mx.array(sel)])
    tk = mx.array(np.asarray(x["tk"])[row_i[sel]]); lp = mx.array(lpn[row_i[sel]])
    l, _ = C._kl_hit(hn, tk, lp, mx.ones((len(sel),), mx.float32), mx.array(ids[tgt_i[sel]]))
    return LAM * l * (len(sel) / max(float(m_np.sum()), 1.0))
