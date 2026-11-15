"""MTP Head lever 'walkw' (WEEK98 S5 WALKW; REPORT 'WEEK98 S5 WALKW'): walk-weighted TTT, i.e. train where you draft.
At run time every draft starts at a check start: the anchor right after the previous check's bonus token. These are
harder than the average anchor (EvalSuite 9480ba4: HE+ depth 0 91.6 % there vs 97.9 % elsewhere). Plain TTT weighs
every anchor equally. This lever adds LAM x the same per-depth KL (core._kl_hit, top-32, the same rows and targets as
ttt_step) on the anchors that are check starts in the reference walk (s30's, data/sb98/walk_starts_s30.npz). Those
anchors then weigh (1 + LAM)x at every depth they survive to (accept-until-fail is unchanged). Training-only.
Judged by kept/32 and depth 0 at check starts."""
import os
import numpy as np
import mlx.core as mx

LAM = float(os.environ.get("WALKW_LAM", "2.0"))
TAB = os.path.expanduser(os.environ.get("WALKW_TABLE", "~/TrainLab/data/sb98/walk_starts_s30.npz"))
_S = {}
_D = [None]


def setup(D, args):
    _D[0] = D
    z = np.load(TAB)
    _S.update({k: set(z[k].tolist()) for k in z.files})


def extra(j, h, anc, m, x):
    from sb import core as C
    st = _S.get(os.path.basename(str(x.get("f", ""))))
    if not st:
        return mx.array(0.0)
    anc_np = np.array(anc); m_np = np.array(m)
    sel = np.flatnonzero(np.fromiter((int(a) in st for a in anc_np), bool, len(anc_np)) & (m_np > 0))
    if len(sel) == 0:
        return mx.array(0.0)
    ids = np.asarray(x["ids"]); S = len(ids); T = np.asarray(x["tk"]).shape[0]
    a = anc_np[sel]
    row_i = np.minimum(a + j, T - 1); tgt_i = np.minimum(a + j + 1, S - 1)
    hn = _D[0].norm(h[:, mx.array(sel)])
    tk = mx.array(np.asarray(x["tk"])[row_i]); lp = mx.array(np.asarray(x["lp"])[row_i])
    l, _ = C._kl_hit(hn, tk, lp, mx.ones((len(sel),), mx.float32), mx.array(ids[tgt_i]))
    return LAM * l * (len(sel) / max(float(m_np.sum()), 1.0))
