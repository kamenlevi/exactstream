"""Lever 'feat' (t30f's carried-state feature loss, ported; WEEK98 item 2 'a learned refresher'): at every depth j, the
relative squared error between the head's next input as read, pre_fc_norm_hidden(h_j), and the true one,
pre_fc_norm_hidden(W[s + j]), on the KL's rows (answer targets x accept-until-fail) and weights. lambda = LAM (fixed).
On the 125B, t30f was NO EFFECT; this screen checks whether the sandbox agrees (a transfer check) and is the template for
learned-stand-in levers."""
import mlx.core as mx

LAM = 0.5


def extra(j, h, anc, m, x):
    """h_j row i is depth j + 1's stand-in for W[anc[i] + j]"""
    D = _D[0]
    W = x["W"]
    idx = mx.minimum(anc + j, W.shape[1] - 1)
    pred = D.pre_fc_norm_hidden(h.astype(mx.float32))
    tgt = D.pre_fc_norm_hidden(W[:, idx].astype(mx.float32))
    rel = ((pred - tgt) ** 2).sum(-1) / mx.maximum((tgt ** 2).sum(-1), 1e-6)
    return LAM * (rel[0] * m).sum() / mx.maximum(m.sum(), 1.0)


_D = [None]


def setup(D, args):
    _D[0] = D
