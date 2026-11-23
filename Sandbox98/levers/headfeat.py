"""Lever 'headfeat' (orchestrator 20:2x, item 2: STATE TARGET IN HEAD SPACE). t30f's / 'feat' feature loss compared the
stand-in with the true state after the head's input norm only (wd space) and was NO EFFECT on both models. This moves the
target into the head's own fused input space, BigMoE's r2_head: head_in(v) = fc_hidden(pre_fc_norm_hidden(v)) (here the
hidden half of fc). At every depth j the stand-in h_j (depth j+1's input) is pulled toward head_in(W[s + j]) with
LAM x (1 - R2_head) on the KL's rows, the variance taken over the whole window. The projection
and the norm are under stop-gradient, so only the chain (the stand-in) moves, not the reading of it."""
import os
import mlx.core as mx

LAM = float(os.environ.get("HEADFEAT_LAM", "0.5"))
_D = [None]


def setup(D, args):
    _D[0] = D


def extra(j, h, anc, m, x):
    """LAM x mean over the KL's rows of ||head_in(h) - head_in(W)||^2, divided by the window's centred variance of
    head_in(W) over ALL its rows (a per-depth batch variance collapses when only a few rows are still alive)"""
    D = _D[0]
    W = x["W"]
    idx = mx.minimum(anc + j, W.shape[1] - 1)
    nrm = D.pre_fc_norm_hidden
    gw, Fh = mx.stop_gradient(nrm.weight), mx.stop_gradient(D.fc.weight[:, D.d:])      # fc input = [emb, hidden]

    def head_in(v):
        return mx.fast.rms_norm(v.astype(mx.float32), gw, nrm.eps) @ Fh.T
    key = (x["f"], x["a0"])
    if _V.get("k") != key:
        HY = mx.stop_gradient(head_in(W)[0])
        _V.update(k=key, var=mx.stop_gradient(((HY - HY.mean(0, keepdims=True)) ** 2).sum(-1).mean()))
    pred = head_in(h)[0]
    tgt = mx.stop_gradient(head_in(W[:, idx]))[0]
    num = (((pred - tgt) ** 2).sum(-1) * m).sum() / mx.maximum(m.sum(), 1.0)
    return LAM * num / mx.maximum(_V["var"], 1e-6)


_V = {}
