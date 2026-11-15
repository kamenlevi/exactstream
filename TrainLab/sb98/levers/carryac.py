"""MTP Head lever 'carryac' (WEEK98 screen 2, THE CARRY; REPORT 'WEEK98 S2 CARRY400'): carrya's adapter A, plus a
state-consistency loss on the CARRY only. headfeat pulled the head's own output h (which is also the guess) toward the
true state and lost 1.5 pt; here only the carry A(h) is pulled, in headfeat's space (what the head reads:
head_in(v) = fc_hidden(pre_fc_norm_hidden(v)), under stop-gradient), with LAM x ||head_in(A(h)) - head_in(W[s + j])||^2
over the KL's rows, divided by the window's centred variance of head_in(W). Judged by read-outs only."""
import os
import mlx.core as mx
import mlx.nn as nn

PREFIX, MODULE_LR = "mhca", float(os.environ.get("CARRYA_LR", "1e-4"))
LAM = float(os.environ.get("CARRYAC_LAM", "0.5"))
HID = 1024
_D = [None]
_V = {}


class Adapter(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = mx.ones((d,))
        self.v = nn.Linear(d, HID, bias=False)
        self.u = nn.Linear(HID, d, bias=False)
        self.u.weight = mx.random.normal(self.u.weight.shape, key=mx.random.key(1115)) * 1e-3

    def __call__(self, h):
        return h + self.u(nn.gelu(self.v(mx.fast.rms_norm(h.astype(mx.float32), self.g, 1e-6))))


def setup(D, args):
    D.mhca = [Adapter(D.d)]
    _D[0] = D


def standin(j, h, anc):
    return _D[0].mhca[0](h)


def extra(j, h, anc, m, x):
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
    pred = head_in(D.mhca[0](h))[0]
    tgt = mx.stop_gradient(head_in(W[:, idx]))[0]
    num = (((pred - tgt) ** 2).sum(-1) * m).sum() / mx.maximum(m.sum(), 1.0)
    return LAM * num / mx.maximum(_V["var"], 1e-6)
