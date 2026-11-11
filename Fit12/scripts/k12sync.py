"""Fit12 K12 (Tue 3 Nov 23:0x; the orchestrator's yes 22:5x; PLAN.md 'T3R2' [K] CHECK): Leap's [K] (50de28c, S7B-PROBE) says
a CPU rewrite of a slot the GPU already read inside one command buffer is not seen by later dispatches. wrap(mx) puts a full
mx.eval at the entry and the exit of every MoE block call (mtplx qwen4_exp.SparseMoeBlock.__call__, i.e. bt_b12's block with
the T3 patches, installed by load_b12): each command buffer then holds one block's slot reads, and a late part (applied in the
next layer's call, before its block) runs in a buffer where its pool was not read before. Same prompts / arms / order as an
unsynced run = the same requests and evictions, so the bits must match unless the unsynced run read stale lines. Call it
after B.load_b12. No engine file is changed."""
N = [0]
_ORIG = [None]
_W = [None]


def wrap(mx, Q=None):
    if Q is None:
        from mtplx.models import qwen4_exp as Q
    assert _W[0] is None, "k12sync.wrap: already wrapped"
    blk = Q.SparseMoeBlock.__call__

    def _k12_block(self, x, *a, **kw):
        mx.eval(x)                       # everything encoded before this block (attention, a late part) is finished
        y = blk(self, x, *a, **kw)
        mx.eval(y)                       # this block's slot reads are finished before anything else is encoded
        N[0] += 1
        return y

    _ORIG[0], _W[0] = (Q, blk), _k12_block
    Q.SparseMoeBlock.__call__ = _k12_block
    return _k12_block


def installed():
    return _W[0] is not None and _ORIG[0][0].SparseMoeBlock.__call__ is _W[0]
