"""S10b trace (NewIdeas QUEUE.md row S10b; slim spec 10-26 00:xx): data for TrainLab's long-lead expert predictor.
Per decode/verify call of a streamed MoE layer (rows <= 8): the pass seq, the ext layer index and every row's routed ids
in the tier block's order (score descending); for the predictor layers (BIGMOE_S10B_LAYERS) also the rows' MoE input
(the router's input: layer L's hyper-connection mix, bf16). The arrays are kept as lazy references and read back only by
flush(), which the harness calls once a prompt is done (all of it evaluated), so the trace adds no sync point to the
decode. The loader's per-(pass, layer) miss flags come from the ext (BIGMOE_S10B_MISS, written at exit).
Enabled by BIGMOE_S10B=1 (stream.py calls rec() from the decode/verify branch)."""
import os

import mlx.core as mx
import numpy as np

LAYERS = tuple(int(v) for v in os.environ.get("BIGMOE_S10B_LAYERS", "2,8,14,20,26,32,38,44").split(","))
_pend = []


def rec(bm, x, indices):
    L = bm["L"]
    _pend.append((bm["gate"].seq, L, indices, x if L in LAYERS else None))


def drop():
    _pend.clear()


def flush():
    """The calls since the last flush / drop as numpy arrays (everything they reference must be evaluated already):
    seq, L, rows per call; ids [sum rows, K] int16 with row_off per call; x [rows of predictor-layer calls, D] as bf16
    bits (uint16) with x_off per call (-1: not a predictor layer)."""
    ids = [ind.reshape(-1, ind.shape[-1]).astype(mx.int16) for _, _, ind, _ in _pend]
    xs = [x.reshape(-1, x.shape[-1]).astype(mx.bfloat16).view(mx.uint16) for _, _, _, x in _pend if x is not None]
    mx.eval(ids + xs)                                   # one batch of tiny copies, no model work
    ids = [np.array(a) for a in ids]
    xs = [np.array(a) for a in xs]
    rows = np.array([a.shape[0] for a in ids], np.int16)
    x_off, n, j = [], 0, 0
    for (_, _, _, x), a in zip(_pend, ids):
        if x is None:
            x_off.append(-1)
        else:
            assert xs[j].shape[0] == a.shape[0], (xs[j].shape, a.shape)
            x_off.append(n); n += xs[j].shape[0]; j += 1
    out = dict(seq=np.array([p[0] for p in _pend], np.int32), L=np.array([p[1] for p in _pend], np.int16), rows=rows,
               row_off=(np.cumsum(rows, dtype=np.int64) - rows), ids=np.concatenate(ids) if ids else np.zeros((0, 10), np.int16),
               x_off=np.array(x_off, np.int64), x=np.concatenate(xs) if xs else np.zeros((0, 0), np.uint16))
    _pend.clear()
    return out
