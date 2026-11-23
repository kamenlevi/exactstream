"""moe32.py (Leap, Sun 15 Nov 2026 17:5x; D8: a correctness fix for v32's V32_MOE1 resident-expert path at 17-32 rows).
qmmg25's grouped expert GEMM runs each segment (one expert's (row, expert) pairs, from gmr24.segments) as ONE 16-row tile
and stores at most 16 rows of it. At <= 16 rows per check an expert has at most 16 pairs, so that always fits. At 17-32
rows an expert picked by more than 16 rows had its 17th+ pairs never computed (zeros, init_value 0). D8 found it on the
12-layer slice: layer 8, rows 26 / 29 / 30. The streamed exact-45 engine's 32-row block uses BigMoE's tier kernels, not
this path. Fix: cut every expert's run of pairs into segments of <= 16. A pair's arithmetic depends only on its own row,
the expert's weights and the fixed tile (padding rows copy a real row and are never stored), so any such cut gives every
pair the bits of a <= 16-row call.
segments_cap(ids, cap=16) -> (order, sexp, prow, seg, P) as gmr24.segments; experts_nax_cap / experts_split_cap as
qmmg25.experts_nax / ref2.experts_split with it."""
import os
import sys
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98a/Leap_scripts"))
from qmmg25 import qmmg


def segments_cap(ids, cap=16):
    """gmr24.segments with every expert's run of pairs cut into segments of <= cap pairs"""
    k = ids.shape[-1]
    flat = ids.flatten().astype(mx.int32)
    P = flat.shape[0]
    order = mx.argsort(flat)
    sexp = flat[order]
    prow = (order // k).astype(mx.int32)
    ar = mx.arange(P, dtype=mx.int32)
    first = mx.concatenate([mx.array([True]), sexp[1:] != sexp[:-1]])
    run0 = mx.cummax(mx.where(first, ar, 0))               # the index of each pair's run start
    start = first | (((ar - run0) % cap) == 0)
    sid = mx.cumsum(start.astype(mx.int32)) - 1
    tgt = mx.where(start, sid, P + 1)
    seg = mx.full((P + 2,), P, mx.int32).at[tgt].minimum(ar)
    return order, sexp, prow, seg, P


def experts_nax_cap(x, gu, dn, ids, cfg_a=(1, 1), cfg_b=(1, 1)):
    R = x.shape[0]
    order, sexp, prow, seg, P = segments_cap(ids)
    g = qmmg(x, *gu, prow, sexp, seg, P, *cfg_a)
    gate, up = mx.split(g, 2, axis=-1)
    h = nn.silu(gate) * up
    y = qmmg(h, *dn, mx.arange(P, dtype=mx.int32), sexp, seg, P, *cfg_b)
    inv = mx.argsort(order)
    return y[inv].reshape(R, ids.shape[-1], -1)


def experts_split_cap(x2d, gate, up, down, ids, cfg_a, cfg_b):
    R_ = x2d.shape[0]
    order, sexp, prow, seg, P = segments_cap(ids)
    g = qmmg(x2d, *gate, prow, sexp, seg, P, *cfg_a)
    u = qmmg(x2d, *up, prow, sexp, seg, P, *cfg_a)
    h = nn.silu(g) * u
    y = qmmg(h, *down, mx.arange(P, dtype=mx.int32), sexp, seg, P, *cfg_b)
    inv = mx.argsort(order)
    return y[inv].reshape(R_, ids.shape[-1], -1)
