"""cand_f2c.py (Breakthrough, Tue 17 Nov 2026 12:1x; MOE BLOCK DIAG): bench arm 'f2c' = F2's block under mx.compile,
traced once per layer and row count, so a call replays the graph without running F2's Python. The kernels and their
order are F2's, so the output must equal f2 bitwise (bench.py checks that before timing). Only the CPU cost per call
changes."""
import moeblk as MB
import cand_f2

mx = MB.mx
_C = {}


def block(lay, x):
    k = id(lay)
    if k not in _C:
        _C[k] = (lay, mx.compile(lambda xx: cand_f2.block(lay, xx)))
    return _C[k][1](x)
