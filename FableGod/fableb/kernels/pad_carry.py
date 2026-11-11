"""pad_carry.py (Fable B, DENSE_REBUILD v2): SKETCH of removing the 2 -> 16-row pads in ref2 without a kernel. Not applied.
Today every <= 16-row ref2 call pads its input to 16 rows (mx Full + Concatenate, ~300 launches per 2-row check, 1-3 us
each [K part B]) and slices its 16-row output back to the real rows. Between two ref2 calls the slice and the pad cancel.
Idea: when a ref2 call slices its 16-row result, remember the 16-row parent; when the next call would pad that same
array, hand the parent back instead. No op is launched. Padding rows then hold the previous kernel's padding-row
outputs instead of zeros. That is safe wherever a row's result does not depend on its companions (R-62b / R-71 / R-71b
INV: PASS for attention, the GDN in-proj and the dense kernels); the GDN recurrence and the commit only ever read the
real rows. Bits of the real rows are unchanged, so this is exact-safe by the same INV argument.
It only helps checks with fewer than 16 rows: at 16 rows there are no pads. Worth ~0.5 ms per 2-row check [K part B].
Sketch (drop-in for ref2._pad16 and for the `y[:rows]` slices of its wrappers):"""
from collections import OrderedDict

_PARENT = OrderedDict()                     # id(real-row array) -> (the array, its 16-row parent); a few entries only
_KEEP = 16


def rows_of(y16, rows, shape):
    """slice a 16-row kernel result to its real rows and remember the parent"""
    out = y16[:rows].reshape(shape)
    _PARENT[id(out)] = (out, y16)
    while len(_PARENT) > _KEEP:
        _PARENT.popitem(last=False)
    return out


def pad16(x, rows, pad_fn):
    """the 16-row form of x: the remembered parent when x came out of rows_of (same object), else pad_fn(x) as today"""
    e = _PARENT.get(id(x))
    if e is not None and e[0] is x and e[1].shape[0] == 16 and e[1].size == 16 * (x.size // rows):
        return e[1].reshape(16, -1)
    return pad_fn(x.reshape(rows, -1))
# The identity test (`e[0] is x`) makes a stale id harmless: a recycled id points at another object and falls through
# to the pad. Any mx op between two ref2 calls (an add, an astype) makes a new array and also falls through.
