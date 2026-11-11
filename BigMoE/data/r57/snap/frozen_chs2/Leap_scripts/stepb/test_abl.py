"""CPU unit test of S2b's memo stub (flat2_probe._memo / _rewrap / _flat; Leap, Sat 2026-11-07 00:5x). MLX on the CPU, tiny
arrays, ~1 s. Checks: (1) the first call runs the piece and returns its real outputs; (2) later calls return equal
values; (3) the graph of a stubbed output still holds the upstream ops (mx.depends: the producer of the piece's input is
not dropped), while the piece's own ops are gone; (4) an instance outside ids passes through; (5) a module function
(ids None) memoizes too."""
import os
import re
import sys
import tempfile

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flat2_probe as FP

mx.set_default_device(mx.cpu)
ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}")


class Piece:
    def __init__(self, w):
        self.w = w

    def __call__(self, x, pending=None):
        y = mx.sin(x @ self.w)                           # the piece's own ops: Matmul, Sin, Cos, Multiply
        return y, mx.cos(x) * 2.0


def labels(arrs):
    p = os.path.join(tempfile.mkdtemp(), "g.dot")
    mx.export_to_dot(p, *arrs)
    return set(re.findall(r'label ="([^"]+)"', open(p).read()))


w = mx.ones((4, 4)) * 0.1
p1, p2 = Piece(w), Piece(w)
orig = Piece.__call__
x0 = mx.arange(8, dtype=mx.float32).reshape(2, 4)
ref = orig(p1, mx.exp(x0))
mx.eval(ref)
Piece.__call__ = FP._memo(mx, orig, {id(p1)})
try:
    a = p1(mx.exp(x0))                                   # 1st: real
    check("first call returns the real outputs", all(bool(mx.array_equal(u, v).item()) for u, v in zip(a, ref)))
    h = mx.exp(x0 + 1.0)                                 # a different producer, same shape
    b = p1(h)                                            # stubbed: cached values, depending on h
    lb = labels(list(b))                                 # the graph first: evaluating b detaches it
    check("the stubbed output's graph keeps the upstream ops (Exp, Add, Depends)", {"Exp", "Add", "Depends"} <= lb)
    check("the piece's own ops are gone from it (no Sin / Matmul)", not any(("Sin" in s) or ("Matmul" in s) for s in lb))
    check("later calls return the cached values", all(bool(mx.array_equal(u, v).item()) for u, v in zip(b, ref)))
    c = p2(h)                                            # not in ids: passes through
    lc = labels(list(c))
    check("an instance outside ids computes for real", any("Sin" in s for s in lc))
    x1 = mx.exp(x0)[:1]
    d = p1(x1)                                           # a new input shape: a new key, real again (evaluated in memo)
    check("a new input shape runs the piece again", d[0].shape == (1, 4)
          and bool(mx.allclose(d[0], orig(p1, x1)[0]).item()))
finally:
    Piece.__call__ = orig


for mode in ("bar", "cal"):                              # Sat 01:2x: the barrier-preserving stub and its calibration
    Piece.__call__ = FP._memo(mx, orig, {id(p1)}, mode=mode)
    try:
        a = p1(mx.exp(x0))                               # bar: 1st call real; cal: always real (+ the zero term)
        mx.eval(a)
        h = mx.exp(x0 + 1.0)
        b = p1(h)
        lb = labels(list(b))
        if mode == "bar":
            check("bar: the output reads the input's buffer (Exp -> Slice -> Multiply x 0 -> Add) and not the piece "
                  "(no Sin / Matmul)", {"Exp", "Slice", "Multiply", "Add"} <= lb and not any(
                      ("Sin" in s) or ("Matmul" in s) for s in lb))
            check("bar: values == the kept outputs", all(bool(mx.array_equal(u, v).item()) for u, v in zip(b, ref)))
        else:
            check("cal: the real piece runs (Sin in the graph) with the zero term", any("Sin" in s for s in lb) and "Add" in lb)
            check("cal: values == the real outputs of this input", all(bool(mx.allclose(u, v).item())
                                                                    for u, v in zip(b, orig(p1, h))))
    finally:
        Piece.__call__ = orig


def fn(x, y):
    return x * y + 1.0


m = FP._memo(mx, fn, None)
r1 = m(mx.ones((3,)), mx.ones((3,)) * 2.0)
r2 = m(mx.zeros((3,)) - 5.0, mx.ones((3,)))
check("a module function memoizes (ids None)", bool(mx.array_equal(r1, r2).item()) and float(r2[0].item()) == 3.0)
check("_flat walks tuples / lists / dicts", len(FP._flat(mx, (mx.ones(1), [mx.ones(2), {"k": mx.ones(3)}], None, 7))) == 3)
print("TEST_ABL", "PASS" if ok else "FAIL")
