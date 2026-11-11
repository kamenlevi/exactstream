# the dwin fix on mlx_lm's RotatingKVCache alone (CPU, no model): positions as values; normal flow vs dwin flow
import sys
import mlx.core as mx
mx.set_default_device(mx.cpu)
from mlx_lm.models.cache import RotatingKVCache
def kv(a, b):
    x = mx.arange(a, b, dtype=mx.float32).reshape(1, 1, -1, 1)
    return x, x
def held(c):
    k = c._temporal_order(c.keys) if c.keys is not None else None
    return sorted(int(v) for v in k.reshape(-1).tolist()[: min(c.offset, c.max_size)]) if k is not None else []
def upd(c, a, b):
    c.update_and_fetch(*kv(a, b))
SWIN = 8; bad = 0
for P in range(1, 26):
    start = max(0, P - (SWIN - 1))
    n = RotatingKVCache(max_size=SWIN - 1, keep=0); n.offset = start
    upd(n, start, P)                                    # normal: the window T[start:P] in the first propose
    old = new = "skip"
    if P - 1 > start:                                   # the old dwin (start .. P-2), then the 1-position update
        o = RotatingKVCache(max_size=SWIN - 1, keep=0); o.offset = start
        upd(o, start, P - 1)
        try: upd(o, P - 1, P); old = "ok" if held(o) == held(n) else f"DIFF {held(o)}"
        except ValueError as e: old = "CRASH"
    s0 = max(0, P - SWIN)
    if P - 1 > s0:                                      # the fixed dwin (s0 .. P-2), then the 1-position update
        f = RotatingKVCache(max_size=SWIN - 1, keep=0); f.offset = s0
        upd(f, s0, P - 1)
        try:
            upd(f, P - 1, P); new = "ok" if held(f) == held(n) else f"DIFF {held(f)} vs {held(n)}"
            upd(f, P, P + 1); upd(n, P, P + 1)           # one more 1-position update (a k = 0 check) on both
            new += "" if held(f) == held(n) else f" / next DIFF {held(f)} vs {held(n)}"
        except ValueError as e: new = "CRASH"
    bad += new not in ("ok", "skip")
    print(f"P {P:2d} start {start:2d} s0 {s0:2d}: old dwin {old:5s} fixed dwin {new}")
print("FIXED OK" if bad == 0 else f"FIXED BAD {bad}"); sys.exit(bad > 0)
