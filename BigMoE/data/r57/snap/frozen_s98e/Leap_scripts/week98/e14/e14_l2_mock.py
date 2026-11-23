"""e14_l2_mock.py (Leap, Sun 15 Nov 2026 21:3x; E14 amendment 1): a desk test of e14_engine._tier_block_l2 (CHECK55 lever 2
at 17-32 rows on the streamed engine's K-73 path) against v32's own _tier_block, with no engine: a mock MoE block with
the 125B's real shapes and random quantized weights (8-bit router 512 x 2,560; 4-bit shared gate / up 640, shared gate 1,
down 2,560 x 640), top-10, norm_topk_prob, and a stand-in tier switch returning per-(row, expert) outputs looked up by
expert id (so the experts' ids and the scores reach the output through the combine, as do g and s through moe_tail).
ref2.apply(), v32 + v32one installed and ON (as C32onem), inside attention_phase("decode_verify"). Rows 32 and 24.
Line: the block outputs bitwise equal (and no lever-2 error) -> the wiring holds; anything else -> fix before Monday.
usage: [BigMoE venv] python -B e14_l2_mock.py   (MEM < 1 GB; share lock)"""
import os
import sys
import time

for k_, v_ in (("REF2_SB_FIRST", "1"), ("REF2_PLE_LATE", "1"), ("REF2_KEEP_CACHES", "1"), ("HF_HUB_OFFLINE", "1")):
    os.environ.setdefault(k_, v_)
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98e/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98e/Leap_scripts/week98"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98e/Leap_scripts/week98/e14"))
import mlx.core as mx
import mlx.nn as nn
from mtplx.attention_context import attention_phase
import ref2

ref2.apply()
import v32

v32.install(ref2)
import v32one

v32one.install(v32, ref2)
v32one.ON[0] = True
import e14_engine as E

t0 = time.time()
BF = mx.bfloat16


def qlin(N, K, bits, key):
    lin = nn.Linear(K, N, bias=False)
    lin.weight = (mx.random.normal((N, K), key=key) * 0.02).astype(BF)
    return nn.QuantizedLinear.from_linear(lin, group_size=64, bits=bits)


class SE:
    pass


class Switch:
    def __init__(self, key):
        self._bm = {"L": 0, "tier": {}}
        self.table = (mx.random.normal((512, 2560), key=key) * 0.1).astype(BF)

    def __call__(self, x, inds):                     # per-(row, expert) outputs: [..., k, 2560]
        return mx.take(self.table, inds.astype(mx.int32), axis=0)


class Blk:
    pass


ks = mx.random.split(mx.random.key(21), 8)
blk = Blk()
blk.gate = qlin(512, 2560, 8, ks[0])
se = SE()
se.gate_proj, se.up_proj, se.down_proj = qlin(640, 2560, 4, ks[1]), qlin(640, 2560, 4, ks[2]), qlin(2560, 640, 4, ks[3])
blk.shared_expert = se
blk.shared_expert_gate = qlin(1, 2560, 4, ks[4])
blk.top_k, blk.norm_topk_prob = 10, True
blk.switch_mlp = Switch(ks[5])


def md():
    d = os.path.expanduser("~/Breakthrough/scripts/m1/check55")
    if d not in sys.path:
        sys.path.append(d)
    import moe_dense
    return moe_dense


E.S.update(v32=v32, G={"_e14_md": md}, tier0=v32._tier_block)
ok_all = True
lines = [f"E14 L2 MOCK {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}"]
for rows in (32, 24):
    x = (mx.random.normal((1, rows, 2560), key=mx.random.key(100 + rows)) * 0.5).astype(BF)
    with attention_phase("decode_verify"):
        E.L2["on"] = False
        y0 = v32._tier_block(blk, x, "k73")
        E.L2["on"] = True
        y1 = E._tier_block_l2(blk, x, "k73")
        E.L2["on"] = False
        mx.eval(y0, y1)
    eq = bool(mx.all(y0 == y1).item())
    nrow = int(mx.sum(mx.all(y0.reshape(rows, -1) == y1.reshape(rows, -1), axis=-1)).item())
    ok_all &= eq
    lines.append(f"  rows {rows}: block output bitwise {eq} ({nrow} / {rows} rows); shape {tuple(y1.shape)}")
lines.append(f"  lever 2 calls {E.L2['n']}, errors {E.L2['err'][:2]}; v32 tierblk {v32.CNT.get('tierblk')}")
ok_all &= not E.L2["err"] and E.L2["n"] == 2
lines.append(f"E14 L2 MOCK verdict: {'WIRING HOLDS (bitwise, no errors)' if ok_all else 'FIX BEFORE MONDAY'} ({time.time() - t0:.0f} s)")
out = "\n".join(lines)
print(out)
open(os.path.expanduser("~/Leap/data/week98/e14/l2_mock.txt"), "a").write(out + "\n")

# ---- part 2 (21:4x): the <= 16-row patch, e14_loop.py's _exact_block, extracted and run with its globals stubbed
import textwrap
import types

import numpy as np
from router_fused import router_fused
from glue_fused import moe_tail, shared_cat

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e14_loop.py")).read()
a = src.index("    def _exact_block(self, x):")
b = src.index("    _QX.SparseMoeBlock.__call__ = _exact_block")
FB = []
ns = dict(np=np, mx=mx, router_fused=router_fused, shared_cat=shared_cat, moe_tail=moe_tail, K73N=[0],
          PR={"on": False, "cur": None, "y": {}, "blk": {}},
          EM=types.SimpleNamespace(STATE=types.SimpleNamespace(sorted=True, capture=None)),
          _BT=types.SimpleNamespace(_COMBINE=True, _GLU="3c"),
          _fb=lambda self, x: FB.append(1) or None,
          E14_L2=[False], E14_L2N=[0], E14_L2ERR=[], _e14_md=md)
exec(textwrap.dedent(src[a:b]), ns)
xb = ns["_exact_block"]
lines2 = []
ok2 = True
for rows in (16, 9):
    x = (mx.random.normal((1, rows, 2560), key=mx.random.key(300 + rows)) * 0.5).astype(BF)
    with attention_phase("decode_verify"):
        ns["E14_L2"][0] = False
        y0 = xb(blk, x)
        ns["E14_L2"][0] = True
        y1 = xb(blk, x)
        ns["E14_L2"][0] = False
        mx.eval(y0, y1)
    eq = bool(mx.all(y0 == y1).item())
    ok2 &= eq
    lines2.append(f"  _exact_block rows {rows}: lever 2 == K-73's pair bitwise {eq}")
ok2 &= not FB and not ns["E14_L2ERR"] and ns["E14_L2N"][0] == 2
lines2.append(f"  lever 2 calls {ns['E14_L2N'][0]}, errors {ns['E14_L2ERR'][:2]}, fallbacks to _fb {len(FB)}")
lines2.append(f"E14 L2 MOCK part 2 verdict: {'WIRING HOLDS (bitwise, no errors)' if ok2 else 'FIX BEFORE MONDAY'}")
out2 = "\n".join(lines2)
print(out2)
open(os.path.expanduser("~/Leap/data/week98/e14/l2_mock.txt"), "a").write(out2 + "\n")
