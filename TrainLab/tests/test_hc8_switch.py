"""CPU tests for sticky/hc8_switch.py (S5: HC8 per arm), tiny real qwen4_exp GatedResidual mixers + PLE-shaped key / value
projections and an 'mtp' mixer that must stay bf16:
1. build() swaps what the pinned hc8.install swaps (count, the mtp mixer untouched); use(False) right after it = the
   loaded model bit for bit.
2. use(True) == a second, identical tree after plain hc8.install, bit for bit (the pinned code's 8-bit modules).
3. use(False) after use(True) = the loaded model again, bit for bit; the 8-bit outputs differ from bf16 (it really switched).
usage: [BigMoE venv] python tests/test_hc8_switch.py"""
import os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sticky"))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
mx.set_default_device(mx.cpu)
from mtplx.models import qwen4_exp as Q  # noqa: E402
import hc8_3055bf7 as H8  # noqa: E402
import hc8_switch as HC  # noqa: E402

ARGS = types.SimpleNamespace(hc_count=4, hidden_size=64, rms_norm_eps=1e-6, hc_lowrank=64)
ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(f"{'ok  ' if cond else 'FAIL'} {name}")


class Tree(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.layers = [Q.GatedResidual(ARGS), Q.GatedResidual(ARGS, use_combine=False)]
        ple = Q.PLELayer.__new__(Q.PLELayer)                   # only key_proj / value_proj matter to hc8.install
        nn.Module.__init__(ple)
        ple.key_proj, ple.value_proj = nn.Linear(64, 256, bias=False), nn.Linear(64, 64, bias=False)
        self.ple = ple
        self.mtp = Q.GatedResidual(ARGS)
        g = np.random.default_rng(seed)
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear):
                m.weight = mx.array(g.normal(0, 0.05, m.weight.shape).astype(np.float32)).astype(mx.bfloat16)


def outputs(t, x, e):
    ys = []
    for gr in (*t.layers, t.mtp):
        y = gr(x)
        ys += list(y) if isinstance(y, tuple) else [y]
    ys += [t.ple.key_proj(e), t.ple.value_proj(e)]
    mx.eval(ys)
    return ys


same = lambda a, b: all(u.shape == v.shape and mx.array_equal(u, v).item() for u, v in zip(a, b))
g = np.random.default_rng(7)
x = mx.array(g.normal(0, 1, (1, 3, 256)).astype(np.float32)).astype(mx.bfloat16)
e = mx.array(g.normal(0, 1, (1, 3, 64)).astype(np.float32)).astype(mx.bfloat16)
A, B = Tree(1), Tree(1)
orig = outputs(A, x, e)
check("two trees from one seed agree (bf16)", same(orig, outputs(B, x, e)))
mtp0 = A.mtp.input_mix_weight_down
n, mb0, mb1 = HC.build(A)
check(f"build: {n} modules swapped by the pinned install ({mb0 * 1e3:.0f} kB -> {mb1 * 1e3:.0f} kB), mtp mixer left bf16",
      n == 6 and A.mtp.input_mix_weight_down is mtp0 and not hasattr(mtp0, "scales"))
check("use(False) after build = the loaded tree, bit for bit", same(orig, outputs(A, x, e)))
nB = H8.install(B)[0]
HC.use(True)
on = outputs(A, x, e)
check(f"use(True) = plain hc8.install on an identical tree, bit for bit ({nB} swapped there)", nB == n and same(on, outputs(B, x, e)))
diff = max(float(mx.abs(u.astype(mx.float32) - v.astype(mx.float32)).max()) for u, v in zip(on, orig))
check(f"8-bit outputs differ from bf16 (max abs diff {diff:.1e}) and are close (relative)",
      0 < diff and all(float(mx.abs(u.astype(mx.float32) - v.astype(mx.float32)).max()) <= 0.05 * float(mx.abs(v.astype(mx.float32)).max()) + 1e-3
                       for u, v in zip(on, orig)))
HC.use(False)
check("use(False) after use(True) = the loaded tree, bit for bit", same(orig, outputs(A, x, e)) and not HC.SW["on"])
print("ALL OK" if ok_all else "SOME FAILED")
