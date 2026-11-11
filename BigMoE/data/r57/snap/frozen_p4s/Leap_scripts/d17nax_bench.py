"""D17-NAX bench (Leap, Fri 2026-11-06 08:3x; TABLE_350 'D17-NAX', pre-registered): the D7b drafter's block projections at
17 rows, MLX's quantized_matmul (today, R57_D17) vs d7b_nax's NaxQL with D7BNAX_32=1 (two 16-row qmm16c halves, per-shape
ks), on the drafter's real shapes (4-bit g64, bf16 scales, random weights).
  bits: max |y - y_mlx| / max |y_mlx| at 17 rows (drafts only: the drafter's outputs are verified), and at 2 rows the
        ks path vs ks 1
  time: dependent chains over distinct weight copies (cold), round-robin over the two forms, median of 7 rounds; per
        propose = calls per propose x per-call saving
The decisive line is the engine A/B (drafter probe walls, quiet lock). -> ~/Leap/data/d17nax_bench.txt"""
import importlib.util, os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p4s/Leap_scripts"))
import mlx.core as mx
import mlx.nn as nn
os.environ["D7BNAX_32"] = "1"
_p = os.environ.get("D7BNAX_PATH", os.path.expanduser("~/Leap/data/d17nax/d7b_nax_cand.py"))
_spec = importlib.util.spec_from_file_location("d7b_nax_cand", _p)
DN = importlib.util.module_from_spec(_spec); sys.modules["d7b_nax_cand"] = DN; _spec.loader.exec_module(DN)
assert DN.D32[0], "D7BNAX_32 not on"
OUT = os.path.expanduser("~/Leap/data/d17nax_bench.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nD17-NAX bench {time.strftime('%a %H:%M:%S')} ({os.environ.get('D17_COND', 'share lock')}; candidate {_p})")
SHAPES = [("gate|up", 7680, 2560, 10), ("down", 2560, 7680, 5), ("q", 4096, 2560, 5), ("o", 2560, 4096, 5),
          ("k|v", 1024, 2560, 10), ("2560->640", 640, 2560, 10)]
rng = np.random.default_rng(5)
clock = time.perf_counter


def make(N, K, n):
    out = []
    for _ in range(n):
        ql = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=4)
        w = mx.array(rng.standard_normal((N, K)).astype(np.float32) * 0.02).astype(mx.bfloat16)
        q, s, b = mx.quantize(w, group_size=64, bits=4)
        ql.weight, ql.scales, ql.biases = q, s.astype(mx.bfloat16), b.astype(mx.bfloat16)
        mx.eval(ql.parameters())
        out.append(ql)
    return out


def chain(mods, x0, nax, reps=1):
    tt = []
    for _ in range(reps):
        for calls in (False, True):
            prev = mx.zeros((1,), mx.float32)
            for m in mods:
                x = x0 + (prev * 0).astype(x0.dtype)
                if calls:
                    y = DN.NaxQL.__call__(m, x) if nax else nn.QuantizedLinear.__call__(m, x)
                    prev = y.reshape(-1)[:1].astype(mx.float32)
                else:
                    prev = x.reshape(-1)[:1].astype(mx.float32)
            t0 = clock(); mx.eval(prev); tt.append((calls, clock() - t0))
    tn = np.median([t for c, t in tt if not c]); tc = np.median([t for c, t in tt if c])
    return 1e6 * (tc - tn) / len(mods)


tot = 0.0
for name, N, K, per in SHAPES:
    mb = (N * K // 2 + 2 * N * (K // 64) * 2) / 1e6
    n = int(min(64, max(8, np.ceil(96.0 / mb))))
    mods = make(N, K, n)
    x17 = mx.array(rng.standard_normal((1, 17, K)).astype(np.float32)).astype(mx.bfloat16)
    x2 = x17[:, :2]
    mx.eval(x17, x2)
    m = mods[0]
    y_mlx = nn.QuantizedLinear.__call__(m, x17).astype(mx.float32)
    y_nax = DN.NaxQL.__call__(m, x17).astype(mx.float32)
    DN.D32[0] = False
    y2_ks1 = DN.NaxQL.__call__(m, x2).astype(mx.float32)
    DN.D32[0] = True
    y2_ks = DN.NaxQL.__call__(m, x2).astype(mx.float32)
    mx.eval(y_mlx, y_nax, y2_ks1, y2_ks)
    e17 = float(mx.max(mx.abs(y_nax - y_mlx)).item() / max(1e-30, mx.max(mx.abs(y_mlx)).item()))
    e2 = float(mx.max(mx.abs(y2_ks - y2_ks1)).item() / max(1e-30, mx.max(mx.abs(y2_ks1)).item()))
    chain(mods, x17, False); chain(mods, x17, True)          # warm-up
    a, b = [], []
    for r in range(7):
        for nax in ((False, True) if r % 2 == 0 else (True, False)):
            (b if nax else a).append(chain(mods, x17, nax))
    ua, ub = float(np.median(a)), float(np.median(b))
    tot += per * (ua - ub) / 1e3
    say(f"  {name:10s} N {N:5d} K {K:5d} ({mb:5.2f} MB, ks {DN._ks(N, K)}, {per} per propose): MLX 17 rows {ua:6.1f} us "
        f"({mb * 1e-3 / max(1e-9, ua * 1e-6):4.0f} GB/s), NaxQL halves {ub:6.1f} us ({mb * 1e-3 / max(1e-9, ub * 1e-6):4.0f} GB/s); "
        f"bits: 17 rows vs MLX {e17:.1e}, 2 rows ks vs ks 1 {e2:.1e}")
    del mods
    mx.clear_cache()
say(f"  per propose (the profiled call counts): {tot:.2f} ms saved; COUNTS {DN.COUNTS}")
