"""moe32_test (Leap, Sun 15 Nov 2026 17:5x; D8's fix): the resident expert path at 32 rows, one call vs two 16-row calls.
Real expert shapes (512 experts; fused gate | up 1,280 x 2,560 and down 2,560 x 640, 4-bit g64; random weights), top-10
distinct experts per row, and expert 7 forced into 20 of the 32 rows (more than one 16-row tile).
  old: qmmg25.experts_nax at 32 rows vs two 16-row experts_nax calls -> the pairs past 16 per expert are lost (D8);
  new: moe32.experts_nax_cap / experts_split_cap at 32 rows vs the same -> bitwise every (row, expert) pair;
  <= 16 rows: experts_nax_cap == experts_nax (the same segmentation when no expert has more than 16 pairs).
usage: [BigMoE venv] moe32_test.py   (MEM ~1 GB; share lock)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98n/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98n/Leap_scripts/week98"))
from qmmg25 import experts_nax
from moe32 import experts_nax_cap, experts_split_cap

BF = mx.bfloat16
E, K, NI, D, TOPK = 512, 2560, 640, 2560, 10
t0 = time.time()


def q4(shape_nk, key):
    k1, k2, k3, k4 = mx.random.split(key, 4)
    lead, (N, Kk) = shape_nk[:-2], shape_nk[-2:]
    half = lambda k: mx.random.randint(0, 65536, lead + (N, Kk // 8), dtype=mx.int32, key=k).astype(mx.uint32)
    w = (half(k1) << 16) | half(k4)
    s = (mx.abs(mx.random.normal(lead + (N, Kk // 64), key=k2)) * 0.002 + 0.006).astype(BF)
    b = (mx.random.normal(lead + (N, Kk // 64), key=k3) * 0.01 - 0.06).astype(BF)
    return w, s, b


ks = mx.random.split(mx.random.key(5), 4)
gu = q4((E, 2 * NI, K), ks[0])
dn = q4((E, D, NI), ks[1])
gate = tuple(a[:, :NI] for a in gu)
up = tuple(a[:, NI:] for a in gu)
mx.eval(gu, dn, gate, up)
rng = np.random.default_rng(3)
ids = np.stack([rng.permutation(E)[:TOPK] for _ in range(32)]).astype(np.int32)
for r in range(20):                                    # expert 7 in 20 of the 32 rows, still distinct per row
    if 7 not in ids[r]:
        ids[r, 0] = 7
ids = mx.array(ids)
x = (mx.random.normal((32, K), key=ks[2]) * 0.5).astype(BF)
cnt7 = int(mx.sum(ids == 7).item())
lines = [f"moe32_test {time.strftime('%a %H:%M:%S')}: expert 7 in {cnt7} of 32 rows"]
for cfg, cfgb in (((1, 1), (1, 1)), ((2, 1), (2, 1)), ((4, 1), (1, 1)), ((4, 1), (5, 1))):   # down K 640: ks | 10
    ref = mx.concatenate([experts_nax(x[:16], gu, dn, ids[:16], cfg, cfgb), experts_nax(x[16:], gu, dn, ids[16:], cfg, cfgb)])
    old = experts_nax(x, gu, dn, ids, cfg, cfgb)
    new = experts_nax_cap(x, gu, dn, ids, cfg, cfgb)
    spl = experts_split_cap(x, gate, up, dn, ids, cfg, cfgb)
    pair_eq = lambda a: int(mx.sum(mx.all(a == ref, axis=-1)).item())
    small = bool(mx.all(experts_nax_cap(x[:16], gu, dn, ids[:16], cfg, cfgb) == ref[:16]).item())
    lines.append(f"  cfg {cfg} / {cfgb}: pairs bitwise vs 2 x 16 (of {32 * TOPK}): old {pair_eq(old)}, cap {pair_eq(new)}, "
                 f"split_cap {pair_eq(spl)}; 16 rows cap == experts_nax: {small}")
ok = all(f"cap {32 * TOPK}," in l and f"split_cap {32 * TOPK};" in l and l.endswith("True") for l in lines[1:])
lines.append(f"moe32 verdict: {'FIX BITWISE (every pair == the 16-row calls)' if ok else 'NOT BITWISE'} ({time.time() - t0:.0f} s)")
out = "\n".join(lines)
print(out)
open(os.path.expanduser("~/Leap/data/week98/screens/moe32_test.txt"), "a").write(out + "\n")
