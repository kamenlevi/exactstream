"""NAX-SEP (Leap, Sun 15 Nov 2026 18:3x; CHECK55 track A, desk-to-bench question): is the NAX MMA's accumulation separable?
S4 (the HC read; REF2-T pins MLX's split-K replica at 4,096-key partitions) chains 32-wide k-steps into one fp32 tile
per partition: Dt = mma(Dt, A_s, B_s), s = 0, 1, ... If each mma call equals Dt + (that step's product rounded to fp32 on
its own), then a partition's result equals the ordered fp32 fold of independently computed k-step terms. A BOF-style
kernel (k-step terms in parallel, folded in order) would then reproduce REF2-T's S4 exactly while fixing its occupancy
(33 simdgroups today). Attention's P.V (MLX's split-K NAX nn) would be the next candidate.
Test, with no new kernel: ck.s4_partials_ps at PS = K (one partition, the k-steps chained inside the MMA unit) vs PS = 32
(one k-step per partition, then an ordered fp32 fold), compared bitwise on the fp32 partials. Also PS = 4,096 (REF2-T's
partitions) vs the fold of its k-steps per partition. K in {128, 512, 4096, 10240}, N 324, random x / w at 3 scales.
Line: every case bitwise -> SEPARABLE (BOF-S4 worth building); any difference -> NOT separable (S4 / P.V speedups then
need the contract's new reference).
usage: [BigMoE venv] nax_sep.py   (MEM < 1 GB; share lock)"""
import os, sys, time
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1a/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1a/Leap_scripts/week98"))
from ck import s4_partials_ps

BF = mx.bfloat16
t0 = time.time()
lines = [f"NAX-SEP {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}"]


def fold(cs, start, n):
    """ordered fp32 fold of cs[start .. start + n - 1] from 0 (leapk._ACCUM's order)"""
    y = mx.zeros(cs.shape[1:], mx.float32)
    for p in range(start, start + n):
        y = y + cs[p]
    return y


total, same = 0, 0
for K in (128, 512, 4096, 10240):
    for sc in (0.02, 1.0, 30.0):
        for seed in (1, 2):
            k1, k2 = mx.random.split(mx.random.key(1000 * K + seed + int(sc * 10)))
            w = (mx.random.normal((324, K), key=k1) * sc).astype(BF)
            x = (mx.random.normal((1, 16, K), key=k2) * 0.5).astype(BF)
            steps, _ = s4_partials_ps(x, w, 32)                       # [K / 32, 16, N]: one k-step per partition
            if K <= 4096:
                chained, _ = s4_partials_ps(x, w, K)                   # [1, 16, N]: all k-steps chained
                ok = bool(mx.all(chained[0] == fold(steps, 0, K // 32)).item())
                total += 1; same += ok
                lines.append(f"  K {K:5d} scale {sc:5.2f} seed {seed}: one partition chained == ordered fold of its "
                             f"{K // 32} k-steps: {ok}")
            if K > 4096:                                               # REF2-T's partitions (4,096 keys) vs their folds
                parts, NP = s4_partials_ps(x, w, 4096)
                oks = []
                for p in range(NP):
                    n = min(4096, K - 4096 * p) // 32
                    oks.append(bool(mx.all(parts[p] == fold(steps, 4096 * p // 32, n)).item()))
                ok = all(oks)
                total += 1; same += ok
                lines.append(f"  K {K:5d} scale {sc:5.2f} seed {seed}: REF2-T's {NP} partitions == ordered folds of their "
                             f"k-steps: {oks}")
lines.append(f"NAX-SEP verdict: {same} / {total} cases bitwise -> " +
             ("SEPARABLE (BOF-S4 worth building)" if same == total else "NOT separable (S4 / P.V need the contract)"))
lines.append(f"done in {time.time() - t0:.0f} s")
out = "\n".join(lines)
print(out)
open(os.path.expanduser("~/Leap/data/week98/screens/nax_sep.txt"), "a").write(out + "\n")
