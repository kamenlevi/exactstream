"""PF-B INVARIANCE (Leap, Sun 2026-11-08 08:1x; BigMoE's PFX: PF-A plm1 = 195 / 368 / 441 tok/s, long prompts compute-bound).
PF-A keeps the screen's 1,024-row chunks per MoE call to stay bitwise (~20 rows per expert per call). MLX 0.32.2's sorted
gather (affine_gather_qmm_rhs_nax) runs, for every BM-row tile, one full K loop over the WHOLE tile per expert segment inside
it, so at ~20 rows per expert ~2/3 of the matrix work is recomputed rows; each row's own accumulation (NAXTile loads,
tile_matmad_nax, K order) does not depend on the tile's other rows. Claim: a layer's MoE for all its chunks in ONE sorted
call is bitwise == the per-chunk calls, and faster. Test at the model's shapes (E 512, top-10, 2560 -> 640 -> 2560, 4-bit
g64, bf16, SwitchGLU's own path: _gather_sort, up / gate / down gather_qmm sorted, SwiGLU, _scatter_unsort).
Pre-registered line: PASS = every (token, rank) row bitwise equal between 1,024-token calls and one call over all tokens
(also 4,096) at 1.6K / 4.7K / 7.9K tokens AND the one-call MoE >= 1.5x faster than the chunked calls at 7.9K. -> stdout"""
import os
import sys
import time

import numpy as np
import mlx.core as mx
from mlx_lm.models.switch_layers import SwitchGLU

E, TOPK, D, F = 512, 10, 2560, 640
rng = np.random.default_rng(11)


def build():
    """SwitchGLU with QuantizedSwitchLinear pieces whose 4-bit codes / scales / biases are drawn directly (the lazy fp32
    initialisers are replaced before any eval, so they are never computed: ~2.5 GB of weights in all)"""
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear
    m = SwitchGLU(D, F, E)
    for p, (o, i) in (("gate_proj", (F, D)), ("up_proj", (F, D)), ("down_proj", (D, F))):
        q = QuantizedSwitchLinear(i, o, E, bias=False, group_size=64, bits=4)
        q.weight = mx.array(rng.integers(0, 2 ** 32, (E, o, i // 8), dtype=np.uint32))
        q.scales = mx.random.uniform(0.002, 0.02, (E, o, i // 64)).astype(mx.bfloat16)
        q.biases = (-7.5 * q.scales.astype(mx.float32)).astype(mx.bfloat16)
        mx.eval(q.weight, q.scales, q.biases)
        setattr(m, p, q)
    return m


def routing(T):
    """realistic top-10 routing: distinct experts per token, skewed popularity (Zipf-ish) like a trained router"""
    pop = 1.0 / np.arange(1, E + 1) ** 0.6
    pop = pop[rng.permutation(E)]
    pop /= pop.sum()
    ids = np.stack([rng.choice(E, TOPK, replace=False, p=pop) for _ in range(T)]).astype(np.int32)
    return ids


def run_chunks(m, x, ids, chunk):
    ys = []
    for a in range(0, x.shape[0], chunk):
        y = m(x[a:a + chunk], mx.array(ids[a:a + chunk]))
        mx.eval(y)
        ys.append(y)
    return mx.concatenate(ys, axis=0)


def timed(fn, reps=3):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


m = build()
print(f"PF-B INVARIANCE {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}, SwitchGLU E {E} top-{TOPK} {D}->{F}->{D} 4-bit g64 bf16",
      flush=True)
ok_all, rows = True, []
for T in (1565, 4730, 7932):
    x = (mx.random.normal((T, D)) * 0.5).astype(mx.bfloat16)
    ids = routing(T)
    mx.eval(x)
    y1 = run_chunks(m, x, ids, 1024)
    mx.eval(y1)
    res = {}
    for chunk in (4096, T):
        yc = run_chunks(m, x, ids, chunk)
        mx.eval(yc)
        eq = bool(mx.array_equal(yc, y1).item())
        nd = int((yc != y1).sum().item())
        res[chunk] = (eq, nd)
        ok_all &= eq
    t1 = timed(lambda: run_chunks(m, x, ids, 1024))
    t4 = timed(lambda: run_chunks(m, x, ids, 4096))
    tT = timed(lambda: run_chunks(m, x, ids, T))
    gfl = T * TOPK * 3 * D * F * 2 / 1e9
    rows.append((T, t1, t4, tT))
    print(f"  T {T:5d}: bitwise vs 1,024-row calls: 4,096 {'==' if res[4096][0] else 'DIFF'} ({res[4096][1]} diffs), "
          f"one call {'==' if res[T][0] else 'DIFF'} ({res[T][1]} diffs); MoE time 1,024 {t1 * 1e3:.1f} ms "
          f"({gfl / t1 / 1e3:.1f} TF) / 4,096 {t4 * 1e3:.1f} ms ({gfl / t4 / 1e3:.1f} TF) / one {tT * 1e3:.1f} ms "
          f"({gfl / tT / 1e3:.1f} TF); one-call speedup x{t1 / tT:.2f}", flush=True)
T, t1, t4, tT = rows[-1]
sp = t1 / tT
v = "PASS" if ok_all and sp >= 1.5 else ("BETWEEN" if ok_all else "KILL (not bitwise)")
print(f"PF-B INVARIANCE verdict (pre-registered): {v}: bitwise {'all' if ok_all else 'NOT all'}; one-call speedup at 7.9K "
      f"x{sp:.2f} (line x1.5); per layer at 7.9K: {t1 * 1e3:.0f} -> {tT * 1e3:.0f} ms = {48 * (t1 - tT):.1f} s per prompt "
      f"over 48 layers; peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
