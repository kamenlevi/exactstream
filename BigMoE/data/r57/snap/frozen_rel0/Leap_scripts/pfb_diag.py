"""PF-B DIAG (Leap, Sun 08:1x): pfb_inv.py found whole (token, rank) rows differing between 1,024-token SwitchGLU calls and
bigger calls at 4.7K / 7.9K (bitwise at 1.6K). Which op, deterministic or not, and which rows?
(a) the one-call twice and the 1,024-token calls twice (determinism); (b) the up_proj sorted gather_qmm alone on the same
sorted rows, chunked by 1,024 tokens vs one call; (c) differing rows by their expert's segment length in the one call."""
import time

import numpy as np
import mlx.core as mx
from mlx_lm.models.switch_layers import SwitchGLU, QuantizedSwitchLinear, _gather_sort

E, TOPK, D, F = 512, 10, 2560, 640
rng = np.random.default_rng(11)


def build():
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
    pop = 1.0 / np.arange(1, E + 1) ** 0.6
    pop = pop[rng.permutation(E)]
    pop /= pop.sum()
    return np.stack([rng.choice(E, TOPK, replace=False, p=pop) for _ in range(T)]).astype(np.int32)


def chunks(m, x, ids, c):
    ys = []
    for a in range(0, x.shape[0], c):
        y = m(x[a:a + c], mx.array(ids[a:a + c]))
        mx.eval(y)
        ys.append(y)
    return mx.concatenate(ys, axis=0)


m = build()
T = 7932
x = (mx.random.normal((T, D)) * 0.5).astype(mx.bfloat16)
ids = routing(T)
mx.eval(x)
print(f"PF-B DIAG {time.strftime('%H:%M:%S')}: T {T}", flush=True)
y1a, y1b = chunks(m, x, ids, 1024), chunks(m, x, ids, 1024)
yoa, yob = chunks(m, x, ids, T), chunks(m, x, ids, T)
mx.eval(y1a, y1b, yoa, yob)
eqr = lambda a, b: int(mx.all(a == b, axis=-1).sum().item())   # rows equal (a row = one (token, rank) output of D)
R = T * TOPK
print(f"  (a) 1,024-token calls twice: {eqr(y1a, y1b)} / {R} rows equal; one call twice: {eqr(yoa, yob)} / {R}; "
      f"one vs 1,024: {eqr(yoa, y1a)} / {R}", flush=True)
# (b) up_proj alone on identical sorted rows: chunk the SORTED one-call rows at 1,024-token boundaries of the original order
up = m.up_proj
xe = mx.expand_dims(x, (-2, -3))
outs_c = []
for a in range(0, T, 1024):
    xs, idx, inv = _gather_sort(xe[a:a + 1024], mx.array(ids[a:a + 1024]))
    outs_c.append(up(xs, idx, sorted_indices=True)[inv].reshape(-1, F))
yc = mx.concatenate(outs_c, axis=0)
xs, idx, inv = _gather_sort(xe, mx.array(ids))
yo = up(xs, idx, sorted_indices=True)[inv].reshape(-1, F)
mx.eval(yc, yo)
rows_eq = np.array(mx.all(yc == yo, axis=-1))
print(f"  (b) up_proj gather_qmm alone: {int(rows_eq.sum())} / {R} rows equal (chunked vs one call)", flush=True)
# (c) which rows differ: by the expert's segment length in the one call and in the chunk
e_flat = ids.reshape(-1)
seg_one = np.bincount(e_flat, minlength=E)
bad = ~rows_eq
for lo, hi in ((0, 64), (64, 128), (128, 192), (192, 256), (256, 10 ** 9)):
    sel = (seg_one[e_flat] >= lo) & (seg_one[e_flat] < hi)
    if sel.sum():
        print(f"      experts with {lo}-{hi if hi < 10 ** 9 else 'max'} rows in the one call: {int(sel.sum())} rows, "
              f"{int(bad[sel].sum())} differ", flush=True)
ex_bad = np.unique(e_flat[bad])
ex_all_bad = [e for e in ex_bad if bad[e_flat == e].all()]
print(f"      experts with any differing row: {len(ex_bad)} of {E}; with ALL rows differing: {len(ex_all_bad)}; "
      f"segment lengths of all-differing experts (min / max): "
      f"{seg_one[ex_all_bad].min() if ex_all_bad else '-'} / {seg_one[ex_all_bad].max() if ex_all_bad else '-'}; "
      f"of the rest (min / max): {seg_one[np.setdiff1d(np.arange(E), ex_bad)].min()} / "
      f"{seg_one[np.setdiff1d(np.arange(E), ex_bad)].max()}", flush=True)
