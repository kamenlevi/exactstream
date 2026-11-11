"""Row 15a speed: compiled GatedResidual read (BigMoE reference, dup-row) vs the same with PackedLinear mixers,
over 16 real HC modules (layers 0-7, attn + mlp) chained so each read depends on the previous one.
Usage: time_packed_hc.py [S]   (GPU, quiet lock; ~0.25 GB of weights)"""
import sys, time
import mlx.core as mx

sys.argv = [sys.argv[0], "0"] + sys.argv[1:]
S = int(sys.argv[2]) if len(sys.argv) > 2 else 1
import test_packed_hc as T   # noqa: E402

mods = {False: [], True: []}
for L in range(8):
    T.LAYER = L
    for part in ("attn_hyper_connection", "mlp_hyper_connection"):
        for packed in (False, True):
            _, f0, _ = T.build(part, packed)
            mods[packed].append(f0)


def run(fs, h):
    for f in fs:
        mixed, _, inj = f(h)
        h = mx.tile(mixed, (1, 1, 4)) + inj.sum() * 0          # next read depends on this one
    return h


h0 = (mx.random.normal((1, S, 10240)) * 0.5).astype(mx.bfloat16)
for packed in (False, True, False, True):
    fs = mods[packed]
    for _ in range(3):
        mx.eval(run(fs, h0))
    ts = []
    for _ in range(21):
        a = time.perf_counter(); mx.eval(run(fs, h0)); ts.append(time.perf_counter() - a)
    t = sorted(ts)[10] / len(fs) * 1e6
    print(f"S={S} {'packed   ' if packed else 'reference'}: {t:6.1f} µs per HC read (16 modules, dependent chain)")
