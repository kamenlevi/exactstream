"""Reference-v2 candidate HCReadV2 vs BigMoE's current reference (compiled GatedResidual, dup-row), on real weights:
closeness (it is a new reference, so not bitwise), determinism, batch invariance, and speed over 16 real HC modules.
Usage: test_v2.py   (small GPU job)"""
import os, sys, time
import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/MoonShots/patches/hc_exp_pack"))
sys.path.insert(0, os.path.expanduser("~/MoonShots/patches/hc_verify_read"))
sys.argv = [sys.argv[0], "0"]
import test_packed_hc as T                     # noqa: E402  compiled dup-row reference builder
from hc_read_v2 import HCReadV2                # noqa: E402
from mtplx.kernels import hc_verify_read as V  # noqa: E402

SIG = V.sigmoid_table(mx.bfloat16)


def ulps(a, b):
    a = np.array(a.astype(mx.float32)).ravel(); b = np.array(b.astype(mx.float32)).ravel()
    ai = a.view(np.int32).astype(np.int64) >> 16; bi = b.view(np.int32).astype(np.int64) >> 16   # bf16 bit patterns
    ai = np.where(ai < 0, -(ai & 0x7FFF), ai); bi = np.where(bi < 0, -(bi & 0x7FFF), bi)
    d = np.abs(ai - bi)
    return float((d == 0).mean()), int(d.max())


for part in ("attn_hyper_connection", "mlp_hyper_connection"):
    hc, r0, r1 = T.build(part, False)
    v2 = HCReadV2(hc, T.args.rms_norm_eps)
    for S in (1, 3):
        h = (mx.random.normal((1, S, 4 * T.args.hidden_size)) * 0.5).astype(mx.bfloat16)
        bo = (mx.random.normal((1, S, T.args.hidden_size)) * 0.5).astype(mx.bfloat16)
        inj = (mx.random.uniform(shape=(1, S, 4)) * 2).astype(mx.bfloat16)
        for tag, a, b, b2 in (("read", r0(h), v2(h), v2(h)), ("write+read", r1(h, bo, inj), v2(h, (bo, inj)), v2(h, (bo, inj)))):
            det = all(mx.array_equal(x, y).item() for x, y in zip(b, b2))
            stats = [ulps(x, y) for x, y in zip(a, b)]
            print(f"{part} S={S} {tag:10s}: equal-to-reference {', '.join(f'{f:.1%} (max {m} ulp)' for f, m in stats)}"
                  f" [mixed, written, inject]; deterministic {det}", flush=True)
    h3 = (mx.random.normal((1, 3, 4 * T.args.hidden_size)) * 0.5).astype(mx.bfloat16)
    a3 = v2(h3); a1 = v2(h3[:, :1])
    print(f"{part}: batch-invariant (row 0 at S=3 == S=1): {all(mx.array_equal(x[:, :1], y).item() for x, y in zip(a3, a1))}")

mods = []
for L in range(8):
    T.LAYER = L
    for part in ("attn_hyper_connection", "mlp_hyper_connection"):
        hc, f0, _ = T.build(part, False)
        mods.append((hc, f0, HCReadV2(hc, T.args.rms_norm_eps)))


def vread(hc, h):
    S = h.shape[1]; x = h.reshape(S, -1)
    if S == 1:
        x = mx.concatenate([x, x])
    m, w, i = V.read_rows(x, hc.hc_norm.weight, hc.input_mix_weight_down.weight, hc.input_mix_weight_up.weight,
                          hc.block_inject_weight.weight, None, None, hc=4, eps=T.args.rms_norm_eps, sigmoid=SIG)
    return m[:S].reshape(1, S, -1), w[:S].reshape(1, S, -1), i[:S].reshape(1, S, -1)


for S in (1, 3):
    h0 = (mx.random.normal((1, S, 4 * T.args.hidden_size)) * 0.5).astype(mx.bfloat16)
    for name in ("reference", "verify_read", "v2 packed", "reference", "verify_read", "v2 packed"):
        def run():
            h = h0
            for hc, f0, v2 in mods:
                mixed, _, inj = f0(h) if name == "reference" else (vread(hc, h) if name == "verify_read" else v2(h))
                h = mx.tile(mixed, (1, 1, 4)) + inj.sum() * 0
            mx.eval(h)
        for _ in range(3):
            run()
        ts = []
        for _ in range(21):
            a = time.perf_counter(); run(); ts.append(time.perf_counter() - a)
        print(f"S={S} {name:11s}: {sorted(ts)[10] / len(mods) * 1e6:6.1f} µs per HC read", flush=True)
