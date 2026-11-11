"""HCReadV3 (2 dispatches, norm fused) vs reference / hc_verify_read / HCReadV2: closeness, determinism,
batch invariance, and speed over 16 real HC modules at S = 1 and 3. Usage: test_v3.py   (small GPU job)"""
import os, sys, time
import numpy as np
import mlx.core as mx
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.expanduser("~/MoonShots/patches/hc_exp_pack"))
sys.argv = [sys.argv[0], "0"]
import test_packed_hc as T                     # noqa: E402
from hc_read_v2 import HCReadV2                # noqa: E402
from hc_read_v3 import HCReadV3                # noqa: E402
from mtplx.kernels import hc_verify_read as V  # noqa: E402
SIG = V.sigmoid_table(mx.bfloat16)


def rel(a, b):
    a = np.array(a.astype(mx.float32)); b = np.array(b.astype(mx.float32))
    return float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-9)), float((a == b).mean())


for part in ("attn_hyper_connection", "mlp_hyper_connection"):
    hc, r0, r1 = T.build(part, False)
    v3 = HCReadV3(hc, T.args.rms_norm_eps)
    for S in (1, 3):
        h = (mx.random.normal((1, S, 10240)) * 0.5).astype(mx.bfloat16)
        bo = (mx.random.normal((1, S, 2560)) * 0.5).astype(mx.bfloat16)
        inj = (mx.random.uniform(shape=(1, S, 4)) * 2).astype(mx.bfloat16)
        for tag, a, b, b2 in (("read", r0(h), v3(h), v3(h)), ("write+read", r1(h, bo, inj), v3(h, (bo, inj)), v3(h, (bo, inj)))):
            det = all(mx.array_equal(x, y).item() for x, y in zip(b, b2))
            st = [rel(x, y) for x, y in zip(a, b)]
            print(f"{part} S={S} {tag:10s}: rel err / equal-share vs reference "
                  f"{', '.join(f'{r:.1e}/{e:.0%}' for r, e in st)} [mixed, written, inject]; deterministic {det}", flush=True)
    h3 = (mx.random.normal((1, 3, 10240)) * 0.5).astype(mx.bfloat16)
    print(f"{part}: batch-invariant: {all(mx.array_equal(x[:, :1], y).item() for x, y in zip(v3(h3), v3(h3[:, :1])))}")

mods = []
for L in range(8):
    T.LAYER = L
    for part in ("attn_hyper_connection", "mlp_hyper_connection"):
        hc, f0, _ = T.build(part, False)
        mods.append((hc, f0, HCReadV2(hc, T.args.rms_norm_eps), HCReadV3(hc, T.args.rms_norm_eps)))


def vread(hc, h):
    S = h.shape[1]; x = h.reshape(S, -1)
    if S == 1:
        x = mx.concatenate([x, x])
    m, w, i = V.read_rows(x, hc.hc_norm.weight, hc.input_mix_weight_down.weight, hc.input_mix_weight_up.weight,
                          hc.block_inject_weight.weight, None, None, hc=4, eps=T.args.rms_norm_eps, sigmoid=SIG)
    return m[:S].reshape(1, S, -1), w[:S].reshape(1, S, -1), i[:S].reshape(1, S, -1)


for S in (1, 3):
    h0 = (mx.random.normal((1, S, 10240)) * 0.5).astype(mx.bfloat16)
    for name in ("reference", "verify_read", "v2", "v3") * 2:
        def run():
            h = h0
            for hc, f0, v2, v3 in mods:
                mixed, _, inj = {"reference": lambda: f0(h), "verify_read": lambda: vread(hc, h),
                                 "v2": lambda: v2(h), "v3": lambda: v3(h)}[name]()
                h = mx.tile(mixed, (1, 1, 4)) + inj.sum() * 0
            mx.eval(h)
        for _ in range(3):
            run()
        ts = []
        for _ in range(21):
            a = time.perf_counter(); run(); ts.append(time.perf_counter() - a)
        print(f"S={S} {name:11s}: {sorted(ts)[10] / len(mods) * 1e6:6.1f} µs per HC read", flush=True)
