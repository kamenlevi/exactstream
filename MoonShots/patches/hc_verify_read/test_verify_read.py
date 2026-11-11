"""Can BigMoE's HC reads use MTPLX's hc_verify_read (3 dispatches instead of ~11) and stay bit-identical to its
compiled dup-row reference? MTPLX claims bit-for-bit equality with the compiled stock chain at 2-8 rows; under dup-row
S=1 runs at 2 rows ([x; x], keep row 0). Checks S = 1, 3, 6 (read and write+read) on real layer weights, then times
16 real HC modules (layers 0-7) as a dependent chain: reference vs verify-read.
Usage: test_verify_read.py   (small GPU job)"""
import os, sys, time
import mlx.core as mx

sys.path.insert(0, os.path.expanduser("~/MoonShots/patches/hc_exp_pack"))
sys.argv = [sys.argv[0], "0"]
import test_packed_hc as T                               # builds the compiled dup-row reference (duprow() applied)
from mtplx.kernels import hc_verify_read as V            # noqa: E402

SIG = V.sigmoid_table(mx.bfloat16)


def vread(hc, h, pending=None):
    """GatedResidual read via hc_verify_read.read_rows; h (1, S, hc*width) -> same outputs as the module."""
    S = h.shape[1]
    x = h.reshape(S, -1)
    bo = inj = None
    if pending is not None:
        bo, inj = (p.reshape(S, -1) for p in pending)
    if S == 1:                                           # dup-row: the reference runs these rows at width 2
        x = mx.concatenate([x, x]); bo = None if bo is None else mx.concatenate([bo, bo])
        inj = None if inj is None else mx.concatenate([inj, inj])
    mixed, written, inject = V.read_rows(x, hc.hc_norm.weight, hc.input_mix_weight_down.weight,
                                         hc.input_mix_weight_up.weight, hc.block_inject_weight.weight, bo, inj,
                                         hc=hc.hc_count, eps=T.args.rms_norm_eps, sigmoid=SIG)
    r = lambda a: a[:S].reshape(1, S, -1)
    return r(mixed), r(written), r(inject)


ok = True
for part in ("attn_hyper_connection", "mlp_hyper_connection"):
    hc, r0, r1 = T.build(part, False)
    for S in (1, 3, 6):
        h = (mx.random.normal((1, S, 4 * T.args.hidden_size)) * 0.5).astype(mx.bfloat16)
        bo = (mx.random.normal((1, S, T.args.hidden_size)) * 0.5).astype(mx.bfloat16)
        inj = (mx.random.uniform(shape=(1, S, 4)) * 2).astype(mx.bfloat16)
        for tag, a, b in (("read", r0(h), vread(hc, h)), ("write+read", r1(h, bo, inj), vread(hc, h, (bo, inj)))):
            eq = all(mx.array_equal(x, y).item() for x, y in zip(a, b))
            ok &= eq
            print(f"{part} S={S} {tag:10s}: hc_verify_read == compiled dup-row reference, bitwise: {eq}", flush=True)
print("ALL BIT-IDENTICAL" if ok else "MISMATCH")

# timing over 16 real modules
mods = []
for L in range(8):
    T.LAYER = L
    for part in ("attn_hyper_connection", "mlp_hyper_connection"):
        hc, f0, _ = T.build(part, False)
        mods.append((hc, f0))
for S in (1, 3):
    h0 = (mx.random.normal((1, S, 4 * T.args.hidden_size)) * 0.5).astype(mx.bfloat16)
    for name in ("reference", "verify_read", "reference", "verify_read"):
        def run():
            h = h0
            for hc, f0 in mods:
                mixed, _, inj = f0(h) if name == "reference" else vread(hc, h)
                h = mx.tile(mixed, (1, 1, 4)) + inj.sum() * 0
            mx.eval(h)
        for _ in range(3):
            run()
        ts = []
        for _ in range(21):
            a = time.perf_counter(); run(); ts.append(time.perf_counter() - a)
        print(f"S={S} {name:11s}: {sorted(ts)[10] / len(mods) * 1e6:6.1f} µs per HC read", flush=True)
