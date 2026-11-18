"""PK v2: qmv_pf / qmv_wide_impl (pk_ops.HEADER) vs MLX's own quantized_matmul / gather_qmm on every GEMV shape of
the live Q8H32 pop, bit for bit, on random packed weights and heavy-tailed inputs (MoonShots, Wed 18 Nov 2026).
usage: ~/BigMoE/.venv/bin/python -I -B pk_qbit.py [--reps 3]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.expanduser("~/MoonShots/week98/draft"))
import mlx.core as mx  # noqa: E402
import pk_ops  # noqa: E402

# name, N, K, bits, gs, fast (the census: qmv_fast x 11, qmv x 4, the head + extras 4-bit, experts gathered)
SHAPES = [("fc_emb", 2560, 2560, 8, 64, True), ("q", 12288, 2560, 8, 64, True), ("k", 512, 2560, 8, 64, True),
          ("o", 2560, 6144, 8, 64, True), ("router", 512, 2560, 8, 64, True), ("se_gate", 640, 2560, 8, 64, True),
          ("hc_down", 320, 10240, 8, 64, True), ("hc_up", 10240, 320, 8, 64, False), ("se_down", 2560, 640, 8, 64, False),
          ("head", 32768, 2560, 4, 64, True), ("extras", 4096, 2560, 4, 64, True)]
GATHER = [("exp_gate", 640, 2560, 4, 32, True), ("exp_down", 2560, 640, 4, 32, False)]

SRC = """
  const uint tg = threadgroup_position_in_grid.x, sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  const uint nsg = threads_per_threadgroup.x / 32;
  float tv = -INFINITY; int ti = 0x7FFFFFFF;
  const uint ntask = uint(NN) / uint(RR);
  for (uint u = tg * nsg + sg; u < ntask; u += threadgroups_per_grid.x * nsg) {
    const uint e = uint(eidx[0]);
    qmv_pf<bfloat16_t, GS, BITS, RR, UBB, FASTB, false>(w + size_t(e) * WSTR, s + size_t(e) * SSTR, b + size_t(e) * SSTR,
        x, y, KK, int(u) * RR, lane, tv, ti);
  }
"""
KER = mx.fast.metal_kernel(name="pk_qbit", input_names=["w", "s", "b", "x", "eidx"], output_names=["y"], source=SRC,
                           header=pk_ops.HEADER)
WSRC = """
  const uint tg = threadgroup_position_in_grid.x, sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  const uint nsg = threads_per_threadgroup.x / 32;
  for (uint u = tg * nsg + sg; u < uint(NN) / 4u; u += threadgroups_per_grid.x * nsg)
    qmv_wide_impl<bfloat16_t, 64, 8, 4, 8>(w, s, b, x, y, KK, NN, 4, uint3(0, u >> 1, 0), u & 1u, lane);
"""
WKER = mx.fast.metal_kernel(name="pk_qbit_wide", input_names=["w", "s", "b", "x"], output_names=["y"], source=WSRC,
                            header=pk_ops.HEADER)


def rand_q(N, K, bits, gs, E=1):
    w = mx.random.randint(0, 2 ** 31, (E, N, K * bits // 32), dtype=mx.uint32)
    w = (w * 2 + mx.random.randint(0, 2, (E, N, K * bits // 32), dtype=mx.uint32)).astype(mx.uint32)
    s = (mx.random.normal((E, N, K // gs)) * 0.01).astype(mx.bfloat16)
    b = (mx.random.normal((E, N, K // gs)) * 0.01).astype(mx.bfloat16)
    return w, s, b


def heavy(K):
    return (mx.random.normal((1, K)) * mx.exp(mx.random.normal((1, K)))).astype(mx.bfloat16)


def run(w, s, b, x, N, K, bits, gs, fast, R, UB, e=0):
    wstr, sstr = w.shape[1] * w.shape[2], s.shape[1] * s.shape[2]
    return KER(inputs=[w.reshape(-1), s.reshape(-1), b.reshape(-1), x.reshape(-1), mx.array([e], mx.int32)],
               template=[("NN", N), ("KK", K), ("GS", gs), ("BITS", bits), ("RR", R), ("UBB", UB), ("FASTB", fast),
                         ("WSTR", wstr), ("SSTR", sstr)],
               grid=(20 * 1024, 1, 1), threadgroup=(1024, 1, 1), output_shapes=[(N,)], output_dtypes=[mx.bfloat16])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    mx.random.seed(7)
    tot = ok = 0
    for name, N, K, bits, gs, fast in SHAPES:
        w, s, b = rand_q(N, K, bits, gs)
        for rep in range(a.reps):
            x = heavy(K)
            ref = mx.quantized_matmul(x, w[0], scales=s[0], biases=b[0], transpose=True, group_size=gs, bits=bits)[0]
            for R, UB in ((1, 1), (1, 2), (2, 2), (4, 1), (1, 4)):
                y = run(w, s, b, x, N, K, bits, gs, fast, R, UB)
                eq = bool(mx.array_equal(y, ref).item())
                tot += 1; ok += int(eq)
                if not eq:
                    nd = int((y != ref).sum().item())
                    print(f"  MISMATCH {name} R{R} UB{UB} rep {rep}: {nd} / {N} rows differ", flush=True)
        print(f"{name} [{N} x {K}] {bits}-bit g{gs} {'fast' if fast else 'qmv'}: done", flush=True)
    E = 8
    for name, N, K, bits, gs, fast in GATHER:
        w, s, b = rand_q(N, K, bits, gs, E)
        for rep in range(a.reps):
            x = heavy(K)
            e = rep % E
            ref = mx.gather_qmm(x[None], w, s, b, rhs_indices=mx.array([[e]], mx.uint32), transpose=True,
                                group_size=gs, bits=bits).reshape(-1)
            for R, UB in ((1, 1), (2, 2), (4, 1)):
                y = run(w, s, b, x, N, K, bits, gs, fast, R, UB, e)
                eq = bool(mx.array_equal(y, ref).item())
                tot += 1; ok += int(eq)
                if not eq:
                    print(f"  MISMATCH {name} R{R} UB{UB} rep {rep}: {int((y != ref).sum().item())} / {N}", flush=True)
        print(f"{name} (gather, E {E}) [{N} x {K}] {bits}-bit g{gs}: done", flush=True)
    w, s, b = rand_q(2560, 2560, 8, 64)
    for rep in range(a.reps):                                   # fc_hidden: 4 streams (qmv_wide nv 4, k_lanes 8)
        x4 = (mx.random.normal((4, 2560)) * mx.exp(mx.random.normal((4, 2560)))).astype(mx.bfloat16)
        ref = mx.quantized_matmul(x4, w[0], scales=s[0], biases=b[0], transpose=True, group_size=64, bits=8)
        y = WKER(inputs=[w.reshape(-1), s.reshape(-1), b.reshape(-1), x4.reshape(-1)], template=[("NN", 2560), ("KK", 2560)],
                 grid=(20 * 1024, 1, 1), threadgroup=(1024, 1, 1), output_shapes=[(4 * 2560,)],
                 output_dtypes=[mx.bfloat16])[0].reshape(4, 2560)
        eq = bool(mx.array_equal(y, ref).item())
        tot += 1; ok += int(eq)
        if not eq:
            print(f"  MISMATCH fc_hidden x4 rep {rep}: {int((y != ref).sum().item())} / 10240", flush=True)
    print(f"fc_hidden x4 (qmv_wide nv4 kl8): done", flush=True)
    print(f"[pk_qbit] bitwise {ok} / {tot}", flush=True)


if __name__ == "__main__":
    main()
