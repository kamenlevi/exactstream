"""Skinny-M quantized matmul on Metal: read each weight once for M = 1..16 rows.
Compares MLX's quantized_matmul with a custom kernel that dequantizes once and
multiplies against all M rows. Usage: skinny.py [check|bench] [bits] [IN] [OUT] [NMAT]"""
import sys, time
import mlx.core as mx

mode = sys.argv[1] if len(sys.argv) > 1 else "check"
BITS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
IN = int(sys.argv[3]) if len(sys.argv) > 3 else 5120
OUT = int(sys.argv[4]) if len(sys.argv) > 4 else 17408
NMAT = int(sys.argv[5]) if len(sys.argv) > 5 else 24
G = 64
ROWS_PER_SG, SGS = 4, 8          # 4 output rows per simdgroup, 8 simdgroups -> 32 rows per threadgroup

SRC = """
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32, lane = tid % 32;
    uint row0 = threadgroup_position_in_grid.x * (R * SG) + sg * R;
    const uint KW = IN * BITS / 32;
    const uint NG = IN / 64;
    float acc[R][M];
    for (int r = 0; r < R; r++) for (int m = 0; m < M; m++) acc[r][m] = 0.0f;
    for (uint k = lane * 8; k < IN; k += 256) {
        uint g = k / 64;
        bool first = (k % 64) == 0;
        float4 xa[M], xb[M];
        for (int m = 0; m < M; m++) {
            const device float4* xp = (const device float4*)(x + m * IN + k);
            xa[m] = xp[0]; xb[m] = xp[1];
        }
        for (int r = 0; r < R; r++) {
            uint row = row0 + r;
            float4 qa, qb;
            if (BITS == 8) {
                const device uint* wp = w + row * KW + k / 4;
                uint w0 = wp[0], w1 = wp[1];
                qa = float4(w0 & 0xff, (w0 >> 8) & 0xff, (w0 >> 16) & 0xff, w0 >> 24);
                qb = float4(w1 & 0xff, (w1 >> 8) & 0xff, (w1 >> 16) & 0xff, w1 >> 24);
            } else {
                uint w0 = w[row * KW + k / 8];
                qa = float4(w0 & 0xf, (w0 >> 4) & 0xf, (w0 >> 8) & 0xf, (w0 >> 12) & 0xf);
                qb = float4((w0 >> 16) & 0xf, (w0 >> 20) & 0xf, (w0 >> 24) & 0xf, w0 >> 28);
            }
            float sc = float(s[row * NG + g]);
            float bi = float(b[row * NG + g]);
            for (int m = 0; m < M; m++) {
                float d = dot(qa, xa[m]) + dot(qb, xb[m]);
                acc[r][m] += sc * d + (first ? bi * xs[m * NG + g] : 0.0f);
            }
        }
    }
    for (int r = 0; r < R; r++) for (int m = 0; m < M; m++) {
        float v = simd_sum(acc[r][m]);
        if (lane == 0) y[m * OUT + row0 + r] = v;
    }
"""
kern = mx.fast.metal_kernel(name="skinny_qmv", input_names=["w", "s", "b", "x", "xs"],
                            output_names=["y"], source=SRC)


def skinny(w, s, b, x):
    M = x.shape[0]
    xf = x.astype(mx.float32)
    xs = xf.reshape(M, IN // G, G).sum(-1)
    return kern(inputs=[w, s, b, xf, xs],
                template=[("M", M), ("IN", IN), ("OUT", OUT), ("BITS", BITS), ("R", ROWS_PER_SG), ("SG", SGS)],
                grid=(OUT // (ROWS_PER_SG * SGS) * 256, 1, 1), threadgroup=(256, 1, 1),
                output_shapes=[(M, OUT)], output_dtypes=[mx.float32])[0]


mats = []
for i in range(NMAT if mode == "bench" else 1):
    W = (mx.random.normal((OUT, IN)) * 0.02).astype(mx.bfloat16)
    wq, sc, bi = mx.quantize(W, group_size=G, bits=BITS)
    mx.eval(wq, sc, bi)
    mats.append((wq, sc, bi))


def mlx_mm(p, x):
    return mx.quantized_matmul(x, *p, transpose=True, group_size=G, bits=BITS)


if mode == "check":
    for M in (1, 2, 3, 5, 8, 16):
        x = mx.random.normal((M, IN)).astype(mx.bfloat16)
        ref = mlx_mm(mats[0], x).astype(mx.float32)
        got = skinny(*mats[0], x)
        err = (mx.abs(ref - got).max() / mx.abs(ref).max()).item()
        print(f"M={M} max rel err {err:.2e}")
    sys.exit()

Ms = (1, 2, 3, 4, 5, 8, 12, 16)


def bench(fn, M, reps=15):
    x = mx.random.normal((M, IN)).astype(mx.bfloat16)

    def run():
        mx.eval([fn(p, x) for p in mats])
    for _ in range(3):
        run()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); run(); ts.append(time.perf_counter() - t)
    return sorted(ts)[reps // 2]


gb = sum(p[0].nbytes + p[1].nbytes + p[2].nbytes for p in mats) / 1e9
res = {M: (bench(mlx_mm, M), bench(lambda p, x: skinny(*p, x), M)) for M in Ms}
a1, c1 = res[1]
print(f"{BITS}-bit {IN}->{OUT} x{NMAT} ({gb:.2f} GB/pass)  MLX M1 {a1*1e3:.1f} ms = {gb/a1:.0f} GB/s, "
      f"skinny M1 {c1*1e3:.1f} ms = {gb/c1:.0f} GB/s")
print("M    " + " ".join(f"{M:>6}" for M in Ms))
print("MLX  " + " ".join(f"{res[M][0]/a1:5.2f}x" for M in Ms))
print("skin " + " ".join(f"{res[M][1]/a1:5.2f}x" for M in Ms) + "   (both relative to MLX M=1)")
