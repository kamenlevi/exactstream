"""PF-B dense invariance (Leap, Sun 08:5x; BigMoE's R57_PFLM=3 risk): are MLX 0.32.2 matmuls per-row invariant between 1,024-row
calls and 4,096 / 7,932 / 8,192-row calls at the MoE block's dense shapes (router 2560->512, shared expert 2560->640 x2 and
640->2560, shared-expert gate 2560->1), for bf16 matmul and 4- / 8-bit quantized_matmul (g64)? Correctness only (no timing)."""
import numpy as np
import mlx.core as mx

SH = (("router", 512, 2560), ("shared gate/up", 640, 2560), ("shared down", 2560, 640), ("shared-expert gate", 1, 2560))
for T in (4096, 7932, 8192):
    x = (mx.random.normal((T, 2560)) * 0.5).astype(mx.bfloat16)
    xd = (mx.random.normal((T, 640)) * 0.5).astype(mx.bfloat16)
    mx.eval(x, xd)
    out = []
    for name, N, K in SH:
        xi = xd if K == 640 else x
        w = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
        mx.eval(w)
        forms = [("bf16", lambda a, w=w: a @ w.T)]
        for bits in (4, 8):
            if K % 64 == 0 and N >= 32:
                wq, s, b = mx.quantize(w, group_size=64, bits=bits)
                mx.eval(wq, s, b)
                forms.append((f"q{bits}", lambda a, wq=wq, s=s, b=b, bits=bits:
                              mx.quantized_matmul(a, wq, s, b, transpose=True, group_size=64, bits=bits)))
        for fn_name, fn in forms:
            yc = mx.concatenate([fn(xi[a:a + 1024]) for a in range(0, T, 1024)], 0)
            yo = fn(xi)
            mx.eval(yc, yo)
            nd = int((~mx.all(yc == yo, axis=-1)).sum().item())
            out.append(f"{name} {fn_name}: {'==' if nd == 0 else f'{nd} rows DIFF'}")
    print(f"T {T}: " + "; ".join(out), flush=True)
