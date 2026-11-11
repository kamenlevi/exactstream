"""QMM26M (Leap, Sun 2026-11-08; CHAT-150 G1 exact piece): qmm26's dense native kernel with the row padding done IN the
kernel. x has M <= 16 rows (no materialized 16-row Concatenate pad); lanes whose fragment rows are >= M load constant
zeros (the same +0.0 the pad held), and only M output rows are written. The A fragments, the code MMA and every
accumulation are those of qmm16n on the padded x, so each row's bits are identical (check() proves it). Removes the
2 -> 16-row pads (~1.4 ms per 2-row chat check: 219 + 97 + 48 Concatenates). qmm16n_m(x [M, K], wn, sn, bn, N, ks, nt)
-> y [M, N] == qmm26.qmm16n(x, ...)[:M] bitwise."""
from functools import lru_cache

import mlx.core as mx

import qmm26 as Q26
from qmm16 import _HDR

_SUBS = [("const int p0 = 0, cnt = 16, e = 0;", "const int p0 = 0, cnt = M_C, e = 0;"),
         ("const int r0 = fm, r1 = fm + 8;",
          "const bool ok0 = fm < M_C, ok1 = fm + 8 < M_C;\n    const int r0 = ok0 ? fm : 0, r1 = ok1 ? fm + 8 : 0;"),
         ("const T a0 = x0[4 * st + j], a1 = x1[4 * st + j];",
          "const T a0 = ok0 ? x0[4 * st + j] : static_cast<T>(0.0f), a1 = ok1 ? x1[4 * st + j] : static_cast<T>(0.0f);")]


@lru_cache(None)
def _k():
    src = Q26._SRC.replace("#if GATHER", "#if 0")
    for a, b in _SUBS:
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    return mx.fast.metal_kernel(name="leap_qmm26_dm", input_names=["x", "wn", "sn", "bn"], output_names=["y"],
                                source=src, header=_HDR)


def qmm16n_m(x, wn, sn, bn, N, ks=1, nt=1):
    M, K = x.shape
    assert 1 <= M <= 16 and N % (32 * nt) == 0 and K % (64 * ks) == 0
    (y,) = _k()(inputs=[x, wn, sn, bn], template=[("T", x.dtype), ("K_C", K), ("N_C", N), ("KS", ks), ("NT", nt), ("P", 16),
                                                   ("M_C", M)],
                grid=(N // 32 * 32 * ks, 1, 1), threadgroup=(32 * ks * nt, 1, 1), output_shapes=[(M, N)],
                output_dtypes=[x.dtype])
    return y


def check(seeds=2):
    """qmm16n_m vs qmm16n on padded x: M 1-16, shapes in_proj / out_proj / shared, ks 1 / 2 / 6 / 8 as DN-KS uses"""
    import numpy as np
    rng = np.random.default_rng(5)
    bad = n = 0
    for K, N, kss in ((2560, 16384, (1, 2)), (6144, 2560, (1, 6)), (2560, 1280, (1, 8)), (640, 2560, (1, 5)), (2560, 512, (1, 10))):
        w = mx.array((rng.standard_normal((N, K)) * 0.02).astype(np.float32)).astype(mx.bfloat16)
        wq, s, b = mx.quantize(w, group_size=64, bits=4)
        wn, sn, bn = Q26.repack(wq, s, b)
        mx.eval(wn, sn, bn)
        for _ in range(seeds):
            X = mx.array(rng.standard_normal((16, K)).astype(np.float32)).astype(mx.bfloat16)
            for M in range(1, 17):
                for ks in kss:
                    if K % (64 * ks):
                        continue
                    a = Q26.qmm16n(X[:M], wn, sn, bn, N, ks)
                    c = qmm16n_m(X[:M], wn, sn, bn, N, ks)
                    mx.eval(a, c)
                    n += 1
                    bad += not (a.shape == c.shape and bool(mx.array_equal(a, c)))
    print(f"QMM26M check: {n - bad} of {n} cases bitwise == qmm16n on the padded x", flush=True)
    return bad


if __name__ == "__main__":
    check()
