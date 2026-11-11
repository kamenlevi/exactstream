"""M-34 SURE2B's quality-screen helper (MoonShots, Wed 2026-11-04 06:0x; NewIdeas 05:51 for Breakthrough's
qom_0_live_bq_t2 arm): records3's 3-bit MSB part served at 2 bits by the pre-registered rule (QUEUE M-34)
    w = s3 * (2 * (q3 >> 1) + 0.5) + b3,
as an MLX affine 2-bit triple: codes q2 = q3 >> 1, scale s2 = 2 * s3 (exact in bf16), bias b2 = b3 + 0.5 * s3 (one bf16
rounding). bias="mean" replaces the 0.5 by the dropped bit's exact per-group mean, b2 = b3 + s3 * mean(q3 & 1): records3's
own rule for its 3-from-4 part (dm_t3's docstring); that is a separate arm, not the registered one.
Use in dm_t3._proj (Breakthrough's harness; dm_t3.read's W per projection = {w3 uint32, s3 / b3 bf16}, leading [U]):
    if bits == 2: return to2bit.proj2(h, W, idx)
Self-test: python to2bit.py [layer expert]: reads one record through dm_t3.read, checks the 3-bit unpack against
mx.dequantize(bits=3) bit for bit, the 2-bit triple against the rule, and gather_qmm(bits=2) against a dense matmul."""
import os, sys
import mlx.core as mx

_SH3 = [(3 * j // 32, 3 * j % 32) for j in range(32)]             # (word, shift) of weight j in a 3-word group


def unpack3(w3):
    """MLX bits=3 codes [..., 3n] uint32 (32 weights per 3 words, a little-endian bitstream) -> [..., 32n] uint32 in 0..7"""
    sh = w3.shape
    g = w3.reshape(*sh[:-1], sh[-1] // 3, 3)
    words = (g[..., 0], g[..., 1], g[..., 2])
    vals = []
    for word, s in _SH3:
        v = words[word] >> s
        if s > 29:                                                  # the weight spans two words
            v = v | (words[word + 1] << (32 - s))
        vals.append(v & 7)
    return mx.stack(vals, axis=-1).reshape(*sh[:-1], sh[-1] // 3 * 32)


def pack2(q2):
    """[..., 16n] values in 0..3 -> MLX bits=2 codes [..., n] uint32 (16 weights per word, weight j at bits 2j..2j+1)"""
    sh = q2.shape
    g = q2.astype(mx.uint32).reshape(*sh[:-1], sh[-1] // 16, 16)
    return (g << mx.arange(0, 32, 2, dtype=mx.uint32)).sum(axis=-1).astype(mx.uint32)


def to2bit(W, bias="mid"):
    """dm_t3.read's projection dict {w3, s3, b3} -> {w2, s2, b2} for gather_qmm(bits=2, group_size=64)"""
    q3 = unpack3(W["w3"])
    s3, b3 = W["s3"], W["b3"]
    s2 = s3 * mx.array(2, dtype=s3.dtype)
    if bias == "mid":
        b2 = (b3.astype(mx.float32) + 0.5 * s3.astype(mx.float32)).astype(b3.dtype)
    else:
        m = (q3 & 1).astype(mx.float32).reshape(*q3.shape[:-1], q3.shape[-1] // 64, 64).mean(axis=-1)
        b2 = (b3.astype(mx.float32) + s3.astype(mx.float32) * m).astype(b3.dtype)
    return {"w2": pack2(q3 >> 1), "s2": s2, "b2": b2}


def proj2(h, W, idx, bias="mid"):
    T = to2bit(W, bias)
    return mx.gather_qmm(h, T["w2"], T["s2"], T["b2"], rhs_indices=idx, transpose=True, group_size=64, bits=2)


def _test(L=12, e=7):
    sys.path.insert(0, os.path.expanduser("~/FableGod/decision_map"))
    import numpy as np
    import dm_t3                                                    # read-only import: Breakthrough's record reader
    W = dm_t3.read(L, np.array([e]))
    ok = True
    for proj, P in W.items():
        w3, s3, b3 = P["w3"], P["s3"], P["b3"]
        q3 = unpack3(w3)
        rep = lambda a: mx.repeat(a.astype(mx.float32), 64, axis=-1)
        ref3 = mx.dequantize(w3, s3, b3, group_size=64, bits=3)
        mine3 = (q3.astype(mx.float32) * rep(s3) + rep(b3)).astype(ref3.dtype)
        exact3 = float((mine3 == ref3).astype(mx.float32).mean())
        d3 = float(mx.abs(mine3.astype(mx.float32) - ref3.astype(mx.float32)).max())
        out = []
        for bias in ("mid", "mean"):
            T = to2bit(P, bias)
            deq2 = mx.dequantize(T["w2"], T["s2"], T["b2"], group_size=64, bits=2).astype(mx.float32)
            if bias == "mid":
                rule = rep(s3) * (2 * (q3 >> 1).astype(mx.float32) + 0.5) + rep(b3)
            else:
                m = (q3 & 1).astype(mx.float32).reshape(*q3.shape[:-1], q3.shape[-1] // 64, 64).mean(axis=-1)
                rule = rep(s3) * (2 * (q3 >> 1).astype(mx.float32)) + rep(b3) + rep(s3) * mx.repeat(m, 64, axis=-1)
            rel = float(mx.abs(deq2 - rule).max() / mx.abs(rule).max())
            r32 = ref3.astype(mx.float32)
            err = float(mx.sqrt(((deq2 - r32) ** 2).mean()) / mx.sqrt((r32 ** 2).mean()))
            x = mx.random.normal((1, 1, w3.shape[-1] // 3 * 32), key=mx.random.key(1)).astype(mx.bfloat16)
            y = mx.gather_qmm(x, T["w2"], T["s2"], T["b2"], rhs_indices=mx.array([[0]]), transpose=True, group_size=64,
                              bits=2).astype(mx.float32).reshape(-1)
            yd = (x.astype(mx.float32).reshape(1, -1) @ deq2[0].T).reshape(-1)
            gq = float(mx.abs(y - yd).max() / mx.abs(yd).max())
            out.append(f"{bias}: 2-bit vs rule max rel {rel:.2e}, vs 3-bit weights rel RMS {err:.3f}, gather_qmm vs dense {gq:.2e}")
            ok &= rel < 1e-2 and gq < 2e-2
        ok &= exact3 == 1.0
        print(f"  L{L} e{e} {proj}: unpack3 vs dequantize bits=3 exact {100 * exact3:.2f} % (max diff {d3:.1e}); " + "; ".join(out))
    print("to2bit self-test:", "OK" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:3]]
    sys.exit(0 if _test(*a) else 1)
