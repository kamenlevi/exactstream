"""R-73 exact glue kernels (Leap, Sun 2026-11-01; exact-safe candidates): small fused kernels that replay MLX 0.32's eager
elementwise ops bit for bit (each op's result rounded to its dtype, the same functors and math overloads).
- moe_tail(y, g, s) = y + sigmoid(g) * s   (EvalSuite's / BigMoE's MoE block tail: the shared expert scaled by its
  sigmoid gate, added to the routed experts' combined rows; eager: Sigmoid, Multiply, Add = 3 kernels).
  MLX's Sigmoid functor: y = 1 / (1 + exp(abs(x))); x < 0 ? y : 1 - y, in bfloat16_t arithmetic (bf16_math.h:
  exp(bfloat) = bfloat(__metal_exp(float(x), __METAL_MAYBE_FAST_MATH__))). EXP selects the exp spelling (the probe
  picks the one that matches mx.sigmoid on all 65,536 bf16 values).
moe_tail(y [.., D], g [.., 1], s [.., D]) -> [.., D] (bf16)."""
from functools import lru_cache
import mlx.core as mx

_TAIL = r"""
    const uint i = thread_position_in_grid.x;
    if (i >= n_el[0]) return;
    const uint row = i / D_C;
    const T x = g[row];
    const T ax = metal::abs(x);
    const T e = EXPF(ax);
    const T yy = T(1) / (T(1) + e);
    const T sg = (x < T(0)) ? yy : T(1) - yy;
    const T m = sg * s[i];
    out[i] = y[i] + m;
"""

_EXPS = {"maybe": "metal::exp", "precise": "metal::precise::exp", "fast": "metal::fast::exp"}


@lru_cache(None)
def _tail_k(exp="maybe"):
    return mx.fast.metal_kernel(name=f"moe_tail_{exp}", input_names=["y", "g", "s", "n_el"], output_names=["out"],
                                source=_TAIL.replace("EXPF", _EXPS[exp]))


def moe_tail(y, g, s, exp="maybe"):
    D = y.shape[-1]
    assert g.shape[-1] == 1 and s.shape == y.shape and y.dtype == g.dtype == s.dtype == mx.bfloat16
    n = y.size
    (o,) = _tail_k(exp)(inputs=[y, g, s, mx.array([n], mx.uint32)], template=[("T", y.dtype), ("D_C", D)],
                        grid=(n, 1, 1), threadgroup=(256, 1, 1), output_shapes=[y.shape], output_dtypes=[y.dtype])
    return o


def moe_tail_ref(y, g, s):
    return y + mx.sigmoid(g) * s


def shared_cat(blk, x):
    """R-74 (exact-safe): the MoE input's three small 4-bit projections (shared expert gate_proj N 640, up_proj N 640,
    shared_expert_gate N 1) as ONE MLX call on the row-concatenated packs (N 1281: not a multiple of 64, so REF2 keeps
    MLX's 16-row call; every part bitwise == the separate calls, R-74) -> (g, s) with s = the shared expert's output
    (Qwen3NextMLP: down_proj(swiglu(gate, up))). The packs are concatenated once per block (~1.8 MB per layer)."""
    import mlx.nn as nn
    from mlx_lm.models.activations import swiglu
    se = blk.shared_expert
    k = blk.__dict__.get("_k74")
    if k is None:
        ps = [se.gate_proj, se.up_proj, blk.shared_expert_gate]
        assert all(isinstance(p, nn.QuantizedLinear) and p.bits == 4 and p.group_size == 64 and "bias" not in p
                   for p in ps), [type(p) for p in ps]
        n = [int(p.weight.shape[0]) for p in ps]
        assert sum(n) % 64 != 0, n
        cat = tuple(mx.concatenate([p[key] for p in ps]) for key in ("weight", "scales", "biases"))
        mx.eval(cat)
        k = (cat, n)
        object.__setattr__(blk, "_k74", k)
    cat, n = k
    try:                                                 # SMALL-N: mark the resident concat for ref2 (inert unless
        import ref2 as _r2                               # CFG small_native: ref2 then runs it on qmm16n, FAST)
        prev = _r2._DN_SRC[0]
        _r2._DN_SRC[0] = cat[0]
    except Exception:
        _r2 = None
    try:
        y = mx.quantized_matmul(x, *cat, transpose=True, group_size=64, bits=4)
    finally:
        if _r2 is not None:
            _r2._DN_SRC[0] = prev
    a, b = n[0], n[0] + n[1]
    return y[..., b:b + n[2]], se.down_proj(swiglu(y[..., :a], y[..., a:b]))
