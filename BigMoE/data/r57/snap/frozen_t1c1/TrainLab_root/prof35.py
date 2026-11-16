"""Where does a 35B training step go? (TrainLab, GPU, ~3 min). Times (median of 3 after a warm-up): teacher forward,
eval-path forward, train-path forward (no grad), full value_and_grad step, at batch 2 x 1024 and 4 x 1024; one GDN layer
(conv1d / chunked recurrence / projections, forward and backward) and one MoE layer (forward and backward)."""
import os, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train35 as T
from gdn_chunk import gated_delta_chunked


def timeit(name, f, n=3):
    f(); ts = []
    for _ in range(n):
        t = time.time(); f(); ts.append(time.time() - t)
    print(f"{name:48s} {np.median(ts) * 1000:8.1f} ms   peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
    return np.median(ts)


def main():
    T.limit_memory()
    flow = T.Flow(sys.argv[1] if len(sys.argv) > 1 else "drop_w2_C13")
    model, tok = T.setup(targets=["gdn", "attn", "shared", "router"], r=16, alpha=32, pre_flows=[flow.spec])
    stream = np.load(os.path.join(T.TL, "data/train_tokens.npy"), mmap_mode="r")
    for B in (2,):
        ids = mx.array(np.stack([np.asarray(stream[i * 5000:i * 5000 + 1024]) for i in range(B)]).astype(np.int32))
        top, tlp = T.teacher_top(model, ids)
        timeit(f"B{B} teacher_top (eval path, exact, LoRA off)", lambda: T.teacher_top(model, ids))
        timeit(f"B{B} eval-path forward ({flow.spec})", lambda: mx.eval(T.forward(model, ids, flow, False)[0]))
        timeit(f"B{B} train-path forward, no grad", lambda: mx.eval(T.forward(model, ids, flow, True)[0]))
        lf = nn.value_and_grad(model, lambda m: T.student_loss(m, ids, top, tlp, flow, 1.0)[0])
        l0, g0 = lf(model)
        from mlx.utils import tree_flatten
        bad = [k for k, v in tree_flatten(g0) if not bool(mx.isfinite(v).all().item())]
        print(f"B{B} train-path loss {l0.item():.5f} (finite {bool(mx.isfinite(l0).item())}), non-finite grads {len(bad)}", flush=True)
        timeit(f"B{B} value_and_grad (checkpointed)", lambda: mx.eval(lf(model)))
        lf2 = nn.value_and_grad(model, lambda m: T.student_loss(m, ids, top, tlp, flow, 1.0, ckpt=False)[0])
        if B == 2:
            try:
                timeit(f"B{B} value_and_grad (no checkpoint)", lambda: mx.eval(lf2(model)))
            except Exception as e:
                print("no-checkpoint failed:", str(e)[:120])
        mx.clear_cache(); mx.reset_peak_memory()
    # one GDN layer and one MoE layer, B=2 x 1024
    layer = model.language_model.model.layers[0]
    la = layer.linear_attn
    x = mx.random.normal((2, 1024, 2048)).astype(mx.bfloat16)
    timeit("GDN layer eval (kernel)", lambda: mx.eval(la(x, None, None)))
    timeit("GDN layer train path fwd", lambda: mx.eval(T.gdn_train(la, x)))
    g = mx.grad(lambda xx: T.gdn_train(la, xx).astype(mx.float32).sum())
    timeit("GDN layer train path fwd+bwd (dx)", lambda: mx.eval(g(x)))
    qkv = la.in_proj_qkv(x)
    cs = mx.zeros((2, 3, la.conv_dim), dtype=x.dtype)
    gc = mx.grad(lambda q: la.conv1d(mx.concatenate([cs, q], axis=1)).astype(mx.float32).sum())
    timeit("  conv1d fwd", lambda: mx.eval(la.conv1d(mx.concatenate([cs, qkv], axis=1))))
    timeit("  conv1d fwd+bwd", lambda: mx.eval(gc(qkv)))
    q = mx.random.normal((2, 1024, 16, 128)) * 0.01; k = mx.fast.rms_norm(mx.random.normal((2, 1024, 16, 128)), None, 1e-6) / 11.3
    v = mx.random.normal((2, 1024, 32, 128)); lg = -mx.abs(mx.random.normal((2, 1024, 32))) * 0.1; bt = mx.sigmoid(mx.random.normal((2, 1024, 32)))
    timeit("  chunked recurrence fwd", lambda: mx.eval(gated_delta_chunked(q, k, v, lg, bt)[0]))
    gr = mx.grad(lambda qq, kk, vv: gated_delta_chunked(qq, kk, vv, lg, bt)[0].sum(), argnums=(0, 1, 2))
    timeit("  chunked recurrence fwd+bwd", lambda: mx.eval(gr(q, k, v)))
    moe = layer.mlp
    xm = mx.random.normal((2, 1024, 2048)).astype(mx.bfloat16)
    fl = T.Flow("exact")
    timeit("MoE layer fwd", lambda: mx.eval(T.moe(moe, 0, xm, fl, None)[0]))
    gm = mx.grad(lambda xx: T.moe(moe, 0, xx, fl, None)[0].astype(mx.float32).sum())
    timeit("MoE layer fwd+bwd (dx)", lambda: mx.eval(gm(xm)))
    gsw = mx.grad(lambda xx: moe.switch_mlp(xx, mx.stop_gradient(mx.argpartition(moe.gate(xx), kth=-8, axis=-1)[..., -8:])).astype(mx.float32).sum())
    timeit("  switch_mlp fwd+bwd (dx)", lambda: mx.eval(gsw(xm)))


if __name__ == "__main__":
    main()
