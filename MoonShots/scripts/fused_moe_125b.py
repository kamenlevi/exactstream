"""MTPLX's fused routed-expert kernels (moe_glu_decode / moe_glu_verify: 2 dispatches per layer, batch-invariant) vs MLX's
gather_qmm chain, on the 125B MoE shape (512 experts, top-10, 4-bit g64, gate/up 640x2560, down 2560x640), dependent chain over
4 layers' distinct weights. Usage: fused_moe_125b.py   (GPU, quiet lock; ~6 GB of weights)"""
import importlib.util, os, time
import numpy as np
import mlx.core as mx

spec = importlib.util.spec_from_file_location("mgd", os.path.expanduser(
    "~/.cache/uv/archive-v0/dfadjselx9BIIozk/mtplx/kernels/moe_glu_decode.py"))
mgd = importlib.util.module_from_spec(spec); spec.loader.exec_module(mgd)

NE, K, D, F, G, NLAY = 512, 10, 2560, 640, 64, 4
rng = np.random.default_rng(0)
layers = []
for _ in range(NLAY):
    gate = (mx.random.normal((NE, F, D)) * 0.02).astype(mx.bfloat16)
    up = (mx.random.normal((NE, F, D)) * 0.02).astype(mx.bfloat16)
    down = (mx.random.normal((NE, D, F)) * 0.02).astype(mx.bfloat16)
    gq, uq, dq = (mx.quantize(w, group_size=G, bits=4) for w in (gate, up, down))
    gu = mx.quantize(mx.concatenate([gate, up], axis=1), group_size=G, bits=4)     # fused [E, 2F, D/8] pack
    mx.eval(gq, uq, dq, gu)
    del gate, up, down
    layers.append(dict(gq=gq, uq=uq, dq=dq, gu=gu))


def routing(S):
    base = rng.choice(NE, K, replace=False); rows = [base]
    for _ in range(S - 1):
        keep = rng.choice(base, int(0.45 * K), replace=False)
        rows.append(np.concatenate([keep, rng.choice(np.setdiff1d(np.arange(NE), keep), K - len(keep), replace=False)]))
        base = rows[-1]
    return np.stack(rows)


def mlx_moe(x, idx, wts, L):
    flat = idx.reshape(-1); order = mx.argsort(flat); xs = x[order // K][:, None, :]; ids = flat[order]
    kw = dict(transpose=True, group_size=G, bits=4, sorted_indices=True)
    g = mx.gather_qmm(xs, *L["gq"], rhs_indices=ids, **kw)
    u = mx.gather_qmm(xs, *L["uq"], rhs_indices=ids, **kw)
    h = mx.gather_qmm((g * mx.sigmoid(g)) * u, *L["dq"], rhs_indices=ids, **kw)
    inv = mx.argsort(order)
    h = h[inv].reshape(x.shape[0], K, D)
    return (h * wts[..., None]).sum(1).astype(mx.bfloat16)


def fused_moe(x, idx, wts, L):
    S = x.shape[0]
    ex = idx.reshape(-1).astype(mx.uint32); rw = wts.reshape(-1).astype(mx.float32)
    if S == 1:
        return mgd.moe_glu_decode(x[0], *L["gu"], *L["dq"], ex, rw, gu_group_size=G, dn_group_size=G)[None]
    return mgd.moe_glu_verify(x, *L["gu"], *L["dq"], ex, rw, gu_group_size=G, dn_group_size=G)


for S in (1, 2, 3, 4):
    idxs = [mx.array(routing(S).astype(np.int32)) for _ in range(NLAY)]
    wts = [mx.softmax(mx.random.normal((S, K)), axis=-1) for _ in range(NLAY)]
    x0 = (mx.random.normal((S, D)) * 0.5).astype(mx.bfloat16)
    res = {}
    for name, f in (("MLX gather chain", mlx_moe), ("MTPLX fused", fused_moe)):
        def run():
            y = x0
            for L, i, w in zip(layers, idxs, wts):
                y = f(y, i, w, L)
            mx.eval(y)
            return y
        for _ in range(3):
            run()
        ts = []
        for _ in range(15):
            a = time.perf_counter(); run(); ts.append(time.perf_counter() - a)
        res[name] = sorted(ts)[7] / NLAY * 1e3
    a = mlx_moe(x0, idxs[0], wts[0], layers[0]).astype(mx.float32); b = fused_moe(x0, idxs[0], wts[0], layers[0]).astype(mx.float32)
    rel = (mx.abs(a - b).max() / mx.abs(a).max()).item()
    print(f"S={S}: MLX gather chain {res['MLX gather chain']:.3f} ms/layer | MTPLX fused {res['MTPLX fused']:.3f} ms/layer "
          f"({res['MLX gather chain']/res['MTPLX fused']:.2f}x) | rel diff {rel:.1e} -> x48 layers: "
          f"{(res['MLX gather chain']-res['MTPLX fused'])*48:+.1f} ms per pass", flush=True)
