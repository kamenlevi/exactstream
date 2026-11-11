"""PORT unit parity vs MLX (MTP Head, Wed 11-04 20:4x), small and CPU-only (MLX's CPU QuantizedMatmul is single-threaded,
so only a few rows):
  A. optimizer: mlx.optimizers.AdamW (wd 0 and 0.01) + clip_grad_norm(1.0), 6 steps on random tensors with random
     gradients (scaled so clipping engages on some steps) vs port/optim_port.py;
  B. the frozen 4-bit head: port Head("bf16") logits (4 rows x 248320) and input gradient vs float64 truth (MLX's
     GPU kernels accumulate in fp32; MLX 0.32.2's CPU QuantizedMatmul in bf16 is far off and shown for information);
     the 4-bit embedding rows vs MLX-CPU dequantize.
Usage: [BigMoE venv] python -I test_mlx_units.py ref OUT.npz ; [TrainLab .venv-torch] python -I test_mlx_units.py port OUT.npz"""
import os, sys
import numpy as np
mode, path = sys.argv[1], os.path.expanduser(sys.argv[2])
g = np.random.default_rng(7)
P0 = {"a": g.normal(0, 0.1, (64, 32)).astype(np.float32), "b": g.normal(0, 0.1, (128,)).astype(np.float32)}
GR = [{k: (g.normal(0, 1, v.shape) * (0.05 if s % 2 else 0.5)).astype(np.float32) for k, v in P0.items()} for s in range(6)]
LRS = [2e-4, 1e-3, 3e-3, 1e-3, 5e-4, 2e-4]
X = g.normal(0, 1, (4, 2560)).astype(np.float32)
COT = g.normal(0, 1, (4, 248320)).astype(np.float32)
IDS = np.array([0, 1, 1000, 151643, 248077, 248319], np.int32)
SHARD = os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors")

if mode == "ref":
    import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
    from mlx.utils import tree_map
    mx.set_default_device(mx.cpu)
    out = {}
    for wd in (0.0, 0.01):
        p = {k: mx.array(v) for k, v in P0.items()}
        opt = optim.AdamW(learning_rate=LRS[0], weight_decay=wd)
        norms = []
        for s in range(6):
            gr, tn = optim.clip_grad_norm({k: mx.array(v) for k, v in GR[s].items()}, 1.0)
            opt.learning_rate = LRS[s]
            p = opt.apply_gradients(gr, p)
            mx.eval(p)
            norms.append(float(tn))
        for k in p:
            out[f"opt_wd{wd}_{k}"] = np.array(p[k])
        out[f"opt_wd{wd}_norms"] = np.array(norms)
    w = mx.load(SHARD)
    head = nn.QuantizedLinear(2560, 248320, bias=False, group_size=64, bits=4)
    head.load_weights([(s, w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
    emb = nn.QuantizedEmbedding(248320, 2560, group_size=64, bits=4)
    emb.load_weights([(s, w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales", "biases")])
    f = lambda x: (head(x.astype(mx.bfloat16)).astype(mx.float32) * mx.array(COT)).sum()
    x = mx.array(X)
    logits = head(x.astype(mx.bfloat16)).astype(mx.float32)
    dx = mx.grad(f)(x)
    e = emb(mx.array(IDS)).astype(mx.float32)
    mx.eval(logits, dx, e)
    out.update(head_logits=np.array(logits), head_dx=np.array(dx), emb=np.array(e))
    np.savez(path, **out)
    print("ref saved", path)
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    import dflash2_torch as M
    from optim_port import MLXAdamW, clip_grad_norm_mlx
    torch.set_num_threads(4)
    ref = np.load(path)
    ok = True
    for wd in (0.0, 0.01):
        ps = [torch.nn.Parameter(torch.from_numpy(P0[k].copy())) for k in ("a", "b")]
        opt = MLXAdamW(ps, lr=LRS[0], weight_decay=wd)
        norms = []
        for s in range(6):
            for p_, k in zip(ps, ("a", "b")):
                p_.grad = torch.from_numpy(GR[s][k].copy())
            norms.append(clip_grad_norm_mlx(ps, 1.0))
            opt.param_groups[0]["lr"] = LRS[s]
            opt.step()
        dp = max(float(np.abs(p_.detach().numpy() - ref[f"opt_wd{wd}_{k}"]).max()) for p_, k in zip(ps, ("a", "b")))
        dn = float(np.abs(np.array(norms) - ref[f"opt_wd{wd}_norms"]).max())
        ok &= dp < 1e-6 and dn < 1e-5
        print(f"A. AdamW wd {wd}: 6 steps, max |param diff| {dp:.2e} (params ~0.1, steps ~1e-3), max |norm diff| {dn:.2e}; "
              f"clipped on {int((ref[f'opt_wd{wd}_norms'] > 1).sum())}/6 steps")
    eq, hq = M.load_target_q4()
    hq.unpack_u8()
    head = M.Head(hq, "bf16")
    x = torch.from_numpy(X.copy()).requires_grad_(True)
    logits = head(x)
    cot = torch.from_numpy(COT)
    (logits * cot).sum().backward()
    # truth in float64: logits = bf16(bf16(x) @ W^T), d_x = bf16(bf16(cot) @ W) with W = s * q + b exactly
    xb, cb = M.round_bf16(torch.from_numpy(X)).double(), M.round_bf16(cot).double()
    ex_l, ex_dx = [], torch.zeros((4, 2560), dtype=torch.float64)
    for v0 in range(0, hq.rows_n, 16384):
        Wd = hq.deq(r0=v0, r1=v0 + 16384).double()
        ex_l.append(xb @ Wd.T); ex_dx += cb[:, v0:v0 + 16384] @ Wd
    ex_l = M.round_bf16(torch.cat(ex_l, -1).float()).double().numpy(); ex_dx = M.round_bf16(ex_dx.float()).double().numpy()
    pl, pdx, rl, rdx = logits.detach().double().numpy(), x.grad.double().numpy(), ref["head_logits"], ref["head_dx"]
    eq_p, eq_m = float((pl == ex_l).mean()), float((rl == ex_l).mean())
    am_p, am_m = float((pl.argmax(-1) == ex_l.argmax(-1)).mean()), float((rl.argmax(-1) == ex_l.argmax(-1)).mean())
    rel = lambda a: float(np.abs(a - ex_dx).max() / np.abs(ex_dx).max())
    ids = torch.from_numpy(IDS.astype(np.int64))
    e1, e2 = eq.deq_rows_bf16(ids, "once").numpy(), eq.deq_rows_bf16(ids, "twice").numpy()
    n1, tw = float((e1 != ref["emb"]).mean()), bool(np.array_equal(e2, ref["emb"]))
    ok &= eq_p > 0.999 and am_p == 1.0 and rel(pdx) < 1e-2 and tw
    print(f"B. head vs float64 truth: port logits bitwise = bf16(truth) on {eq_p * 100:.2f} % (argmax {am_p:.2f}), input "
          f"gradient max rel diff {rel(pdx):.1e}; MLX-CPU QuantizedMatmul (for information): {eq_m * 100:.2f} % "
          f"(argmax {am_m:.2f}), gradient {rel(rdx):.1e}. Embedding: rounding 'twice' bitwise = MLX-CPU dequantize: "
          f"{tw}; rounding 'once' (the port's default, GPU-like) differs on {n1 * 100:.2f} % of entries by 1 bf16 ulp")
    print("MLX UNITS:", "PASS" if ok else "FAIL")
