"""xk (our routed-expert kernel) vs MLX (mlx-lm SwitchGLU on QuantizedSwitchLinear + score-weighted sum) on REAL Qwen3.8-Flash-Next
125B expert weights (layers 0..NL-1 of ~/BigMoE/models/bare, read-only via mmap), S = 1..6 rows, top-10 with realistic overlap
between consecutive tokens. Dependent chain over the layers. Reports ms/layer, GB/s of unique expert bytes, rel. difference to
MLX, determinism, and batch invariance (row 0 at S=4 == S=1 bitwise). Usage: bench_xk.py [NL]   (GPU, quiet lock, ~1.4 GB/layer)"""
import json, os, struct, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xk
from mlx_lm.models.switch_layers import SwitchGLU, QuantizedSwitchLinear

NL = int(sys.argv[1]) if len(sys.argv) > 1 else 3
D, F, E, KT = 2560, 640, 512, 10
MD = os.path.expanduser("~/BigMoE/models/bare")
idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]


def tensor(name):
    f = f"{MD}/{idx[name]}"
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        m = json.loads(fh.read(n))[name]
    a, b = m["data_offsets"]
    dt = {"U32": np.uint32, "BF16": np.uint16}[m["dtype"]]
    arr = mx.array(np.array(np.memmap(f, dt, "r", offset=8 + n + a, shape=((b - a) // np.dtype(dt).itemsize,))).reshape(m["shape"]))
    return arr.view(mx.bfloat16) if m["dtype"] == "BF16" else arr


layers = []
for L in range(NL):
    pre = f"language_model.model.layers.{L}.mlp.switch_mlp."
    w = {p: tuple(tensor(pre + f"{p}.{f}") for f in ("weight", "scales", "biases")) for p in ("gate_proj", "up_proj", "down_proj")}
    sw = SwitchGLU(D, F, E)
    for p in ("gate_proj", "up_proj", "down_proj"):
        q = QuantizedSwitchLinear(D if p != "down_proj" else F, F if p != "down_proj" else D, E, bias=False, group_size=64, bits=4)
        q.weight, q.scales, q.biases = w[p]
        setattr(sw, p, q)
    mx.eval(sw.parameters())
    layers.append((sw, w))
per_expert = sum(a.nbytes for t in layers[0][1].values() for a in t) / E
print(f"{NL} real 125B MoE layers loaded; {per_expert/1e6:.2f} MB per expert", flush=True)
rng = np.random.default_rng(0)


def routing(S):
    base = rng.choice(E, KT, replace=False); rows = [base]
    for _ in range(S - 1):
        keep = rng.choice(base, int(0.45 * KT), replace=False)
        rows.append(np.concatenate([keep, rng.choice(np.setdiff1d(np.arange(E), keep), KT - len(keep), replace=False)]))
        base = rows[-1]
    return np.stack(rows).astype(np.int32)


def mlx_moe(x, ids, sc, L):
    sw = layers[L][0]
    y = sw(x, ids)                                     # [S, KT, D] bf16 (SwitchGLU: sorts only for >= 64 indices)
    return (y * sc[..., None]).sum(axis=-2).astype(mx.bfloat16)


def xk_moe(x, ids, sc, L):
    w = layers[L][1]
    return xk.experts(x, ids, sc, w["gate_proj"], w["up_proj"], w["down_proj"], KT=KT)


for S in (1, 2, 3, 4, 6):
    ids = [mx.array(routing(S)) for _ in range(NL)]
    scs = [mx.softmax(mx.random.normal((S, KT)), axis=-1) for _ in range(NL)]
    uniq = np.mean([len(np.unique(np.array(i))) for i in ids])
    x0 = (mx.random.normal((S, D)) * 0.5).astype(mx.bfloat16)
    res = {}
    for name, f in (("mlx", mlx_moe), ("xk", xk_moe), ("mlx", mlx_moe), ("xk", xk_moe)):
        def run():
            y = x0
            for L in range(NL):
                y = f(y, ids[L], scs[L], L)
            mx.eval(y)
        for _ in range(3):
            run()
        ts = []
        for _ in range(15):
            a = time.perf_counter(); run(); ts.append(time.perf_counter() - a)
        res.setdefault(name, []).append(sorted(ts)[7] / NL * 1e3)
    a = mlx_moe(x0, ids[0], scs[0], 0).astype(mx.float32); b = xk_moe(x0, ids[0], scs[0], 0).astype(mx.float32)
    rel = (mx.abs(a - b).max() / mx.abs(a).max()).item()
    det = mx.array_equal(xk_moe(x0, ids[0], scs[0], 0), xk_moe(x0, ids[0], scs[0], 0)).item()
    t_m, t_x = min(res["mlx"]), min(res["xk"])
    gb = uniq * per_expert / 1e9
    print(f"S={S}: union {uniq:4.1f}  MLX {t_m:.3f} ms/layer ({gb/t_m*1e3:4.0f} GB/s)  xk {t_x:.3f} ms/layer ({gb/t_x*1e3:4.0f} GB/s)"
          f"  -> {t_m/t_x:.2f}x, x48 layers {(t_m-t_x)*48:+.1f} ms/pass | rel diff {rel:.1e} | deterministic {det}", flush=True)

# batch invariance: row 0 alone vs inside a 4-row batch
ids4 = mx.array(routing(4)); sc4 = mx.softmax(mx.random.normal((4, KT)), axis=-1)
x4 = (mx.random.normal((4, D)) * 0.5).astype(mx.bfloat16)
y4 = xk_moe(x4, ids4, sc4, 0); y1 = xk_moe(x4[:1], ids4[:1], sc4[:1], 0)
print(f"batch-invariant (row 0 at S=4 == S=1, bitwise): {mx.array_equal(y4[:1], y1).item()}")
