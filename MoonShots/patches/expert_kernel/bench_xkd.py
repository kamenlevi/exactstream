"""v2 down projection (xkd) vs MLX gather_qmm on REAL 125B layer-0 down weights, verify shapes S=1..6 (top-10, realistic
overlap). Reports time per call in a dependent chain, GB/s of distinct down bytes, max relative difference to MLX, determinism
and batch invariance (pair 0 alone == pair 0 inside the batch). usage: bench_xkd.py   (GPU, quiet lock, ~1-2 min)"""
import json, os, struct, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xkd
MD = os.path.expanduser("~/BigMoE/models/bare")
idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]


def tensor(name):
    f = f"{MD}/{idx[name]}"
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; m = json.loads(fh.read(n))[name]
    a, b = m["data_offsets"]; dt = {"U32": np.uint32, "BF16": np.uint16}[m["dtype"]]
    arr = mx.array(np.array(np.memmap(f, dt, "r", offset=8 + n + a, shape=((b - a) // np.dtype(dt).itemsize,))).reshape(m["shape"]))
    return arr.view(mx.bfloat16) if m["dtype"] == "BF16" else arr


dn = tuple(tensor(f"language_model.model.layers.0.mlp.switch_mlp.down_proj.{f}") for f in ("weight", "scales", "biases"))
mx.eval(dn)
TPG = int(os.environ.get("XKD_TPG", "128"))
packed = None
P4T = int(os.environ.get("XKD_P4T", "32"))
packed4 = xkd.pack4(*dn, tpg=P4T)
mx.eval(packed4)
packedr = xkd.pack_rev(*dn)
mx.eval(packedr)
E, N, F, KT = 512, 2560, 640, 10
per = sum(a.nbytes for a in dn) / E
rng = np.random.default_rng(0)


def routing(S):
    base = rng.choice(E, KT, replace=False); rows = [base]
    for _ in range(S - 1):
        keep = rng.choice(base, int(0.45 * KT), replace=False)
        rows.append(np.concatenate([keep, rng.choice(np.setdiff1d(np.arange(E), keep), KT - len(keep), replace=False)])); base = rows[-1]
    return np.stack(rows).astype(np.uint32)


def chain(f, x0, n=20):
    def run():
        x = x0
        for _ in range(n): x = (f(x)[:, :F] * 0.01).astype(mx.bfloat16)
        mx.eval(x)
    for _ in range(3): run()
    ts = []
    for _ in range(9):
        a = time.perf_counter(); run(); ts.append((time.perf_counter() - a) / n)
    return sorted(ts)[4]


for S in (1, 2, 3, 4, 6):
    ids = mx.array(routing(S)); P = S * KT; u = len(np.unique(np.array(ids)))
    h = (mx.random.normal((P, F)) * 0.5).astype(mx.bfloat16)
    mlx_f = lambda x: mx.gather_qmm(x.reshape(1, S, KT, 1, F), *dn, rhs_indices=ids[None], transpose=True, group_size=64, bits=4).reshape(P, N)
    res = {"mlx": chain(mlx_f, h)}
    ref = mlx_f(h).astype(mx.float32)
    line = f"S={S} ({u} distinct, {P} pairs): MLX {res['mlx']*1e3:.3f} ms ({u*per/res['mlx']/1e9:3.0f} GB/s)"
    for name, kw in (("pack4", None), ("p4-mlxlayout", "m"), ("rev", "r")):
        if kw is None:
            f = lambda x: xkd.down4(x, ids, packed4, pmax=2, tpg=P4T)
        elif kw == "m":
            f = lambda x: xkd.down4m(x, ids, dn, pmax=2, tpg=P4T)
        else:
            f = lambda x: xkd.down_rev(x, ids, packedr, pmax=min(8, S))
        t = chain(f, h)
        y = f(h); y2 = f(h)
        rel = (mx.abs(y.astype(mx.float32) - ref).max() / mx.abs(ref).max()).item()
        if kw == "m": rel = mx.array_equal(y, xkd.down4(h, ids, packed4, pmax=2, tpg=P4T)).item()   # == pack4 bitwise?
        det = mx.array_equal(y, y2).item()
        one = (xkd.down4(h[:1], ids[:1, :1], packed4, pmax=2, tpg=P4T) if kw is None
               else xkd.down4m(h[:1], ids[:1, :1], dn, pmax=2, tpg=P4T) if kw == "m"
               else xkd.down_rev(h[:1], ids[:1, :1], packedr, pmax=1))            # pair 0 alone
        inv = mx.array_equal(one[0], y[0]).item()
        line += f" | {name} {t*1e3:.3f} ms ({u*per/t/1e9:3.0f} GB/s) {'same as pack4 %s' % rel if kw == "m" else 'rel %.1e' % rel} det {det} inv {inv}"
    print(line, flush=True)
