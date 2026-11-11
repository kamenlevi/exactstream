"""Where does MLX's verify MoE time go? Per projection on REAL 125B layer 0 weights at S=4 (top-10, realistic overlap):
gate (= up) gather_qmm and down gather_qmm, each as 20 dependent calls (output feeds the next input), vs their bytes.
Then xk3's down kernel at several rows-per-simdgroup (RSG), same check of exactness against MLX's down.
usage: prof_proj.py   (GPU, quiet lock, ~1 min)"""
import os, sys, time, json, struct
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xk2
MD = os.path.expanduser("~/BigMoE/models/bare")
idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]
def tensor(name):
    f = f"{MD}/{idx[name]}"
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; m = json.loads(fh.read(n))[name]
    a, b = m["data_offsets"]; dt = {"U32": np.uint32, "BF16": np.uint16}[m["dtype"]]
    arr = mx.array(np.array(np.memmap(f, dt, "r", offset=8 + n + a, shape=((b - a) // np.dtype(dt).itemsize,))).reshape(m["shape"]))
    return arr.view(mx.bfloat16) if m["dtype"] == "BF16" else arr
pre = "language_model.model.layers.0.mlp.switch_mlp."
W = {p: tuple(tensor(pre + f"{p}.{f}") for f in ("weight", "scales", "biases")) for p in ("gate_proj", "down_proj")}
mx.eval(W)
rng = np.random.default_rng(0); E, KT, S, D, F = 512, 10, 4, 2560, 640
base = rng.choice(E, KT, replace=False); rows = [base]
for _ in range(S - 1):
    keep = rng.choice(base, int(0.45 * KT), replace=False)
    rows.append(np.concatenate([keep, rng.choice(np.setdiff1d(np.arange(E), keep), KT - len(keep), replace=False)])); base = rows[-1]
ids = mx.array(np.stack(rows).astype(np.uint32)[None]); u = len(np.unique(np.stack(rows)))
gq = lambda x, w: mx.gather_qmm(x, *w, rhs_indices=ids, transpose=True, group_size=64, bits=4)
def chain(f, x0, n=20):
    def run():
        x = x0
        for _ in range(n): x = f(x)
        mx.eval(x)
    for _ in range(3): run()
    ts = []
    for _ in range(9):
        a = time.perf_counter(); run(); ts.append((time.perf_counter() - a) / n)
    return sorted(ts)[4]
xg = (mx.random.normal((1, S, 1, 1, D)) * 0.5).astype(mx.bfloat16)
fg = lambda x: (gq(x, W["gate_proj"]).astype(mx.float32).sum() * 0 + x).astype(mx.bfloat16)   # next input depends on it
xd = (mx.random.normal((1, S, KT, 1, F)) * 0.5).astype(mx.bfloat16)
fd = lambda h: (gq(h, W["down_proj"])[..., :F] * 0.01).astype(mx.bfloat16)
per = lambda w: sum(a.nbytes for a in w) / E
for name, f, x0, w in (("gate", fg, xg, W["gate_proj"]), ("down", fd, xd, W["down_proj"])):
    t = chain(f, x0)
    print(f"MLX {name} gather_qmm S={S} ({u} distinct experts, 40 pairs): {t*1e3:.3f} ms -> {u*per(w)/t/1e9:.0f} GB/s distinct, "
          f"{40*per(w)/t/1e9:.0f} GB/s incl. repeats", flush=True)
# xk3 down at several RSG (exactness vs MLX down)
h = xd.reshape(S * KT, F)
ref = gq(xd, W["down_proj"]).reshape(S * KT, D)
idsf = ids.reshape(-1).astype(mx.int32)
for rsg in (4, 8, 16):
    for sgtg in (2, 4):
        n_sg = S * KT * (D // rsg)
        call = lambda hh, rsg=rsg, sgtg=sgtg, n_sg=n_sg: xk2._KB3(inputs=[hh, idsf, *W["down_proj"]],
                    template=[("K", F), ("N", D), ("P", S * KT), ("RSG", rsg), ("PMAX", 2), ("SGTG", sgtg)],
                    grid=(-(-n_sg // sgtg) * 32 * sgtg, 1, 1), threadgroup=(32 * sgtg, 1, 1),
                    output_shapes=[(S * KT, D)], output_dtypes=[mx.bfloat16])[0]
        ok = mx.array_equal(call(h), ref).item()
        t = chain(lambda hh, c=call: (c(hh)[:, :F] * 0.01).astype(mx.bfloat16), h)
        print(f"xk3 down RSG {rsg:2d} SGTG {sgtg}: {t*1e3:.3f} ms -> {u*per(W['down_proj'])/t/1e9:.0f} GB/s distinct, exact {ok}", flush=True)
