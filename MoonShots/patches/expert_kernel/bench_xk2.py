"""xk2 (dedup, MLX-exact per pair) vs mlx-lm SwitchGLU on REAL Qwen3.8-Flash-Next 125B experts (layers 0..NL-1 of
~/BigMoE/models/bare, read-only mmap). S = 1..6 verify rows, top-10 with realistic overlap between consecutive tokens.
Checks: SwitchGLU output bit-identical (every S, several routings); timing per layer over a dependent chain of NL layers;
GB/s of distinct expert bytes; and the bandwidth one large MLX qmv reaches on the same byte count (what the hardware gives
a single dispatch of this size). usage: bench_xk2.py [NL] [RSG_A RSG_B SGTG]   (GPU: quiet lock for the timings)"""
import json, os, struct, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xk2
from mlx_lm.models.switch_layers import SwitchGLU, QuantizedSwitchLinear

NL = int(sys.argv[1]) if len(sys.argv) > 1 else 3
if len(sys.argv) > 4:
    xk2.RSG_A, xk2.RSG_B, xk2.SGTG = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
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
    sw = SwitchGLU(D, F, E)
    for p in ("gate_proj", "up_proj", "down_proj"):
        q = QuantizedSwitchLinear(D if p != "down_proj" else F, F if p != "down_proj" else D, E, bias=False, group_size=64, bits=4)
        q.weight, q.scales, q.biases = (tensor(pre + f"{p}.{f}") for f in ("weight", "scales", "biases"))
        setattr(sw, p, q)
    mx.eval(sw.parameters())
    layers.append(sw)
per_expert = sum(a.nbytes for p in ("gate_proj", "up_proj", "down_proj") for a in
                 (getattr(layers[0], p).weight, getattr(layers[0], p).scales, getattr(layers[0], p).biases)) / E
print(f"{NL} real 125B MoE layers; {per_expert/1e6:.2f} MB per expert; xk2 RSG_A {xk2.RSG_A} RSG_B {xk2.RSG_B} SGTG {xk2.SGTG}", flush=True)
rng = np.random.default_rng(0)


def routing(S):
    base = rng.choice(E, KT, replace=False); rows = [base]
    for _ in range(S - 1):
        keep = rng.choice(base, int(0.45 * KT), replace=False)
        rows.append(np.concatenate([keep, rng.choice(np.setdiff1d(np.arange(E), keep), KT - len(keep), replace=False)]))
        base = rows[-1]
    return np.stack(rows).astype(np.uint32)[None]          # [1, S, KT] like the MoE block's argpartition output


def moe(f, x, ids, sc, L):                                  # x [1, S, D] -> [1, S, D] (combine as in qwen4_exp)
    y = f(layers[L], x, ids)
    return (y * sc[..., None]).sum(axis=-2)


mlx_f = lambda sw, x, ids: sw(x, ids)
xk2_f = lambda sw, x, ids: xk2.switchglu(sw, x, ids)
xk3_f = lambda sw, x, ids: xk2.switchglu3(sw, x, ids)
NCH = int(os.environ.get("XK2_CHUNKS", "2"))


def chunk_f(sw, x, ids):          # exact: every pair still runs MLX's own gather_qmv; chunks along top-k can overlap on the GPU
    cuts = [round(i * KT / NCH) for i in range(NCH + 1)]
    return mx.concatenate([sw(x, ids[..., a:b]) for a, b in zip(cuts, cuts[1:])], axis=-2)

# exactness first: several routings per S
bad = 0
for S in (1, 2, 3, 4, 5, 6):
    for t in range(4):
        ids = mx.array(routing(S)); x = (mx.random.normal((1, S, D)) * 0.5).astype(mx.bfloat16)
        a, b = mlx_f(layers[t % NL], x, ids), xk2_f(layers[t % NL], x, ids)
        c3 = xk3_f(layers[t % NL], x, ids)
        same = mx.array_equal(a, b).item() and mx.array_equal(a, c3).item()
        if not mx.array_equal(a, c3).item(): print(f"  xk3 differs at S={S}", flush=True)
        if not same:
            bad += 1
            d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
            print(f"  S={S} routing {t}: NOT identical, max abs diff {d.max().item():.3e}, differing {int((d > 0).sum().item())}/{d.size}", flush=True)
print(f"exactness: {'ALL bit-identical' if bad == 0 else f'{bad} mismatching cases'} (SwitchGLU output, S=1..6 x 4 routings)", flush=True)

for S in (1, 2, 3, 4, 6):
    ids = [mx.array(routing(S)) for _ in range(NL)]
    scs = [mx.softmax(mx.random.normal((1, S, KT)), axis=-1).astype(mx.bfloat16) for _ in range(NL)]
    uniq = np.mean([len(np.unique(np.array(i))) for i in ids])
    x0 = (mx.random.normal((1, S, D)) * 0.5).astype(mx.bfloat16)
    res = {}
    for name, f in (("mlx", mlx_f), ("xk2", xk2_f), ("xk3", xk3_f), ("mlx", mlx_f), ("xk2", xk2_f), ("xk3", xk3_f)):
        def run():
            y = x0
            for L in range(NL):
                y = moe(f, y, ids[L], scs[L], L)
            mx.eval(y)
            return y
        for _ in range(3): run()
        ts = []
        for _ in range(15):
            a = time.perf_counter(); run(); ts.append(time.perf_counter() - a)
        res.setdefault(name, []).append(sorted(ts)[7] / NL * 1e3)
    t_m, t_x, t_c = min(res["mlx"]), min(res["xk2"]), min(res["xk3"])
    gb = uniq * per_expert / 1e9
    print(f"S={S}: distinct {uniq:4.1f}  MLX {t_m:.3f} ms/layer ({gb/t_m*1e3:4.0f} GB/s distinct)  xk2 {t_x:.3f} ({gb/t_x*1e3:4.0f})  "
          f"xk3 {t_c:.3f} ({gb/t_c*1e3:4.0f})  -> best vs MLX x48 layers {(t_m-min(t_x, t_c))*48:+.1f} ms/pass", flush=True)

# what one large MLX qmv reaches on the bytes of ~22 distinct experts' gate+up (one dispatch, ~40 MB)
nE = 22
w = mx.concatenate([layers[0].gate_proj.weight[:nE].reshape(-1, D // 8), layers[0].up_proj.weight[:nE].reshape(-1, D // 8)])
s = mx.concatenate([layers[0].gate_proj.scales[:nE].reshape(-1, D // 64), layers[0].up_proj.scales[:nE].reshape(-1, D // 64)])
b = mx.concatenate([layers[0].gate_proj.biases[:nE].reshape(-1, D // 64), layers[0].up_proj.biases[:nE].reshape(-1, D // 64)])
mx.eval(w, s, b)
xs = [(mx.random.normal((1, D)) * 0.5).astype(mx.bfloat16) for _ in range(20)]
mx.eval(xs)
qmv = lambda: mx.eval([mx.quantized_matmul(xv, w, s, b, transpose=True, group_size=64, bits=4) for xv in xs])
for _ in range(3): qmv()
ts = []
for _ in range(10):
    a = time.perf_counter(); qmv(); ts.append((time.perf_counter() - a) / len(xs))
nb = w.nbytes + s.nbytes + b.nbytes
def dep():
    y = xs[0]
    for _ in range(len(xs)):
        o = mx.quantized_matmul(y, w, s, b, transpose=True, group_size=64, bits=4)
        y = (o[:, :D] * 0.001).astype(mx.bfloat16)          # next input depends on this output
    mx.eval(y)
for _ in range(3): dep()
td = []
for _ in range(10):
    a = time.perf_counter(); dep(); td.append((time.perf_counter() - a) / len(xs))
print(f"same qmv as a dependent chain: {sorted(td)[5]*1e3:.3f} ms each -> {nb/sorted(td)[5]/1e9:.0f} GB/s", flush=True)
print(f"one MLX qmv over {nb/1e6:.0f} MB ({nE} experts' gate+up as one matrix), 20 back to back: {sorted(ts)[5]*1e3:.3f} ms each "
      f"-> {nb/sorted(ts)[5]/1e9:.0f} GB/s", flush=True)
