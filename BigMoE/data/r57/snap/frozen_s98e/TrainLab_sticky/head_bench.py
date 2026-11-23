"""HEADBENCH (MTP Head, Tue 11-03; information, < 1 GB GPU, ~20 s): what the MTP draft step's head costs. The live step
(d23c_mtptrunc.MTPHead.draft_cut) runs mtp_forward (embed, the MTP layer, the HC mixer, the full 248,320-token lm_head at
4-bit g64) and then argmax + an f32 softmax max, with one mx.eval sync per step. Here: the lm_head alone from the shard, one
row, the same post-ops and sync, vs variants (no softmax; p1 by logsumexp; an empty sync). D-23 measured ~2.2 ms per step.
Output: data/head_bench.txt (appended)."""
import os, time, json
import mlx.core as mx
MD = os.path.expanduser("~/BigMoE/models/bare")
wm = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
z = mx.load(os.path.join(MD, wm["language_model.lm_head.weight"]))
W, S, B = (z[f"language_model.lm_head.{k}"] for k in ("weight", "scales", "biases"))
del z
mx.eval(W, S, B)
print("lm_head", W.shape, W.dtype, S.shape, f"{(W.nbytes + S.nbytes + B.nbytes) / 1e6:.0f} MB", flush=True)
x = mx.random.normal((1, 1, 2560)).astype(mx.bfloat16)
mx.eval(x)
def head(x):
    return mx.quantized_matmul(x, W, S, B, transpose=True, group_size=64, bits=4)
def full(x):                                       # the live post-ops
    lx = head(x)[0, -1].astype(mx.float32); return mx.argmax(lx), mx.softmax(lx).max()
def lse(x):
    lx = head(x)[0, -1].astype(mx.float32); return mx.argmax(lx), mx.exp(lx.max() - mx.logsumexp(lx))
def amax(x):
    return (mx.argmax(head(x)[0, -1]),)
def empty(x):
    return (x[0, 0, :1] + 1,)
lines = []
for nm, f in (("live: head + argmax + f32 softmax max", full), ("head + argmax + p1 by logsumexp", lse), ("head + argmax only", amax), ("empty op + sync", empty)):
    for _ in range(20):
        mx.eval(*f(x))
    ts = []
    for _ in range(200):
        t0 = time.perf_counter(); r = f(x); mx.eval(*r); [float(v.item()) for v in r]; ts.append(time.perf_counter() - t0)
    ts.sort()
    lines.append(f"  {nm:40s}: median {1e3 * ts[100]:.3f} ms, p10 {1e3 * ts[20]:.3f}, p90 {1e3 * ts[180]:.3f}")
    print(lines[-1], flush=True)
bw = (W.nbytes + S.nbytes + B.nbytes) / 1e9
lines.append(f"  lm_head bytes {bw:.3f} GB; live head time implies {bw / (float(lines[0].split('median ')[1].split(' ms')[0]) / 1e3):.0f} GB/s")
print(lines[-1])
open(os.path.expanduser("~/TrainLab/data/head_bench.txt"), "a").write(f"HEADBENCH {time.strftime('%m-%d %H:%M')}\n" + "\n".join(lines) + "\n\n")
