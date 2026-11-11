"""CPUMTP step 1 (MTP Head, Tue 11-03; REPORT 'CPUMTP', pre-registered): the lm_head part of an MTP draft step on the CPU.
The head (248,320 x 2560, 4-bit g64, 358 MB) is a floor for a CPU MTP step. Variants: MLX CPU stream with the 4-bit weights
(quantized_matmul), bf16 and fp32 dequantized weights; numpy (Accelerate) fp32 matvec; a 32K-row subset (a reduced draft
vocabulary) in the best full-vocab path. Each timed alone, then beside a GPU load (env CPU_BENCH_GPULOAD=1: the parent
starts a child process running the 4-bit lm_head on the GPU in a loop for the duration). Median of N steps, each followed
by an argmax (as a draft step needs). Output: data/cpu_head_bench.txt (appended)."""
import os, sys, time, json, subprocess
import numpy as np
import mlx.core as mx
MD = os.path.expanduser("~/BigMoE/models/bare")
N = 9
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

if len(sys.argv) > 1 and sys.argv[1] == "gpuload":                 # the child: the GPU head in a loop until killed
    wm = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
    z = mx.load(os.path.join(MD, wm["language_model.lm_head.weight"]))
    W, S, B = (z[f"language_model.lm_head.{k}"] for k in ("weight", "scales", "biases")); del z
    x = mx.random.normal((1, 2560)).astype(mx.bfloat16)
    while True:
        mx.eval(mx.argmax(mx.quantized_matmul(x, W, S, B, transpose=True, group_size=64, bits=4), -1))

wm = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
z = mx.load(os.path.join(MD, wm["language_model.lm_head.weight"]))
W, S, B = (z[f"language_model.lm_head.{k}"] for k in ("weight", "scales", "biases")); del z
mx.eval(W, S, B)
Wd = mx.dequantize(W, S, B, group_size=64, bits=4)               # [248320, 2560]
W16 = Wd if Wd.dtype == mx.bfloat16 else Wd.astype(mx.bfloat16); del Wd
W32 = W16.astype(mx.float32); mx.eval(W16, W32)                   # peak ~4.2 GB (4-bit + bf16 + fp32 copies)
W32np = np.asarray(W32)                                           # a view where MLX allows it
SUB = 32768
W32sub = W32[:SUB]; mx.eval(W32sub)
x = mx.random.normal((1, 2560)); x16 = x.astype(mx.bfloat16); mx.eval(x, x16)
xnp = np.array(x, dtype=np.float32)


def t_mlx(fn, n=N, cap_s=20.0):
    ts, t_start = [], time.perf_counter()
    for _ in range(n):
        t0 = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t0)
        if time.perf_counter() - t_start > cap_s:
            break
    return 1e3 * float(np.median(ts)), len(ts)


def t_np(fn, n=N):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts)), len(ts)


cpu = mx.cpu
Q4R = 8192                                                        # the CPU 4-bit path is a reference loop (M-19): time a slice, scale up
V = {
    "MLX CPU 4-bit (8K rows x 30.3)": lambda: (lambda r: (r[0] * W.shape[0] / Q4R, r[1]))(t_mlx(lambda: mx.argmax(mx.quantized_matmul(x16, W[:Q4R], S[:Q4R], B[:Q4R], transpose=True, group_size=64, bits=4, stream=cpu), -1, stream=cpu), n=3, cap_s=15.0)),
    "MLX CPU bf16": lambda: t_mlx(lambda: mx.argmax(mx.matmul(x16, W16.T, stream=cpu), -1, stream=cpu)),
    "MLX CPU fp32": lambda: t_mlx(lambda: mx.argmax(mx.matmul(x, W32.T, stream=cpu), -1, stream=cpu)),
    "numpy (Accelerate) fp32": lambda: t_np(lambda: np.argmax(W32np @ xnp[0])),
    f"MLX CPU fp32, {SUB // 1024}K-row subset": lambda: t_mlx(lambda: mx.argmax(mx.matmul(x, W32sub.T, stream=cpu), -1, stream=cpu)),
    f"numpy fp32, {SUB // 1024}K-row subset": lambda: t_np(lambda: np.argmax(W32np[:SUB] @ xnp[0])),
}
res = {}
for mode in ("alone", "beside a GPU load"):
    child = None
    if mode != "alone":
        child = subprocess.Popen([sys.executable, os.path.abspath(__file__), "gpuload"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(4.0)
    try:
        for name, fn in V.items():
            ms, n = fn()
            res[(mode, name)] = ms
            say(f"  {mode:18s} {name:34s}: median {ms:9.2f} ms ({n} steps)")
    finally:
        if child is not None:
            child.kill(); child.wait()
full = [res[("beside a GPU load", k)] for k in V if "subset" not in k]
sub = [res[("beside a GPU load", k)] for k in V if "subset" in k]
v = "PASS" if min(full) <= 3.0 else ("KILL" if (min(full) > 5.6 and min(sub) > 3.0) else "BETWEEN")
say(f"  LINE (best full-vocab head beside the GPU load <= 3.0 ms PASS; > 5.6 ms and the 32K subset > 3.0 ms KILL): full {min(full):.2f} ms, "
    f"subset {min(sub):.2f} ms -> {v}  (GPU 4-bit head alone: 1.47 ms, HEADBENCH)")
open(os.path.expanduser("~/TrainLab/data/cpu_head_bench.txt"), "a").write(f"CPUMTP step 1 {time.strftime('%m-%d %H:%M')}\n" + "\n".join(lines) + "\n\n")
