"""WEEK98 BET 7, M7a: the dispatch floor (MoonShots, Mon 16 Nov 2026; information, pre-registered in LEVERS_A
"BET 7", commit 6bb1a0c, before any code ran).

A chain of N dependent minimal kernels per "level", L levels per block, async_eval per level and one sync per block
(M1's protocol in p3_m012.py). Each step reads the previous step's [2560] bf16 output and writes a new one, so every
step depends on the one before it. Arms:
  stock     N MLX elementwise ops (x = x * c): one stock binary kernel each
  custom    N mx.fast.metal_kernel calls, 10 threadgroups x 256 threads (out = in * c)
  wide      the same with 64 threadgroups x 256 threads, 40 outputs per threadgroup (a GEMV-wide grid)
  compiled  the custom chain under mx.compile (one trace per N)
Reported per arm and N: wall ms per level (median over rounds), CPU ms per level spent building the graph and in
async_eval; then the fit wall = a + b N per arm (b = ms per dependent step).
Under a share lock beside other GPU jobs: INDICATIVE (GPU busy % logged per round). The verdict uses a quiet slot.
usage: ~/BigMoE/.venv/bin/python -I -B fl_floor.py [--rounds 12] [--L 21] [--smoke] [--out <json>]"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=12)
ap.add_argument("--L", type=int, default=21)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--out", default=os.path.expanduser("~/MoonShots/week98/bet7/fl_floor.json"))
ap.add_argument("--level", action="store_true",
                help="also time the stock C0 level (fl_stock.py) at F = 1 the same way: wall vs CPU per level, so "
                     "today's fixed cost splits into its CPU and GPU parts (+2.5 GB)")
A = ap.parse_args()
NS = [8, 32] if A.smoke else [8, 16, 24, 32, 48, 64, 96, 144]
R = 3 if A.smoke else A.rounds
D = 2560


def say(s=""):
    print(s, flush=True)


def gpu_busy():
    try:
        o = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], capture_output=True,
                           text=True, timeout=5).stdout
        m = re.search(r'"Device Utilization %"=(\d+)', o)
        return int(m.group(1)) if m else -1
    except Exception:
        return -1


_K_NARROW = mx.fast.metal_kernel(
    name="fl_step_narrow", input_names=["inp"], output_names=["out"],
    source="""
        uint i = thread_position_in_grid.x;
        if (i < 2560) { out[i] = inp[i] * static_cast<T>(1.0001f); }
    """)
_K_WIDE = mx.fast.metal_kernel(
    name="fl_step_wide", input_names=["inp"], output_names=["out"],
    source="""
        uint g = threadgroup_position_in_grid.x;
        uint t = thread_position_in_threadgroup.x;
        if (t < 40) { uint i = g * 40 + t; out[i] = inp[i] * static_cast<T>(1.0001f); }
    """)


def k_narrow(x):
    return _K_NARROW(inputs=[x], template=[("T", mx.bfloat16)], grid=(2560, 1, 1), threadgroup=(256, 1, 1),
                     output_shapes=[x.shape], output_dtypes=[x.dtype])[0]


def k_wide(x):
    return _K_WIDE(inputs=[x], template=[("T", mx.bfloat16)], grid=(64 * 256, 1, 1), threadgroup=(256, 1, 1),
                   output_shapes=[x.shape], output_dtypes=[x.dtype])[0]


C = mx.array(1.0001, dtype=mx.bfloat16)


def chain_stock(x, n):
    for _ in range(n):
        x = x * C
    return x


def chain_custom(x, n):
    for _ in range(n):
        x = k_narrow(x)
    return x


def chain_wide(x, n):
    for _ in range(n):
        x = k_wide(x)
    return x


COMPILED = {n: mx.compile(lambda x, n=n: chain_custom(x, n)) for n in NS}
ARMS = {"stock": chain_stock, "custom": chain_custom, "wide": chain_wide,
        "compiled": lambda x, n: COMPILED[n](x)}

if A.level:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fl_stock                                          # noqa: E402
    ST = fl_stock.Stock(nh=32768, T0=400)


def run_level_block():
    """the stock C0 level at F = 1, K = 8: L levels, async_eval per level, one sync (M1); CPU per level split
    into graph building and async_eval"""
    s = ST
    h, tok, sc = s.h0, mx.array([[1000]], dtype=mx.int32), mx.zeros((1,), mx.float32)
    t_build = t_sub = 0.0
    keep = []
    t0 = time.perf_counter()
    for _ in range(A.L):
        a = time.perf_counter()
        o = s.level(h, tok, sc, 1, 8)
        h, tok, sc = o["h"], o["tok"], o["sc"]
        b = time.perf_counter()
        mx.async_eval(h, tok, sc, o["kidx"], o["klp"])
        t_build += b - a
        t_sub += time.perf_counter() - b
        keep.append(o["kidx"])
    mx.eval(keep, h, sc)
    t1 = time.perf_counter()
    s.cache[0].trim(A.L)
    return (t1 - t0) / A.L, t_build / A.L, t_sub / A.L


def run_block(fn, n, x0):
    """L levels of n dependent steps; async_eval per level, one sync at the end"""
    t_build = t_sub = 0.0
    x = x0
    t0 = time.perf_counter()
    for _ in range(A.L):
        a = time.perf_counter()
        x = fn(x, n)
        b = time.perf_counter()
        mx.async_eval(x)
        t_build += b - a
        t_sub += time.perf_counter() - b
    mx.eval(x)
    t1 = time.perf_counter()
    return (t1 - t0) / A.L, t_build / A.L, t_sub / A.L, x


# self-check: every arm gives the same bits (same elementwise op, same dtype)
x0 = (mx.random.normal((D,), key=mx.random.key(7)) * 0.5).astype(mx.bfloat16)
mx.eval(x0)
ref = None
chk = {}
for name, fn in ARMS.items():
    y = run_block(fn, NS[0], x0)[3]
    mx.eval(y)
    if ref is None:
        ref = y
    chk[name] = bool(mx.array_equal(y, ref).item())
say(f"self-check (all arms bit-equal to stock after {A.L} x {NS[0]} steps): {chk}")

T = {(a, n): [] for a in ARMS for n in NS}
B = {(a, n): [] for a in ARMS for n in NS}
S = {(a, n): [] for a in ARMS for n in NS}
LV = {"wall": [], "build": [], "submit": []}
BUSY = []
keys = [(a, n) for n in NS for a in ARMS] + ([("level", 0)] if A.level else [])
t00 = time.time()
for r in range(R):
    BUSY.append(gpu_busy())
    order = keys if r % 2 == 0 else keys[::-1]
    for a, n in order:
        if a == "level":
            w, bld, sub = run_level_block()
            if r >= 2 or A.smoke:
                LV["wall"].append(1e3 * w)
                LV["build"].append(1e3 * bld)
                LV["submit"].append(1e3 * sub)
            continue
        w, bld, sub, _ = run_block(ARMS[a], n, x0)
        if r >= 2 or A.smoke:
            T[(a, n)].append(1e3 * w)
            B[(a, n)].append(1e3 * bld)
            S[(a, n)].append(1e3 * sub)
    say(f"round {r + 1}/{R} at {time.time() - t00:.0f} s (GPU busy at round start {BUSY[-1]} %)")

res = {"arms": {}, "fits": {}, "gpu_busy": BUSY, "L": A.L, "rounds": R, "self_check": chk}
say("")
say("ms per level (median): wall | CPU build | CPU in async_eval")
say("N    " + "  ".join(f"{a:>26s}" for a in ARMS))
for n in NS:
    cells = []
    for a in ARMS:
        w, bld, sub = (float(np.median(T[(a, n)])), float(np.median(B[(a, n)])), float(np.median(S[(a, n)])))
        res["arms"].setdefault(a, {})[n] = dict(wall=round(w, 4), build=round(bld, 4), submit=round(sub, 4),
                                               p10=round(float(np.percentile(T[(a, n)], 10)), 4),
                                               p90=round(float(np.percentile(T[(a, n)], 90)), 4))
        cells.append(f"{w:7.4f} | {bld:6.4f} | {sub:6.4f}")
    say(f"{n:<4d} " + "  ".join(f"{c:>26s}" for c in cells))
say("")
for a in ARMS:
    xs = np.array(NS, float)
    ys = np.array([res["arms"][a][n]["wall"] for n in NS])
    cs = np.array([res["arms"][a][n]["build"] + res["arms"][a][n]["submit"] for n in NS])
    b, a0 = np.polyfit(xs, ys, 1)
    bc, ac = np.polyfit(xs, cs, 1)
    res["fits"][a] = dict(us_per_step=round(1e3 * b, 3), intercept_ms=round(a0, 4), cpu_us_per_step=round(1e3 * bc, 3),
                          cpu_intercept_ms=round(ac, 4))
    say(f"{a:9s}: wall = {a0:.4f} ms + {1e3 * b:.3f} us x N; CPU (build + async_eval) = {ac:.4f} ms + {1e3 * bc:.3f} us x N")
if A.level:
    lv = {k: round(float(np.median(x)), 4) for k, x in LV.items()}
    lv["mb"] = 177.6
    lv["wall_minus_bytes_at_292.7"] = round(lv["wall"] - 177.6 / 292.7, 4)
    res["stock_level_C0_F1"] = lv
    say(f"stock C0 level, F = 1, T ~400: wall {lv['wall']:.4f} ms per level; CPU build {lv['build']:.4f} + async_eval "
        f"{lv['submit']:.4f} = {lv['build'] + lv['submit']:.4f} ms; wall - 177.6 MB / 292.7 GB/s = "
        f"{lv['wall_minus_bytes_at_292.7']:.4f} ms")
json.dump(res, open(A.out, "w"), indent=1)
say(f"wrote {A.out}")
