"""rec_loader_toy.py (Breakthrough, Tue 17 Nov 2026 01:3x; MODULE 1a milestone 2 groundwork, docs/MODULE1A.md): a live CPU
loader inside a replay (the engine's routing-row handshake), pre-registered in docs/MODULE1A.md (Tue 01:3x).
post (coherent(system) row write) -> C loader pthread (release [seq, 3 v + 1]) -> spin (coherent(system) polls) ->
y = x * slot + 1. Run like today's engine (the post in its own command buffer). Lines: (1) MLX's own run 16/16; (2) the
segmented ICB replay 16/16 bitwise; (3) the one-buffer ICB replay 16/16 bitwise; (4) loader stopped -> timeouts 4/4;
(5) info: a device-scope post in the one-buffer replay. Tiny memory; one simdgroup spins (bounded)."""
import os
import resource
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtlrec as R
R.load()
import mlx.core as mx
import numpy as np

mx.set_cache_limit(0)
N = 4096
CAP = 2_000_000
CHECKS = list(range(2, 18))
rng = np.random.default_rng(20261117)

HDR = """#pragma METAL internals : enable
#include <metal_stdlib>
#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
namespace metal { constexpr constant thread_scope thread_scope_system = static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__); }
#endif
"""
POST_SYS = """
  uint t = thread_position_in_grid.x;
  if (t == 0) {
    volatile coherent(system) device int* r = (volatile coherent(system) device int*)(row);
    int s = seqa[0];
    int v = int(x[0] * 1000.0f) & 0xFFFF;
    r[1] = v;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
    r[0] = s;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
    done[0] = v;
  }
"""
POST_DEV = """
  uint t = thread_position_in_grid.x;
  if (t == 0) {
    device atomic_int* r = (device atomic_int*)(row);
    int s = seqa[0];
    int v = int(x[0] * 1000.0f) & 0xFFFF;
    atomic_store_explicit(&r[1], v, memory_order_relaxed);
    atomic_store_explicit(&r[0], s, memory_order_relaxed);
    done[0] = v;
  }
"""
SPIN = f"""
  uint lane = thread_index_in_simdgroup;
  volatile coherent(system) device int* r = (volatile coherent(system) device int*)(rel);
  int s = seqa[0];
  int C = cap[0];
  int it = 0;
  bool got = false;
  for (; it < C; it++) {{
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
    if (simd_all(r[0] == s)) {{ got = true; break; }}
  }}
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
  int slot = got ? r[1] : -1;
  for (uint j = lane; j < {N}; j += 32) y[j] = x[j] * float(slot) + 1.0f;
  if (lane == 0) {{ flag[0] = got ? 1 : 0; flag[1] = it; flag[2] = slot; }}
"""
post_sys = mx.fast.metal_kernel(name="bt_toy_post_sys", input_names=["x", "row", "seqa"], output_names=["done"],
                                header=HDR, source=POST_SYS)
post_dev = mx.fast.metal_kernel(name="bt_toy_post_dev", input_names=["x", "row", "seqa"], output_names=["done"],
                                header=HDR, source=POST_DEV)
spin = mx.fast.metal_kernel(name="bt_toy_spin", input_names=["x", "rel", "seqa", "cap"], output_names=["y", "flag"],
                            header=HDR, source=SPIN)

row = mx.zeros((8,), dtype=mx.int32)
rel = mx.zeros((8,), dtype=mx.int32)
mx.eval(row, rel)


def u8(a):
    return np.frombuffer(memoryview(a).cast("B"), dtype=np.uint8)


def i32(a):
    return np.frombuffer(memoryview(a).cast("B"), dtype=np.int32)


def addr(a):
    return u8(a).ctypes.data


def check(post, s, x, cap):
    seqa = mx.array([s], dtype=mx.int32)
    mx.eval(seqa)
    d = post(inputs=[x, row, seqa], grid=(32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(1,)],
             output_dtypes=[mx.int32])[0]
    mx.eval(d)                                     # the post's own command buffer (like the engine's bump commit)
    y, flag = spin(inputs=[x, rel, seqa, cap], grid=(32, 1, 1), threadgroup=(32, 1, 1),
                   output_shapes=[(N,), (3,)], output_dtypes=[mx.float32, mx.int32])
    mx.eval(y, flag)
    return seqa, d, y, flag


def newx():
    x = mx.array(rng.uniform(0.1, 0.9, N).astype(np.float32))
    mx.eval(x)
    return x


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


rc = R.toyloader_start(addr(row), addr(rel), 3, 1)
print(f"[ldr] toy loader started: rc {rc}")
cap = mx.array([CAP], dtype=mx.int32)
mx.eval(cap)


def record(post, s):
    x = newx()
    c = mx.array([CAP], dtype=mx.int32)
    mx.eval(c)
    mx.synchronize()
    R.start()
    seqa, d, y, flag = check(post, s, x, c)
    mx.synchronize()
    R.stop()
    return dict(x=x, cap=c, seqa=seqa, d=d, y=y, flag=flag)


rec = record(post_sys, 1)
fl = i32(rec["flag"]).tolist()
print(f"[ldr] record (post_sys, seq 1): {R.counts()}; flag {fl}; kernels {R.disp_names()}")
n_icb, err = R.build_icb()
print(f"[ldr] ICB: {n_icb} commands, {R.icb_nseg()} segments {('(error: ' + err + ')') if n_icb < 0 else ''}")

seq = 100
res = {"mlx": [], "seg": [], "one": []}
for k in CHECKS:
    x = newx()
    seq += 1
    _, d, y, flag = check(post_sys, seq, x, cap)
    f_ = i32(flag).tolist()
    v = int(i32(d)[0])
    ok1 = f_[0] == 1 and f_[2] == 3 * v + 1
    y_ref = u8(y).copy()
    res["mlx"].append((ok1, f_[1]))
    u8(rec["x"])[:] = u8(x)
    row_form = {}
    for form in ("seg", "one"):
        seq += 1
        i32(rec["seqa"])[0] = seq
        u8(rec["y"])[:] = 0
        i32(rec["flag"])[:] = -7
        if form == "seg":
            r_, st = R.replay_icb_seg(0)
        else:
            r_, st = R.replay_icb()
        g = i32(rec["flag"]).tolist()
        ok = r_ == 0 and g[0] == 1 and g[2] == 3 * v + 1 and np.array_equal(u8(rec["y"]), y_ref)
        res[form].append((ok, g[1]))
        row_form[form] = (ok, g)
    print(f"[ldr] check {k}: v {v}; MLX ok {ok1} (polls {f_[1]}); segmented {row_form['seg'][0]} {row_form['seg'][1]}; "
          f"one-buffer {row_form['one'][0]} {row_form['one'][1]}")


def tally(form):
    oks = [o for o, _ in res[form]]
    polls = [p for _, p in res[form]]
    return sum(oks), len(oks), int(np.median(polls)), int(np.max(polls))


for form, line in (("mlx", 1), ("seg", 2), ("one", 3)):
    a, n, med, mx_ = tally(form)
    print(f"[ldr] ({line}) {form}: {a}/{n} -> {'PASS' if a == n else 'FAIL'}; polls median {med}, max {mx_}")
line1, line2, line3 = (tally(f)[0] == len(CHECKS) for f in ("mlx", "seg", "one"))

# (5) info: a device-scope post (no POST_SYS), one-buffer replay
rec5 = record(post_dev, 5000)
n5, err5 = R.build_icb()
oks5, polls5 = 0, []
for k in range(8):
    x = newx()
    u8(rec5["x"])[:] = u8(x)
    seq += 1
    i32(rec5["seqa"])[0] = seq
    i32(rec5["flag"])[:] = -7
    r_, st = R.replay_icb()
    g = i32(rec5["flag"]).tolist()
    oks5 += (r_ == 0 and g[0] == 1)
    polls5.append(g[1])
print(f"[ldr] (5) info: device-scope post, one-buffer replay: release seen {oks5}/8; polls {polls5}")

# (4) control: loader stopped, cap 200,000 -> the spin must time out (on the post_dev record: same spin)
nrel = R.toyloader_stop()
print(f"[ldr] toy loader stopped after {nrel} releases")
i32(rec5["cap"])[0] = 200_000
to = 0
for k in range(4):
    seq += 1
    i32(rec5["seqa"])[0] = seq
    i32(rec5["flag"])[:] = -7
    r_, st = R.replay_icb()
    g = i32(rec5["flag"]).tolist()
    to += (g[0] == 0 and g[1] == 200_000)
    print(f"[ldr] (4) control {k}: flag {g}, gpu {st['gpu_us']:.0f} us")
line4 = to == 4
print(f"[ldr] (4) loader stopped -> timeouts {to}/4 -> {'PASS' if line4 else 'FAIL'}")
verdict = "PASS" if (line1 and line2 and line3 and line4) else "FAIL"
print(f"[ldr] RESULT: {verdict}: (1) {line1} (2) {line2} (3) {line3} (4) {line4}; (5) info {oks5}/8; "
      f"lifetime peak {peak_gb():.2f} GB")
