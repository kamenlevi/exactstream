"""V1 for docs/CHECK55_PROFILE.md: the hook's own cost per dispatch. Separate processes, the dylib loaded (arg 'hook') or
not ('bare'); 40 x 500 tiny dispatches (x = x + 1 on [16, 2560] bf16, one eval per 500), wall and thread CPU per dispatch.
Usage: [venv] python -I -B hook_overhead.py hook|bare"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if sys.argv[1] == "hook":
    import mtlprof as MP
    MP.load(2)
import mlx.core as mx
import numpy as np
x = mx.zeros((16, 2560), mx.bfloat16); mx.eval(x)
for _ in range(5):
    for i in range(500):
        x = x + 1
    mx.eval(x)
W, U = [], []
for r in range(40):
    w0 = time.perf_counter(); u0 = time.thread_time()
    for i in range(500):
        x = x + 1
    mx.eval(x)
    W.append((time.perf_counter() - w0) / 500); U.append((time.thread_time() - u0) / 500)
print(f"{sys.argv[1]} wall {1e6*np.median(W):.3f} us/dispatch, thread CPU {1e6*np.median(U):.3f} us/dispatch")
