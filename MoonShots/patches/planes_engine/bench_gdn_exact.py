"""Speed of the exact GDN prework (one dispatch) vs the eager chain it replaces, at S = 1, 2, 4, 6 rows: 36 GDN layers'
worth of dependent calls per command buffer (each layer's input = the previous layer's v output, mixed into qkv), so
the dependent-kernel latency shows as it does in a pass. usage: bench_gdn_exact.py   (GPU; quiet lock, daytime)"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import mlx.nn as nn
from gdn_exact import gdn_prework_exact
from test_gdn_exact import eager, conv, C, KW
scale = mx.array(1.0, mx.bfloat16) * (128 ** -0.5)
convs = []
for i in range(36):
    w = (mx.random.normal(conv.weight.shape) * 0.3).astype(mx.bfloat16)
    convs.append(w)
mx.eval(convs)


def run(S, fused, reps=20):
    qkv = (mx.random.normal((1, S, C))).astype(mx.bfloat16)
    st = (mx.random.normal((1, KW - 1, C))).astype(mx.bfloat16)
    mx.eval(qkv, st)
    best = 1e9
    for r in range(reps):
        x = qkv
        t = time.perf_counter()
        for i in range(36):
            if fused:
                q, k, v, ns = gdn_prework_exact(x.reshape(S, C), st.reshape(KW - 1, C), convs[i].reshape(C, KW), scale)
                x = mx.concatenate([q, k, v], axis=-1).reshape(1, S, C)
            else:
                conv.weight = convs[i]
                q, k, v, ns = eager(x, st)
                x = mx.concatenate([q.reshape(1, S, -1), k.reshape(1, S, -1), v.reshape(1, S, -1)], axis=-1)
        mx.eval(x)
        best = min(best, time.perf_counter() - t)
    return best / 36 * 1e3


for S in (1, 2, 4, 6):
    e, f = run(S, False), run(S, True)
    print(f"S={S}: eager chain {e:.3f} ms/layer | exact fused {f:.3f} ms/layer | saved {e - f:.3f} ms/layer = {36 * (e - f):.1f} ms/pass (36 GDN layers)", flush=True)
