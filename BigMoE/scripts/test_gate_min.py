"""Small gate check: NLT chained layers, loader delayed 20 ms each. Correct gate => results ==
loader slot ids, elapsed >= NLT*20 ms. Layer L+1's ids depend on layer L's gated output."""
import os, sys, time
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import numpy as np, mlx.core as mx
from bigmoe.gated import Gate, ext
NLT = int(os.environ.get("NLT", 4))
fd = os.open(sys.argv[0], os.O_RDONLY)
pools = [mx.zeros((64, 1024), mx.uint32) for _ in range(NLT)]; mx.eval(*pools)
for L in range(NLT): ext.add_layer(64, 512, [fd], [0], [0], [pools[L]])
gate = Gate(NLT)
for rep in range(3):
    gate.new_pass(); outs = []; dep = mx.array([0], mx.int32); idl = []
    for L in range(NLT):
        ids_np = np.array([3 * L + rep, 200 + L, 400 + rep], np.int32); idl.append(ids_np)
        ids = mx.array(ids_np) + (dep.sum() * 0)                 # depend on previous layer's output
        g = gate.route_and_gate(L, ids); o = g + 0; outs.append(o); dep = o
    t = time.perf_counter(); mx.eval(*outs); dt = time.perf_counter() - t
    good = sum(np.array(outs[L]).tolist() == np.array(ext.ensure(L, mx.array(idl[L]))).tolist() for L in range(NLT))
    print(f"rep {rep}: {good}/{NLT} layers correct, elapsed {dt*1e3:.0f} ms", flush=True)
gate.check(); gate.stop()
