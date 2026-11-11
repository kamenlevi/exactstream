import os, sys, time
os.environ["BIGMOE_TRACE"] = "1"
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import numpy as np, mlx.core as mx
from bigmoe.gated import Gate, ext
fd = os.open(sys.argv[0], os.O_RDONLY)
pool = mx.zeros((64, 1024), mx.uint32); mx.eval(pool)
ext.add_layer(64, 512, [fd], [0], [0], [pool])
gate = Gate(1)
for rep in range(3):
    gate.new_pass()
    g = gate.route_and_gate(0, mx.array(np.array([3 + rep, 9], np.int32)))
    t_ns = time.clock_gettime_ns(time.CLOCK_UPTIME_RAW) * 1e-9
    t = time.perf_counter(); mx.eval(g); dt = time.perf_counter() - t
    tr = [e for e in ext.trace() if e[2] == 1]
    print("  loader saw row at +%.1f ms after eval start" % ((tr[-1][3] - t_ns) * 1e3) if tr else "  no trace")
    sv = np.frombuffer(memoryview(gate.status), np.int32); st = sv[0]
    cnt = np.frombuffer(memoryview(gate.bump), np.int32)[0]
    print(f"  copy kernels run so far {cnt}; spin of this pass saw {sv[1 + (gate.seq & 7)]} at its start")
    print(f"rep {rep}: out {np.array(g).tolist()} elapsed {dt*1e3:.1f} ms, loader gated {ext.stats()['gated']}, timeouts so far {st}", flush=True)
ext.stop()
