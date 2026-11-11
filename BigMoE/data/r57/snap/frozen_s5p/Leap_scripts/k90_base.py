"""K90-0 baseline (Leap, Wed 2026-11-04 18:1x; MISSION 33): NAX6 and NAX6-N (the exact kernels) on layer 20's real records
and routing, per (trace, S): the call's record bytes, (expert, tier) groups, ms per call, GB/s, us per group, and the share
of the machine's streaming rate (C2: 258-287 GB/s; 284 used). The 90 % line per call = bytes / (0.9 x 284 GB/s).
usage: [BigMoE venv] k90_base.py [S=8,16,32] [passes=6] -> ~/Leap/data/k90_base.txt (GPU, MEM ~3.5 GB; share lock)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5p/Leap_scripts"))
import k90 as H

SS = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "8,16,32").split(",")]
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 6
OUT = os.path.expanduser("~/Leap/data/k90_base.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nK90-0 baseline {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}; layer {H.LAYER}, plan {H.PLAN}")
t0 = time.time()
H.load(native=True, records3=True)
say(f"  loaded L{H.LAYER} records3 + native in {time.time() - t0:.0f} s")
BW = 284e9
ok = True
for S in SS:
    for name in ("code", "chat"):
        rows = []
        for tr, ids, ms_, ls_, x in H.passes(S, NP, traces=(name,)):
            y6, yn = H.nax6(x, ms_, ls_), H.nax6n(x, ms_, ls_)
            mx.eval(y6, yn)
            ok &= H.same_bits(y6, yn)
            t = H.timeit({"NAX6": lambda: H.nax6(x, ms_, ls_), "NAX6-N": lambda: H.nax6n(x, ms_, ls_)})
            rows.append((H.call_bytes(ids), H.call_groups(ids), t["NAX6"], t["NAX6-N"]))
        b, g, t6, tn = (float(np.mean([r[i] for r in rows])) for i in range(4))
        say(f"{name} S {S:2d}: {b / 1e6:.0f} MB, {g:.0f} groups per call; NAX6 {1e3 * t6:.3f} ms ({b / t6 / 1e9:.0f} GB/s, "
            f"{1e6 * t6 / g:.2f} us/group, {100 * b / t6 / BW:.0f} %), NAX6-N {1e3 * tn:.3f} ms ({b / tn / 1e9:.0f} GB/s, "
            f"{1e6 * tn / g:.2f} us/group, {100 * b / tn / BW:.0f} %); 90 % line {1e3 * b / (0.9 * BW):.3f} ms")
say(f"  NAX6-N == NAX6 bitwise on every pass: {ok}; mlx peak {mx.get_peak_memory() / 2**30:.2f} GB; {time.time() - t0:.0f} s")
