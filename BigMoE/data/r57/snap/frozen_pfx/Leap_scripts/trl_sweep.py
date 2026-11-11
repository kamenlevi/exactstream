"""TRL-G geometry sweep (Leap, Wed 2026-11-04 20:5x; INFORMATIONAL, after the pre-registered KILL at 1.41x): threadgroup sizes
(nsg_a / nsg_b) and rows per pass for kernel A + B vs NAX6-N, S 16, layer 18 code / chat routing, k90's chained timing.
usage: [BigMoE venv] trl_sweep.py -> ~/Leap/data/trl_sweep.txt (GPU, MEM ~4 GB)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_scripts"))
import k90 as H
import tierglu as TG
import tierglu_nax6n as NN
import tierglu_trl as TR

LAYER, PLAN, R, NCH = 18, H.PLAN, 10, 16
OUT = os.path.expanduser("~/Leap/data/trl_sweep.txt")
def say(s):
    print(s, flush=True); open(OUT, "a").write(s + "\n")
_tr = H.trace
H.trace = lambda name, layer=LAYER: _tr(name, layer)
say(f"\nTRL-G sweep {time.strftime('%a %H:%M:%S')} {sys.argv[1] if len(sys.argv) > 1 else ''}: layer {LAYER} S 16 (informational)")
H.load(layer=LAYER, native=True, records3=False)
MN, LN = H.L["msb_n"], H.L["lsb_n"]
PM = TR.random_pools(E=512, kg=2, kd=3, seed=1, L=LAYER)
GEO = {f"a{a} b{b} rp{rp}": dict(nsg_a=a, nsg_b=b, rp=rp) for a, b, rp in
       ((4, 4, 4), (2, 2, 4), (8, 8, 4), (4, 2, 4), (4, 8, 4), (2, 4, 4), (4, 4, 2), (8, 4, 2))}
ARMS = ["NAX6-N"] + list(GEO)
for trace in ("code", "chat"):
    acc = {k: [] for k in ARMS}
    for _, ids, ms_, ls_, x0 in H.passes(16, 2, traces=(trace,), seed=316):
        sc = mx.softmax(mx.random.normal((16, R)), axis=-1).astype(mx.bfloat16)
        one = mx.ones((16, R), mx.int32)
        mx.eval(sc, one, x0)
        comb = lambda y: TG.combine_phase(y.reshape(-1, 2560), y.reshape(-1, 2560), one, sc)
        F = {"NAX6-N": lambda xi: comb(NN.tier_glu_nax6n(xi, MN, LN, ms_, ls_, plan=PLAN))}
        for k, kw in GEO.items():
            F[k] = (lambda kw: lambda xi: TR.trl_moe(xi, PM, ms_, sc, **kw))(kw)
        def chain(f):
            out = None
            for _ in range(NCH):
                out = f(x0 if out is None else mx.depends(x0, out))
            return out
        for k in ARMS:
            mx.eval(chain(F[k]))
        ts = {k: [] for k in ARMS}
        for r in range(9):
            for k in (ARMS if r % 2 == 0 else ARMS[::-1]):
                mx.synchronize(); t1 = time.perf_counter(); mx.eval(chain(F[k])); mx.synchronize()
                ts[k].append((time.perf_counter() - t1) / NCH)
        for k in ARMS:
            acc[k].append(float(np.median(ts[k])))
    t = {k: float(np.mean(v)) for k, v in acc.items()}
    say(f"{trace} S 16: NAX6-N {1e3 * t['NAX6-N']:.3f} ms | " + ", ".join(f"{k} {t[k] / t['NAX6-N']:.2f}x" for k in GEO))
