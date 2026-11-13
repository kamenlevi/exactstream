"""K90 probes 0b / 0c (Leap, Wed 2026-11-04 18:2x; MISSION 33, Fable A 18:1x): loads-only kernels on layer 20's real
records + routing (scripts/k90k.py stream_probe; every STREAM variant runs its grouping kernel inside the call, so they
compare among themselves; arm 0's R excludes the grouping).
 0b: STREAM with U = 1 / 2 / 4 quant groups of loads issued per simdgroup before the first use (same addresses, grid);
     line (Fable A): R(U 2) >= +8 % over U 1 -> the loads are concurrency-bound -> arm 9 (NAX8-D2); else not;
 0c: STREAM as persistent kernels (P1's loop around the loads-only body), RES 3/4, 4/6, 8/12: the structure's ceiling
     on the load side (informational);
 ext: NAX6-N vs P1 at RES 6/8 and 8/12 (P1's residency trend past the pre-registered sweep; informational).
Dependent chains of 16, interleaved, medians of 15; traces code / chat, S 8 / 16 / 32, NP passes.
usage: [BigMoE venv] k90_probe0.py [NP=2] [tag] -> ~/Leap/data/k90_probe0.txt (GPU, MEM ~3 GB; share lock)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_scripts"))
import k90 as H
import k90k as KK
import tierglu as TG
import tierglu_nax6n as NN

NP = int(sys.argv[1]) if len(sys.argv) > 1 else 2
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
OUT = os.path.expanduser("~/Leap/data/k90_probe0.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nK90 probes 0b / 0c {time.strftime('%a %H:%M:%S')} {TAG}: MLX {mx.__version__}; layer {H.LAYER}")
t00 = time.time()
H.load(native=True, records3=False)
MN, LN = H.L["msb_n"], H.L["lsb_n"]
PLAN, R, NCH = H.PLAN, 10, 16
SARMS = {"S u1": dict(u=1), "S u2": dict(u=2), "S u4": dict(u=4),
         "SP 3/4": dict(res=(3, 4)), "SP 4/6": dict(res=(4, 6)), "SP 8/12": dict(res=(8, 12))}
ARMS = list(SARMS) + ["NAX6-N", "P1 6/8", "P1 8/12"]
summ = {}
for S in (16, 8, 32):
    for trace in ("code", "chat"):
        acc = {k: [] for k in ARMS}
        bs = []
        for _, ids, ms_, ls_, x0 in H.passes(S, NP, traces=(trace,), seed=200 + S):
            sc = mx.softmax(mx.random.normal((S, R)), axis=-1).astype(mx.bfloat16)
            one = mx.ones((S, R), mx.int32)
            mx.eval(sc, one, x0)
            comb = lambda y: TG.combine_phase(y.reshape(-1, 2560), y.reshape(-1, 2560), one, sc)
            F = {k: (lambda kw: lambda xi: KK.stream_probe(xi, MN, LN, ms_, ls_, PLAN, **kw))(kw) for k, kw in SARMS.items()}
            F["NAX6-N"] = lambda xi: comb(NN.tier_glu_nax6n(xi, MN, LN, ms_, ls_, plan=PLAN))
            F["P1 6/8"] = lambda xi: comb(KK.tier_glu_nax8p1(xi, MN, LN, ms_, ls_, plan=PLAN, res_a=6, res_b=8))
            F["P1 8/12"] = lambda xi: comb(KK.tier_glu_nax8p1(xi, MN, LN, ms_, ls_, plan=PLAN, res_a=8, res_b=12))

            def chain(f):
                out = None
                for _ in range(NCH):
                    out = f(x0 if out is None else mx.depends(x0, out))
                return out
            for k in ARMS:
                mx.eval(chain(F[k]))
            ts = {k: [] for k in ARMS}
            for r in range(15):
                for k in (ARMS if r % 2 == 0 else ARMS[::-1]):
                    mx.synchronize(); t1 = time.perf_counter(); mx.eval(chain(F[k])); mx.synchronize()
                    ts[k].append((time.perf_counter() - t1) / NCH)
            for k in ARMS:
                acc[k].append(float(np.median(ts[k])))
            bs.append(H.call_bytes(ids))
        b = float(np.mean(bs))
        t = {k: float(np.mean(v)) for k, v in acc.items()}
        summ[(trace, S)] = t
        say(f"{trace} S {S:2d}: B {b / 1e6:.0f} MB | " + ", ".join(f"{k} {b / t[k] / 1e9:.0f} GB/s" for k in SARMS)
            + f" | 0b: u2 {100 * (t['S u1'] / t['S u2'] - 1):+.1f} %, u4 {100 * (t['S u1'] / t['S u4'] - 1):+.1f} %"
            + f" | 0c best SP {100 * (t['S u1'] / min(t[k] for k in SARMS if k.startswith('SP')) - 1):+.1f} %"
            + f" | ext: NAX6-N {1e3 * t['NAX6-N']:.3f} ms, P1 6/8 {100 * (1 - t['P1 6/8'] / t['NAX6-N']):+.1f} %, "
              f"P1 8/12 {100 * (1 - t['P1 8/12'] / t['NAX6-N']):+.1f} %")
g16 = float(np.mean([summ[(tr, 16)]["S u1"] / summ[(tr, 16)]["S u2"] - 1 for tr in ("code", "chat")]))
say(f"0b verdict (S 16, R(u2) vs R(u1), mean of code / chat; arm 9 if >= +8 %): {'arm 9' if g16 >= 0.08 else 'no arm 9'} ({100 * g16:+.1f} %)")
say(f"  ({time.time() - t00:.0f} s; mlx peak {mx.get_peak_memory() / 2**30:.2f} GB)")
