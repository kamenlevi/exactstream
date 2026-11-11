"""K90 single-arm runner (Leap, Wed 2026-11-04 18:3x; MISSION 33): one candidate kernel vs NAX6-N on layer 20's real
records + routing. bits: candidate == NAX6 (row-major) on 96 cases (4 traces x S 8 / 16 / 32 x 8 passes; S 32 skipped for
arms limited to S <= 16); timing: the whole call + combine in dependent chains of 16, NAX6-N / candidate / STREAM,
traces code / chat, interleaved, medians of 15. Lines (Fable A): arm 7 GU PASS <= -4 %, KILL > -1 %; arms 8a / 8b PASS
<= -5 %, BETWEEN -2..-5, KILL > -2 % (time vs NAX6-N at S 16, mean of code / chat).
usage: [BigMoE venv] k90_arm.py ARM [NP=3] [tag]   (ARM: gu | 8b | 8a) -> ~/Leap/data/k90_arm_<ARM>.txt (GPU, MEM ~3.5 GB)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_scripts"))
import k90 as H
import k90k as KK
import tierglu as TG
import tierglu_nax6 as N6
import tierglu_nax6n as NN

ARM = sys.argv[1]
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 3
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
FN = {"gu": KK.tier_glu_nax8gu, "8b": getattr(KK, "tier_glu_nax8b", None), "8a": getattr(KK, "tier_glu_nax8a", None), "mf": getattr(KK, "tier_glu_nax8mf", None), "lut": getattr(KK, "tier_glu_nax8lut", None), "lutd": getattr(KK, "tier_glu_nax8lutd", None), "pf6": getattr(KK, "tier_glu_nax8pf6", None)}[ARM]
SMAX = {"gu": 32, "8b": 16, "8a": 32, "mf": 32, "lut": 32, "lutd": 32, "pf6": 32}[ARM]
LINE = {"gu": (0.04, 0.01), "8b": (0.04, 0.01), "8a": (0.04, 0.01), "mf": (0.04, 0.01), "lut": (0.04, 0.01), "lutd": (0.04, 0.01), "pf6": (0.05, 0.01)}[ARM]   # Fable A 18:2x: every arm PASS >= 4 %, KILL < 1 %
OUT = os.path.expanduser(f"~/Leap/data/k90_arm_{ARM}.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nK90 arm {ARM} {time.strftime('%a %H:%M:%S')} {TAG}: MLX {mx.__version__}; layer {H.LAYER}, plan {H.PLAN}")
t00 = time.time()
H.load(native=True, records3=True)
MS, LS, MN, LN = H.L["msb"], H.L["lsb"], H.L["msb_n"], H.L["lsb_n"]
PLAN, R, NCH = H.PLAN, 10, 16
SS = [s for s in (8, 16, 32) if s <= SMAX]

nb = neq = 0
for trace in ("code", "chat", "math", "bulgarian"):
    for S in SS:
        for _, ids, ms_, ls_, x in H.passes(S, 8, traces=(trace,), seed=S):
            y6, yc = H.nax6(x, ms_, ls_), FN(x, MN, LN, ms_, ls_, plan=PLAN)
            mx.eval(y6, yc)
            nb += 1
            neq += H.same_bits(yc, y6)
say(f"bits: {ARM} == NAX6 in {neq} / {nb} (4 traces x S {SS} x 8 passes)")

ARMS = ["STREAM", "NAX6-N", ARM]
summ = {}
for S in SS:
    for trace in ("code", "chat"):
        acc = {k: [] for k in ARMS}
        bs = []
        for _, ids, ms_, ls_, x0 in H.passes(S, NP, traces=(trace,), seed=100 + S):
            grp = N6.gpu_grp(ms_, ls_, PLAN)
            sc = mx.softmax(mx.random.normal((S, R)), axis=-1).astype(mx.bfloat16)
            one = mx.ones((S, R), mx.int32)
            mx.eval(grp, sc, one, x0)
            comb = lambda y: TG.combine_phase(y.reshape(-1, 2560), y.reshape(-1, 2560), one, sc)
            F = {"STREAM": lambda xi: KK.stream(xi, MN, LN, ms_, ls_, grp, PLAN),
                 "NAX6-N": lambda xi: comb(NN.tier_glu_nax6n(xi, MN, LN, ms_, ls_, plan=PLAN)),
                 ARM: lambda xi: comb(FN(xi, MN, LN, ms_, ls_, plan=PLAN))}

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
        say(f"{trace} S {S:2d}: NAX6-N {1e3 * t['NAX6-N']:.3f} ms ({100 * t['STREAM'] / t['NAX6-N']:.0f} % of R), {ARM} "
            f"{1e3 * t[ARM]:.3f} ms ({100 * t['STREAM'] / t[ARM]:.0f} % of R): {100 * (1 - t[ARM] / t['NAX6-N']):+.1f} % "
            f"(R {b / t['STREAM'] / 1e9:.0f} GB/s)")
s16 = float(np.mean([1 - summ[(tr, 16)][ARM] / summ[(tr, 16)]["NAX6-N"] for tr in ("code", "chat")]))
v = "BLOCKED (bits)" if neq < nb else ("PASS" if s16 >= LINE[0] else "KILL" if s16 < LINE[1] else "BETWEEN")
say(f"arm {ARM} verdict (S 16, mean of code / chat; PASS >= {100 * LINE[0]:.0f} % saved, KILL < {100 * LINE[1]:.0f} %): {v} ({100 * s16:+.1f} %)")
say(f"  ({time.time() - t00:.0f} s; mlx peak {mx.get_peak_memory() / 2**30:.2f} GB)")
