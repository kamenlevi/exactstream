"""K90 arm 4, NAX8-P1 (Leap, Wed 2026-11-04 18:3x; MISSION 33, ~/FableGod/KERNEL90.md section 1 arm 4): NAX6-N's kernels
A and B as persistent kernels (scripts/k90k.py tier_glu_nax8p1: work items from device counters zeroed in the grouping
kernel, NAX6-N's inner loop and tails verbatim), grid 20 cores x RES threadgroups, RES_A in {2, 3, 4} x RES_B in {3, 4, 6}.
 bits: P1 (RES 3 / 4) == NAX6 (row-major) on 96 cases (4 traces x S 8 / 16 / 32 x 8 passes); every other RES pair on 6;
 timing: the whole call + combine in dependent chains of 16, NAX6-N vs the 9 P1 configs (+ STREAM for the share of R),
   traces code / chat, S 8 / 16 / 32, NP passes, interleaved, medians of 15.
Lines (Fable A, pre-registered): bits 96 / 96 else BLOCKED; time saved vs NAX6-N at S 16 (best config, mean of code /
chat) >= 6 % PASS, 2-6 % BETWEEN, < 2 % KILL.
usage: [BigMoE venv] k90_p1.py [NP=3] [tag] -> ~/Leap/data/k90_p1.txt (GPU, MEM ~3.5 GB; share lock)"""
import itertools, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_f24k/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_f24k/Leap_scripts"))
import k90 as H
import k90k as KK
import tierglu as TG
import tierglu_nax6 as N6
import tierglu_nax6n as NN

NP = int(sys.argv[1]) if len(sys.argv) > 1 else 3
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
OUT = os.path.expanduser("~/Leap/data/k90_p1.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nK90 arm 4 NAX8-P1 {time.strftime('%a %H:%M:%S')} {TAG}: MLX {mx.__version__}; layer {H.LAYER}, plan {H.PLAN}")
t00 = time.time()
H.load(native=True, records3=True)
MS, LS, MN, LN = H.L["msb"], H.L["lsb"], H.L["msb_n"], H.L["lsb_n"]
PLAN, R, NCH = H.PLAN, 10, 16
CFGS = list(itertools.product((2, 3, 4), (3, 4, 6)))

nb = neq = 0
other = {c: [0, 0] for c in CFGS if c != (3, 4)}
for trace in ("code", "chat", "math", "bulgarian"):
    for S in (8, 16, 32):
        for k_, (_, ids, ms_, ls_, x) in enumerate(H.passes(S, 8, traces=(trace,), seed=S)):
            y6, yp = H.nax6(x, ms_, ls_), KK.tier_glu_nax8p1(x, MN, LN, ms_, ls_, plan=PLAN, res_a=3, res_b=4)
            mx.eval(y6, yp)
            nb += 1
            neq += H.same_bits(yp, y6)
            if k_ == 0 and trace in ("code", "chat"):
                for c in other:
                    yc = KK.tier_glu_nax8p1(x, MN, LN, ms_, ls_, plan=PLAN, res_a=c[0], res_b=c[1])
                    mx.eval(yc)
                    other[c][0] += H.same_bits(yc, y6); other[c][1] += 1
say(f"bits: P1 (RES 3 / 4) == NAX6 in {neq} / {nb}; other RES pairs: " + ", ".join(f"{c} {v[0]} / {v[1]}" for c, v in other.items()))


def xyz(S):
    return S * 2560 * 2 + S * R * 640 * 2 + S * R * 2560 * 2


ARMS = ["STREAM", "NAX6-N"] + [f"P1 {a}/{b}" for a, b in CFGS]
summ = {}
for S in (16, 8, 32):
    for trace in ("code", "chat"):
        acc = {k: [] for k in ARMS}
        bs, gs = [], []
        for _, ids, ms_, ls_, x0 in H.passes(S, NP, traces=(trace,), seed=100 + S):
            grp = N6.gpu_grp(ms_, ls_, PLAN)
            sc = mx.softmax(mx.random.normal((S, R)), axis=-1).astype(mx.bfloat16)
            one = mx.ones((S, R), mx.int32)
            mx.eval(grp, sc, one, x0)
            comb = lambda y: TG.combine_phase(y.reshape(-1, 2560), y.reshape(-1, 2560), one, sc)
            F = {"STREAM": lambda xi: KK.stream(xi, MN, LN, ms_, ls_, grp, PLAN),
                 "NAX6-N": lambda xi: comb(NN.tier_glu_nax6n(xi, MN, LN, ms_, ls_, plan=PLAN))}
            for a, b in CFGS:
                F[f"P1 {a}/{b}"] = (lambda a_, b_: lambda xi: comb(KK.tier_glu_nax8p1(xi, MN, LN, ms_, ls_, plan=PLAN,
                                                                                       res_a=a_, res_b=b_)))(a, b)

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
            bs.append(H.call_bytes(ids) + xyz(S)); gs.append(H.call_groups(ids))
        b, g = float(np.mean(bs)), float(np.mean(gs))
        t = {k: float(np.mean(v)) for k, v in acc.items()}
        summ[(trace, S)] = t
        best = min((k for k in ARMS if k.startswith("P1")), key=t.get)
        say(f"{trace} S {S:2d}: B {b / 1e6:.0f} MB, {g:.0f} groups; R {b / t['STREAM'] / 1e9:.0f} GB/s | NAX6-N {1e3 * t['NAX6-N']:.3f} ms "
            f"({100 * t['STREAM'] / t['NAX6-N']:.0f} % of R) | " + ", ".join(
                f"{k[3:]} {100 * (1 - t[k] / t['NAX6-N']):+.1f} %" for k in ARMS if k.startswith("P1"))
            + f" | best {best[3:]}: {1e3 * t[best]:.3f} ms, {100 * t['STREAM'] / t[best]:.0f} % of R")
sv = {k: float(np.mean([1 - summ[(tr, 16)][k] / summ[(tr, 16)]["NAX6-N"] for tr in ("code", "chat")])) for k in ARMS if k.startswith("P1")}
bk = max(sv, key=sv.get)
v = "BLOCKED (bits)" if neq < nb else ("PASS" if sv[bk] >= 0.06 else "KILL" if sv[bk] < 0.02 else "BETWEEN")
say(f"arm 4 verdict (S 16, mean of code / chat, best config; PASS >= 6 %, KILL < 2 %): {v} ({bk[3:]} {100 * sv[bk]:+.1f} %)")
say(f"  ({time.time() - t00:.0f} s; mlx peak {mx.get_peak_memory() / 2**30:.2f} GB)")
