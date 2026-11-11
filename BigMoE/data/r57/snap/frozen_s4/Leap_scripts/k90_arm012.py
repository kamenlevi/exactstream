"""K90 arms 0-2 (Leap, Wed 2026-11-04 18:2x; MISSION 33, ~/FableGod/KERNEL90.md section 1): STREAM (the denominator R for
NAX6-N's own access pattern, scripts/k90k.py), NAX6-N (exact after Friday's conversion), NAX6 (row-major; bits reference),
NAX6-NH (FAST), on layer 20's real records (records3 + the native repack) and real routing.
 bits: NAX6-N == NAX6 on 96 cases (traces code / chat / math / bulgarian x S 8 / 16 / 32 x 8 passes);
 timing: the whole call + combine (gpu_grp, rowsum64s, A, B, combine_phase) in dependent chains of 16 (n7_ks.py's method),
   STREAM = kernels A + B on a precomputed grouping; traces code / chat, S 8 / 16 / 32, NP passes each, interleaved rounds,
   medians of 15; per (trace, S): B (record bytes from the routing + x / h / y), groups, ms, GB/s = B / ms, share of R =
   t(STREAM) / t(arm), us per group.
usage: [BigMoE venv] k90_arm012.py [NP=4] [tag] -> ~/Leap/data/k90_arm012.txt (GPU, MEM ~3.5 GB; share lock)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Leap/third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
import k90 as H
import k90k as KK
import tierglu as TG
import tierglu_nax6 as N6
import tierglu_nax6n as NN
import tierglu_nax6h as NH

NP = int(sys.argv[1]) if len(sys.argv) > 1 else 4
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
OUT = os.path.expanduser("~/Leap/data/k90_arm012.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nK90 arms 0-2 {time.strftime('%a %H:%M:%S')} {TAG}: MLX {mx.__version__}; layer {H.LAYER}, plan {H.PLAN}")
t00 = time.time()
H.load(native=True, records3=True)
MS, LS, MN, LN = H.L["msb"], H.L["lsb"], H.L["msb_n"], H.L["lsb_n"]
PLAN, R, NCH = H.PLAN, 10, 16

# ---- bits: NAX6-N == NAX6 on 96 cases ----
nb = neq = 0
for trace in ("code", "chat", "math", "bulgarian"):
    for S in (8, 16, 32):
        for _, ids, ms_, ls_, x in H.passes(S, 8, traces=(trace,), seed=S):
            yn, y6 = H.nax6n(x, ms_, ls_), H.nax6(x, ms_, ls_)
            mx.eval(yn, y6)
            nb += 1
            neq += H.same_bits(yn, y6)
say(f"bits: NAX6-N == NAX6 in {neq} / {nb} cases (4 traces x S 8 / 16 / 32 x 8 passes)")


def xyz(S):
    return S * 2560 * 2 + S * R * 640 * 2 + S * R * 2560 * 2


ARMS = ["STREAM", "NAX6-N", "NAX6", "NAX6-NH"]
summ = {}
for S in (8, 16, 32):
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
                 "NAX6-N": lambda xi: comb(NN.tier_glu_nax6n(xi, MN, LN, ms_, ls_, plan=PLAN)),
                 "NAX6": lambda xi: comb(N6.tier_glu_nax6(xi, MS, LS, ms_, ls_, plan=PLAN)),
                 "NAX6-NH": lambda xi: comb(NH.tier_glu_nax6nh(xi, MN, LN, ms_, ls_, plan=PLAN))}

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
        summ[(trace, S)] = (b, g, t)
        say(f"{trace} S {S:2d}: B {b / 1e6:.0f} MB, {g:.0f} groups; R (STREAM) {b / t['STREAM'] / 1e9:.0f} GB/s ({1e3 * t['STREAM']:.3f} ms) | "
            + " | ".join(f"{k} {1e3 * t[k]:.3f} ms, {b / t[k] / 1e9:.0f} GB/s, {100 * t['STREAM'] / t[k]:.0f} % of R, "
                         f"{1e6 * t[k] / g:.2f} us/group" for k in ARMS[1:]))
say(f"  ({time.time() - t00:.0f} s; mlx peak {mx.get_peak_memory() / 2**30:.2f} GB)")
