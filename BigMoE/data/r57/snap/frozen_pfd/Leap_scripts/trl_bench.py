"""TRL-G speed (Leap, Wed 2026-11-04 20:3x; pre-registered in TABLE_350 'TRELLIS FULL-STORE PREP (2)'): the grouped trellis
GLU (tierglu_trl.py, the mix: gate / up k 2, down k 3; rotations, grouping and combine included) vs NAX6-N (layer 18's
records3, native repack, plan 4433333333) per call, on layer 18's real routing (Breakthrough's traces125, code / chat),
resident pools (TRL: 512 random-stream experts; decode cost does not depend on the values). k90's method: dependent
chains of 16, interleaved medians of 15 rounds, NP passes per trace. Also TRL-3 (k 3 everywhere) for the record.
Line (S 16, mean of code / chat): PASS <= 1.00x NAX6-N, BETWEEN <= 1.25x, KILL > 1.25x.
usage: [BigMoE venv] trl_bench.py [NP=3] [tag] -> ~/Leap/data/trl_bench.txt (GPU, MEM ~5 GB; share or quiet lock)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_scripts"))
import k90 as H
import tierglu as TG
import tierglu_nax6n as NN
import tierglu_trl as TR

NP = int(sys.argv[1]) if len(sys.argv) > 1 else 3
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
LAYER, PLAN, R, NCH = 18, H.PLAN, 10, 16
OUT = os.path.expanduser("~/Leap/data/trl_bench.txt")


def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")


say(f"\nTRL-G bench {time.strftime('%a %H:%M:%S')} {TAG}: MLX {mx.__version__}; layer {LAYER}, NAX6-N plan {PLAN}")
t00 = time.time()
H.LAYER = LAYER
_tr = H.trace
H.trace = lambda name, layer=LAYER: _tr(name, layer)      # passes() looks trace up by name; its default was 20
H.load(layer=LAYER, native=True, records3=False)
MN, LN = H.L["msb_n"], H.L["lsb_n"]
TA = os.environ.get("TRL_TA") == "1"
NAXM = os.environ.get("TRL_NAX") == "1"
if TA:                                                    # TRL-TA: v2 pools only (MEM: NAX6-N 1.5 + 1.9 + 1.6 GB)
    PM2 = TR.random_pools_v2(E=512, kg=2, kd=3, seed=1, L=LAYER)
    P32 = TR.random_pools_v2(E=512, kg=3, kd=3, seed=2, L=LAYER, share_d=PM2)
    PM = P3 = PM2
else:
    PM = TR.random_pools(E=512, kg=2, kd=3, seed=1, L=LAYER)
    P3 = dict(PM)
    rng = np.random.default_rng(2)
    for nm, (O, I) in (("g", (640, 2560)), ("u", (640, 2560))):
        P3["w" + nm] = mx.array(rng.integers(0, 1 << 32, (512, O // 16, I // 16, 25), dtype=np.uint32))
    P3["kg"] = 3
    mx.eval([v for v in P3.values() if isinstance(v, mx.array)])
_W = ("wg2", "wu2", "wd2") if TA else ("wg", "wu", "wd")
say(f"pools: NAX6-N layer {LAYER} native; TRL mix {sum(PM[k].nbytes for k in _W + ('sg', 'su', 'sd')) / 512 / 1e6:.3f} "
    f"MB per expert, TRL-3 {sum((P32 if TA else P3)[k].nbytes for k in _W + ('sg', 'su', 'sd')) / 512 / 1e6:.3f}; "
    f"records3 {(H.MSB_B) / 1e6:.3f} MB MSB + {H.LSB_B / 1e6:.3f} LSB; loaded in {time.time() - t00:.0f} s")
ARMS = ["NAX6-N", "TRL mix", "TRL-NAX ks4", "TRL-NAX ks2", "TRL-3 NAX ks4"] if NAXM else ["NAX6-N", "TA rp2", "TA rp3", "TA rp4", "TRL-3 TA rp3"] if TA else ["NAX6-N", "TRL mix", "TRL mix A2", "TRL mix A2 rp8", "TRL-3"]
summ = {}
for S in (16, 8):
    for trace in ("code", "chat"):
        acc = {k: [] for k in ARMS}
        for _, ids, ms_, ls_, x0 in H.passes(S, NP, traces=(trace,), seed=300 + S):
            sc = mx.softmax(mx.random.normal((S, R)), axis=-1).astype(mx.bfloat16)
            one = mx.ones((S, R), mx.int32)
            mx.eval(sc, one, x0)
            comb = lambda y: TG.combine_phase(y.reshape(-1, 2560), y.reshape(-1, 2560), one, sc)
            F = {"NAX6-N": lambda xi: comb(NN.tier_glu_nax6n(xi, MN, LN, ms_, ls_, plan=PLAN)),
                 "TRL mix": lambda xi: TR.trl_moe(xi, PM, ms_, sc),
                 "TRL mix A2": lambda xi: TR.trl_moe(xi, PM, ms_, sc, a2=True, nsg_a=2),
                 "TRL mix A2 rp8": lambda xi: TR.trl_moe(xi, PM, ms_, sc, a2=True, nsg_a=2, rp=8),
                 "TRL-3": lambda xi: TR.trl_moe(xi, P3, ms_, sc),
                 "TRL-NAX ks4": lambda xi: TR.trl_moe_nax(xi, PM, ms_, sc, ks=4),
                 "TRL-NAX ks2": lambda xi: TR.trl_moe_nax(xi, PM, ms_, sc, ks=2),
                 "TRL-3 NAX ks4": lambda xi: TR.trl_moe_nax(xi, P3, ms_, sc, ks=4),
                 "TA rp2": lambda xi: TR.trl_moe_ta(xi, PM2, ms_, sc, rp=2),
                 "TA rp3": lambda xi: TR.trl_moe_ta(xi, PM2, ms_, sc, rp=3),
                 "TA rp4": lambda xi: TR.trl_moe_ta(xi, PM2, ms_, sc, rp=4),
                 "TRL-3 TA rp3": lambda xi: TR.trl_moe_ta(xi, P32, ms_, sc, rp=3)}

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
        t = {k: float(np.mean(v)) for k, v in acc.items()}
        summ[(trace, S)] = t
        if TA or NAXM:
            say(f"{trace} S {S:2d}: NAX6-N {1e3 * t['NAX6-N']:.3f} ms | " + ", ".join(f"{k} {1e3 * t[k]:.3f} ms ({t[k] / t['NAX6-N']:.2f}x)" for k in ARMS[1:]))
            continue
        say(f"{trace} S {S:2d}: NAX6-N {1e3 * t['NAX6-N']:.3f} ms, TRL mix {1e3 * t['TRL mix']:.3f} ms "
            f"({t['TRL mix'] / t['NAX6-N']:.2f}x), A2 {1e3 * t['TRL mix A2']:.3f} ms ({t['TRL mix A2'] / t['NAX6-N']:.2f}x), A2 rp8 {1e3 * t['TRL mix A2 rp8']:.3f} ms ({t['TRL mix A2 rp8'] / t['NAX6-N']:.2f}x), TRL-3 {1e3 * t['TRL-3']:.3f} ms ({t['TRL-3'] / t['NAX6-N']:.2f}x)")
VK = ("TRL-NAX ks4", "TRL-NAX ks2") if NAXM else ("TA rp2", "TA rp3", "TA rp4") if TA else ("TRL mix", "TRL mix A2", "TRL mix A2 rp8")
r16 = float(np.mean([min(summ[(tr, 16)][k] for k in VK) / summ[(tr, 16)]["NAX6-N"]
                     for tr in ("code", "chat")]))                      # the better TRL mix kernel (A or A2) per trace
v = "PASS" if r16 <= 1.0 else ("BETWEEN" if r16 <= 1.25 else "KILL")
say(f"{'TRL-NAX' if NAXM else 'TRL-TA' if TA else 'TRL-G'} verdict (S 16, best variant / NAX6-N, mean of code / chat; PASS <= 1.00, BETWEEN <= 1.25, KILL above): {v} ({r16:.2f}x)")
say(f"  ({time.time() - t00:.0f} s; mlx peak {mx.get_peak_memory() / 2**30:.2f} GB)")
