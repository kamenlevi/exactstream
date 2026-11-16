"""qmm32n_bench.py (Leap, Mon 16 Nov 2026 12:0x; CHECK25 step 1; doubles as the WEEK98 "32 vs 16 rows" microbench):
per FAST dense shape, the per-check set of DISTINCT native 4-bit weights (48-layer call counts, so every weight is read
cold, as in a check), timed in 7 interleaved rounds (median of the last 6) for:
  16-row  one qmm16n_m call on 16 rows (A16's cost)
  split   two qmm16n_m calls (rows 0-15, 16-31): the only > 16-row FAST form without the DNR unpack
  q32n    qmm32n at DN-KS v1's ks (bitwise == split by construction)
  q32n kN qmm32n at other ks (FAST: the K order changes vs the 16-row call; reported, not adopted)
Two modes: 'ind' = the calls independent in one graph (one32_bench's method; overlap allowed), 'chain' = each call's
input depends on the previous call's output through a ~2-3 us link op (as in a forward), so it is latency as in a check.
Reported per shape: ms per 32-row check for that shape's calls, us per call, and x the byte floor (280 GB/s) and x the
matrix floor (14.0 T MAC/s bf16, M27-P) at 32 rows. Weights are random native words + small bf16 scales / biases (timing
only; bits and errors are qmm32n.check()'s).
usage: [BigMoE venv] qmm32n_bench.py [ind|chain|both] [shape substring]   (MEM ~2 GB; quiet ticket for clean ms)"""
import os
import sys
import time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts/week98"))
from qmm26m import qmm16n_m
import qmm32n as Q
import dn32 as DN

BF = mx.bfloat16
ROUNDS = 7
BW, MMA = 280e9, 14.0e12
OUT = os.path.expanduser("~/Leap/data/week98/check25/qmm32n_bench.txt")
L = []


def say(s):
    print(s, flush=True)
    L.append(s)


def native(N, K, key):
    k1, k2, k3 = mx.random.split(key, 3)
    wn = mx.random.randint(0, 2 ** 31 - 1, (N // 32, K // 64, 32, 8), key=k1).astype(mx.uint32)
    sn = (mx.random.uniform(0.002, 0.004, (N // 32, K // 64, 32), key=k2)).astype(BF)
    bn = (mx.random.uniform(-0.02, 0.0, (N // 32, K // 64, 32), key=k3)).astype(BF)
    return wn, sn, bn


def run(forms, ws, x, chain):
    T = {f: [] for f in forms}
    names = list(forms)
    for r in range(ROUNDS):
        order = names[r % len(names):] + names[:r % len(names)]
        for f in order:
            mx.synchronize()
            t0 = time.perf_counter()
            if chain:
                h = x
                for W in ws:
                    y = forms[f](h, W)
                    h = x + (y[:1, :1] * 0).astype(BF)         # the link: the next call waits for this one
                mx.eval(h)
            else:
                mx.eval([forms[f](x, W) for W in ws])
            mx.synchronize()
            T[f].append(time.perf_counter() - t0)
    return {f: 1e3 * float(np.median(T[f][1:])) for f in forms}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    sub = sys.argv[2] if len(sys.argv) > 2 else ""
    modes = ("ind", "chain") if mode == "both" else (mode,)
    say(f"QMM32N-BENCH {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; {ROUNDS} interleaved rounds, median of "
        f"the last {ROUNDS - 1}; ms per 32-row check for each shape's calls (cold distinct weights); modes {modes}")
    x_all = (mx.random.normal((32, 6144), dtype=BF, key=mx.random.key(5)) * 0.7)
    key = mx.random.key(1116)
    tot = {}
    for name, N, K, cnt in Q.SHAPES:
        if sub and sub not in name:
            continue
        ks0 = Q.ks_of(N, K)
        g = K // 64
        kss = [k for k in (1, 2, 4, 5, 8, 10, 16) if g % k == 0 and k != ks0 and k <= 16]
        ws = []
        for i in range(cnt):
            key, k_ = mx.random.split(key)
            ws.append(native(N, K, k_))
        mx.eval(ws)
        x = x_all[:, :K]
        forms = {
            "16-row": lambda x, W, N=N, ks=ks0: qmm16n_m(x[:16], *W, N, ks),
            "split": lambda x, W, N=N, ks=ks0: mx.concatenate([qmm16n_m(x[:16], *W, N, ks), qmm16n_m(x[16:], *W, N, ks)], 0),
            f"q32n k{ks0}": lambda x, W, N=N, ks=ks0: Q.qmm32n(x, *W, N, ks),
        }
        for k in kss:
            forms[f"q32n k{k}"] = (lambda x, W, N=N, ks=k: Q.qmm32n(x, *W, N, ks))
        # dn32 / dn32q (dn32.py): the same random words serve the NAX-dense layout (timing only); dn32q's 'q' forms
        # include the per-call activation quantizer (quant64), its plain forms read a pre-quantized x (shared input)
        xq0 = DN.quant64(x_all[:, :K])
        mx.eval(xq0)
        for k in [k for k in (1, 2, 4, 5, 8, 10) if g % k == 0 and DN._tg_ok(k, 1)]:
            forms[f"dn32 k{k}"] = (lambda x, W, N=N, ks=k: DN.dn32(x, *W, N, ks))
            forms[f"dn32q k{k}"] = (lambda x, W, N=N, ks=k, xq0=xq0: DN.dn32q(*xq0, *W, N, ks))
            forms[f"dn32q+q k{k}"] = (lambda x, W, N=N, ks=k: DN.dn32q(*DN.quant64(x), *W, N, ks))
            forms[f"dn32n k{k}"] = (lambda x, W, N=N, ks=k: DN.dn32(x, *W, N, ks, native=True))
            forms[f"dn32qn+q k{k}"] = (lambda x, W, N=N, ks=k: DN.dn32q(*DN.quant64(x), *W, N, ks, native=True))
        by = N * K * 0.5625
        fl_b = by / BW * 1e6
        fl_m = 32 * N * K / MMA * 1e6
        for md in modes:
            fm_ = {k: v for k, v in forms.items() if not (md == "chain" and k.startswith("dn32q k"))}   # no chain input
            r = run(fm_, ws, x, md == "chain")
            best = min((v, f) for f, v in r.items() if f.startswith("q32n"))
            bdn = min((v, f) for f, v in r.items() if f.startswith("dn32 "))
            bdq = min((v, f) for f, v in r.items() if f.startswith("dn32q+q"))
            bdq0 = min(((v, f) for f, v in r.items() if f.startswith("dn32q k")), default=(float("nan"), "-"))
            bdn_n = min((v, f) for f, v in r.items() if f.startswith("dn32n "))
            bdq_n = min((v, f) for f, v in r.items() if f.startswith("dn32qn+q"))
            say(f"  [{md:5s}] {name:34s} N {N:6d} K {K:5d} x{cnt:2d} (floors per call: bytes {fl_b:6.1f} us, matrix@32 {fl_m:6.1f} us)")
            say("          " + " | ".join(f"{f} {v:7.3f} ms ({1e3 * v / cnt:6.1f} us)" for f, v in r.items()))
            for lab, bb in (("qmm32n", best), ("dn32", bdn), ("dn32 native", bdn_n), ("dn32q+quant", bdq),
                            ("dn32q native+quant", bdq_n), ("dn32q pre-quantized", bdq0)):
                if bb[1] == "-":
                    continue
                say(f"          best {lab:20s} {bb[1]:12s}: {1e3 * bb[0] / cnt:7.1f} us/call = x{1e3 * bb[0] / cnt / fl_b:.2f} "
                    f"bytes, x{1e3 * bb[0] / cnt / max(fl_b, fl_m):.2f} max(bytes, matrix@32); split / it {r['split'] / bb[0]:.2f}")
            t = tot.setdefault(md, {"16-row": 0.0, "split": 0.0, "q32n bitwise": 0.0, "q32n best": 0.0, "dn32 best": 0.0,
                                    "dn32 native best": 0.0, "dn32q+q best": 0.0, "dn32q native+q best": 0.0,
                                    "dn32q pre-q best": 0.0, "floor": 0.0})
            t["16-row"] += r["16-row"]
            t["split"] += r["split"]
            t["q32n bitwise"] += r[f"q32n k{ks0}"]
            t["q32n best"] += best[0]
            t["dn32 best"] += bdn[0]
            t["dn32 native best"] += bdn_n[0]
            t["dn32q native+q best"] += bdq_n[0]
            t["dn32q+q best"] += bdq[0]
            t["dn32q pre-q best"] += bdq0[0]
            t["floor"] += cnt * max(fl_b, fl_m) / 1e3
        del ws
        mx.clear_cache()
    for md, t in tot.items():
        say(f"TOTAL [{md}] these calls per check: " + " | ".join(f"{k} {v:.2f} ms" for k, v in t.items()))
    say(f"(peak {mx.get_peak_memory() / 1e9:.1f} GB)")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "a").write("\n".join(L) + "\n\n")


if __name__ == "__main__":
    main()
