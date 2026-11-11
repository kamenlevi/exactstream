"""DN-KS step 1 (Leap, Fri 2026-11-06 08:1x; TABLE_350 'STEP 0 PART B RESULT', pre-registered there): does a per-shape
K split (qmm26.qmm16n's ks) fix the narrow native GEMVs that part B found K-serial at ks 1?
Kinds (real shapes, 4-bit g64, native layout; calls per 2-row check): out_proj / o_proj N 2560 K 6144 (48), a|b N 96 K 2560
(36), attention k / v N 512 K 2560 (24), shared gate|up N 1312 K 2560 (48), shared down N 2560 K 640 (48); controls: in_proj
N 16384 K 2560 (36), attention q N 12288 K 2560 (12).
Per kind and ks (round-robin over ks, DNKS_ROUNDS rounds, median): a dependent chain over distinct weight copies (>= 64 MB of them, so reads are cold as in the check),
link = x0 + 0 x the last output's first value (3 tiny kernels), per call = (chain - the links alone) / n, median of 5.
Bits: max |y_ks - y_1| / max |y_1| per kind (the FAST line: <= 2e-3).
Line (pre-registered 08:0x): PASS >= 5.0 ms per 2-row check saved over the 5 narrow kinds; KILL < 2.5. Prediction -6..-8.
-> ~/Leap/data/dnks_bench.txt. Share lock, MEM < 1.5 GB."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs4/Leap_scripts"))
import mlx.core as mx
import qmm26

OUT = os.path.expanduser("~/Leap/data/dnks_bench.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")

KINDS = [("out_proj/o_proj", 2560, 6144, 48), ("a|b", 96, 2560, 36), ("att k/v", 512, 2560, 24),
         ("shared gate|up", 1312, 2560, 48), ("shared down", 2560, 640, 48), ("in_proj (control)", 16384, 2560, 36),
         ("att q (control)", 12288, 2560, 12)]
clock = time.perf_counter
ROUNDS = int(os.environ.get("DNKS_ROUNDS", "9"))
say(f"\nDN-KS step 1 {time.strftime('%a %H:%M:%S')} (MLX {mx.__version__}; {os.environ.get('DNKS_COND', 'share lock')})")
rng = np.random.default_rng(7)


def make(N, K, n):
    """n distinct native-layout weight copies of [N, K] at 4-bit g64 (random bf16 quantized by MLX)"""
    out = []
    for i in range(n):
        w = mx.array(rng.standard_normal((N, K)).astype(np.float32) * 0.02).astype(mx.bfloat16)
        q, s, b = mx.quantize(w, group_size=64, bits=4)
        wn, sn, bn = qmm26.repack(q, s, b)
        mx.eval(wn, sn, bn)
        out.append((wn, sn, bn))
        del w, q, s, b
    return out


def chain(W, N, x0, ks, reps=5):
    def build(calls):
        prev = mx.zeros((1,), mx.float32)
        for wn, sn, bn in W:
            x = x0 + (prev * 0).astype(x0.dtype)
            if calls:
                y = qmm26.qmm16n(x, wn, sn, bn, N, ks, 1)
                prev = y.reshape(-1)[:1].astype(mx.float32)
            else:
                prev = x.reshape(-1)[:1].astype(mx.float32)
        return prev
    tc, tn = [], []
    for _ in range(reps):
        for calls, acc in ((False, tn), (True, tc)):
            p = build(calls)
            t0 = clock(); mx.eval(p); acc.append(clock() - t0)
    return 1e6 * (float(np.median(tc)) - float(np.median(tn))) / len(W)


res = {}
x0 = mx.array(rng.standard_normal((2, 6144)).astype(np.float32)).astype(mx.bfloat16)
mx.eval(x0)
for name, N, K, calls in KINDS:
    mb = (N * K // 2 + 2 * N * (K // 64) * 2) / 1e6
    n = int(min(300, max(12, np.ceil(64.0 / mb))))
    if name.startswith("in_proj") or name.startswith("att q"):
        n = 12
    W = make(N, K, n)
    x = x0[:, :K]
    mx.eval(x)
    g = K // 64
    cands = [k for k in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16) if g % k == 0 and 32 * k <= 1024]
    y1 = qmm26.qmm16n(x, *W[0], N, 1, 1).astype(mx.float32)
    mx.eval(y1)
    errs = {}
    for ks in cands:
        yk = qmm26.qmm16n(x, *W[0], N, ks, 1).astype(mx.float32)
        errs[ks] = float(mx.max(mx.abs(yk - y1)).item() / max(1e-30, mx.max(mx.abs(y1)).item()))
    for ks in cands:                                     # warm-up: every variant compiled and run once
        chain(W, N, x, ks, reps=1)
    acc = {ks: [] for ks in cands}
    for r_ in range(ROUNDS):                             # round-robin: slow drifts hit every ks alike
        for ks in (cands if r_ % 2 == 0 else cands[::-1]):
            acc[ks].append(chain(W, N, x, ks, reps=1))
    row = {ks: (float(np.median(acc[ks])), errs[ks]) for ks in cands}
    best = min(row, key=lambda k: row[k][0])
    res[name] = (calls, mb, row, best)
    say(f"  {name:18s} N {N:5d} K {K:5d} {mb:6.2f} MB x {n:3d} copies, {calls} calls per check: " + "; ".join(
        f"ks {k}: {row[k][0]:6.1f} us ({mb * 1e-3 / max(1e-9, row[k][0] * 1e-6):4.0f} GB/s, err {row[k][1]:.1e})" for k in cands)
        + f" -> best ks {best}")
    del W
    mx.clear_cache()
narrow = [k for k in res if "control" not in k]
sav = sum(res[k][0] * (res[k][2][1][0] - res[k][2][res[k][3]][0]) for k in narrow) / 1e3
ctl = sum(res[k][0] * (res[k][2][1][0] - res[k][2][res[k][3]][0]) for k in res if "control" in k) / 1e3
bad = [k for k in res if res[k][2][res[k][3]][1] > 2e-3]
verdict = "PASS" if sav >= 5.0 and not bad else ("KILL" if sav < 2.5 else "BETWEEN")
say(f"  saved per 2-row check over the 5 narrow kinds at their best ks: {sav:.2f} ms (controls: {ctl:.2f} ms more); "
    f"bits over 2e-3: {bad or 'none'}")
say(f"RESULT {verdict} (line: PASS >= 5.0 ms and bits <= 2e-3, KILL < 2.5 ms); best ks: "
    + ", ".join(f"{k} {res[k][3]}" for k in res))
