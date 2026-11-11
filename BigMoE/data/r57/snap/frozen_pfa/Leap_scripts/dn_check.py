"""DENSE-N (Leap, Wed 2026-11-04 13:5x; BUILD_DAY_1104 #1, the dense pass; TABLE_350 'DENSE-N', pre-registered 14:0x):
L1-26's fragment-native 4-bit layout (qmm26.qmm16n; repacked ONCE at load, since the dense weights are resident) against
REF2's exact dense call (qmm16c at ks 1, nt 1; ref2._qmm) on the 125B's REF2 4-bit dense set at 16 rows: random weights
at the real shapes, one distinct matrix per call of a check (cold, as the check streams them):
 (1) bits, every shape: qmm16n at (1, 1) / (1, 2) / (1, 4) == qmm16c (1, 1) bitwise (nt only groups tiles per
     threadgroup, each tile keeps its own K walk: exact-safe), and qmm16c (1, 2) / (1, 4) == qmm16c (1, 1);
     the FAST arm nF (native at ks 2) == qmm16c at the same (2, nt) (its own numerics: ks 2 sums two K halves);
 (2) speed: ab28's interleaved A/B (shuffled rounds x median of 7, paired ratios vs c11), ms per check per shape;
     arms c11 (REF2 today), c12, c14, n11, n12, n14 (exact), nF22 (FAST); floor = bytes / 280 GB/s.
Lines: PASS = (1) AND the best exact arm with the native layout saves >= 1.0 ms per check over c11 (one nt for the
whole set: qkv, z, out_proj + o_proj, q_proj, shared down); KILL = not bitwise, or < 0.4 ms; else BETWEEN.
lm_head and R-83's qkv | z concat are reported beside the set (not in the line). FAST is reported against the floor.
Usage: [BigMoE venv] [DN_ROUNDS=10] [DN_COND=...] dn_check.py -> ~/Leap/data/dn_check.txt (GPU, MEM <= 2.5 GB;
quiet lock; ~2 min)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_scripts"))
from qmm16 import qmm16c
from qmm26 import repack, qmm16n

OUT = os.path.expanduser("~/Leap/data/dn_check.txt")
ROUNDS = int(os.environ.get("DN_ROUNDS", "10"))
DRY = os.environ.get("DN_DRY") == "1"                         # shapes / bytes only, no GPU work
if DRY:
    OUT = os.devnull
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")

# (name, N, K, calls per 16-row check, in the line?)
SHAPES = [("in_proj_qkv", 10240, 2560, 36, True), ("in_proj_z", 6144, 2560, 36, True),
          ("out_proj+o_proj", 2560, 6144, 48, True), ("q_proj", 12288, 2560, 12, True),
          ("shared down", 2560, 640, 48, True), ("qkv|z concat (R-83)", 16384, 2560, 36, False),
          ("lm_head", 248320, 2560, 1, False)]
nbytes = lambda N, K: N * K // 2 + 2 * (N * K // 64) * 2     # codes + bf16 scales + biases

def rbits(shape):
    """random 32-bit words (this MLX has no mx.random.bits): two 16-bit draws"""
    hi = mx.random.randint(0, 1 << 16, shape).astype(mx.uint32)
    lo = mx.random.randint(0, 1 << 16, shape).astype(mx.uint32)
    return mx.bitwise_or(mx.left_shift(hi, 16), lo)

def t_once(fn, inner=7):
    mx.eval(fn())
    ts = []
    for _ in range(inner):
        mx.synchronize(); t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    return sorted(ts)[inner // 2] * 1e3

def ab(fns, rounds, seed):
    rng = np.random.default_rng(seed)
    res = {k: [] for k in fns}
    for _ in range(rounds):
        for k in rng.permutation(list(fns)):
            res[k].append(t_once(fns[k]))
    return {k: np.array(v) for k, v in res.items()}

eqb = lambda a, b: bool(a.shape == b.shape and mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())

say(f"\nDENSE-N {time.strftime('%a %H:%M:%S')} ({os.environ.get('DN_COND', 'no lock noted')}); rounds {ROUNDS} x median of 7")
tot = {}
bits_ok = True
mx.random.seed(1104)
for si, (name, N, K, cnt, inline) in enumerate(SHAPES):
    floor = cnt * nbytes(N, K) / 280e9 * 1e3
    if DRY:
        say(f"  {name} {N}x{K} x{cnt}: {cnt * nbytes(N, K) / 1e6:.0f} MB per check, floor {floor:.3f} ms")
        continue
    t0 = time.time()
    Ws = []
    for _ in range(cnt):                                      # random 4-bit codes + affine g64 scales / biases (no fp32
        w = rbits((N, K // 8))                                # N x K temporaries: lm_head stays small)
        s = (mx.random.uniform(0.5, 1.5, (N, K // 64)) * 0.004).astype(mx.bfloat16)
        bb = (-8.0 * s.astype(mx.float32) + mx.random.normal((N, K // 64)) * 0.001).astype(mx.bfloat16)
        W = (w, s, bb)
        mx.eval(W); Ws.append(W)
    Wn = []
    for W in Ws:
        r = repack(*W); mx.eval(r); Wn.append(r)
    tb = time.time() - t0
    x = (mx.random.normal((16, K)) * 0.5).astype(mx.bfloat16)
    mx.eval(x)
    # (1) bits on the first two matrices
    b = {}
    for i in range(min(2, cnt)):
        ref = qmm16c(x, *Ws[i], ks=1, nt=1)
        for nt in (1, 2, 4):
            if N % (32 * nt) == 0:
                b.setdefault(f"n1{nt}", []).append(eqb(qmm16n(x, *Wn[i], N, 1, nt), ref))
                if nt > 1:
                    b.setdefault(f"c1{nt}", []).append(eqb(qmm16c(x, *Ws[i], ks=1, nt=nt), ref))
        if K % 128 == 0 and N % 64 == 0:
            b.setdefault("nF22==c22", []).append(eqb(qmm16n(x, *Wn[i], N, 2, 2), qmm16c(x, *Ws[i], ks=2, nt=2)))
            b.setdefault("c22==c11", []).append(eqb(qmm16c(x, *Ws[i], ks=2, nt=2), ref))
    ok = all(all(v) for k, v in b.items() if k != "c22==c11")
    bits_ok &= ok or not inline
    # (2) speed
    fns = {"c11": lambda: [qmm16c(x, *W, ks=1, nt=1) for W in Ws]}
    for nt in (2, 4):
        if N % (32 * nt) == 0:
            fns[f"c1{nt}"] = (lambda nt_: lambda: [qmm16c(x, *W, ks=1, nt=nt_) for W in Ws])(nt)
    for nt in (1, 2, 4):
        if N % (32 * nt) == 0:
            fns[f"n1{nt}"] = (lambda nt_: lambda: [qmm16n(x, *W, N, 1, nt_) for W in Wn])(nt)
    if K % 128 == 0 and N % 64 == 0:
        fns["nF22"] = lambda: [qmm16n(x, *W, N, 2, 2) for W in Wn]
    res = ab(fns, ROUNDS if cnt > 1 else max(ROUNDS // 2, 3), seed=si)
    med = {k: float(np.median(v)) for k, v in res.items()}
    tot[name] = (med, floor, inline)
    cells = []
    for k, v in res.items():
        r = res["c11"] / v
        cells.append(f"{k} {med[k]:.3f} ({floor / med[k]:.0%}; x{np.median(r):.2f} IQR {np.percentile(r, 25):.2f}-"
                     f"{np.percentile(r, 75):.2f})")
    say(f"  {name} {N}x{K} x{cnt} (floor {floor:.3f} ms; built + repacked in {tb:.1f} s): bits "
        + ", ".join(f"{k} {sum(v)}/{len(v)}" for k, v in b.items()) + f" -> {'OK' if ok else 'DIFF'}")
    say("    ms per check: " + "; ".join(cells))
    del Ws, Wn, fns, res
    mx.clear_cache()
if not DRY:
    line = [n for n, (_, _, il) in tot.items() if il]
    S = lambda arm, names: sum(tot[n][0][arm] if arm in tot[n][0] else tot[n][0]["n11" if arm[0] == "n" else "c11"] for n in names)
    fl = sum(tot[n][1] for n in line)
    arms = ["c11", "c12", "c14", "n11", "n12", "n14", "nF22"]
    say("  the set (" + ", ".join(line) + f"; floor {fl:.2f} ms): " + ", ".join(f"{a} {S(a, line):.2f}" for a in arms)
        + "  (an arm missing for a shape counts that shape at its nt 1 form)")
    best = min(("n11", "n12", "n14"), key=lambda a: S(a, line))
    bestc = min(("c11", "c12", "c14"), key=lambda a: S(a, line))
    save = S("c11", line) - S(best, line)
    v = "KILL" if not bits_ok or save < 0.4 else "PASS" if save >= 1.0 else "BETWEEN"
    say(f"  best exact native {best}: -{save:.2f} ms per check vs c11 ({S('c11', line) / S(best, line):.2f}x; "
        f"{fl / S(best, line):.0%} of the floor); best qmm16c nt {bestc}: -{S('c11', line) - S(bestc, line):.2f} ms; "
        f"FAST nF22 -{S('c11', line) - S('nF22', line):.2f} ms ({fl / S('nF22', line):.0%} of the floor)")
    say(f"DENSE-N verdict: {v} (bits {'OK' if bits_ok else 'DIFF'}; saving {save:.2f} ms per check)")
