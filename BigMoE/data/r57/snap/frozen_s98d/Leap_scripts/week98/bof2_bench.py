"""BOF2 (Leap, Sun 15 Nov 2026 18:1x; CHECK55 track A, 32 rows): BOF's one-weight-pass 32-row twin (bof.qmm16bof2) vs
REF2-T's 32-row forms (v32one: one32.qmm16c2a at ks 1 for N x K >= 8M, else one32.qmm16c2), same 5 shapes as BOF.
Bits: every TERM 0 config (bs, rg) in {(2, 1), (4, 1), (8, 1), (4, 2)} vs qmm16c ks 1 on each 16-row half AND vs REF2-T's
32-row form, at 32 and 24 rows. Timing (quiet): CB1's method at 32 rows (cold pools, dependent chains, 7 rotated rounds):
REF2-T's form vs every BOF2 config vs the contract's ck.qmm16c2a_ks (for reference).
PRE-REGISTERED (check55_plan.md 'BOF2', before any run): not bitwise -> KILL; saving per 32-row check over the 5 shapes
>= 1.5 ms -> PASS (with BOF into E2's C32 arm); 0.5-1.5 BETWEEN; < 0.5 KILL. Prediction: o_proj / out_proj 112 -> 55-65 us;
total -2..-3 ms.
usage: [BigMoE venv] bof2_bench.py [--bits]   (MEM ~1.5 GB; --bits: share lock; else quiet)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98d/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98d/Leap_scripts/week98"))
from qmm16 import qmm16c
from bof import qmm16bof2
import one32
from ck import qmm16c2a_ks

BITS_ONLY = "--bits" in sys.argv
OUT = os.path.expanduser(f"~/Leap/data/week98/screens/bof2{'_bits' if BITS_ONLY else ''}.txt")
BF = mx.bfloat16
POOL = 128 * 2 ** 20
ROUNDS = int(os.environ.get("BOF_ROUNDS", "7"))
t00 = time.time()


def say(s):
    print(s, flush=True)
    with open(OUT, "a") as f:
        f.write(s + "\n")


SHAPES = [("gdn_qz", 16384, 2560, 36), ("o_out", 2560, 6144, 48), ("qsa_qkv", 13312, 2560, 12),
          ("sh_down", 2560, 640, 48), ("lm_head", 248320, 2560, 1)]
CFGS = [(2, 1), (4, 1), (8, 1), (4, 2)]
KC = {640: 10}


def ref32(x, W):                                       # REF2-T's 32-row form (v32one W1)
    N, K = int(W[0].shape[0]), int(x.shape[-1])
    return one32.qmm16c2a(x, *W) if N * K >= 8_000_000 else one32.qmm16c2(x, *W)


def q4_pool(N, K, seed, D):
    k1, k2, k3, k4 = mx.random.split(mx.random.key(seed), 4)
    half = lambda k: mx.random.randint(0, 65536, (D * N, K // 8), dtype=mx.int32, key=k).astype(mx.uint32)
    w = (half(k1) << 16) | half(k4)
    s = (mx.abs(mx.random.normal((D * N, K // 64), key=k2)) * 0.002 + 0.006).astype(BF)
    b = (mx.random.normal((D * N, K // 64), key=k3) * 0.01 - 0.06).astype(BF)
    mx.eval(w, s, b)
    C = [tuple(a[i * N:(i + 1) * N] for a in (w, s, b)) for i in range(D)]
    mx.eval(C)
    return C


def run_chain(fn, x0, n, off, D):
    x = x0
    for j in range(n):
        y = fn(x, (off + j) % D)
        x = x0 + y[:1, :1] * 0
    return x


def run_links(x0, n):
    x = x0
    for _ in range(n):
        x = x0 + x[:1, :1] * 0
    return x


def wall(build):
    out = build()
    mx.synchronize()
    t = time.perf_counter()
    mx.eval(out)
    mx.synchronize()
    return time.perf_counter() - t


def bench(forms, K, n, D, seed):
    rng = np.random.default_rng(seed)
    X = (mx.random.normal((32, K), key=mx.random.key(seed)) * 0.5).astype(BF)
    mx.eval(X)
    nch = max(n, 3)
    for fn in forms.values():
        mx.eval(run_chain(fn, X, 2, 0, D))
    T = {k: [] for k in forms}
    L = []
    names = list(forms)
    for rd in range(ROUNDS):
        L.append(wall(lambda: run_links(X, nch)))
        for name in names[rd % len(names):] + names[:rd % len(names)]:
            off = int(rng.integers(0, D))
            T[name].append(wall(lambda: run_chain(forms[name], X, nch, off, D)))
    lk = float(np.median(L))
    return {k: max(0.0, (float(np.median(v)) - lk) / nch * 1e6) for k, v in T.items()}


say(f"\nBOF2 {'BITS ' if BITS_ONLY else ''}{time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; "
    f"{os.environ.get('BOF_COND', 'share lock')}")
allok = True
POOLS = {}
for si, (key, N, K, cnt) in enumerate(SHAPES):
    D = 1 if BITS_ONLY else max(1, -(-POOL // int(N * K * 0.5625)))
    C = q4_pool(N, K, 400 + si, D)
    POOLS[key] = C
    W = C[0]
    x = (mx.random.normal((32, K), key=mx.random.key(40 + si)) * 0.5).astype(BF)
    good = 0
    for bs, rg in CFGS:
        ok = True
        for M in (32, 24):
            y = qmm16bof2(x[:M], *W, bs=bs, rg=rg)
            h = mx.concatenate([qmm16c(x[:16], *W), qmm16c(x[16:M], *W)])
            ok &= bool(mx.all(y == h).item()) and bool(mx.all(y == ref32(x[:M], W)).item())
        good += ok
    allok &= good == len(CFGS)
    say(f"{key} {N} x {K}: BOF2 bitwise vs qmm16c ks 1 halves and REF2-T's 32-row form, M 32 / 24: {good} / {len(CFGS)}")
if BITS_ONLY or not allok:
    if not allok:
        say("BOF2 verdict (pre-registered): KILL (not bitwise)")
    say(f"done in {time.time() - t00:.0f} s; peak {mx.get_peak_memory() / 1e9:.2f} GB")
    sys.exit(0)
tr, tb, tc = 0.0, 0.0, 0.0
for si, (key, N, K, cnt) in enumerate(SHAPES):
    C = POOLS[key]
    D = len(C)
    kc = KC.get(K, 4)
    forms = {"ref32": lambda x, i: ref32(x, C[i]), "contract": lambda x, i, k=kc: qmm16c2a_ks(x, *C[i], ks=k)}
    for bs, rg in CFGS:
        forms[f"bof2_{bs}x{rg}"] = lambda x, i, b=bs, r=rg: qmm16bof2(x, *C[i], bs=b, rg=r)
    r = bench(forms, K, cnt, D, 4000 + si)
    best = min((k for k in r if k.startswith("bof2")), key=lambda k: r[k])
    tr += cnt * r["ref32"] / 1e3; tb += cnt * r[best] / 1e3; tc += cnt * r["contract"] / 1e3
    say(f"  {key:8s} x{cnt:2d}: REF2-T 32-row {r['ref32']:7.1f} us | " + " ".join(f"{k} {v:.1f}" for k, v in r.items()
                                                                             if k.startswith("bof2")) +
        f" | best {best} {r[best]:.1f} | contract {r['contract']:.1f}")
    POOLS[key] = None
    mx.clear_cache()
d = tb - tr
say(f"per 32-row check over the 5 shapes: REF2-T {tr:.2f} ms -> BOF2 (bitwise) {tb:.2f} ({d:+.2f}); the contract {tc:.2f}")
say("BOF2 verdict (pre-registered): " + ("PASS (>= 1.5 ms)" if -d >= 1.5 else "BETWEEN (0.5-1.5 ms)" if -d >= 0.5 else
                                         "KILL (< 0.5 ms)") + "; standalone, the transfer rule applies")
say(f"done in {time.time() - t00:.0f} s; peak {mx.get_peak_memory() / 1e9:.2f} GB")
