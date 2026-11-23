"""BOF (Leap, Sun 15 Nov 2026 17:5x; CHECK55 16-row lever (b), track A): is qmm16c ks 1 with a blocked ordered fold
(bof.qmm16bof) bitwise with REF2-T's qmm16c ks 1, and how much faster on the narrow shapes?
Bits: every TERM variant x (bs, rg) in {(4, 1), (8, 1), (8, 2), (16, 1)} vs qmm16c(ks 1) at 16 rows (+ random-padding rows), on
the 5 shapes REF2-T sends to qmm16c ks 1: GDN qkv | z 16,384 x 2,560 (36 per check), o_proj / out_proj 2,560 x 6,144 (48),
QSA qkv_fused 13,312 x 2,560 (12), shared down 2,560 x 640 (48), lm_head 248,320 x 2,560 (1).
Timing (quiet ticket): CB1's method (cold pools >= 128 MB, dependent chains of a check's calls, (chain - links) / n,
7 rotated rounds): qmm16c ks 1 (REF2-T) vs every bitwise BOF config vs qmm16c ks 4 (the contract's, for reference).
PRE-REGISTERED (check55_plan.md 'BOF', before any run):
  no TERM variant bitwise on all 5 shapes -> KILL (REF2-T's arithmetic can't be matched this way);
  else saving per 16-row check over the 5 shapes (best bitwise config per shape): >= 1.5 ms PASS (a ref2 switch, a
  slice A/B Monday, then E1); 0.5-1.5 BETWEEN; < 0.5 KILL.
  prediction: TERM 0 bitwise; o_proj / out_proj 87 -> 50-60 us; in-proj 118 -> 105-110; total -1.8..-2.5 ms.
usage: [BigMoE venv] bof_bench.py [--bits]   (MEM ~1 GB; --bits: bits only, share lock; else bits + timing, quiet)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98k/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98k/Leap_scripts/week98"))
from qmm16 import qmm16c
from bof import qmm16bof

BITS_ONLY = "--bits" in sys.argv
OUT = os.path.expanduser(f"~/Leap/data/week98/screens/bof{'_bits' if BITS_ONLY else ''}.txt")
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
CFGS = [(4, 1), (8, 1), (8, 2), (16, 1)]


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
    X = (mx.random.normal((16, K), key=mx.random.key(seed)) * 0.5).astype(BF)
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


say(f"\nBOF {'BITS ' if BITS_ONLY else ''}{time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; "
    f"{os.environ.get('BOF_COND', 'share lock')}")
OK = {t: True for t in range(4)}
POOLS = {}
for si, (key, N, K, cnt) in enumerate(SHAPES):
    D = 1 if BITS_ONLY else max(1, -(-POOL // int(N * K * 0.5625)))
    C = q4_pool(N, K, 300 + si, D)
    POOLS[key] = C
    W = C[0]
    x = (mx.random.normal((16, K), key=mx.random.key(30 + si)) * 0.5).astype(BF)
    pad = (mx.random.normal((15, K), key=mx.random.key(31 + si)) * 0.5).astype(BF)
    ref = qmm16c(x, *W)
    ref1 = qmm16c(x[:1], *W)
    line = []
    for term in range(4):
        good = 0
        for bs, rg in CFGS:
            ok = (bool(mx.all(qmm16bof(x, *W, bs=bs, rg=rg, term=term) == ref).item())
                  and bool(mx.all(qmm16bof(mx.concatenate([x[:1], pad]), *W, bs=bs, rg=rg, term=term)[:1] == ref1).item()))
            good += ok
        OK[term] &= good == len(CFGS)
        line.append(f"TERM {term}: {good} / {len(CFGS)}")
    say(f"{key} {N} x {K}: bitwise vs qmm16c ks 1 (configs, incl. random padding): " + "; ".join(line))
bit_terms = [t for t, v in OK.items() if v]
say(f"TERM variants bitwise on every shape and config: {bit_terms if bit_terms else 'NONE'}")
if BITS_ONLY or not bit_terms:
    if not bit_terms:
        say("BOF verdict (pre-registered): KILL (no TERM variant matches qmm16c ks 1's compiled arithmetic)")
    say(f"done in {time.time() - t00:.0f} s; peak {mx.get_peak_memory() / 1e9:.2f} GB")
    sys.exit(0)
term = bit_terms[0]
tot_ref, tot_bof, tot_c4 = 0.0, 0.0, 0.0
for si, (key, N, K, cnt) in enumerate(SHAPES):
    C = POOLS[key]
    D = len(C)
    kc = 10 if K == 640 else 4                        # the contract's ks (CB1): shared down K 640 takes ks 10
    forms = {"ref_ks1": lambda x, i: qmm16c(x, *C[i]), "contract_ks4": lambda x, i, k=kc: qmm16c(x, *C[i], ks=k)}
    for bs, rg in CFGS:
        forms[f"bof{bs}x{rg}"] = lambda x, i, b=bs, r=rg: qmm16bof(x, *C[i], bs=b, rg=r, term=term)
    r = bench(forms, K, cnt, D, 3000 + si)
    best = min((k for k in r if k.startswith("bof")), key=lambda k: r[k])
    tot_ref += cnt * r["ref_ks1"] / 1e3; tot_bof += cnt * r[best] / 1e3; tot_c4 += cnt * r["contract_ks4"] / 1e3
    say(f"  {key:8s} x{cnt:2d}: ks 1 {r['ref_ks1']:7.1f} us | " + " ".join(f"{k} {v:.1f}" for k, v in r.items()
                                                                       if k.startswith("bof")) +
        f" | best {best} {r[best]:.1f} | contract ks 4 {r['contract_ks4']:.1f}")
    POOLS[key] = None
    mx.clear_cache()
d = tot_bof - tot_ref
say(f"per 16-row check over the 5 shapes: REF2-T ks 1 {tot_ref:.2f} ms -> BOF (TERM {term}, bitwise) {tot_bof:.2f} "
    f"({d:+.2f}); the contract's ks (4; 10 for shared down) {tot_c4:.2f} (for reference)")
say(f"BOF verdict (pre-registered): " + ("PASS (>= 1.5 ms)" if -d >= 1.5 else "BETWEEN (0.5-1.5 ms)" if -d >= 0.5 else
                                         "KILL (< 0.5 ms)") + f"; standalone, the transfer rule applies")
say(f"done in {time.time() - t00:.0f} s; peak {mx.get_peak_memory() / 1e9:.2f} GB")
