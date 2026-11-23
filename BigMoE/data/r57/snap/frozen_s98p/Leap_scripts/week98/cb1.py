"""CB1 (Leap, Sun 15 Nov 2026 17:3x; CHECK55 track B = NewIdeas' CHECK55_LEVERS levers 6-8): the reference-contract bench.
Track B re-certifies the exact reference ONCE next week (Kamen: the switch is next week). The contract (NewIdeas' lever 6):
every dense GEMM, the HC read and the head compute each output with (a) a FIXED K-chunk per shape (independent of the row
count M and of the N tiling), (b) inside a chunk qmm16c's code-MMA order (S4: MLX's NAX k-steps of 32), (c) the chunk
partials summed in increasing order in fp32, (d) one bf16 rounding. SD == AR then holds by construction: M = 1 pads to
16, M = 32 is two 16-row tiles with one weight pass, and each row's arithmetic is the same at every M.
Question: per real shape of the 125B (safetensors headers; per-check counts for 36 GDN + 12 QSA layers), what does the best
fixed chunk cost at 16 and at 32 rows, against REF2-T's forms today (16 rows: qmm16c ks 1, MLX's padded call, S4 at MLX's
4,096-key partitions; 32 rows: v32one's per-shape forms)? Cold weights (pools >= 128 MB per shape, rotated per round), a
dependent chain per check's worth of calls (per call = (chain - its links alone) / n), forms round-robin with the order
rotated per round, median of ROUNDS.
Forms:
  4-bit shapes, 16 rows: qmm16c (ks, nt) and the native layout qmm16n (ks, nt) (bit-identical by design);
  4-bit shapes, 32 rows: qmm16c2a_ks (ks, nt), NEW here: v32one's one-weight-pass qmm16c2a with qmm16c's K split, the two
    row tiles' partials reduced one after the other in qmm16c's order;
  fusions the contract allows: b | a into the GDN in-proj (N 16,480, padded to 16,512); the shared cat padded to 1,344;
  HC down (S4, the merged down + inject, 324 x 10,240 bf16): partition size PS in {4096 (today), 2048, 1024, 512, 256},
    S4 at 16 rows and S4 2i at 32 rows, then the ordered partition sum.
ks per shape: divisors of K / 64 with (ks - 1) x nt <= 10 (<= 20 KB of partials, DN-KS v2's NAX-OCC limit), nt in {1, 2}.
Bits, every contract config: rows bitwise across M = 1 (zero and random padding) / 16 / 32, across nt, and qmm16c ==
qmm16n at the same ks; S4: 16 == 32 rows (2i) == M 1 at each PS. Sanity vs an fp32 reference (relative max error).
PRE-REGISTERED in data/week98/check55_plan.md 'CB1' (before any run).
usage: [BigMoE venv] cb1.py [--smoke]   (MEM ~1.5 GB peak; smoke = bits + every path, share lock; timing = quiet ticket)
-> data/week98/screens/cb1.txt (cb1_smoke.txt)"""
import os, sys, time
from functools import lru_cache
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98p/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98p/Leap_scripts/week98"))
import qmm16 as Q16
from qmm16 import qmm16c
import qmm26
import leapk as LK
import one32

SMOKE = "--smoke" in sys.argv
OUT = os.path.expanduser(f"~/Leap/data/week98/screens/cb1{'_smoke' if SMOKE else ''}.txt")
BF = mx.bfloat16
POOL = (8 if SMOKE else 128) * 2 ** 20
ROUNDS = 2 if SMOKE else int(os.environ.get("CB1_ROUNDS", "7"))
t00 = time.time()


def say(s):
    print(s, flush=True)
    with open(OUT, "a") as f:
        f.write(s + "\n")


from ck import qmm16c2a_ks, s4_partials_ps, s4_partials2i_ps     # the contract kernels (moved to ck.py, Sun 17:4x)


# ---------------------------------------------------------------- S4 at a chosen partition size, 16 or 32 (2i) rows
_ACC = mx.fast.metal_kernel(name="leap_cb1_accum", input_names=["cs"], output_names=["y"], source="""
    const uint i = thread_position_in_grid.x;
    if (i >= R * N) return;
    float out = 0;
    for (int p = 0; p < NP; p++) {
        out += cs[size_t(p) * R * N + i];
    }
    y[i] = static_cast<T>(out);
""")


def s4_ps(x, w, ps):
    """x [R, K] (R 16 or 32) bf16, w [N, K] bf16 -> y [R, N]: S4 (R 16) / S4 2i (R 32) partials at partition size ps,
    then the ordered fp32 partition sum (leapk._ACCUM's order) and one bf16 rounding"""
    R, K = x.shape
    N = w.shape[0]
    cs, NP = (s4_partials_ps if R == 16 else s4_partials2i_ps)(x.reshape(1, R, K), w, ps)
    (y,) = _ACC(inputs=[cs], template=[("T", BF), ("N", N), ("NP", NP), ("R", R)], grid=(R * N, 1, 1),
                threadgroup=(256, 1, 1), output_shapes=[(R, N)], output_dtypes=[BF])
    return y


# ---------------------------------------------------------------- shapes (real: ~/BigMoE/models/bare headers)
# key, N (pool), N real, K, calls per check (36 GDN + 12 QSA layers), REF2-T 16-row form, REF2-T 32-row form
Q4 = [("gdn_qz", 16384, 16384, 2560, 36, "c", "2a"),          # InProjExact qkv | z: one qmm16c call
      ("gdn_ba", 96, 96, 2560, 36, "mlx", ("qmvw", 2)),         # b | a: MLX's padded 16-row call / qmvw32 vpt 2
      ("gdn_qzba", 16512, 16480, 2560, 36, None, None),         # contract fusion: b | a folded in (16,480 -> 16,512)
      ("o_out", 2560, 2560, 6144, 48, "c", "2a"),               # GDN out_proj x 36 + QSA o_proj x 12
      ("qsa_qkv", 13312, 13312, 2560, 12, "c", "2a"),           # the engine's qkv_fused (q 12,288 + k 512 + v 512)
      ("sh_cat", 1344, 1281, 2560, 48, "mlx", ("qmvw", 8)),     # shared gate | up | gate (1,281; contract pads to 1,344)
      ("sh_down", 2560, 2560, 640, 48, "c", "2"),               # N x K < BIG_NK: qmm16c2 at 32 rows
      ("lm_head", 248320, 248320, 2560, 1, "c", "2a")]
HC = ("hc_down", 324, 10240, 97)                                  # 96 layer reads + the model-level mixer
PSS = (4096, 2048, 1024, 512, 256)


def ks_list(K, nt):
    g = K // 64
    return [k for k in range(1, 21) if g % k == 0 and (k - 1) * nt <= 10 and 32 * k * nt <= 1024]


def q4_pool(N, K, seed):
    D = max(1, -(-POOL // int(N * K * 0.5625)))
    k1, k2, k3, k4 = mx.random.split(mx.random.key(seed), 4)
    half = lambda k: mx.random.randint(0, 65536, (D * N, K // 8), dtype=mx.int32, key=k).astype(mx.uint32)
    w = (half(k1) << 16) | half(k4)                    # random 32-bit words = 8 random 4-bit codes
    s = (mx.abs(mx.random.normal((D * N, K // 64), key=k2)) * 0.002 + 0.006).astype(BF)
    b = (mx.random.normal((D * N, K // 64), key=k3) * 0.01 - 0.06).astype(BF)
    mx.eval(w, s, b)
    return D, (w, s, b)


def copies(arrs, D, N, Nreal=None):
    n = Nreal or N
    out = [tuple(a[i * N:i * N + n] for a in arrs) for i in range(D)]
    mx.eval(out)
    return out


def bits_eq(a, b):
    return bool(mx.all(a == b).item()) if a.shape == b.shape else False


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
    """forms: {name: (rows, fn(x, i))}; -> {name: median us per call}"""
    rng = np.random.default_rng(seed)
    X = {r: (mx.random.normal((r, K), key=mx.random.key(seed + r)) * 0.5).astype(BF) for r in (16, 32)}
    mx.eval(X)
    nch = max(n, 3)
    for name, (r, fn) in forms.items():                       # compile + warm each form once
        mx.eval(run_chain(fn, X[r], 2, 0, D))
    T = {k: [] for k in forms}
    L = {16: [], 32: []}
    names = list(forms)
    for rd in range(ROUNDS):
        order = names[rd % len(names):] + names[:rd % len(names)]
        for r in (16, 32):
            L[r].append(wall(lambda: run_links(X[r], nch)))
        for name in order:
            r, fn = forms[name]
            off = int(rng.integers(0, D))
            T[name].append(wall(lambda: run_chain(fn, X[r], nch, off, D)))
    lk = {r: float(np.median(v)) for r, v in L.items()}
    return {k: max(0.0, (float(np.median(v)) - lk[forms[k][0]]) / nch * 1e6) for k, v in T.items()}


# ---------------------------------------------------------------- per shape: bits, then timing
say(f"\nCB1 {'SMOKE ' if SMOKE else ''}{time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; pools "
    f"{POOL >> 20} MB per shape; {ROUNDS} rounds; {os.environ.get('CB1_COND', 'share lock')}")
RES, BITS = {}, []
for si, (key, N, Nr, K, cnt, f16, f32) in enumerate(Q4):
    t0 = time.time()
    D, pool = q4_pool(N, K, 100 + si)
    C = copies(pool, D, N)
    Cr = copies(pool, D, N, Nr) if Nr != N else C
    native = key != "lm_head"
    if native:
        wn, sn, bn = qmm26.repack(pool[0].reshape(D, N, K // 8), pool[1].reshape(D, N, K // 64),
                                  pool[2].reshape(D, N, K // 64))
        mx.eval(wn, sn, bn)
        CN = [(wn[i], sn[i], bn[i]) for i in range(D)]
        mx.eval(CN)
    # bits on copy 0
    x32 = (mx.random.normal((32, K), key=mx.random.key(7 + si)) * 0.5).astype(BF)
    pad = (mx.random.normal((15, K), key=mx.random.key(70 + si)) * 0.5).astype(BF)
    W = C[0]
    NC = min(N, 4096)                                  # the fp32 sanity reference on the first <= 4,096 columns
    deq = mx.dequantize(*(a[:NC] for a in W), group_size=64, bits=4).astype(mx.float32)
    ref = (x32.astype(mx.float32) @ deq.T)
    rmax = float(mx.abs(ref).max())
    nb, bad, errmax = 0, [], 0.0
    for nt in (1, 2):
        if N % (32 * nt):
            continue
        for ks in ks_list(K, nt):
            a16 = qmm16c(x32[:16], *W, ks=ks, nt=1)
            b16 = qmm16c(x32[16:], *W, ks=ks, nt=1)
            checks = {"nt": bits_eq(qmm16c(x32[:16], *W, ks=ks, nt=nt), a16),
                      "M1z": bits_eq(qmm16c(x32[:1], *W, ks=ks, nt=nt)[:1], a16[:1]),
                      "M1r": bits_eq(qmm16c(mx.concatenate([x32[:1], pad]), *W, ks=ks, nt=nt)[:1], a16[:1]),
                      "M32": bits_eq(qmm16c2a_ks(x32, *W, ks=ks, nt=nt), mx.concatenate([a16, b16]))}
            if native:
                checks["native"] = bits_eq(qmm26.qmm16n(x32[:16], *CN[0], N, ks, nt), a16)
            errmax = max(errmax, float(mx.abs(mx.concatenate([a16, b16])[:, :NC].astype(mx.float32) - ref).max()) / rmax)
            nb += len(checks)
            bad += [f"ks{ks}nt{nt}:{c}" for c, ok in checks.items() if not ok]
    BITS.append((key, nb, bad))
    say(f"{key} N {Nr} (pool {N}) K {K}: {D} copies; bits {nb - len(bad)} / {nb} configs x checks"
        f"{'' if not bad else ' FAIL ' + ' '.join(bad[:8])}; contract vs fp32 max err {errmax:.2e} x max|y|")
    # timing forms
    forms = {}
    if f16 == "c":
        forms["ref16"] = (16, lambda x, i: qmm16c(x, *C[i]))
    elif f16 == "mlx":
        forms["ref16"] = (16, lambda x, i: mx.quantized_matmul(x, *Cr[i], transpose=True, group_size=64, bits=4))
    if f32 == "2a":
        forms["ref32"] = (32, lambda x, i: one32.qmm16c2a(x, *C[i]))
    elif f32 == "2":
        forms["ref32"] = (32, lambda x, i: one32.qmm16c2(x, *C[i]))
    elif isinstance(f32, tuple):
        vpt = f32[1]
        forms["ref32"] = (32, lambda x, i, v=vpt: one32.qmvw32(x, *Cr[i], 4, 8, v))
    for nt in (1, 2):
        if N % (32 * nt):
            continue
        for ks in ks_list(K, nt):
            forms[f"c{ks}n{nt}"] = (16, lambda x, i, k=ks, t=nt: qmm16c(x, *C[i], ks=k, nt=t))
            if native:
                forms[f"N{ks}n{nt}"] = (16, lambda x, i, k=ks, t=nt: qmm26.qmm16n(x, *CN[i], N, k, t))
            forms[f"a{ks}n{nt}"] = (32, lambda x, i, k=ks, t=nt: qmm16c2a_ks(x, *C[i], ks=k, nt=t))
    RES[key] = bench(forms, K, cnt, D, 1000 + si)
    del C, Cr, pool
    if native:
        del CN, wn, sn, bn
    mx.clear_cache()
    say(f"  timed {len(forms)} forms in {time.time() - t0:.0f} s")

# HC down (S4)
key, N, K, cnt = HC
D = max(1, -(-POOL // (N * K * 2)))
wpool = (mx.random.normal((D * N, K), key=mx.random.key(5)) * 0.02).astype(BF)
mx.eval(wpool)
CW = [wpool[i * N:(i + 1) * N] for i in range(D)]
mx.eval(CW)
x32 = (mx.random.normal((32, K), key=mx.random.key(6)) * 0.5).astype(BF)
pad = (mx.random.normal((15, K), key=mx.random.key(60)) * 0.5).astype(BF)
ref = x32.astype(mx.float32) @ CW[0].astype(mx.float32).T
rmax = float(mx.abs(ref).max())
nb, bad, errmax = 0, [], 0.0
for ps in PSS:
    a16, b16 = s4_ps(x32[:16], CW[0], ps), s4_ps(x32[16:], CW[0], ps)
    checks = {"M32": bits_eq(s4_ps(x32, CW[0], ps), mx.concatenate([a16, b16])),
              "M1z": bits_eq(s4_ps(mx.concatenate([x32[:1], mx.zeros((15, K), BF)]), CW[0], ps)[:1], a16[:1]),
              "M1r": bits_eq(s4_ps(mx.concatenate([x32[:1], pad]), CW[0], ps)[:1], a16[:1])}
    if ps == 4096:                                     # today's S4 (leapk) at 16 rows: the replica is the same kernel
        cs, NP = LK.s4_partials(x32[:16].reshape(1, 16, K), CW[0])
        checks["leapk"] = bits_eq(LK._ACCUM(inputs=[cs], template=[("T", BF), ("N", N), ("NP", NP)], grid=(16 * N, 1, 1),
                                            threadgroup=(256, 1, 1), output_shapes=[(1, 16, N)],
                                            output_dtypes=[BF])[0].reshape(16, N), a16)
    errmax = max(errmax, float(mx.abs(mx.concatenate([a16, b16]).astype(mx.float32) - ref).max()) / rmax)
    nb += len(checks)
    bad += [f"ps{ps}:{c}" for c, ok in checks.items() if not ok]
BITS.append((key, nb, bad))
say(f"{key} N {N} K {K} bf16: {D} copies; bits {nb - len(bad)} / {nb}{'' if not bad else ' FAIL ' + ' '.join(bad)}; "
    f"vs fp32 max err {errmax:.2e} x max|y|")
forms = {}
for ps in PSS:
    forms[f"s{ps}"] = (16, lambda x, i, p=ps: s4_ps(x, CW[i], p))
    forms[f"s{ps}x2i"] = (32, lambda x, i, p=ps: s4_ps(x, CW[i], p))
RES[key] = bench(forms, K, cnt, D, 2000)
del CW, wpool
mx.clear_cache()

# ---------------------------------------------------------------- summary
GBs = lambda nbytes, us: nbytes / (us * 1e-6) / 1e9 if us > 0 else float("nan")
say("\nper call (us), cold, dependent chain; GB/s in brackets (DRAM ~276 measured, floor line 280)")
say(f"{'shape':9s} {'n':>3s} | {'REF2-T 16':>16s} {'REF2-T 32':>16s} | joint ks: {'16-row form':>20s} {'32-row form':>20s} | "
    f"best 16 / best 32 (form)")
TOT = {"r16": 0.0, "r32": 0.0, "c16": 0.0, "c32": 0.0}
JOINT = {}
for key, N, Nr, K, cnt, f16, f32 in Q4:
    r = RES[key]
    nbytes = Nr * K * 0.5625
    by_ks = {}
    for name, us in r.items():
        if name[0] in "cNa":
            ks = int(name[1:].split("n")[0])
            slot = by_ks.setdefault(ks, {"16": None, "32": None})
            col = "32" if name[0] == "a" else "16"
            if slot[col] is None or us < slot[col][1]:
                slot[col] = (name, us)
    full = {k: v for k, v in by_ks.items() if v["16"] and v["32"]}
    ksj = min(full, key=lambda k: full[k]["16"][1] + full[k]["32"][1])
    j16, j32 = full[ksj]["16"], full[ksj]["32"]
    b16 = min((v for v in r.items() if v[0][0] in "cN"), key=lambda v: v[1])
    b32 = min((v for v in r.items() if v[0][0] == "a"), key=lambda v: v[1])
    JOINT[key] = (ksj, j16, j32)
    rt = lambda k: f"{r[k]:7.1f} [{GBs(nbytes, r[k]):4.0f}]" if k in r else f"{'-':>16s}"
    say(f"{key:9s} {cnt:3d} | {rt('ref16'):>16s} {rt('ref32'):>16s} | ks {ksj:2d}: {j16[0]:>7s} {j16[1]:7.1f} "
        f"[{GBs(nbytes, j16[1]):4.0f}] {j32[0]:>7s} {j32[1]:7.1f} [{GBs(nbytes, j32[1]):4.0f}] | "
        f"{b16[0]} {b16[1]:.1f} / {b32[0]} {b32[1]:.1f}")
r = RES["hc_down"]
nbytes = HC[1] * HC[2] * 2
psj = min(PSS, key=lambda p: r[f"s{p}"] + r[f"s{p}x2i"])
say(f"{'hc_down':9s} {HC[3]:3d} | " + " ".join(f"PS {p}: {r[f's{p}']:.1f} / {r[f's{p}x2i']:.1f} [{GBs(nbytes, r[f's{p}']):.0f} / "
                                              f"{GBs(nbytes, r[f's{p}x2i']):.0f}]" for p in PSS) + f" | joint PS {psj}")
# totals per check (ms): REF2-T vs the contract at the joint ks (b | a folded into the in-proj only if that is cheaper)
sep = sum(JOINT[k][1][1] for k in ("gdn_qz", "gdn_ba")), sum(JOINT[k][2][1] for k in ("gdn_qz", "gdn_ba"))
fz = JOINT["gdn_qzba"][1][1], JOINT["gdn_qzba"][2][1]
fold = fz[0] + fz[1] < sep[0] + sep[1]
for key, N, Nr, K, cnt, f16, f32 in Q4:
    r = RES[key]
    if key == "gdn_qzba":
        if fold:
            TOT["c16"] += cnt * fz[0] / 1e3; TOT["c32"] += cnt * fz[1] / 1e3
        continue
    TOT["r16"] += cnt * r["ref16"] / 1e3; TOT["r32"] += cnt * r["ref32"] / 1e3
    if key in ("gdn_qz", "gdn_ba") and fold:
        continue
    TOT["c16"] += cnt * JOINT[key][1][1] / 1e3; TOT["c32"] += cnt * JOINT[key][2][1] / 1e3
rh = RES["hc_down"]
TOT["r16"] += HC[3] * rh["s4096"] / 1e3; TOT["r32"] += HC[3] * rh["s4096x2i"] / 1e3
TOT["c16"] += HC[3] * rh[f"s{psj}"] / 1e3; TOT["c32"] += HC[3] * rh[f"s{psj}x2i"] / 1e3
d16, d32 = TOT["c16"] - TOT["r16"], TOT["c32"] - TOT["r32"]
say(f"\nper check (ms, this set: 7 four-bit kinds + HC down; router / indexer 8-bit, HC up, PLE, experts not in it):")
say(f"  16 rows: REF2-T {TOT['r16']:.2f} -> contract {TOT['c16']:.2f} ({d16:+.2f}); b | a folded into the in-proj: "
    f"{'yes' if fold else 'no'} ({fz[0] + fz[1]:.1f} vs {sep[0] + sep[1]:.1f} us per 16 + 32-row pair)")
say(f"  32 rows: REF2-T {TOT['r32']:.2f} -> contract {TOT['c32']:.2f} ({d32:+.2f})")
say(f"  M = 1 (AR, chat, the commit's 1-row calls): the contract pads to 16, so it costs the 16-row time; today's REF2-T "
    f"1-row path also pads to 16 ({TOT['r16']:.2f} ms over this set) -> the contract's 1-row line holds iff its 16 does")
allbits = all(not b for _, _, b in BITS)
v16 = "PASS" if -d16 >= 4.0 else "KILL" if -d16 < 2.0 else "BETWEEN"
v32 = "PASS" if -d32 >= 6.0 else "KILL" if -d32 < 3.0 else "BETWEEN"
say(f"CB1 verdict (pre-registered; standalone, the transfer rule applies): bits {'ALL BITWISE' if allbits else 'FAIL (a bug: no verdict)'}; "
    f"16 rows {v16} ({d16:+.2f} ms; PASS <= -4.0, KILL > -2.0); 32 rows {v32} ({d32:+.2f} ms; PASS <= -6.0, KILL > -3.0)")
say(f"done in {time.time() - t00:.0f} s; peak {mx.get_peak_memory() / 1e9:.2f} GB")
