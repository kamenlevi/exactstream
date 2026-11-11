#!/usr/bin/env python3 -I
"""expert_kernel.py (FableGod, mission 27). CPU only (numpy; no GPU; no model weights). Five parts:
1. The accounting of the 16-row expert kernels from Leap's measurements: a least-squares fit of NAX4's ms per layer
   against its (expert, tier) group count (K-62f, layer 20, 4 traces x S 16 / 32) and of 3c's against its (row, rank)
   pairs; the per-group MMA floor (K-62e: kernels A + B at the matrix-unit peak, 0.60 ms at 82 groups); the per-group
   bytes (k62p: MB per layer / groups) at the measured 258-294 GB/s; the 85 %-of-streaming budget per group.
2. Real routing (the decision map's 36 REF2-T captures, ids [tokens, 48 layers, 10 ranks]): per layer at S 8 / 16 / 32,
   distinct experts, (expert, tier) groups (ranks 0-1 at 4 bits = MSB + LSB, ranks 2-9 at 3 bits; groups capped at
   16 pairs), pairs per group, bytes; per check (48 layers): bytes, the byte floor, the MMA floor, today's NAX6 by the
   fit, the tier-merge saving, and the designs' arithmetic.
3. The tier-merge arithmetic (design B): s3 (x . q3) + 0.5 s3 (x . l) vs 0.5 s3 (x . (2 q3 + l)) in fp32, same group
   order, on real activations: bf16 bits differing, relative rms.
4. Design C's int8 activations (per-row, per-64 scale) on real activations vs fp64, next to the bf16-activation
   reference's own deviation (the metric of K-62g: mean |err| / output RMS).
5. How often a 32-term bf16 x uint4 sum (one tensor op's K) depends on its order in fp32, on real activations: the
   fraction of 32-term chunks where sequential, reversed and pairwise fp32 sums differ."""
import numpy as np, glob, os, time
t0 = time.time()
rng = np.random.default_rng(27)
MB_MSB, MB_LSB = 2_150_400, 768_000          # k62p_floor.py: 3 x (640 x 2560 x 3 / 8 + 2 x 640 x 2560 / 64 x 2), 3 x (640 x 2560 / 8 + ...)
NL, R, R4, I, H = 48, 10, 2, 640, 2560
BW = (258, 284, 294)

def bf16(x):
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32)); u = x.view(np.uint32).astype(np.uint64)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return u.astype(np.uint32).view(np.float32)

print("## 1. Leap's measurements, fitted [K -> P]")
# K-62f, layer 20 (data/k62f_real_L20.txt): (S, distinct, groups, tier_glu ms, NAX4 ms)
k62f = [(16, 68.8, 82.5, 1.184, 0.930), (32, 99.8, 123.5, 2.319, 1.345), (16, 60.5, 73.5, 1.174, 0.837), (32, 86.6, 110.8, 2.304, 1.220),
        (16, 63.2, 75.4, 1.177, 0.864), (32, 93.8, 115.1, 2.313, 1.270), (16, 48.9, 58.2, 1.162, 0.687), (32, 79.7, 100.2, 2.290, 1.108)]
G = np.array([r[2] for r in k62f]); P = np.array([r[0] * R for r in k62f]); T3 = np.array([r[3] for r in k62f]); TN = np.array([r[4] for r in k62f])
bn, an = np.polyfit(G, TN, 1); b3, a3 = np.polyfit(P, T3, 1)
rn = TN - (an + bn * G); r3 = T3 - (a3 + b3 * P)
print(f"  NAX4 ms per layer = {an:.3f} + {1e3 * bn:.2f} us x groups  (8 points, residual rms {1e3 * rn.std():.0f} us; distinct experts do not enter)")
print(f"  3c   ms per layer = {a3:.3f} + {1e3 * b3:.2f} us x (row, rank) pairs  (residual rms {1e3 * r3.std():.0f} us; groups do not enter)")
mma_us = 0.60e3 / 82                                   # K-62e: kernels A + B at the matrix-unit peak, 0.40 + 0.20 ms at 82 groups
print(f"  per-group MMA floor at the probe's peak (K-62e: 0.60 ms / 82 groups): {mma_us:.2f} us  = 9,600 ops of 16 x 32 x 16 at 1.3 G op/s = {9600 / 1.3e9 * 1e6:.2f} us")
# k62p layer 20 S 16: 143 MB over 4 traces; k62f layer 20 S 16 groups avg 72.4, distinct 60.4
grp16 = np.mean([r[2] for r in k62f if r[0] == 16]); dis16 = np.mean([r[1] for r in k62f if r[0] == 16])
mb_g = 143e6 / grp16
print(f"  bytes per group (k62p layer 20 S 16: 143 MB / {grp16:.1f} groups): {mb_g / 1e6:.2f} MB -> {', '.join(f'{mb_g / b / 1e3:.2f} us at {b} GB/s' for b in BW)}; per distinct expert {143e6 / dis16 / 1e6:.2f} MB")
print(f"  measured per group (the fit): {1e3 * bn:.1f} us = MMA floor x {1e3 * bn / mma_us:.2f}; the fixed {an:.3f} ms per call = {1e3 * an / grp16:.1f} us per group at {grp16:.0f} groups")
budget = mb_g / 284e3 / 0.85
print(f"  85 % of streaming at 284 GB/s allows {budget:.2f} us per group all-in; the MMA floor alone is {mma_us:.2f} us ({mma_us / budget:.2f} x the budget): structurally over it at the probe's peak")
print(f"  K-62e's zero-DRAM run (codes from one row): 0.70 of 0.85-0.90 ms -> exposed DRAM ~{(0.875 - 0.70) * 1e3 / 82:.1f} us per group; A-operand marshalling ~25 % of the MMA time (nax_tput mode 2) ~{0.25 * mma_us:.1f} us")

print("\n## 2. Real routing: the decision map's 36 REF2-T captures (ids [tokens, 48, 10], rank-sorted) [K -> P]")
caps = sorted(glob.glob(os.path.expanduser("~/FableGod/decision_map/out/routing/*.npz")))
per = {}                                               # (dom, S) -> list of per-window dicts summed over layers
for f in caps:
    z = np.load(f); ids = z["ids"].astype(np.int32); n = ids.shape[0]
    dom = os.path.basename(f).split("_")[1]
    for S in (8, 16, 32):
        starts = list(range(0, n - S + 1, max(S // 2, 1)))[:24]
        for s in starts:
            w = ids[s:s + S]                             # [S, 48, 10]
            d = dict(dist=0, lsb=0, grp=0, grp_m=0, pairs=S * R * NL, bytes=0, g1=0, gn=0, g2=0, g1m=0, g2m=0)
            for L in range(NL):
                e4 = w[:, L, :R4].reshape(-1); e3 = w[:, L, R4:].reshape(-1)
                u4, c4 = np.unique(e4, return_counts=True); u3, c3 = np.unique(e3, return_counts=True)
                ua, ca = np.unique(w[:, L, :].reshape(-1), return_counts=True)
                d["dist"] += len(ua); d["lsb"] += len(u4)
                d["grp"] += int(np.ceil(c4 / 16).sum() + np.ceil(c3 / 16).sum())       # (expert, tier) groups, capped at 16 pairs
                d["grp_m"] += int(np.ceil(ca / 16).sum())                               # design B: one group per expert
                d["bytes"] += MB_MSB * len(ua) + MB_LSB * len(u4)
                d["g1"] += int((c4 == 1).sum() + (c3 == 1).sum()); d["gn"] += len(u4) + len(u3)
                d["g2"] += int((c4 == 2).sum() + (c3 == 2).sum()); d["g1m"] += int((ca == 1).sum()); d["g2m"] += int((ca == 2).sum())
            per.setdefault((dom, S), []).append(d)
def agg(key_f):
    out = {}
    for (dom, S), L in per.items():
        if key_f(dom):
            out.setdefault(S, []).extend(L)
    return out
rows = []
print("  | domain | S | windows | distinct experts per check | groups | pairs per group | 1-pair groups | bytes GB | byte floor 284 GB/s | MMA floor (groups x 7.3 us) | NAX6 by the fit (0.10 + 10.1 us x groups) | 3c by the fit | tier-merge: groups -> | MMA floor merged |")
print("  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for dom in ("code", "ifeval", "multi", "all"):
    A = agg((lambda d: d == dom) if dom != "all" else (lambda d: True))
    for S in (8, 16, 32):
        L = A[S]; m = {k: np.mean([d[k] for d in L]) for k in L[0]}
        mma = m["grp"] * mma_us / 1e3; mma_m = m["grp_m"] * mma_us / 1e3
        nax = NL * an + m["grp"] * bn; c3 = NL * a3 + m["pairs"] * b3
        fl = m["bytes"] / 284e6
        rows.append((dom, S, m, fl, mma, nax))
        print(f"  | {dom} | {S} | {len(L)} | {m['dist']:.0f} | {m['grp']:.0f} | {m['pairs'] / m['grp']:.2f} | {100 * m['g1'] / m['gn']:.0f} % | {m['bytes'] / 1e9:.2f} | {fl:.1f} ms | {mma:.1f} ms | {nax:.1f} ms ({100 * fl / nax:.0f} % of streaming) | {c3:.1f} ms | {m['grp_m']:.0f} (-{100 * (1 - m['grp_m'] / m['grp']):.0f} %) | {mma_m:.1f} ms |")
print("  cross-check at S 16, all domains: C2 measured 7.38 GB per check (Breakthrough's expA traces, layers 6 / 20 / 40 x 48) and NAX6 43.2 ms; 3c 50.5")

print("\n## 2b. The designs per check, from the per-group arithmetic [P/G]: time = max(byte floor, 48 x fixed + groups x per-group) ; share = byte floor / time")
def hyb(m, ca, cg, gk, k1, k2, cb=2.8):
    """per group: min(pairs x ca + cb, cg) for 1- and 2-pair groups, cg otherwise, + 0.08 ms per call"""
    g1, g2, G = m[k1], m[k2], m[gk]
    return NL * 0.08 + (g1 * min(ca + cb, cg) + g2 * min(2 * ca + cb, cg) + (G - g1 - g2) * cg) * 1e-3
def price(S, dom="all"):
    A = agg(lambda d: True); L = A[S]; m = {k: np.mean([d[k] for d in L]) for k in L[0]}
    fl = m["bytes"] / 284e6
    designs = [("today NAX6 (fit)", NL * an + m["grp"] * bn),
               ("today 3c (fit)", NL * a3 + m["pairs"] * b3),
               ("A NAX7, probe peak holds: 8.5 us per group + 0.08 ms per call", NL * 0.08 + m["grp"] * 8.5e-3),
               ("A NAX7, true peak 2x the probe: 6.0 us + 0.08", NL * 0.08 + m["grp"] * 6.0e-3),
               ("B tier merge on NAX7 (probe peak): 8.5 us x merged groups + 0.08", NL * 0.08 + m["grp_m"] * 8.5e-3),
               ("B on NAX7 at 2x peak: 6.0 us x merged + 0.08", NL * 0.08 + m["grp_m"] * 6.0e-3),
               ("C int8 activations (if int8 ops run 2x): 5.5 us x merged + 0.08", NL * 0.08 + m["grp_m"] * 5.5e-3),
               ("D ALU kernel at 3.2 us per pair + 0.08", NL * 0.08 + m["pairs"] * 3.2e-3),
               ("H hybrid on NAX7: 1-2-pair groups' gate/up on the ALU at 3.0 us per pair, down on the MMA (2.8), else 8.5", hyb(m, 3.0, 8.5, "grp", "g1", "g2")),
               ("H hybrid on NAX7, ALU at 4.8 us per pair (3c's per-MAC rate on 2/3 of the MACs)", hyb(m, 4.8, 8.5, "grp", "g1", "g2")),
               ("H hybrid on NAX7, ALU at 6.5 us per pair (bound)", hyb(m, 6.5, 8.5, "grp", "g1", "g2")),
               ("H + B: hybrid on merged groups, ALU 4.8", hyb(m, 4.8, 8.5, "grp_m", "g1m", "g2m")),
               ("H + B at 2x peak: NAX7 6.0 (A 4.0 + B 2.0), ALU 4.8", hyb(m, 4.8, 6.0, "grp_m", "g1m", "g2m", 2.0)),
               ("D ALU kernel at 4.5 us per pair + 0.08 (bound)", NL * 0.08 + m["pairs"] * 4.5e-3)]
    print(f"  S {S}: bytes {m['bytes'] / 1e9:.2f} GB, byte floor {fl:.1f} ms at 284 GB/s, 85 % line {fl / 0.85:.1f} ms; groups {m['grp']:.0f} of which 1-pair {100 * m['g1'] / m['grp']:.0f} %, 2-pair {100 * m['g2'] / m['grp']:.0f} %; merged {m['grp_m']:.0f}: 1-pair {100 * m['g1m'] / m['grp_m']:.0f} %, 2-pair {100 * m['g2m'] / m['grp_m']:.0f} %")
    for name, t in designs:
        tt = max(t, fl); print(f"    {name:<72} compute {t:5.1f} ms -> {tt:5.1f} ms, {100 * fl / tt:3.0f} % of streaming")
for S in (8, 16, 32): price(S)

print("\n## 3. Design B's arithmetic: the 4-bit pairs as two products vs one nibble product (fp32, same group order) [K numeric]")
tap = np.load(os.path.expanduser("~/TrainLab/data/d02a/agent_bash17.npz"))["f"]        # [n, 5 taps, 2560] fp16: real hidden states
X = bf16(tap[200:216, 2, :].astype(np.float32))                                          # 16 rows at the layer-25 tap, as the MoE input's proxy
W = rng.normal(0, 0.02, (I, H)).astype(np.float32)                                       # one expert's gate weights (dequantized scale)
Wg = W.reshape(I, H // 64, 64); mn, mx_ = Wg.min(-1, keepdims=True), Wg.max(-1, keepdims=True)
s4 = bf16((mx_ - mn) / 15); b4 = bf16(mn); q4 = np.clip(np.round((Wg - b4) / s4), 0, 15).astype(np.int32)
q3, l1 = q4 >> 1, q4 & 1; s3 = bf16(2 * s4)                                              # records3: MSB-3 code + LSB-1, scale s3 = 2 s4, 4-bit scale = s3 / 2
Xg = X.reshape(16, H // 64, 64).astype(np.float32); xs = Xg.sum(-1, dtype=np.float32)  # rowsum64
def dot(Xg, Q):                                                                          # [16, groups, out] fp32 fixed order
    return np.einsum("rgk,ogk->rgo", Xg, Q.astype(np.float32), dtype=np.float32, optimize=False)
pn = dot(Xg, 2 * q3 + l1); p3 = dot(Xg, q3); pl = dot(Xg, l1)
S4 = (s4[None, :, :, 0].transpose(0, 2, 1)); B4 = b4[None, :, :, 0].transpose(0, 2, 1)  # [1, groups, out]
y_nib = np.zeros((16, I), np.float32); y_two = np.zeros((16, I), np.float32)
for g in range(H // 64):
    y_nib = y_nib + (S4[:, g] * pn[:, g] + B4[:, g] * xs[:, g][:, None]).astype(np.float32)
    y_two = y_two + ((s3[None, :, g, 0] * p3[:, g] + (0.5 * s3[None, :, g, 0]) * pl[:, g]) + B4[:, g] * xs[:, g][:, None]).astype(np.float32)
yb_n, yb_t = bf16(y_nib), bf16(y_two)
print(f"  gate output of one expert, 16 real rows: bf16 bits differ {100 * np.mean(yb_n.view(np.uint32) != yb_t.view(np.uint32)):.2f} %, rel rms {np.sqrt(np.mean((y_nib - y_two) ** 2)) / np.sqrt(np.mean(y_nib ** 2)):.1e} (fp32 reorder level; all within 1 bf16 ulp: {bool(np.all(np.abs(yb_n - yb_t) <= np.abs(yb_n) * 2 ** -7 + 1e-30))})")

print("\n## 4. Design C: int8 activations (per row, per 64-group scale) vs the bf16-activation reference, both against fp64 [K numeric]")
Xq_s = (np.abs(Xg).max(-1, keepdims=True) / 127).astype(np.float32); Xq = np.clip(np.round(Xg / Xq_s), -127, 127)
Wd = (s3[:, :, 0, None] * q3 + (b4 + 0.5 * s4)[:, :, 0, None]).astype(np.float64)                 # [out, groups, 64]
y64 = np.einsum("rgk,ogk->ro", Xg.astype(np.float64), Wd)                                           # exact with bf16 x
yref = np.zeros((16, I), np.float32)
for g in range(H // 64): yref = yref + (s3[None, :, g, 0] * p3[:, g] + (b4 + 0.5 * s4)[None, :, g, 0] * xs[:, g][:, None]).astype(np.float32)
y8 = np.zeros((16, I), np.float32)
pi = np.einsum("rgk,ogk->rgo", Xq, q3.astype(np.float64)).astype(np.float32)                      # exact integer sums (fit fp32 exactly: < 2^24)
xs8 = (Xq * Xq_s).sum(-1, dtype=np.float32)
for g in range(H // 64): y8 = y8 + (s3[None, :, g, 0] * Xq_s[:, g] * pi[:, g] + (b4 + 0.5 * s4)[None, :, g, 0] * xs8[:, g][:, None]).astype(np.float32)
rms = np.sqrt(np.mean(y64 ** 2))
e_ref = np.mean(np.abs(yref - y64)) / rms; e_8 = np.mean(np.abs(y8 - y64)) / rms
print(f"  mean |err| / output RMS: bf16-x fp32 reference {e_ref:.2e}; int8-x {e_8:.2e} ({e_8 / max(e_ref, 1e-12):.0f} x); K-62g measured on the GPU with random x: 3c 1.23e-2, NAX4 3.6e-3 (against fp64 of the same weights)")
print(f"  max |err| / RMS: reference {np.max(np.abs(yref - y64)) / rms:.2e}, int8-x {np.max(np.abs(y8 - y64)) / rms:.2e}; activation range per row max/rms {np.mean(np.abs(X).max(1) / np.sqrt(np.mean(X ** 2, 1))):.1f}")

print("\n## 5. Order sensitivity of one tensor op's 32-term sum (bf16 x times uint4 code, fp32) on real activations [K numeric]")
Xc = X.reshape(16, H // 32, 32).astype(np.float32); Q = rng.integers(0, 16, (64, H // 32, 32)).astype(np.float32)
prod = Xc[:, None] * Q[None]                                                                        # [16, 64 cols, chunks, 32]: exact products
seq = np.zeros(prod.shape[:3], np.float32); rev = np.zeros_like(seq)
for k in range(32): seq = seq + prod[..., k]
for k in range(31, -1, -1): rev = rev + prod[..., k]
pw = prod.sum(-1, dtype=np.float32)                                                                 # numpy's pairwise order
ex = prod.astype(np.float64).sum(-1)
diff = (seq != rev) | (seq != pw)
print(f"  chunks where the fp32 result depends on the order: {100 * diff.mean():.2f} % of {diff.size}; chunks whose exact sum is not an fp32 number: {100 * np.mean(ex.astype(np.float32) != ex):.2f} %")
print(f"  -> an ALU path can match the tensor op bit for bit only where the exact sum is fp32-representable AND the op rounds once; the op's internal order is not public, so the hybrid is not certifiable as bitwise")
print("\n## 6. The hybrid's exactness guard on real activations: per (row, 64-group) the non-zero values' exponent spread must be <= 6 for the fp32 sum of bf16 x uint4 products to be exact in any order [K numeric]")
def spread_fail(A, limit=6):
    A = np.asarray(A, np.float32); Ag = A.reshape(A.shape[0], -1, 64)
    e = np.floor(np.log2(np.where(Ag == 0, np.nan, np.abs(Ag))))
    sp = np.nanmax(e, -1) - np.nanmin(e, -1)
    return np.nan_to_num(sp, nan=0.0) > limit
for li, lay in enumerate((5, 15, 25, 35, 45)):
    Xa = bf16(tap[:, li, :].astype(np.float32)); f = spread_fail(Xa)
    print(f"  tap layer {lay}, {Xa.shape[0]} rows x 40 groups: cells failing {100 * f.mean():.3f} %; rows with any failing cell {100 * f.any(1).mean():.2f} %")
Xa = bf16(tap[:, 2, :].astype(np.float32))[:256]
Wu_ = rng.normal(0, 0.02, (I, H)).astype(np.float32)
g_ = bf16(Xa @ W.T); u_ = bf16(Xa @ Wu_.T); h_ = bf16(bf16(g_ / (1 + np.exp(-g_))) * u_)
fh = spread_fail(h_)
print(f"  h = swiglu(gate, up) for the down projection (256 rows x 10 groups, random weights): cells failing {100 * fh.mean():.1f} %, rows with any failing cell {100 * fh.any(1).mean():.1f} % -> the down projection stays on the MMA in the hybrid")
print(f"  with limit 8 (fp32 headroom if the 64-term sum stays below 2^(e_max + 9)): x cells failing {100 * spread_fail(bf16(tap[:, 2, :].astype(np.float32)), 8).mean():.3f} %, h cells {100 * spread_fail(h_, 8).mean():.1f} %")
print(f"\n({time.time() - t0:.1f} s, numpy {np.__version__}, CPU only; captures {len(caps)})")
