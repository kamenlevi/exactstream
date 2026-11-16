"""check25_bound.py (Leap, Mon 16 Nov 2026 11:5x; the orchestrator 11:4x-11:5x: "the check at 25 ms is yours"). Desk only, CPU.

What is the fastest a 32-row LOSSY check can be on this Mac, for a given byte budget, if every kernel ran at the
measured hardware peaks? A check is layer-sequential: in each MoE layer the routed experts need the router, the router
needs the attention / GDN output, and the next layer needs the experts. So the dense side and the experts cannot
overlap, and each part takes at least max(its matrix-unit time, its DRAM time).

Inputs (labels: [K] measured record, [P] priced from measured parts, [est] assumption):
  - bytes per 32-row check on the code windows [P]: Fit12 ~/Fit12/data/lossy32/lossy32_price.json (v2)
  - DRAM: 280 GB/s (the WEEK98 floor convention); best measured streaming ~295 GB/s [K] (TABLE_350 bw_probe best
    292-294, the 4-bit kinds' fit B ~295)
  - matrix unit: 14.0 T MAC/s bf16 [K] (M27-P, register operands, 16x32x32); int8 x int8 1.73x that [K] (M27-C)
  - dense shapes [K]: one32_bench's per-check call list (scripts/week98/one32_bench.py) + the HC up mixer + V1's head
  - expert slots [P]: 32 rows x top-10, 1c drops 29.1 % of the row-rank slots on code (Fit12 lossy1c_16row.txt)
  - cores [est]: the GDN state read + written once per layer (fp32, 48 x 128 x 128); attention at 1K / 2K keys
"""
import json
import os

OUT = os.path.expanduser("~/Leap/data/week98/check25_bound.txt")
F = json.load(open(os.path.expanduser("~/Fit12/data/lossy32/lossy32_price.json")))
L = []


def p(s=""):
    print(s)
    L.append(s)


BW, BW_BEST = 280e9, 295e9
MMA16 = 14.0e12
MMA8 = 1.73 * MMA16
ROWS = 32

# dense calls per 32-row check: (N, K, calls)
DENSE = [("GDN qkv|z", 16384, 2560, 36), ("GDN b|a", 96, 2560, 36), ("GDN out_proj", 2560, 6144, 36),
         ("att q", 12288, 2560, 12), ("att k / v", 512, 2560, 24), ("att o_proj", 2560, 6144, 12),
         ("indexer qk", 640, 2560, 12), ("8-bit 640", 640, 2560, 12),
         ("router", 512, 2560, 48), ("shared gate|up|gate", 1281, 2560, 48), ("shared down", 2560, 640, 48),
         ("HC down mixer", 324, 10240, 96), ("HC up mixer", 10240, 324, 96),
         ("PLE key", 10240, 2560, 1), ("PLE value", 2560, 2560, 1), ("head (V1 16K list)", 16384 + 64, 2560, 1)]
params = sum(n * k * c for _, n, k, c in DENSE)
dense_mac = params * ROWS
SLOTS = ROWS * 10 * (1 - 0.291)
EXP_MAC = 48 * SLOTS * 3 * 2560 * 512
GDN_STATE = 36 * 2 * 48 * 128 * 128 * 4                     # bytes: read + write once per layer


def att(keys):
    by = 12 * 2 * 256 * 2 * 2 * keys                            # K + V, 2 kv heads, hd 256, bf16
    mac = 12 * 24 * ROWS * keys * 256 * 2                       # q.k and p.v
    return by, mac


p(f"# check25_bound (Leap): the layer-sequential lower bound of a 32-row lossy check on this Mac")
p(f"dense params per check {params / 1e9:.3f} B -> {dense_mac / 1e9:.0f} G MAC at 32 rows = {dense_mac / MMA16 * 1e3:.2f} ms "
  f"bf16 / {dense_mac / MMA8 * 1e3:.2f} ms int8 at the measured matrix peak")
p(f"expert slots per layer {SLOTS:.0f} (1c) -> {EXP_MAC / 1e9:.0f} G MAC = {EXP_MAC / MMA16 * 1e3:.2f} ms bf16 (the experts are "
  f"DRAM-bound)")
for keys in (1024, 2048):
    by, mac = att(keys)
    p(f"cores at {keys} keys: GDN state {GDN_STATE / 1e6:.0f} MB = {GDN_STATE / BW * 1e3:.2f} ms; attention KV {by / 1e6:.0f} MB, "
      f"{mac / 1e9:.1f} G MAC = {max(by / BW, mac / MMA16) * 1e3:.2f} ms")
by2, mac2 = att(1024)
CORES = GDN_STATE / BW + max(by2 / BW, mac2 / MMA16)            # at 1K keys (HumanEval-like contexts)

dn = F["dense"]
eg = F["gb_exp"]["c1_eng"]
e6 = 48 * 33.6 * 2.135e-3 * (F["gb_exp"]["c1_eng"] / F["gb_exp"]["c1_map"])   # Fit12's C6 (lambda 1.8e-2)
CAND = [("C5 = 1c-tiers + V1 + V3 (the target's bytes)", dn["l45_v1v3"], eg),
        ("C6 = C5 at lambda 1.8e-2", dn["l45_v1v3"], e6),
        ("C8 = C5 + V4 (GDN / attention 3-bit)", dn["l45_v1v3v4"], eg),
        ("C6 + V4", dn["l45_v1v3v4"], e6)]
p()
p("| bytes | dense MMA | GB (dense + experts) | byte floor 280 | bound at 100 % of every peak | at 90 % | at 85 % | "
  "efficiency needed for 25 ms |")
p("|---|---|---|---|---|---|---|---|")
rows = []
for name, dgb, egb in CAND:
    for mma, lab in ((MMA16, "bf16"), (MMA8, "int8")):
        d = max(dense_mac / mma, dgb * 1e9 / BW)
        e = max(EXP_MAC / mma, egb * 1e9 / BW)
        t = (d + e + CORES) * 1e3
        need = t / 25.0
        p(f"| {name} | {lab} | {dgb + egb:.2f} ({dgb:.2f} + {egb:.2f}) | {(dgb + egb) / 0.28:.1f} | **{t:.1f}** "
          f"(dense {d * 1e3:.1f}, experts {e * 1e3:.1f}, cores {CORES * 1e3:.1f}) | {t / 0.90:.1f} | {t / 0.85:.1f} | "
          f"{100 * need:.0f} % |")
        rows.append(dict(name=name, mma=lab, gb=round(dgb + egb, 3), bound_ms=round(t, 2), need=round(need, 3)))
p()
c5 = rows[0]
p(f"- At 295 GB/s (the best streaming read) every byte term is x0.949: C5 bf16 bound "
  f"{(max(dense_mac / MMA16, dn['l45_v1v3'] * 1e9 / BW_BEST) + eg * 1e9 / BW_BEST + GDN_STATE / BW_BEST + max(by2 / BW_BEST, mac2 / MMA16)) * 1e3:.1f} ms.")
p("- Today, for scale [P, Fit12 v2]: C5 on today's > 16-row route 62.4-65.9 ms; with the one-pass FAST forms 51.3-55.7;")
p("  whole-check efficiency ~0.35-0.45 of the peaks. Best single kernels today: NAX6-N experts ~74 % of streaming,")
p("  DN-KS out / o_proj 236 GB/s, GDN in_proj 213 GB/s (16 rows); the 32-row c2a in_proj 1.52x its matrix floor.")
json.dump(dict(rows=rows, dense_params=params, dense_mac=dense_mac, exp_mac=EXP_MAC, cores_ms=CORES * 1e3,
               bw=BW, mma_bf16=MMA16, mma_int8=MMA8), open(OUT.replace(".txt", ".json"), "w"), indent=1)
open(OUT, "w").write("\n".join(L) + "\n")
