"""check25_bound_real.py (Leap, Mon 16 Nov 2026 13:4x): check25_bound.py's layer-sequential bound with the REAL expert
piece sizes. Every byte budget so far (Leap's cheapcheck / pipe_price / treek_price / check25_bound, Fit12's lossy32_price,
Breakthrough's LOSSY_CHECK_PRICE) priced an expert at 2.135 MB (4-bit) / 1.661 MB (MSB-3 plane), a constant first written
in Leap's cheapcheck.md on 11-14. The engine reads records3 (native+p11+pad) pieces:
  MSB part 2,064,384 B (3 x w3 614,400 + 6 x s3 / b3 36,864), LSB part 720,048 B (3 x l1 204,800 + 3 x b 35,216)
  -> 4-bit expert 2.784 MB, MSB-3 only 2.064 MB (x1.304 / x1.243); config: 3 x 640 x 2560 at 4-bit g64 = 2.765 MB in MLX.
Checks: records3/index.json layer 20; the c25 engine's pools: C_m 175 + C_l 112 per layer = 21.21 GB over 48 layers
(175 x 2.0644 + 112 x 0.7209 = 442.0 MB per layer). Desk only."""
import json
import os

F = json.load(open(os.path.expanduser("~/Fit12/data/lossy32/lossy32_price.json")))
ix = json.load(open(os.path.expanduser("~/BigMoE/models/bare/records3/index.json")))["layers"]["20"]
E3R = sum(ix["msb_piece_size"]) / 1e9
E4R = E3R + sum(ix["lsb_piece_size"]) / 1e9
E4O, E3O = 2.135e-3, 2.135e-3 * 0.778
BW, BWB, MMA16 = 280e9, 295e9, 14.0e12
MMA8 = 1.73 * MMA16
e4, e3 = F["mean"]["c1_map_e4"], F["mean"]["c1_map_e3"]
eng = F["gb_exp"]["c1_eng"] / F["gb_exp"]["c1_map"]
c5_old = 48 * (e4 * E4O + e3 * E3O) * eng
c5_new = 48 * (e4 * E4R + e3 * E3R) * eng
eq5 = e4 + 0.778 * e3
c6_new = c5_new * 35.7 / eq5                                  # C6 = 4-bit-eq 35.7 per layer on code, C5's tier mix
c6_old = 48 * 35.7 * E4O * eng
DMAC = 3.73e9 * 32                                           # check25_bound: 3.73 B dense params x 32 rows
GDN = 36 * 2 * 48 * 128 * 128 * 4
ATT = max(12 * 2 * 256 * 2 * 2 * 1024 / BW, 12 * 24 * 32 * 1024 * 256 * 2 / MMA16)
CORES = GDN / BW + ATT
dn = F["dense"]
out = [f"# check25 bound with the REAL expert sizes (Leap, Mon 16 Nov 2026): 4-bit {E4R * 1e3:.3f} MB, MSB-3 {E3R * 1e3:.3f} MB "
       f"(the budgets used {E4O * 1e3:.3f} / {E3O * 1e3:.3f})",
       f"C5 map mix per layer: {e4:.2f} at 4-bit + {e3:.2f} at MSB-3 (4-bit-eq {eq5:.1f}); experts per check {c5_old:.2f} GB "
       f"(old) -> {c5_new:.2f} GB (real, x{c5_new / c5_old:.3f}); C6 {c6_old:.2f} -> {c6_new:.2f} GB", "",
       "| bytes | dense MMA | GB real (dense + experts) | byte floor 280 / 295 | bound at 100 % of every peak | efficiency needed for 25 ms |",
       "|---|---|---|---|---|---|"]
for name, dgb, egb in (("C5", dn["l45_v1v3"], c5_new), ("C6", dn["l45_v1v3"], c6_new), ("C5 + V4", dn["l45_v1v3v4"], c5_new),
                       ("C6 + V4", dn["l45_v1v3v4"], c6_new)):
    for mma, lab in ((MMA16, "bf16"), (MMA8, "int8")):
        d = max(DMAC / mma, dgb * 1e9 / BW)
        e = egb * 1e9 / BW
        t = (d + e + CORES) * 1e3
        out.append(f"| {name} | {lab} | {dgb + egb:.2f} ({dgb:.2f} + {egb:.2f}) | {(dgb + egb) / 0.28:.1f} / {(dgb + egb) / 0.295:.1f} | "
                   f"**{t:.1f}** (dense {d * 1e3:.1f}, experts {e * 1e3:.1f}, cores {CORES * 1e3:.1f}) | {100 * t / 25:.0f} % |")
eb = 25e-3 - CORES - max(DMAC / MMA8, dn["l45_v1v3v4"] * 1e9 / BW)
out += ["", f"- For 25 ms at 100 % of every peak with C6 + V4's dense (int8): experts <= {eb * BW / 1e9:.2f} GB per check "
        f"({eb * BW / 1e9 / 48 * 1e3:.0f} MB per layer = {eb * BW / 1e9 / 48 / E4R:.1f} 4-bit or {eb * BW / 1e9 / 48 / E3R:.1f} MSB-3 experts)."
        f" At 85 %: {(25e-3 * 0.85 - CORES - max(DMAC / MMA8, dn['l45_v1v3v4'] * 1e9 / BW)) * BW / 1e9:.2f} GB."]
print("\n".join(out))
open(os.path.expanduser("~/Leap/data/week98/check25_bound_real.txt"), "w").write("\n".join(out) + "\n")
