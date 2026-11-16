"""treek_price2.py (Leap, Mon 16 Nov 2026 00:5x): R(K) for the bigger trees, recalibrated on the ENGINE.
Breakthrough's CHECK55 engine profile (Mon 00:42, quiet ticket, exact-45 loop at C 160, 3 G32T windows x 16 rounds,
48 pairs per arm; ~/Breakthrough/data/m1/check55/engine/c55eng_c55engine.{txt,json}, commit a082f8c) measured:
  - walls (median per check): A16 61.11 ms, C32 (v32's split) 102.09, C32one (step 1 complete) 90.46: R(32) one pass
    1.480 on the engine vs 1.345 on C7c's slice, which compresses ratios (its check is ~half Python build and its
    experts are a smaller share);
  - per class at 16 and 32 rows (the pass-K profile, split path): the experts 33.71 -> 50.65 ms, x1.50 = the measured
    union ratio (109.7 / 73.2), i.e. ~1.26 x their byte floor at both sizes. The rest (dense, cores, head, PLE) 30.62 ->
    54.36 split; C32one takes 9.25 ms of GPU busy off it (-11.64 wall).
Model (engine basis, every K-row value above 32 [est]):
  experts(K) = 33.71 x union(K) / union(16) (the measured unions, treek_union.py);
  rest, one pass: 30.62 x (1 + g1 x (K/16 - 1)), g1 from C32one (one weight pass per op; the cores per row);
  rest, split:    30.62 x (1 + gs x (K/16 - 1)), gs from C32 (today's 16-row calls);
  24 rows: the second tile is padded to 16 rows, so the rest costs as at 32 rows minus the cores' 8 rows (the cores are
  the measured gdn_core + att_core: 3.41 ms per 16 rows);
  each sum is scaled to the wall by the arm's measured wall / class-sum ratio (so 32 rows reproduces 1.671 / 1.480);
  contract = one pass - 4 ms [est, C8]; floor = 3.44 GB dense + 48 x union x 2.784 MB at 280 GB/s (the real records3
  pieces; 2.135 MB until 11-16 13:4x, when the experts read ~1.26x "their floor": on real sizes ~0.97x, ~290 GB/s).
Writes data/week98/treek_costs_v2.json (NewIdeas' costs_v0 format, A16_ms 61.11) and treek_price2.txt.
usage: [BigMoE venv] python -I treek_price2.py"""
import json
import os

ENG = os.path.expanduser("~/Breakthrough/data/m1/check55/engine/c55eng_c55engine.json")
OUT = os.path.expanduser("~/Leap/data/week98")
A16W, C32W, C32ONEW, GPU_ONE = 61.11, 102.09, 90.46, -9.25          # walls (txt) and C32one - C32's GPU busy delta
E4R = 2.784e-3                     # GB per 4-bit expert READ (records3 pieces); was 2.135e-3 until 11-16 13:4x
UNION = {16: 73.2, 24: 93.3, 32: 109.7, 48: 135.5, 64: 156.5, 96: 187.4, 128: 211.1}
KS = sorted(UNION)
pk = json.load(open(ENG))["passk"]["dd__mc_27921"]
cls16 = {k: v[1] for k, v in pk["A16"].items()}
cls32 = {k: v[1] for k, v in pk["C32"].items()}
S16, S32 = sum(cls16.values()), sum(cls32.values())
E16, E32 = cls16["moe_experts"], cls32["moe_experts"]
N16, N32s = S16 - E16, S32 - E32
N32o = N32s + GPU_ONE
CORES16 = cls16["gdn_core"] + cls16["att_core"]
gs, g1 = N32s / N16 - 1, N32o / N16 - 1
k16 = A16W / S16                                    # wall / class-sum, per arm
ks32, ko32 = C32W / S32, C32ONEW / (S32 + GPU_ONE)


def experts(K):
    return E16 * UNION[K] / UNION[16]


def rest(K, g, n32):
    if K == 24:
        return n32 - CORES16 / 2                    # padded second tile: the 32-row rest minus the cores' 8 rows
    return N16 * (1 + g * (K / 16 - 1))


def floor_ms(K):
    return (3.44 + 48 * UNION[K] * E4R) / 280 * 1e3


R = {"floor": {}, "today": {}, "onepass": {}, "contract": {}}
lines = [f"engine classes (pass K, ms): 16 rows sum {S16:.2f} (experts {E16:.2f}, rest {N16:.2f}); 32 rows split sum "
         f"{S32:.2f} (experts {E32:.2f} = x{E32 / E16:.3f}, union x{UNION[32] / UNION[16]:.3f}; rest {N32s:.2f}); one pass "
         f"rest {N32o:.2f}; growth per extra 16-row tile of the rest: split {gs:.3f}, one pass {g1:.3f}; cores {CORES16:.2f} "
         f"per 16 rows",
         f"wall / class-sum: A16 {k16:.3f}, C32 {ks32:.3f}, C32one {ko32:.3f}",
         f"{'K':>4s} {'union':>6s} | {'floor':>6s} | {'split ms':>8s} {'R':>5s} | {'1-pass ms':>9s} {'R':>5s} | {'contract R':>10s}"]
for K in KS:
    if K == 16:
        sp = op = A16W
    else:
        sp = (experts(K) + rest(K, gs, N32s)) * ks32
        op = (experts(K) + rest(K, g1, N32o)) * ko32
    R["floor"][str(K)] = round(floor_ms(K) / A16W, 3)
    R["today"][str(K)] = round(sp / A16W, 3)
    R["onepass"][str(K)] = round(op / A16W, 3)
    R["contract"][str(K)] = round((op - 4.0) / A16W, 3)
    lines.append(f"{K:4d} {UNION[K]:6.1f} | {floor_ms(K) / A16W:6.3f} | {sp:8.2f} {sp / A16W:5.3f} | {op:9.2f} {op / A16W:5.3f} | "
                 f"{(op - 4.0) / A16W:10.3f}")
assert abs(R["today"]["32"] - C32W / A16W) < 0.005 and abs(R["onepass"]["32"] - C32ONEW / A16W) < 0.005
costs = {
    "source": "Leap ~/Leap/data/week98/treek_costs_v2.json (scripts/week98/treek_price2.py), Mon 16 Nov 00:5x: R(K) "
              "recalibrated on the ENGINE (Breakthrough's CHECK55 engine profile, Mon 00:42, a082f8c: A16 61.11, C32 "
              "102.09, C32one 90.46 ms; per-class pass K). Replaces v1, whose 32-row one-pass value (C7c's slice 1.345) "
              "does not hold on the engine (1.480). 32 rows measured; other K [est].",
    "A16_ms": A16W, "overhead_ms": 4.0, "lossy32_ms": 25.0, "lossy_floor32_ms": 20.9,
    "R": R,
    "c": {"16": 1.025, "24": 1.035, "32": 1.04, "48": 1.055, "64": 1.07, "96": 1.09, "128": 1.11},
    "note": "today = REF2-T as the engine runs it (16-row calls; G32T measured 1.625 at 32 rows, this profile 1.671). "
            "onepass = step 1 complete (v32one's forms + seams to 32); beyond 32 rows it assumes one-pass K-row forms "
            "with the 16->32 per-tile growth of the non-expert part. The experts scale with the measured union at ~1.26 "
            "x their byte floor (engine). contract = onepass - 4 ms [est]. c(K) = midpoints of Leap's first-cut ranges.",
}
json.dump(costs, open(os.path.join(OUT, "treek_costs_v2.json"), "w"), indent=1)
out = "\n".join(lines)
print(out)
open(os.path.join(OUT, "treek_price2.txt"), "w").write(out + "\n")
