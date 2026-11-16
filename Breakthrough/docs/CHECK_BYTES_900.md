# CHECK BYTES (the 900 push, PLAN900 bet 5): a code-scoped lossy 32-row check at <= ~5.5 GB, at quality

Owner: Breakthrough. NewIdeas assigned it on Mon 16 Nov 12:2x, after the orchestrator named me.
- Kernels: Leap (CHECK25, 79a850f).
- Gates: EvalSuite, lossygate amendment 5 (1e8b9a5): G1 = C6 + int8 dense, G2 = G1 + V4, each against lossy-45 on a
  new snapshot.
- Builds: BigMoE.

Names follow Fit12's LOSSY32 table (~/Fit12/data/lossy32/lossy32_price.txt, v2):
- C5 = lossy-45 + V1 (16K code lm_head list) + V2 (1c tiers, lambda 1.6e-2) + V3 (HC mixers + PLE at 4-bit);
- C6 = C5 at lambda 1.8e-2;
- V4 = GDN + attention projections at 3-bit;
- int8 = int8 activations on the FAST dense matmuls (Leap's dn32q form).

## 1. Price of C6 / int8 / V4 on code windows (pre-registered Mon 16 Nov 12:4x, CPU, before computing)

**Rows.** All rows are on the map's 24 code windows, 32-row blocks, in the engine's semantics. Dense is 2.13 GB
(lossy-45 + V1 + V3) unless V4 is on.
- C5 (the reference) and C5 + int8.
- C6 (35.7 eq on code) and C6 + int8 (= G1).
- C8 = C5 + V4, and C8 + int8.
- C6 + V4, and C6 + V4 + int8 (= G2).
- The <= 5.5 GB scan without V4, each with and without int8: three tiers at s 0.98, lambda 2.0e-2 and 2.2e-2; three tiers
  at s 0.935, lambda 1.2e-2 and 1.6e-2. Their code 4-bit-eq comes from step A.

**Method.**
- **A. Code 4-bit-eq and predicted code flips (experts only) for the scan options.**
  - Source: scripts/m1/dense_cut_price.py's own tier rule and calibration (7 drop arms, leave-one-arm-out ranges), run
    from its file with the output redirected.
  - The 4-bit-eq is counted on code blocks only (5fc4c34's per-domain count).
  - Self-check: s 0.98 / lambda 1.8e-2 must give 33.6 pooled, 35.7 on code, and code flips 1.21 [1.06, 1.28] /
    0.59 [0.52, 0.63], as published in data/m1/dense_cut_price.out.
- **B. GB, floor and realistic ms: Fit12's v2 ladder.**
  - Its script runs unchanged, as in e62c405, with rows added through its own cand().
  - Expert GB for carried rows = 48 x eq x E4 x ENG_OVER_MAP (Fit12's carry).
  - V4 = Fit12's dense -0.335 GB and its V4_GDN_X.
- **int8 dense activations.**
  - Bytes: no change.
  - Time model [est]: int8 removes a share f of the 16 -> 32-row growth term of the dense matmul classes (hc, attp,
    gdnw, moe, ple, head). The attention / GDN cores and the experts are untouched.
  - Low end of the realistic range, f = 1: the 32-row pass becomes as fast as the 16-row one. Leap's int8 compute bound
    (4.93 ms) is under the 7.6 ms byte floor, so the pass is byte-bound again.
  - High end, f = 1 - 1/1.55 = 0.355: only the added tile's MMA runs faster, at M27-C's lowest int8 rate.
  - No CPU change (fused quantize) [est].
  - Leap's measured dn32q / qmm32n ratio (due this evening) replaces this model.
- **C. Leap's bound (check25_bound.py's method) on code bytes.**
  - Bound = dense max(compute 8.53 ms bf16 or 4.93 int8, bytes / 280) + experts bytes / 280 + cores 1.2 ms.
  - Efficiency needed for 25 ms = bound / 25.

**Self-checks.**
- (a) The published variant is byte-identical to Fit12's txt.
- (b) The C5 and C8 rows equal Fit12's to the printed digit; C6 equals e62c405.
- (c) The bound for C5 equals Leap's 23.9 (bf16) / 23.0 (int8), and for C8 its 23.9 / 21.8.

**Predictions [P].**
- Among the asked combinations, only C6 + V4 is <= 5.5 GB (~5.43). int8 changes no bytes.
- int8 saves ~1.4-3.7 ms realistic on C5.
- Without V4, <= 5.5 GB needs lambda ~2.0e-2 at s 0.98 (~5.46 GB), with predicted code flips above C6's.
- On code, Leap's "efficiency needed" rises from 84 to ~87 % for C6 + int8, and from 79 to ~82 % for C6 + V4 + int8.
  Its C6 rows used the pooled 5.55 GB.

**Quality status, from the record (no new measurement):**
- C5: map PASS, 1.09 / 0.62 on 24 code windows (f755d70).
- C6: unscreened. Predicted 1.65 additive, ~1.33 with the measured dense increment (5fc4c34).
- V4: map KILL. RTN 4.43 / 3.62 and AWQ-style 3.86 / 2.99 on 12 code windows (d9b2db7, 43ef47c), against the lines
  1.34 / 0.87.
- int8 activations: no quality data.

**Action.** Send NewIdeas, the orchestrator and Leap (and EvalSuite, for the scope):
- the table;
- which combination reaches <= 5.5 GB;
- the scope of each change;
- the gate-design notes.

### Result (Mon 16 Nov 12:34, 5 s CPU): only V4 gets an asked combination under 5.5 GB, and V4 is the lever with a map KILL

Data: data/m1/check_bytes_900_price.out (+ .json under data/m1/check_bytes_900/). Script:
scripts/m1/check_bytes_900_price.py.

**Self-checks, all PASS:**
- (A) s 0.98 / lambda 1.8e-2 gives 33.56 pooled / 35.72 code, and flips 1.208 [1.057, 1.284] / 0.588 [0.518, 0.630].
  The dense_cut_price.py re-run output is identical to the committed one.
- (B a) Fit12's published output is reproduced byte for byte.
- (B b) The C5 and C8 rows equal Fit12's cand(); C6 equals e62c405.
- (C) Leap's bound: C5 23.9 / 23.0 and C8 23.9 / 21.8.

**Price on code (32-row check; realistic = Fit12's v2 one-pass ladder, not built; bound = Leap's at 100 % of every
peak):**

| candidate | GB (dense + experts) | <= 5.5 | floor ms | realistic ms | Leap's bound | efficiency for 25 ms | code flips |
|---|---|---|---|---|---|---|---|
| C5 (1c, lambda 1.6e-2) | 6.12 (2.13 + 3.99) | no | 21.9 | 51.3-55.7 | 23.9 | 96 % | MEASURED 1.09 / 0.62 |
| C5 + int8 | 6.12 | no | 21.9 | 47.6-54.3 | 23.0 | 92 % | int8: no data |
| C6 (lambda 1.8e-2) | 5.77 (2.13 + 3.63) | no | 20.6 | 49.0-53.4 | 22.7 | 91 % | 1.33 / 0.71 [composed] |
| **G1 = C6 + int8** | 5.77 | no | 20.6 | **45.3-52.1** | 21.8 | **87 %** | 1.33 / 0.71; int8: no data |
| C8 = C5 + V4 | 5.79 (1.80 + 3.99) | no | 20.7 | 49.1-54.0 | 23.9 | 96 % | V4 map KILL |
| C6 + V4 | 5.43 (1.80 + 3.63) | YES | 19.4 | 46.8-51.8 | 22.7 | 91 % | V4 map KILL |
| **G2 = C6 + V4 + int8** | 5.43 | YES | 19.4 | **43.6-50.5** | 20.6 | **82 %** | V4 map KILL; int8: no data |
| lambda 2.0e-2, s 0.98 (32.7 eq) | 5.46 (2.13 + 3.33) | YES | 19.5 | 47.0-51.5 | 21.6 | 86 % | 1.45 / 0.80 [composed] |
| the same + int8 | 5.46 | YES | 19.5 | 43.4-50.1 | 20.7 | 83 % | int8: no data |
| lambda 2.2e-2, s 0.98 (30.2 eq) | 5.21 | YES | 18.6 | 45.4-49.8 | 20.7 | 83 % | 1.56 / 0.89 (both over) |
| s 0.935, lambda 1.2e-2 (30.9 eq) | 5.28 | YES | 18.8 | 45.9-50.3 | 20.9 | 84 % | 1.42 / 0.78 (s 0.935 reach) |

**Reading.**
- **Which reaches <= 5.5 GB.**
  - Of the asked combinations, only those with V4 (C6 + V4, and G2): 5.43 GB.
  - V4's two measured forms are map KILLs at ~3x the code line (RTN 4.43 / 3.62, AWQ 3.86 / 2.99).
  - Without V4, lambda 2.0e-2 at s 0.98 reaches 5.46 GB. Its predicted code flips are 1.45 / 0.80 [composed]: all-row
    over the 1.34 line, non-tie within 0.87.
  - The composed predictor reads high on C5 (1.20 predicted vs 1.09 measured on 24 windows). With that bias, lambda
    2.0e-2 is ~1.34, i.e. at the line.
- **int8 changes no bytes.** It is worth 1.4-3.7 ms realistic [est] and moves Leap's bound 0.9 ms (C5) to 2.1 ms (C6 +
  V4), where V4's dense is byte-bound.
- **Leap's CHECK25 efficiency on code** (its table used the pooled 5.55 GB): G1 needs 87 %, not 84 %; G2 needs 82 %, not
  79 %.
- **s 0.935 rows** weight quality by today's drafter's reach (0.935^d). They are optimistic for the 97.8 % drafter
  (0.978^d), so they are not candidates for the 900 line as priced.
- **Realistic ms** are Fit12's one-pass forms (not built). Today's > 16-row route adds ~10-11 ms.

## 2. G1 / G2 map pre-screens: 4 arms on 24 code windows (pre-registered Mon 16 Nov 13:0x, before any arm is built or run)

NewIdeas gave the GO at 12:4x. These are priors for EvalSuite's lossygate amendment 5 gates (G1 Tue, G2 Wed). They
are information, not a gate (EvalSuite 12:5x).

**Map.**
- ~/FableGod/decision_map: the exact-45 model, teacher-forced.
- DM_OUT=out_code24, DM_SETS=code, --max 24: the 24 code windows of Sunday's combined-plan run (f755d70).
- Flips are counted against out_code24/baseline.json (exact). Ties = the baseline's own p1 - p2 <= 0.1.
- Ranges: 95 % window bootstrap (10,000 draws, seed 20261116).

**Arms, in run order (one process, one model load):**
1. **bt_g1 = G1.** It has four parts:
   - C6 tiers: s 0.98, lambda 1.8e-2; 4-bit / MSB-3 / drop per 32-row answer block. Masks out_code24/bt/tiers_c6,
     from the exact routing, with code24_comb.py's rule.
   - V3: HC mixers, inject and PLE at 4-bit (bt_mix4).
   - int8 activations on the ANSWER rows only, as Leap's int8d.py (7536125) runs in the verify phase. Rule: per (row,
     64-group), scale = max|x| / 127 (1 if all zero), round, clamp to +-127. It covers every 4-bit g64 projection of
     Leap's list: GDN in / out, attention q / k / v / o, indexer, shared expert, PLE key / value. Not the router, HC
     mixers, head or experts. Prompt rows stay bf16.
   - V1: the 16K code shortlist, applied offline to the recorded top-8 (as f755d70).
2. **bt_c6comb** = arm 1 without int8 (G1's bytes).
3. **bt_dense3** = V4's standalone quality.
   - Leap's DENSE-3 (dense3.py 894b0ee, through 11-04's "dense3" arm in dm_ablations.py).
   - GDN in / out + attention q / k / v / o at the MSB-3 midpoint (codes & 0xEEEEEEEE, biases bf16(b + s / 2)).
   - Exact experts, no shortlist.
4. **bt_l20comb** = arm 2 at lambda 2.0e-2 (the no-V4 route to <= 5.5 GB).
5. **bt_restore** = no perturbation. It is the validity check that every teardown restored the model.

**Lines** (code, point estimates): all-row 1.34 %, non-tie 0.87 %.

**Verdicts and decisions** (NewIdeas 12:3x / 12:4x):
- Arms 1, 2, 4: PASS = both lines met, else OVER (naming the line).
- Arm 3, as registered on 11-04, on all-row flips: PASS <= 1.34, BETWEEN <= 2.01, else FAIL.
- Arm 1 OVER: G1's Tuesday gate is reconsidered (bisect early; arm 2 says whether C6 alone is within).
- Arm 3 PASS: Leap + BigMoE build V4 as a plane-split, code-only, RAM-neutral format.
- Arm 3 BETWEEN or FAIL: G2 is dropped and nobody builds V4.
- Arm 4 informs the no-V4 route to <= 5.5 GB.

**Predictions [P].**
- Arm 2: ~1.22-1.33 / ~0.71. The composed estimate is 1.33 / 0.71; C5 measured 1.09 against a predicted 1.20.
- Arm 1: arm 2 + int8's increment, guessed +0.05-0.3 all-row (Leap: int8's error per projection is ~2x bf16's).
- Arm 3: ~3.5-4 %, so FAIL. On 12 windows RTN gave 4.43 and AWQ 3.86; the MSB-3 midpoint's weight MSE is ~27 % below
  RTN-3 from 4-bit.
- Arm 4: ~1.34 / 0.80, at the all-row line.

**Validity.** A failure voids that arm.
- INVALID if an arm's top-8 ids and log-probs equal the baseline's on every row: the perturbation did not reach the
  forward.
- int8 wiring:
  - every wrapped projection sees each window row exactly once and in order (its per-window row count equals the
    window's);
  - the quantized rows are exactly the answer rows;
  - the census is logged.
- Masks:
  - the builder at lambda 1.6e-2 reproduces out_code24/bt/tiers_1c bit for bit;
  - on the 12 screen windows, the full blocks' 4-bit-eq equals the recount's code values (35.72 at 1.8e-2, 32.7 at
    2.0e-2) within 0.1.
- int8 numerics (CPU, before the run). On random data (w ~ N(0, 0.02) at 4-bit g64, x ~ N(0, 1) bf16), the fake-quant
  path's max error / max |ref| must be ~2x the bf16 path's, as in Leap's int8d check (6.6-7.9e-3 vs 2.9-3.9e-3).
- bt_restore must equal the baseline bit for bit. Otherwise every arm after the first unrestored one is VOID.

**Run gating.**
- Start after both:
  - (a) EvalSuite's T1C_MON2_END: mtime >= 12:01:46, pid 60135 gone, read twice 30 s apart;
  - (b) BigMoE's G1 key-test end marker.
- Not after 22:00. Otherwise Tue ~07:00-08:30, before Leap's engine slot #1.
- End by 22:28 at the latest: TERM, and the finished arms are kept.
- A peer share lock (MEM 16 GB), NO-SWAP >= 24 GB (peak ~16 + 8), a swap guard of +2 GB, HOLD / PAUSE / lid checks,
  one process.

**Read-out.** scripts/m1/g1g2_map.py read -> data/m1/g1g2_map.out. The verdict goes to NewIdeas, the orchestrator and
EvalSuite the same evening.

**Amendment (a), Mon 16 Nov 13:1x, before any arm runs: G1 exactly as gated, and the arm order.**
- **What G1 is.** BigMoE's G1 stack (~/BigMoE/data/stack/g1x45g1a_env.txt, 12:43) is t1c1's line with
  R57_TIERS_LAM 0.018 + R57_INT8D=1 + R57_INT8D_SETS = code sets.
  - That is lossy-45 + C6 tiers + int8, with lossy-45's HC8 mixers and SVH head.
  - It has no V1 (16K list) and no V3 (HC4): BigMoE has not built them this week.
- **What that changes in the price.** G1 as gated reads 2.59 + 3.63 = 6.22 GB per 32-row check. The 5.77 GB of section
  1 needs V1 + V3 built and gated too.
- **New arm 0, bt_g1h8 = G1 as gated.**
  - C6 tiers + hc8w (HC mixers + PLE at 8-bit, lossy-45's HC8) + int8 on the answer rows.
  - With PLE at 8-bit, int8 leaves it, as int8d does.
  - Read-out: the full head (no shortlist; SVH reads 102K rows and is near the full head).
- **The G1 decision reads bt_g1h8.** bt_g1 (with V1 + V3) stays, as the full C6 bytes plan.
- **New order:** bt_g1h8, bt_dense3, bt_g1, bt_restore, bt_c6comb, bt_l20comb.
  - The decision arms come first; the first arm runs on a fresh model.
  - bt_restore checks the teardowns of the three arms before it.
  - The last two are attribution (C6 without int8) and the no-V4 route.
- **Prediction for bt_g1h8 [P]:** ~1.3-1.5 / ~0.65, at the all-row line. The experts alone are predicted at 1.21 / 0.59;
  hc8w alone measured 0.33 / 0.06 (sub-additive); int8 adds a little.
- Lines, verdicts and validity are unchanged.

**Amendment (b), Mon 16 Nov 13:1x, before any arm runs: G2's no-V4 route.**
- **NewIdeas' 13:0x decisions.** V1 + V3 ride with G2 (Wednesday, one stacked gate):
  - G2 = G1 + V1 + V3 + V4 if bt_dense3 PASSes;
  - G2 = G1 + V1 + V3 + lambda 2.0e-2 if it does not.
- **New arm bt_l20g2 = that no-V4 G2.** lambda 2.0e-2 tiers + V3 (bt_mix4) + int8 on the answer rows + V1 offline.
  - 5.46 GB.
  - Same lines; PASS = both met.
- **Order:** bt_g1h8, bt_dense3, bt_l20g2, bt_g1, bt_restore, bt_l20comb, bt_c6comb (~35 min). A late start cuts
  attribution first.
- **Prediction [P]:** bt_l20comb ~1.34 / 0.80 + int8's increment, so ~1.4-1.6 / ~0.85. At or over the all-row line.
- **V1 + V3 builds (NewIdeas 13:1x):** Breakthrough writes both as opt-in installers against the g1a snapshot. BigMoE
  wires, snapshots and key-tests them (Tue evening or Wed morning). G2's MEM counts V3's ~0.4 GB of extra 4-bit copies.
