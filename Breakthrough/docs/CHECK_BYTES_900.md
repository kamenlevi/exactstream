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

**Amendment (c), Mon 16 Nov 13:2x, before any arm runs: bt_g1h8 uses hc8w_mix.**
- g1a's HC8 is `hc8r.hc8_install(model)` with its defaults, ple=False (snapshot r57_screen_g1a.py:945-951). Only the 194
  mixers are 8-bit; block_inject and PLE stay bf16.
- So bt_g1h8 = C6 tiers + hc8w_mix (mixers only) + int8. With PLE in bf16, int8 leaves PLE alone, as int8d does in g1a.
- dm_bt.py is reinstalled (432 lines, blob 05e098f9); CPU test ALL PASS (24). The armed runner loads it at run time.
- Read-out of the shipped stack, for the record:
  - SVH serves only calls of <= 16 rows (the full head above that).
  - SVH never adds the drafted token: a draft outside its set gets -inf.
  - G1's head is therefore lossy-45's SVH as already gated.

**Amendment (d), Mon 16 Nov 13:2x, before any arm runs: the V1 read-out.**
- The registered shortlist is IN SAMPLE (section 3). The bank it was counted from contains the 24 evaluated windows, and
  the "16K" list is really every top-1 token of the bank (8,775), so it misses only 2 of their 6,327 rows by
  construction.
- Arms with V1 (bt_l20g2, bt_g1, bt_c6comb, bt_l20comb) therefore get three head reads:
  - the registered one, kept for continuity;
  - V1-16k + session rows;
  - V1-32k + session rows (the list proposed to ship).
- The out-of-sample lists are data/m1/lmsl_code16k.npy / 32k.npy: Magicoder top-1 tokens + FR-Spec fill. Session rows =
  the prompt + the committed answer prefix. Drafts are not added, so the reads are conservative.
- **G2's V1 decision reads V1-32k.** The runner was disarmed 13:12 for EvalSuite's gate change and re-arms with this.
- The runner's EvalSuite gate is now T1C_RESF2_END (mtime >= 13:15) + no t1c_resume_follower2.sh / t1c_chain2.sh
  (EvalSuite 13:1x: a second pause makes T1C_MON3_END a mid-run marker). BigMoE re-armed on the same gate at 13:14.

## 3. V1's code list was in sample; out-of-sample lists (Mon 16 Nov 13:1x, CPU) [post hoc, unregistered]

- **Found while designing V1.** The "16K code list" (LOSSY_CHECK_PLAN, dense_cut_price.py, the map's bt_comb +
  shortlist, f755d70) was counted from all of ~/datasets/deepdraft/bank.
  - The bank is 400 Magicoder + 100 MBPP + 24 HumanEval files, and the 24 HumanEval files ARE the map's 24 code windows
    (same names; the windows' baseline top-1 sequence sits inside the bank's tk).
  - The bank has only 8,775 distinct top-1 tokens, so "the top 16,384" was all of them. The list's map misses (0.03 %)
    and the shortlist's share of the combined plan's flips were in sample, and say nothing about V1.
  - The combined plan's 1.09 / 0.62 still stands for 1c + V3; the shortlist added ~nothing there.
- **Out of sample (scripts/m1/v1_lmsl_lists.py -> data/m1/v1_lmsl_lists.out).** Lists from the Magicoder part only,
  filled by FR-Spec rank. Forced misses = the baseline top-1 outside list U session rows, on the 24 windows' 6,327
  answer rows, with no drafts:

| list | size | outside the list | outside list + session rows |
|---|---|---|---|
| in-sample all-bank (as used) | 8,775 | 0.03 % | 0.03 % |
| Magicoder only | 8,157 | 8.47 % | 1.03 % |
| Magicoder + FR-Spec to 16,384 | 16,384 | 3.21 % | **0.35 %** |
| Magicoder + FR-Spec to 32,768 | 32,768 | 1.61 % | **0.14 %** |
| SVH base (FR-Spec top 98,304) | 98,304 | 0.08 % | 0.00 % |

- **Proposal: ship V1 at 32K, not 16K.**
  - 16K forces 0.35 % misses, a quarter of the 1.34 % code line, to save ~0.12 GB vs SVH.
  - 32K forces 0.14 % and saves ~0.09 GB (65,536 fewer head rows x 2,560 x 0.5625 B).
  - The extra 0.03 GB is ~0.1 ms at 280 GB/s. G2's no-V4 route stays <= 5.5 GB (5.49).
  - Drafts added per check (V1's rule) lower the misses further: a draft equal to the true token is scored.

**Amendment (e), Mon 16 Nov 13:3x, before any arm runs (NewIdeas 13:2x): bt_g1h8_res = G1 as gated + RES.**
- **What RES is.** Fit12's resident-only 3-bit tier at C 256 (PLAN_12GB_90_fit12.md "RES-1c PRICE RESULT"). 1c replaces
  QOM's resident keep on code, leaving ~60 misses per 32-row check; RES drops an MSB-3-tier expert that is not resident,
  for ~8 misses and about -13 ms. Price: +0.046 / +0.033 pt code flips.
- **The masks.** out_code24/bt/tiers_c6res (scripts/m1/g1g2_map.py resmasks) applies Fit12's rule to the C6 masks: per
  32-row answer block and layer, an MSB-3-tier expert with any non-resident slot is dropped. Residency is out/res_c256
  (my LRU replay of the 45 GB base pool, dm_res_masks.py).
- **Window coverage.** Residency exists only for the 12 screen windows; windows 13-24 keep plain C6 masks.
  - So RES's read is the increment paired on windows 1-12: bt_g1h8_res vs bt_g1h8.
  - Self-check: the dropped share of MSB-3-tier experts is plausible next to Fit12's 3.58 % at 1c.
- **The arm.** tiers_c6res + hc8w_mix + int8 (G1 as gated, + RES). Order: after bt_l20g2, before the rest (~40 min in
  all).
- **Prediction [P]:** a paired increment of ~+0.05 pt. 12 windows (~3,400 rows) cannot resolve that from 0, so the read
  only rules out a large surprise (say > +0.3 pt) before RES joins G2.
- Verdict and lines as for the other arms (bt_g1h8_res: PASS = both met on all 24 windows).

## 4. G2's V1 + V3 installers (Mon 16 Nov 13:2x-13:3x; NewIdeas 13:1x; due Tue 18:00), BUILT and TESTED (210d95d)

- **V1 = scripts/m1/g2/lmsl.py (R57_LMSL).**
  - A second SVH state over data/m1/lmsl_code32k.npy (section 3), registered in ref2._SH. SVH's own feed (the prompt,
    the first token, committed ids) reaches it.
  - ref2._screened is wrapped: ON + the verify phase + the trunk head's <= 16-row call -> logits over list U session U
    drafts. add_drafts(dr) runs per check.
  - Counters: lmsl_rows, and lmsl_fallback (top-2 tie p1 - p2 <= 0.1, counted on the GPU, not acted on).
  - GPU tests 12/12 on the real lm_head: unset path bitwise; engaged values within 3.6e-3 of a float32 reference; argmax
    right; feed / drafts / counters / _qlin / remove.
- **V3 = scripts/m1/g2/hc4.py (R57_HC4).**
  - prepare() runs before hc8_install: 4-bit copies (== bt_mix4) of the 194 mixers, 96 injects and 2 PLE, the PLE DNR.
  - engage() swaps the module objects per item after the prefill. Leap's hc4i / hc4r (7c3e149) read the 4-bit mixers.
  - CPU tests 11/11.
- **Wiring:** scripts/m1/g2/G2_WIRING.md, against g1a's lines, for BigMoE's g2a. Results: data/m1/g2_installer_tests.out.
- **MEM:** + ~0.3-0.4 GB (V3 copies) + 53 MB (V1 state). The V1 list (32K vs 16K) is NewIdeas' call.

**Amendment (f), Mon 16 Nov 13:5x, before any run: the optional comment-QOM arms (NewIdeas 13:4x, information only).**
- **Why.** Fit12's read of the 12 HumanEval windows: 35 of 1c's 38 flips are on "#" comment rows, including all 16
  non-tie flips; code rows flip at QOM's rate. The candidate fix, if 1c-tiers fails RUNAWAY, is comment rows on QOM.
- **Map.** The ORIGINAL 12-window map (DM_OUT=out, DM_SETS=code, --max 12), where Fit12's masks and the residency exist.
  - Masks: copies of Fit12's masks_1c_comqom (md5 537a81b1ab82) and masks_1c_comqom_res (4a085fd69800), in
    out/bt/cq_comqom(_res).
  - Arms registered in memory by scripts/m1/comqom_run_map.py; the armed G1 / G2 runner and dm_bt.py are untouched.
- **Arms** (all with hc8w_mix + int8 on the answer rows, G1's dense form):
  - bt_cq_ctrl = 1c tiers (the matched control);
  - bt_cq = 1c with comment rows on QOM;
  - bt_cq_res = bt_cq + RES.
- **Slot:** Tue ~07:00-07:15, after tonight's screen; Leap's smoke stays as armed. Read-out (scripts/m1/comqom_read.py)
  to NewIdeas, Fit12, EvalSuite and BigMoE by ~07:30.
- **Reading** (NewIdeas' rule; a decision aid for G1's re-plan, not a gate line). On all-row flips, non-tie printed
  beside:
  - CONFIRMED if bt_cq <= 0.5 x bt_cq_ctrl (G1's fix = comment-QOM);
  - NOT CARRIED if bt_cq >= 0.8 x bt_cq_ctrl (the re-plan falls back to a lower lambda);
  - BETWEEN otherwise (Fit12's 4-key kill test decides).
  - bt_cq_res - bt_cq is RES's paired increment, judged against <= +0.3 pt (as G2's rule).
- **Prediction [P]** (Fit12): bt_cq ~0.42 / ~0.06, vs bt_cq_ctrl near 1c's measured 1.13 / 0.48 plus a little from
  HC8 + int8: CONFIRMED.

## 5. Re-price with the engine's real expert pieces (pre-registered Mon 16 Nov 13:5x, CPU, before computing)

- **Leap 13:45:** every check budget priced experts 23-30 % low.
  - The constant used was 2.135 MB (4-bit) / 1.661 MB (MSB-3), from Leap's 11-14 cheapcheck.md, in Fit12's lossy32_price
    E4 / E3 and in my LOSSY_CHECK_PRICE.
  - The engine reads records3 pieces: 4-bit 2.784 MB, MSB-3 2.064 MB (index, pools, config).
  - My check from the checkpoint headers: one expert's 4-bit gate / up / down + bf16 scales / biases = 2.765 MB
    (layer 5, 512 experts), within 1 % of Leap's piece.
- **Method.**
  - check_bytes_900_price.py --pieces re-runs everything with E4 = 2.784e-3 and E3 = 2.064e-3 GB in Fit12's script (its
    E4 / E3 line replaced).
  - Carried lambda rows count bytes as n4 x 2.784 + n3 x 2.064 MB per layer. The policy stays as defined: tiers chosen
    with sizes 1 / 0.778 / 0, the masks unchanged.
- **Self-checks.**
  - C5 experts 5.05 GB and C6 4.60 GB (Leap).
  - Leap's corrected bounds at 100 % of every peak: C5 27.7 (bf16), C6 + int8 25.2, C6 + V4 + int8 24.0, each within 0.1.
- **Prediction.**
  - G1 as built ~7.2 GB; G2 routes ~6.2-6.6 GB. Nothing on the table reaches 5.5 GB.
  - The 25 ms check needs >= ~96 % of every peak even with V4. Bytes stop being "done": the experts dominate.

### Result (Mon 16 Nov 13:5x, CPU): at the engine's real pieces, no lever on the table makes 25 ms reachable

Data: data/m1/check_bytes_900_price_pieces.out (+ .json). All self-checks PASS:
- C5 experts 5.04 GB and C6 4.60 GB, == Leap's;
- Leap's corrected bounds reproduced exactly: C5 27.7 / 26.8, C6 + int8 25.2, C6 + V4 + int8 24.0;
- the normal mode still reproduces ddd7a0b byte for byte.

| candidate (code, 32-row check) | GB (old constant) | GB (real pieces) | floor ms | realistic ms | Leap's bound | efficiency for 25 ms |
|---|---|---|---|---|---|---|
| C5 | 6.12 | **7.18** | 25.6 | 51.3-55.8 | 27.7 | 111 % |
| G1 = C6 + int8, with V1 + V3 | 5.77 | **6.73** | 24.0 | 45.3-52.0 | 25.2 | **101 %** |
| G1 as built (lossy-45 dense 2.59, no V1 / V3) | 6.22 | **7.19** | 25.7 | - | - | - |
| G2 = C6 + V4 + int8 | 5.43 | **6.40** | 22.8 | 43.5-50.4 | 24.0 | **96 %** |
| G2 no-V4 = lambda 2.0e-2 + int8 | 5.46 | **6.35** | 22.7 | 43.3-50.0 | 23.8 | **95 %** |
| lambda 2.2e-2 + int8 (flips 1.56 / 0.89: over) | 5.21 | 6.02 | 21.5 | 42.5-48.3 | 22.6 | 91 % |

**Reading.**
- **Times.** Realistic ms barely move: Fit12's time model is calibrated on measured engine ms, so it was right all
  along. Only the GB, the floors and the bounds were understated (by ~1 GB per check).
- **Bytes are binding again.** 25 ms needs 95-101 % of every peak for G1 / G2. Whole checks today reach 35-45 %; the
  best single kernels reach 74-84 %.
- **What would make 25 ms plausible at ~80-85 %:** a bound of <= ~21 ms. With V4 + int8 that means experts <= ~3.7 GB
  per check, ~28 4-bit-eq per layer vs C6's 34.7. That is past every screened lambda's quality.
- **So the bytes plan needs a new expert lever, not a lower lambda.** Candidates:
  - reach-weighted tiers for TREE checks: off-spine nodes are rarely reached, so their exclusive experts can drop. The
    900 line checks trees;
  - per-layer lambda;
  - a 2-bit tier, which needs a records plane the engine lacks.
- **The <= 5.5 GB target** came from the old constant. At the real pieces it means ~4.6 GB at the old constant's scale.
  It should be restated in ms.

## 6. Per-layer lambda: a band screen (pre-registered Mon 16 Nov 14:0x, before any mask or run; NewIdeas 13:5x GO)

**Question.** The quality law treats every layer alike: flips = a E^b, with E summed over layers / 48. Is any layer band
cheaper in flips per GB saved? If one is, a higher lambda there is the expert lever that 25 ms needs (experts
<= ~3.7 GB vs C6's 4.6).

**Arms** (experts only: the exact dense path, tiers via dm_bt's tiers(); 24 code windows in out_code24, DM_SETS=code,
--max 24; registered in memory, the armed runners untouched):
- bt_c6tiers = C6 everywhere (lambda 1.8e-2): the reference.
- bt_lb0 / bt_lb1 / bt_lb2 = lambda 3.2e-2 on layers 0-15 / 16-31 / 32-47, C6 elsewhere.
- Masks: out_code24/bt/tiers_lb{0,1,2}, g1g2_map.py's rule with a per-layer lambda.

**Bytes** (real pieces, full blocks, 24 windows; computed 14:0x): C6 experts 4.61 GB = L0-15 1.67 + L16-31 1.48 + L32-47
1.46. lambda 3.2e-2 on one band saves 0.70 / 0.61 / 0.56 GB (-42 / -41 / -38 % of that band).

**Uniform-law prediction [P]** (experts only, 12 screen windows): C6 1.21 / 0.59. The band arms:
- L0-15: +0.27 / +0.22 pt;
- L16-31: +0.23 / +0.19 pt;
- L32-47: +0.22 / +0.17 pt;
- about 0.04 pt per 0.1 GB saved in every band.

**Read-out** (scripts/m1/lband_read.py). For each band arm vs bt_c6tiers, paired on the same rows: the all-row and
non-tie increments, the GB saved, and the measured cost per 0.1 GB vs the uniform law's.

**Verdict** (a decision aid for G2's expert lever):
- A band is TOLERANT if its measured all-row cost per 0.1 GB is <= 0.5x the uniform law's, and SENSITIVE if >= 2x.
- **PASS = at least one TOLERANT band.** The lever exists; a confirm arm at that allocation on G2's stack follows (Wed).
- **KILL = no TOLERANT band.** Per-layer lambda gives no lever; the bytes need another route.

**Slot:** Tue morning, after Fit12's 4-key kill test (which follows my COMQOM_END), gated on its end marker. Latest start
09:00, hard stop 09:25 (Leap's slot #1 at 09:30). ~20 min, ~14 GB, NO-SWAP >= 24, share lock.

**Amendment (f'), Mon 16 Nov 14:0x, before data (NewIdeas 14:0x):** the comment-QOM reading uses NON-TIE flips, with
all-row printed beside: hc8w + int8 in both arms dilute an all-row ratio. Same cut-offs (CONFIRMED <= 0.5x the control,
NOT CARRIED >= 0.8x, else BETWEEN).
- **V3's slice test PASS (Leap 14:28, his data/week98/check25/hc4_slice_run4.log; 4 layers, my install order).**
  - Mechanics: engage swaps every mixer. hc4i serves every read at 16 rows (hc4r) and at 32 (read32, over v32 and
    f32one). engage(False) restores bit for bit. Never engaged, the 8-bit reads pass through.
  - Numerics vs MTPLX's eager read of the same 4-bit modules: mixed 0.64-1.01 % rel, inject 1.03 %, written bitwise.
    That is the FAST HC form's own gap (today's HC8 is 0.93 % off its eager 8-bit read).
  - Slice argmax vs all-eager HC: 16/16, 29/32 (v32), 32/32 (f32one).
  - So G2 runs ~1 % off the map's bt_mix4 numerics in the HC mixed output: the same gap HC8 already carries in the gated
    lossy-45.

**Amendment (g), Mon 16 Nov 17:2x, before any arm runs: section 2's slot moves to now, beside EvalSuite's waiting B2.**
- **Why.** BigMoE 17:15: EvalSuite's B2 has waited for memory since 15:04 (NO-SWAP 36-41 GB < its 43; apps stay open
  until 23:00). Its chain may run to the 22:15 no-start cutoff, so BigMoE's key test starts ~22:15-22:30 at the
  earliest. The runner would SKIP its 22:00 latest start, and Tue ~07:00 is taken (comment-QOM -> Fit12's kill test ->
  band screen -> Leap's slot #1 at 09:30).
- **Who agreed.** NewIdeas 17:2x: yes, once EvalSuite has no objection. It checked EvalSuite's runner log at 17:23: B2
  reads 38 GB and needs 43, and re-checks every 10 min, so a yield costs B2 at most ~10 min. EvalSuite 17:2x: no
  objection. Its terms:
  - keep "MEM 18 GB" in the share lock's reason (its runner yields to it at each process start);
  - stop if swap grows by more than 2 GB (the runner's own guard);
  - if the orchestrator grants B2 an at-peak start at NO-SWAP ~38, let the current arm finish and stop by ~18:20
    (finished arms kept), or TERM sooner on EvalSuite's word.
- **Start.** The frozen runner's manual GO (data/m1/g1g2_map/GO). Unchanged: its NO-SWAP >= 26 GB gate, share lock
  (MEM 18 GB), swap guard, HOLD / lid stops and the 22:28 hard stop; the arms, their order, the lines and the read-out.
- **Yield** (scripts/m1/g1g2_yield.sh, beside the run). B2's view without my run = NO-SWAP + the map process's current
  phys_footprint. If that is >= 43 GB (B2's gate) on two reads 30 s apart, the watcher TERMs the map process. The runner
  then unlocks, reads out the finished arms and writes G1G2_MAP_END.
  - Stub simulation 4/4: yield, B2's runner gone, no map process, a normal end.
  - The footprint read was checked on a 1 GB allocation (1.01 GB).
- **After a yield** (NewIdeas 17:2x): the rest re-runs with --skip-done right after BigMoE's key test if it can end
  before M4 (23:05), else Tue 03:00 after E14.
- **Read-out.** The verdicts go to NewIdeas in one message: bt_g1h8, DENSE-3 (PASS <= 1.34 % code flips / BETWEEN /
  FAIL) and RES's increment. NewIdeas orders Tuesday's G1 and Wednesday's G2 from them.
- **Addendum, Mon 17:3x-17:4x: two starts stopped at my own 22 GB footprint guard; run 3 from 17:38.**
  - Run 1 (17:30:41, the GO) and run 2 (17:34:37) each stopped ~55 s in, in bt_g1h8's second window: lifetime
    phys_footprint 22.099 / 22.097 GB, MLX peak active 15.35 GB. Earlier map runs were ~16.0 GB at the same 15.33 GB.
    No arm finished; the read-outs say NOT RUN. Their end markers were moved aside (.selfguard1731 / .selfguard1735).
  - Run 2 capped MLX's buffer cache at 2 GB (scripts/m1/g1g2_run_map.py, G1G2_CACHE_GB, default 2). It did not change
    the peak, so the extra memory is not the cache. The cap stays (bt_restore checks results bit for bit under it).
  - Run 3: BG_PEAK_MAX=30 in the runner's environment; scripts/m1/fp_log.sh samples the map process every 15 s
    (data/m1/g1g2_map/footprint.log). Steady ~19-20 GB, transient peaks to 23.7 GB on a 398-position window (the
    longest of the 24). NO-SWAP was 38 GB at the start; the swap guard (+2 GB) is unchanged.
  - EvalSuite 17:3x stopped its chain (B2 had not started), withdrew the 18:20 stop and set the order: this screen,
    then BigMoE's key test, then its B2 -> V2. The yield watcher exited (B2's runner gone).
    scripts/m1/g1g2_softstop.sh (stop after the next complete arm file, or at a deadline; stub simulation 4/4) was
    built for the withdrawn stop and is not running.

### Section 2 result (Mon 16 Nov 19:33; run 3, 17:38:21-19:33:05, all 8 arms, rc 0)

Data: data/m1/g1g2_map.out (+ .json), data/m1/g1g2_map/run.log, footprint.log and pins.txt (sha256 of the arm files in
~/FableGod/decision_map/out_code24).

**Validity: all PASS.**
- bt_restore equals the baseline bit for bit: top-8 ids and log-probs, every row of the 24 windows.
- Each of the 4 int8 arms quantized exactly the answer rows: 2,667,420 / 2,680,122 row-modules, as wanted, with no
  window mismatch. The census is logged; 2 extra PLE modules appear where PLE is 4-bit.
- No arm equals the baseline on any row (0 of 6,327 rows identical).
- RES masks: 3.11 % of MSB-3-tier experts dropped as non-resident (Fit12 at 1c: 3.58 %).

**Verdicts.** 24 windows, 6,327 rows (6,258 non-tie). The lines are 1.34 % (at most 84 rows) and 0.87 % (at most 54
non-tie rows). Ranges are the 95 % window bootstrap; the full table, with every head read, is in data/m1/g1g2_map.out.

| arm | head read | all-row % (rows) | non-tie % (rows) | verdict | [P] |
|---|---|---|---|---|---|
| bt_g1h8 = G1 as gated (C6 + HC8 mixers + int8) | full | 0.95 [0.49, 1.53] (60) | 0.53 [0.29, 0.81] (33) | **PASS** | 1.3-1.5 / 0.65 |
| bt_g1h8_res = G1 as gated + RES | full | 0.98 [0.49, 1.65] (62) | 0.56 [0.31, 0.85] (35) | **PASS** | - |
| bt_dense3 = DENSE-3 alone (V4's quality) | full | 2.09 [1.10, 3.42] (132) | 1.44 [0.77, 2.30] (90) | **FAIL** (> 2.01) | 3.5-4, FAIL |
| bt_l20g2 = G2 no-V4 (lambda 2.0e-2 + V3 + int8 + V1) | V1-32k (G2's read) | 1.36 [0.66, 2.22] (86) | 0.93 [0.45, 1.47] (58) | **OVER** (both) | 1.4-1.6 / 0.85 |
| bt_g1 = C6 + V3 + int8 + V1 | V1-32k | 1.33 [0.65, 2.20] (84) | 0.90 [0.46, 1.39] (56) | **OVER** (non-tie) | - |
| bt_c6comb = C6 + V3 + V1 | V1-32k | 1.31 [0.69, 2.06] (83) | 0.86 [0.43, 1.32] (54) | PASS | 1.22-1.33 / 0.71 |
| bt_l20comb = lambda 2.0e-2 + V3 + V1 | V1-32k | 1.34 [0.68, 2.23] (85) | 0.85 [0.45, 1.30] (53) | **OVER** (all-row, 1 row) | 1.34 / 0.80 |

- **RES increment**, paired on windows 1-12 (3,366 rows): all-row 1.19 -> 1.25 (+0.06 pt), non-tie 0.63 -> 0.69
  (+0.06 pt). Fit12 priced it at +0.046 / +0.033, and the prediction was ~+0.05. There is no large surprise (< +0.3 pt).
- **V1-16k + session rows is OVER on every arm** (1.50-1.53 / 1.01-1.09). The in-sample registered reads (1.22-1.26 /
  0.74-0.82) are void as V1 evidence (section 3).

**Decisions, as registered.**
- **G1 PASS** (0.95 / 0.53): Tuesday's G1 gate stands as gated, with no bisect.
- **DENSE-3 FAIL** (2.09 > 2.01): nobody builds V4, and G2 takes the no-V4 route, G1 + V1 + V3 + lambda 2.0e-2
  (amendment (b)).
  - It did much better than predicted (3.5-4), since the MSB-3 midpoint beats RTN.
  - It is over BETWEEN's ceiling too, so the decision is the same either way.
- **G2's no-V4 route at its V1 read (32K + session rows, no drafts) is OVER both lines**, by 2 / 4 rows. That read adds
  no drafts, so it is an upper bound on V1's cost (section 3). This is a prior for EvalSuite's Wednesday gate, not a
  gate.
- **RES** (+0.06 pt paired, as priced) can join G2.

**Post hoc** (unregistered): unpaired differences of the arms' points on the same rows; a paired read is still to do.
- **V1 vs each arm's in-sample read** (which misses 2 rows by construction):
  - 32K costs +0.09-0.11 pt all-row (5-6 rows);
  - 16K costs +0.27-0.29.
- **int8 on the answer rows** (bt_g1 - bt_c6comb and bt_l20g2 - bt_l20comb, every head read): +0.00-0.03 pt all-row,
  below the predicted +0.05-0.3; +0.02-0.08 pt non-tie.
- **lambda 2.0e-2 vs C6** (bt_l20comb - bt_c6comb): +0.01-0.03 pt all-row for 0.38 GB per check (section 5).
- **V3 vs G1's HC8 mixers** (the in-sample reads of bt_c6comb / bt_g1 minus bt_g1h8): +0.27-0.29 pt all-row for
  ~0.34 GB per check (section 5: V1 + V3 save 0.46 GB, V1 at 16K ~0.12).
  - Per 0.1 GB saved, V3 (~0.08 pt) and V1-32K (~0.1 pt) cost about the same.
  - lambda 2.0e-2 costs ~0.005 pt per 0.1 GB, ~20x cheaper.
  - So V3 is what puts G2's no-V4 route at the line.

**Run notes.**
- Wall 6,879 s (1 h 55 min, against ~40 min planned).
  - Arms 5-8 (bt_g1, bt_restore, bt_l20comb, bt_c6comb) took 24 / 6 / 26 / 26 min, against ~10 min for each int8 arm
    before them.
  - The GPU was shared with other share-locked runs then (at 19:3x: MTP Head 25 %, Sandbox98 40 %; GPU 89 %).
- The share lock, renewed to 19:15, ran out 18 min before the end. Next time the lock runs to the hard stop.
- Peak footprint 23.69 GB lifetime, MLX peak active 16.76 GB. Swap went 7,650 -> 7,594 MB.

**Amendment (h), Mon 16 Nov 19:5x, before any band arm runs: section 6 moves to tonight (NewIdeas 19:3x).**
- **Why.** It frees ~45 min of Tuesday's crowded morning (B2 -> V2, G1 base and G1 all moved there), so G1's verdict
  lands earlier.
- **No data dependency.** lband_run_map.py and lband_read.py read only out_code24 and the 14:0x masks. Fit12's kill-test
  marker was only the slot gate.
- **Window.**
  - Start after Sandbox98's screen 2 (its share lock, to ~20:15).
  - Latest start 21:00, hard stop 22:50 (M4 at 23:05; apps quit at 23:00).
  - If BigMoE's key test or Leap's gpub start first, wait for them (NewIdeas).
  - Order guard: if NO-SWAP reads >= 40 while BigMoE's g1_follow waits, its gate is open, so its key test gets 8 min to
    start first.
- **Pieces.**
  - scripts/m1/lband_go_1116.sh: the GO gate. It reads lock files and process lists only.
  - data/m1/lband_map/run_lband_1116.frozen.sh: run_lband.frozen.sh with only the share lock changed. The lock reads
    "MEM 24 GB: ..." and runs to the hard stop + 5 min.
    - 24 GB is the same machinery's measured peak today on the same 24 windows (23.7 GB lifetime).
    - The 15-s samples were 18-20 GB in every arm, int8 or not.
  - scripts/m1/lband_tonight_1116.sh: the launcher (gate + runner, NO-SWAP >= 32, self-guard 30 GB). If any arm is
    missing afterwards, a Tuesday fallback on Fit12's KILL_DONE (07:00-09:00, hard stop 09:25) runs only those arms
    (--skip-done).
- Stub simulations 5/5 (data/m1/lband_map/sim_1116.out).
- **The old Tuesday runner (pid 88753) is stopped.** Arms, lines, verdict and read-out are unchanged.

## 7. Paired attribution of the map arms (pre-registered Mon 16 Nov 19:5x, before computing; CPU, the arm files only)

Section 2's post-hoc differences are unpaired points. This reads the same files paired, for NewIdeas' G2 arming. No
decision rests on it alone.

**Reads** (scripts/m1/g1g2_paired.py -> data/m1/g1g2_paired.out; section 2's flip rule, ties, pooled ratio and
window bootstrap: 10,000 draws, seed 20261116, windows resampled with replacement).
- **A full-head read of the four V1 arms** (pick "full" = the recorded top-1). The recorded top-8 is the full head's,
  and V1 is applied offline, so this is the true no-V1 read that replaces the void in-sample one.
- **Paired differences** in flips per row on the same 6,327 rows (6,258 non-tie), all-row and non-tie:
  1. V3 vs HC8 mixers: bt_g1 (full) - bt_g1h8. Both are C6 + int8; they differ in V3 vs hc8w_mix, plus int8 on PLE.
  2. int8: bt_g1 - bt_c6comb, and bt_l20g2 - bt_l20comb (full).
  3. lambda 2.0e-2 vs C6: bt_l20comb - bt_c6comb, and bt_l20g2 - bt_g1 (full).
  4. V1-32k: each V1 arm's 32K read minus its full read.
  5. G2 no-V4 without V1: bt_l20g2 (full), on the lines.

**Predictions [P].**
- Full reads within 2 rows of the in-sample reads.
- 1: about +0.27 pt all-row, with a CI that excludes 0.
- 2 and 3: +0.00-0.03 pt, with CIs that include 0.
- 4: +0.09-0.11 pt, with a CI that excludes 0.
- 5: ~1.26 / ~0.82, PASS.

### Section 7 result (Mon 16 Nov 19:5x, CPU 0.34 s, peak 58 MB): G2's no-V4 route passes with the full head; V1-32k adds 6 rows

Data: data/m1/g1g2_paired.out. Paired, one shared bootstrap draw matrix; CIs are 95 %.

**Full-head reads of the V1 arms.** They are within 1 row of the in-sample reads [P: within 2]:

| arm | all-row % (rows) | non-tie % (rows) | lines |
|---|---|---|---|
| bt_g1 | 1.23 (78) | 0.78 (49) | PASS |
| bt_c6comb | 1.22 (77) | 0.75 (47) | PASS |
| bt_l20comb | 1.22 (77) | 0.72 (45) | PASS |
| bt_l20g2 = G2 no-V4, no V1 | 1.26 (80) | 0.81 (51) | PASS |

So **G2's no-V4 route passes with the full head**, 4 / 3 rows inside the lines [P: ~1.26 / ~0.82, PASS]. SVH serves
32-row checks with the full head, so that is G2 without V1, at +0.09 GB per check (6.35 -> 6.44).

**Paired differences** (pt all-row [CI] / non-tie [CI]):

| difference | all-row | non-tie | [P] |
|---|---|---|---|
| 1. V3 vs HC8 mixers (bt_g1 - bt_g1h8) | +0.284 [-0.015, +0.587] (+18 rows) | +0.256 [+0.016, +0.499] (+16) | ~+0.27, CI excludes 0: all-row CI just touches 0, non-tie excludes it |
| 2a. int8 (bt_g1 - bt_c6comb) | +0.016 [-0.174, +0.212] | +0.032 [-0.103, +0.178] | CI includes 0: yes |
| 2b. int8 (bt_l20g2 - bt_l20comb) | +0.047 [-0.086, +0.229] | +0.096 [-0.033, +0.262] | yes |
| 3a. lambda 2.0e-2 vs C6 (bt_l20comb - bt_c6comb) | +0.000 [-0.182, +0.191] | -0.032 [-0.167, +0.111] | yes |
| 3b. lambda 2.0e-2 vs C6 (bt_l20g2 - bt_g1) | +0.032 [-0.139, +0.229] | +0.032 [-0.128, +0.213] | yes |
| 4. V1-32k vs full (bt_g1, bt_c6comb, bt_l20g2) | +0.095 [+0.000, +0.206] (+6 rows) | +0.112 [+0.029, +0.221] (+7) | +0.09-0.11, CI excludes 0: all-row lower bound at 0, non-tie excludes it |
| 4. V1-32k vs full (bt_l20comb) | +0.126 [+0.031, +0.241] (+8) | +0.128 [+0.032, +0.245] (+8) | yes |

**Reading.**
- **V1-32k is what puts G2's no-V4 route over the lines.**
  - Its cost is the same 6 rows on three of four arms: forced misses where the baseline's top-1 is outside the list and
    the session rows.
  - Drafts are not in this read. V1's rule adds them per check, so the shipped V1's cost lies between 0 and these
    6-8 rows.
- **V3 is the costliest piece per GB**, 18 rows for ~0.34 GB, but under the full head G2 carries it and still passes.
- **lambda 2.0e-2 vs C6 is free within noise** (0-2 rows for 0.38 GB). That matches the band screen's question about
  where the expert lever is cheap.
- **Options for G2's arming** (NewIdeas' call):
  - (a) the SVH / full head instead of V1 (+0.09 GB);
  - (b) V1-32k with drafts, letting EvalSuite's gate measure it;
  - (c) HC8 in place of V3 (+0.34 GB).

**Addendum to (h), Mon 16 Nov 20:5x: tonight's band run STOPPED at the orchestrator's request; 0 arms; slot to NewIdeas.**
- **Run.** GO 20:26:11, when Sandbox98's lock ended. NO-SWAP was 36 GB (< 40, so BigMoE's key-test gate was closed and
  no grace applied). Start 20:27:04, share lock "MEM 24 GB" to 22:54.
- **Stop.** The orchestrator 20:47: Kamen is at the Mac and asked for the lower-RAM tests. TERM at 20:49:13; the runner
  unlocked and wrote LBAND_END "run rc 143" at 20:49:26.
- **No arm finished.** bt_c6tiers was on its last windows (1,286 s in), and run_map writes an arm only at its end. The
  read-out says NOT RUN.
- **Measured.**
  - Experts-only peak footprint 21.68 GB lifetime, MLX peak active 15.35 GB (data/m1/lband_map/tonight1116/
    footprint.log; 15-s samples 15-17 GB). So the int8 arms' extra ~2 GB (23.7) is int8's, and "MEM 22 GB" fits this
    screen.
  - Pace with the GPU shared (MTP Head at 25 %): ~22 min per arm.
- **Disarmed.** The launcher's Tuesday fallback (07:00-09:00 on Fit12's KILL_DONE) was stopped too: Tuesday morning now
  holds EvalSuite's B2 -> V2, and NewIdeas re-orders the slot. The pieces (6ee5d66) re-arm unchanged with a new gate and
  times.

**Amendment (i), Mon 16 Nov 21:0x, before any band arm runs: the band screen's slot is Tuesday morning (NewIdeas 20:5x),
and both of my Tuesday-morning runs follow the away rule.**
- **Slot.** It runs once all of these hold, read twice: Fit12's KILL_DONE AND my COMQOM_END (both newer than arming), no
  'run_ma[p][.]py' process, and Kamen away.
  - Not before Tue 07:00; latest start 08:45; hard stop 09:25. Leap's quiet slot #1 starts at 09:30, and EvalSuite's 4j
    B2 -> V2 follows it.
  - ~40 min on a free GPU. Finished arms are kept (--skip-done).
- **Wednesday fallback** for unfinished arms: only after the morning run's end. Wed 06:30, latest start 07:15, hard
  stop 08:15, so it ends before G2's day chain.
- **Lock and memory.** Share lock "MEM 22 GB" (measured 21.7) to the hard stop; NO-SWAP >= 30; self-guard 30 GB.
> (a quote from the project owner, kept private)
  Mac):
  - no start while ~/.local/share/quiet/away_secs.sh reads < 900 (read twice; no reading counts as away);
  - a running job yields (TERM, finished arms kept, end marker written) when two reads 30 s apart say Kamen is back,
    as EvalSuite's chains do. HOLD / PAUSE / lid / HEALTH ACT as before.
- **The comment-QOM follower** gets the same away rule and a right-sized lock.
  - Its arms run int8, which peaked at 23.7 GB today, so the lock reads "MEM 24 GB" (the old "MEM 16" undercounted)
    and NO-SWAP must be >= 32.
  - Unchanged: its times (Tue 07:00, latest start 08:20, hard stop 08:40), arms, read-out and COMQOM_END (written on
    every end, so Fit12's kill test still chains on it).
  - The old follower (pid 73554) is stopped.
- **Pieces.**
  - data/m1/comqom_map/run_comqom_1117.frozen.sh;
  - data/m1/lband_map/run_lband_1117.frozen.sh;
  - scripts/m1/lband_tue_1117.sh (the launcher: morning run, then the Wednesday fallback only if an arm is missing).
- Stub simulations 5/5 in data/m1/lband_map/sim_1117.out: waits for away, yields mid-run, skips at the latest start,
  both markers, and the Wednesday fallback after a yield.

**Addendum, Tue 17 Nov 01:2x, before the comment-QOM arms run: an engagement add-on (information only).**
- Why: BigMoE's G1 key test on g1b (00:48) failed only on its COMQOM engagement probe, because g1's own answer to the
  probe key had no '#' line to tag. So the comment-QOM map read-out should show how much the lever engages on the map,
  and where its flips land.
- scripts/m1/comqom_engage.py, run by hand after the run. comqom_read.py, which the armed runner calls, is unchanged,
  and so are its reading rule and verdict.
- Engagement before the run (data/m1/comqom_map/engage_prerun.out):
  - the masks of bt_cq (cq_comqom) differ from the control's (tiers_1c) on 505 of 5,989 positions of the 12 code
    windows (8.4 %);
  - all 505 are scored answer rows (15.0 % of the 3,366 scored rows), none in the prompts;
  - 4 windows have no touched position (HumanEval 121, 129, 135, 136); HumanEval 132 has 225.
- So the map run engages, unlike the key test's probe.
- After the run it adds:
  - flips by where the rows sit: touched scored rows / untouched scored rows of touched windows / windows with no
    touched position;
  - a NULL CHECK: on the 4 untouched windows the masks are identical, so bt_cq must equal bt_cq_ctrl row for row. Any
    difference means run noise or a setup difference, and the reading should be held until it is explained.
- Tested on fakes (scratch): flips planted in touched rows and in an untouched window landed in the right columns, and
  the NULL CHECK flagged the planted difference.
