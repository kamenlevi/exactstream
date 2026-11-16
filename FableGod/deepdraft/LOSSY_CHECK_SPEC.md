# LOSSY_CHECK_SPEC: env flags, counters and base for the lossy-check variants (for BigMoE's 1c integration and EvalSuite)

Breakthrough, Sun 15 Nov 2026 01:2x. It answers EvalSuite's three asks (01:1x) and the orchestrator 01:2x. The plan is
in LOSSY_CHECK_PLAN.md; the map screens are in its "Map screens" section.

## The base every variant sits on

**lossy-45** (Leap's pkg_120b `configs/lossy-45.env`, release v1.2.0, tag 21fb59b): config L + COH, C 256.
- It already ships **R57_HC8=1** (HC mixers at 8-bit) and **QOM keep tables on the code sets** (`R57_QOM_SETS`, set by
  run.py), with exact expert reads on chat.
- So the shipped lossy dense read is ~2.81 GB, not exact's 3.44: the plan's "mixers 8-bit" rows are this base.
- EvalSuite gates each variant against lossy-45 run once on the panel, not against exact, because that measures the
  variant's own cost.
- Correction (EvalSuite 02:0x): "-2.65 pt vs exact on HumanEval+ + MBPP" is CERT's forward order only. The reverse
  order is +0.76, and both orders pooled are -0.95 pt (McNemar p 0.38, 8 vs 13). So the shipped stack is not
  significantly below exact.

## The variants (in gate order) and their env flags (names proposed; BigMoE implements)

| variant | adds to lossy-45 | flags | bytes vs the base (32-row check) |
|---|---|---|---|
| V1 | lm_head shortlist on code sets | `R57_LMSL=<code16k.npy>` `R57_LMSL_SETS=humaneval,humanevalx,mbpp` | -0.33 GB |
| V2 | 1c expert tiers on code sets, replacing QOM's keep in the check | `R57_TIERS=1c` `R57_TIERS_LAM=0.016` `R57_TIERS_S=0.98` `R57_TIERS_SETS=humaneval,humanevalx,mbpp` | experts at 36.4 four-bit-equivalent per layer (the plan's 1c) [CORRECTED Mon 16 Nov: 39.2 on code windows; 36.4 was the all-domain mean] |
| V3 | V2 + mixers / inject / PLE at 4-bit | V2's + `R57_HC4=1` `R57_HC4_SETS=humaneval,humanevalx,mbpp` (census mix 194, inject 96, PLE 2) | another -0.34 GB beyond HC8 |
| V4 | V3 + GDN / attention at 3-bit | only if the map's bt_ga3 passes | another -0.33 GB |

The set-scoped switch follows `R57_QOM_SETS`:
- the engine engages the lever only when the current item's set is in the list (the same plumbing run.py uses for QOM);
- outside the list the base path runs unchanged.

## The engagement counter (one runs.json field per prompt)

`"lossy_check": {"checks": n, "e4": n, "e3": n, "edrop": n, "lmsl_rows": n, "lmsl_fallback": n, "hc4": 0|1}`
- e4 / e3 / edrop count the union experts per check by tier (4-bit / MSB-3 / dropped), summed over the prompt's checks.
- lmsl_rows counts rows scored on the shortlist; lmsl_fallback counts rows that fell back to the full head.
- EvalSuite sums `e3 + edrop + lmsl_rows`, plus `hc4` for V3 (PREREG_lossygate amendment 1d, runner 64's key subsets;
  `checks` and `e4` are not in the sum). The sum must be > 0 on the code sets and exactly 0 on IFEval (or any set outside
  the lists).

## Patch notes for the 1c integration (BigMoE, Sun)

1. **Where:** the check's MoE layer, after the router. It works for any check size (16-row today, 32-row once the
   gate is widened); s^d weights the check's rows by depth d (0 = the first verified row).
2. **Rule, per check and per layer:**
   - For every expert e in the union, I_e = sum over rows d of s^d w(d, e)^2 (w = the row's top-10 weights,
     normalised as the block does).
   - Tier: 4-bit if I_e > 2.775 lambda; MSB-3 (read only the records3 MSB plane) if 0.8457 lambda < I_e <= 2.775 lambda;
     else drop. Use the model's own weights, no renorm.
   - At lambda 0.016: 4-bit if I > 0.0444, MSB-3 if 0.0135 < I <= 0.0444.
3. **QOM:** with R57_TIERS set, the tiers replace QOM's keep tables in the check. Don't stack them: the map priced 1c
   alone. The base's other levers (HC8, COH, the relaxed span accept) stay.
4. **Shortlist:**
   - Gather the K rows of lm_head for the set's list (code: the 16,384 most frequent top-1 tokens of the deepdraft bank;
     Breakthrough can export the token-id file).
   - Add each row's drafted token. Take logits over the union and the argmax over the union.
   - Optional fallback to the full head when the best shortlist logit's margin is small (counted in lmsl_fallback).
5. **Self-check before the gate (CPU):** replay the rule on captured routing for the map's 12 code windows. The per-row
   rank tiers must equal ~/FableGod/decision_map/out/bt/tiers_1c/<window>.npz ('keep' = 4-bit, 'keep3' = MSB-3) bit
   for bit. They were made by Breakthrough's scripts/m1/build_tiers_masks.py from the exact routing.
6. **Counters:** as above, so that EvalSuite's tier 0 can confirm engagement in its first minutes.

## The 12 GB engine (the orchestrator's question)

- The 12 GB line already runs HC8 and the adopted rank-based MSB-3 (ranks 2-9 at MSB-3, QOM0 keeps). It has no
  importance-based tiers and no 32-row check (it checks <= 13-16 rows).
- A code-scoped 1c variant could be gated there as a quality pre-check: the tier rule's quality effect does not depend
  on the pools. But the bytes win is a property of the 45 GB 32-row check.
- So integrate and gate at 45 GB first (BigMoE), and keep the 12 GB route as a fallback if the 45 GB engine slips.
