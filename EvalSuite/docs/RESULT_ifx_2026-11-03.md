# RESULT: IFX, the larger IFEval re-test of REF2c-T vs REF2-T = STOP PASS at stage 1 (Tue 2026-11-03 06:37)

Pre-registration: docs/PREREG_ifx_2026-11-02.md. Amendments 1-2 changed only the pin and the launch times, before any
IFX row.
- **Runner 38:** pid 69420, code pin 125db5ca272b. GO from the orchestrator at 03:20; quiet chunks at C 192.
- **Rows:** 220 new IFEval prompts per arm (results/ifx/stage1_keys.txt), plus 2 reproduction rows per arm.
- **The arms:** REF2c-T (`ref2ct.r2`; Leap's REF2c freeze 91cd316: ref2.CFG att_single + gdn_inproj on REF2-T) and
  REF2-T (`ref2t.r2`).

## Verdict: PASS (decision 4's rule on the new rows), and the stopping rule says stop
| rows | REF2c-T | REF2-T | paired delta [90% CI] | McNemar p (b/c) | rule |
|---|---|---|---|---|---|
| new prompts, stage 1 (n 220) | 84.1 % | 83.2 % | +0.91 [-1.5, +3.3] pt | 0.75 (6/4) | **PASS** |

- **Stopping rule:** the predictive probability that decision 4 on all 441 new prompts does not say FAIL is 0.9989 (the
  line is >= 0.95). So STOP PASS; stage 2 (the other 221) does not run.
- **Reproduction:** each arm's first process re-decoded 2 core rows token for token (no VOID).

## Information (no pass line)
- **Pooled with Part 1's 100 core rows (n 320):** 84.4 % vs 85.3 %, -0.94 [-2.9, +1.1] pt, p 0.61 (6/9): PASS.
- **The 100 core rows alone (Part 1):** -5.0 [-8.6, -1.4] pt (0/5): FAIL. That is Part 1's line, which IFX set out to
  re-test. On 220 fresh prompts the sign turns (+0.9, 6/4), so Part 1's 0/5 reads as those items' luck. Its detail file
  had already found 3 of the 7 Part 1 discordant items to be fragile ones REF2-T happened to pass.
- **Token-identical rows:** 20/220.
- **Truncations:** 14 vs 16. **Loops:** 3 vs 3.
- **Instruction misses by group (REF2c-T / REF2-T):** change_case 2/2, combination 8/6, detectable_content 2/2,
  detectable_format 4/5, keywords 4/5, length_constraints 11/13, punctuation 5/5, startend 3/2.
- **Speed (rough, not an A/B):** 13.95 vs 13.31 tok/s (+4.8 %). Part 1's rough AR numbers were +2.1 % math, +6.0 %
  IFEval and +5.1 % HumanEval+.
- **Peak lifetime footprint:** 32.02 vs 32.06 GB.

## What it means
- **On the pre-registered rule, REF2c-T keeps REF2-T's scores on all three sets.** Math (86 vs 88, 1/0) and HumanEval+
  (93.3 vs 95.0, 1/0) passed in Part 1, and IFEval now passes on the new rows.
- **REF2c-T is not bitwise REF2-T** (20/220 identical rows). It is a different exact path: deterministic, with its own
  kernels.
  - Adopting it means a reference change, or the FAST track under Kamen's two-track rule.
  - Its value is Leap's estimate of ~6 ms less per 16-row check (att_single ~5, in_proj ~0.5-0.9).
  - Adoption is NewIdeas' call, then Kamen's.
- **R-56e** (Part 2: REF2c-T in the 16-row live loop with OOO + the fused combine, every token and row against its own
  AR) becomes eligible again. Its PREREG ran it only after a Part 1 PASS, and NewIdeas orders it.
- **Not BIG:** a re-test that turns a borderline FAIL into a PASS for a ~4-6 % path is not next-caliber.

## Files
- results/ifx_stage1.txt, which is also results/ifx.txt.
- results/IFX_DONE.
- results/day38_runner.log.
- The arms' rows: results/ref2ct.r2/ and results/ref2t.r2/ (decode.jsonl).

## Correction (Wed 2026-11-04 15:14): the stage-1 numbers, re-scored deterministically; the verdict is unchanged
- **Re-scored:** 84.5 % vs 83.2 %, delta +1.36 [-0.9, +3.6] pt, McNemar p 0.51 (6/3): PASS. The stopping rule's predictive
  probability is 0.9998, so the call is still STOP PASS. Tuesday's line above (84.1 %, +0.91, p 0.75, 6/4) came from one
  random draw.
- **Cause:** ifevalx/1129 asks for 6 or more exclamation marks: keywords:letter_frequency with letter "!". IFEval's
  reference checker accepts only a-z there and counts a randomly drawn letter instead.
  - So that item scored at random, per evaluation and per arm.
  - Five fresh scorings of the stored rows gave REF2c-T 185 / 186 / 186 / 186 / 186 and REF2-T 182 / 183 / 183 / 183 / 183.
- **Fix:** evalsuite/ifeval/instructions.py now counts a single non-letter character as given, as the prompt words it.
  - Five scorings are then identical (186 / 183), and 1129 passes for both arms.
- **Scope:** 2 of the 441 IFX prompts (1129 '!' in stage 1, 1122 '#' in stage 2) and none of the core 100, so no
  decision-4 verdict on the core 210 changes. tools/d4x.py reproduces REF2c-T's core FAIL exactly (85.0 vs 90.0, 0/5).
