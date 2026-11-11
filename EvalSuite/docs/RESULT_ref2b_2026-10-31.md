# REF2b reference rows + decision 4's same-scores test: results (rows Fri 2026-10-30 17:57 – Sat 03:16)

Pre-registration: `docs/PREREG_ref2b_2026-10-30.md` (74d4f2a; amendments 1-6, all written before the rows they
affect).

## Verdict: PASS (same scores; decision 4's rule, R-40 step 2's rule unchanged)
REF2b against today's stock reference (`ref.r2`), on all 210 core rows:

| set | REF2b | ref.r2 | paired delta [90% CI] | McNemar p | rule (delta > −5 and p >= 0.05) | REF2 (R-40), for comparison |
|---|---|---|---|---|---|---|
| math (n 50) | 88.0 % | 82.0 % | +6.0 [+0.4, +11.6] pt | 0.25 (0/3) | **PASS** (a gain, not significant) | +8.0, p 0.12 |
| IFEval (n 100, strict, prompt level) | 87.0 % | 90.0 % | −3.0 [−6.7, +0.7] pt | 0.38 (4/1) | **PASS** | −3.0, p 0.38 |
| HumanEval+ (n 60, pass@1) | 91.7 % | 91.7 % | 0.0 | 1.00 | **PASS** | 0.0 |

- **The standard table's last line says "FAIL (4/5 rules pass)".** That comes from its truncation/loop rule, which the
  PREREG lists as information to flag, not as a pass line.
  - REF2b loops on 3 rows: HumanEval/93, ifeval/3326 and ifeval/3439. `ref.r2` loops on 1.
  - REF2, which decision 4 approved, looped on 4: these three plus gsm8k/709, whose loop REF2b removes.
  - The floor, the same weights with only a different summation order, loops on 2 of them (ifeval/3326, 3439).
  - So this is the pattern decision 4 already accepted, with one loop fewer.
  - Truncations: REF2b 15, `ref.r2` 15, REF2 12.
- **The other information lines are within the floor's range:**
  - decode answer agreement with `ref.r2` 96.2 % (floor 96.2 %);
  - length ratio ×1.002;
  - hesitation 0.26 vs 0.36 per 1K tokens;
  - token-identical outputs 68/210 (floor 59/210).

## The reference rows
- **68 rows decoded:** the 65 core rows whose AR reaches a bucketed step (n_prompt + n ≥ 497), and 3 spot checks that
  end just short of the boundary. The other 142 were copied from `results/ref2` after the checks.
- **Checks (`results/ref2b_regen.txt`): all pass.**
  - The 3 spot checks are token-identical to REF2.
  - No decoded row differs from REF2 before its first bucketed step.
  - Of the 65, 22 are identical to REF2 and 43 differ after the boundary. The first differences sit at n_prompt + i
    = 498 to 976.
  - This confirms Leap's construction: REF2b = REF2 below 512 keys, and only the rows that pass 512 keys can change.
- **So `results/ref2b` (210 rows) is REF2b's own AR reference.** By NewIdeas' routing it replaces `results/ref2` as the
  45 GB line's exact reference, within Kamen's decision 4.

## REF2b vs REF2 (information)
- math 88.0 vs 90.0 % (−2.0 pt, p 1.00, 2/1 discordant; final-answer agreement 94 %).
- IFEval 87.0 vs 87.0 % (final-answer agreement 100 %). HumanEval+ 91.7 vs 91.7 %.
- 79,628 vs 78,879 generated tokens.

## What it means
- **For the 45 GB route:** REF2b, the reference the live loop needs past 512 keys, scores the same as today's stock
  reference. Its reference rows are in place, so R-56 on REF2b (runner 31, two arms: REF2b, and REF2b + GATE16) can
  now judge the live loop at full length against REF2b's own AR. It started at 03:17.
- **For decision 4:** its condition (same scores on math, code and instructions) holds for REF2b as it did for REF2.

## Notes
- **Code:**
  - REF2b frozen in `snap/leap-faff8e5`; BigMoE frozen in `snap/bigmoe-6c64148`.
  - Rows decoded before 20:43 ran at C 192 (pin e7d0e3c042b8); later rows ran at C 96 (amendment 3, pins
    e874e8c4c0f3, fd5c247326d1). The streamer is exact at any C, so C changes speed only.
  - The checks above cover every decoded row.
- **Interruptions:**
  - Swap-guard stops at 19:33, 20:43, 22:50 and 22:57, with Kamen's own apps (iOS Simulator, a Windows game,
    Chrome) and shared runs beside us.
  - Two shared chunks failed at 21:21 and 21:32, beside MTP Head's shared runs: their rows in flight were not saved and
    were redone.
  - Yields to waiting quiet tickets at 21:51, 23:56, 00:08 and 00:47.
  - Heat pauses: many rows show 80–300 s paused.
  - None of these change a row: a row is saved only from an uninterrupted, gate-checked pass.
- **Rule changes, each recorded before the rows it affects:**
  - memory floors (amendment 2);
  - a shared lock and C 96 (3), per Kamen's FAST INTEGRATION rule;
  - a vm_stat gate (4), then a NO-SWAP gate (5), per the orchestrator's RULES.md updates;
  - back to quiet tickets (6), per the orchestrator at 00:4x.
- **Memory (lifetime footprint):** C 192 rows 34.2–35.6 GB; C 96 rows 21.4–22.3 GB.
- **Speed:** information only, and mostly shared GPU with heat pauses: e.g. 11.6 tok/s at C 192 under a quiet lock,
  ~7 tok/s at C 96 shared.
- **Data:**
  - `results/ref2b/decode.jsonl` (210 rows, 142 marked `copied_from`);
  - `results/ref2b_regen.txt`;
  - `results/report_ref2b_vs_ref.r2.txt`, `results/report_ref2b_vs_ref2.txt`;
  - `results/day30_runner.log`.
