# REF2c-T: same-scores test vs REF2-T: results (Mon 2026-11-02; rows Sun 23:15 – Mon 06:53)

Pre-registration: `docs/PREREG_ref2ct_2026-11-01.md` (bd1c646; amendments 1-3 changed only launch timing, the deadline
and chunk length), written before any REF2c-T row. REF2c-T = REF2-T with Leap's REF2c freeze 91cd316 (`ref2.CFG`
att_single + gdn_inproj).

## Verdict: FAIL (decision 4's rule, REF2c-T − REF2-T on all 210 core rows): IFEval misses the line
| set | REF2c-T | REF2-T | paired delta [90% CI] | McNemar p (b/c) | rule (delta > −5 and p >= 0.05) |
|---|---|---|---|---|---|
| math (n 50) | 86.0 % | 88.0 % | −2.0 [−5.3, +1.3] pt | 1.00 (1/0) | PASS |
| IFEval (n 100, strict, prompt level) | 85.0 % | 90.0 % | −5.0 [−8.6, −1.4] pt | 0.0625 (5/0) | **FAIL** |
| HumanEval+ (n 60, pass@1) | 93.3 % | 95.0 % | −1.7 [−4.4, +1.1] pt | 1.00 (1/0) | PASS |

- IFEval sits exactly on the line: −5.0 is not above −5. Its McNemar p (0.0625) alone would pass.
- The suite's standard table also fails on IFEval only (4/5 rules):
  - answer agreement 96.2 % (floor 98.1 %);
  - length ×1.010;
  - hesitation 0.25 vs 0.23;
  - truncations / loops 13 / 1 vs 13 / 3.
- **What it means:** by the pre-registered rule, REF2c-T does not replace REF2-T. REF2-T stays the 45 GB default and REF2b
  the 4-bit exact mode. Adoption calls are NewIdeas'.
  - R-56e (Part 2) did not run: the PREREG runs it only after a Part 1 PASS.
  - For the 45 GB route: REF2c's expected gain (Leap's estimate ~6 ms less per 16-row check: att_single ~5,
    in_proj ~0.5-0.9) is not available as an exact-mode swap on this evidence.

## The discordant items (`tools/ref2ct_detail.py` -> `results/ref2ct_detail.txt`)
All seven go against REF2c-T. The other same-weights arms are shown beside them: REF2b (4-bit exact mode), `ref.r2`
(stock) and the floor.

| item | rule | REF2c-T | REF2-T | REF2b | ref.r2 | floor | REF2c-T leaves REF2-T at token |
|---|---|---|---|---|---|---|---|
| ifeval/218 | all lowercase (+ 1 placeholder) | fail: one capital ("Peggy guggenheim", day 10 of 10) | pass | pass | pass | pass | 50 |
| ifeval/2439 | >= 2 placeholders (+ 3 paragraphs, first word) | fail: no placeholder (REF2-T: [officer], [city]) | pass | pass | pass | pass | 4 |
| ifeval/3409 | >= 3 highlighted sections | fail: 2 | pass | pass | pass | pass | 15 |
| ifeval/3608 | the letter j >= 20 times (+ lowercase) | fail: few j-words | pass ("jagged", "join", "job" over and over) | pass | pass | pass | 11 |
| ifeval/2497 | >= 1 highlighted section | fail | pass | fail | pass | fail | 0 |
| math algebra/733 | final answer | fail (cut at 1536 tokens) | pass (EOS at 1499) | fail | fail | fail | 230 |
| HumanEval/93 | tests | fail | pass | fail | fail | pass | 146 |

- **Three of the seven are fragile items that REF2-T happened to pass:** ifeval/2497, algebra/733 and HumanEval/93.
  REF2b and at least one other arm fail them too.
- **On four IFEval items REF2c-T alone fails** (218, 2439, 3409, 3608); REF2-T, REF2b, `ref.r2` and the floor all pass.
  They are ordinary instruction misses after an early divergence (tokens 4-50): a single capital letter, a missing
  placeholder, one highlight short, too few j's.
- **IFEval spread between same-weights arms** (b/c = items only the first arm gets right / only the second; the weights
  are equal, only the numerics differ):

  | pair | b/c | token-identical rows |
  |---|---|---|
  | REF2c-T vs REF2-T | 0/5 | 12/100 |
  | REF2c-T vs REF2b | 3/5 | 9/100 |
  | REF2c-T vs ref.r2 | 0/5 | 9/100 |
  | REF2c-T vs floor | 1/4 | 13/100 |
  | REF2-T vs REF2b | 4/1 | 9/100 |
  | REF2-T vs ref.r2 | 0/0 | 8/100 |
  | REF2-T vs floor | 2/0 | 11/100 |
  | REF2b vs ref.r2 | 1/4 | 16/100 |
  | REF2b vs floor | 2/3 | 14/100 |
  | ref.r2 vs floor | 2/0 | 12/100 |

  - IFEval on these 100 items: REF2c-T 85, REF2b 87, floor 88, REF2-T 90, `ref.r2` 90.
  - Items one arm alone gets wrong (the four others right):
    - REF2c-T 4 (218, 2439, 3409, 3608);
    - REF2b 3 (2328, 2534, 3126; and alone right on 2997);
    - the floor 1 (3280);
    - REF2-T 0, `ref.r2` 0.
    REF2b, adopted as the 4-bit exact mode, shows the same kind of lone misses.
- **Reading (information, no pass line):** IFEval rows diverge early in every pair (8-16 of 100 token-identical), so on
  IFEval each arm is in effect a different greedy sample. REF2c-T is the low end of that spread: 5 below REF2-T, 2 below
  REF2b, 3 below the floor. This test can't tell a small real IFEval drop from the low end of the noise. The rule says
  FAIL either way.

## Information (no pass line)
- **vs REF2b:** every set passes decision 4's rule: math 86.0 vs 88.0 (1/0), IFEval 85.0 vs 87.0 (5/3, p 0.73),
  HumanEval+ 93.3 vs 91.7 (0/1). The standard table passes 5/5.
- **vs `ref.r2`:** math +4.0 (0/2), IFEval −5.0 (5/0; the table's one FAIL), HumanEval+ +1.7 (0/1).
- **Token-identical rows vs REF2-T:** 61/210 (the floor vs REF2-T: 54/210); IFEval 12/100.
- **Generated tokens:** 78,667 (REF2-T 77,861).
- **Engagement:** both switches ran in all 210 rows (att_single and gdn_inproj counts > 0 in every row). Each chunk's
  provenance records `ref2.CFG` as run: att_single and gdn_inproj on, gdn_exact and key_buckets on, tier_nax4 off.
- **Memory (lifetime footprint per process):** 31.77 / 32.25 / 31.93 / 32.38 / 32.13 GB for processes of 7 / 8 / 10 /
  149 / 36 rows. REF2-T's R-58 processes: 31.56-32.25 GB. Flat in rows per process: amendment 3's premise holds at 149
  rows.
- **Rough AR speed** (tokens / (row seconds − heat pause)), REF2c-T vs REF2-T:
  - math 13.82 vs 13.53 tok/s (+2.1 %); IFEval 14.17 vs 13.37 (+6.0 %); HumanEval+ 11.58 vs 11.02 (+5.1 %).
  - The direction matches Leap's estimate.
  - A freeze happens inside a heat pause (in every row of both arms frozen_s <= paused_s), so only the pause is
    subtracted.
  - Shared GPU, not an A/B. MTP Head could train beside Part 1 chunks (amendment 2), and NewIdeas' grep loaded the 04:10
    chunk for ~3 min. REF2-T's rows ran at another time (R-58, Sat 05:04-07:05).
  - Leap's A/B is the speed number; this run doesn't measure REF2c's gain.

## Notes
- **Code:**
  - variant `ref2ct` = `snap/leap-91cd316` (Leap's REF2c freeze) with `ref2.CFG` att_single + gdn_inproj;
  - on REF2-T's tiers: BIGMOE_TIER=4433333333, TIER_GLU=3c, TIER_CL=112, TIER_COMBINE unset (asserted);
  - BigMoE frozen in `snap/bigmoe-6c64148`; C 192.
- **Runner 36** (pin 015eb41be555):
  - Chunks: 23:15 (7 rows), 01:04 (8), 04:10 (10), 05:37 (149, the first 60-min chunk), 06:41 (36); 210/210 by
    06:53.
  - No swap trip and no failed chunk. Heat pauses hit 12 rows; tokens don't depend on them.
  - Reports 06:54-06:55; the runner ended 06:55 without Part 2.
  - Launches: 11:59 Sun (pid 24879, stopped in its wait loop, amendment 1); 12:4x (43408, stopped 21:24, amendment 2);
    21:25 (48937, stopped 04:08 inside its own lock, amendment 3); 04:09 Mon (99782).
- **Data:**
  - `results/ref2ct/decode.jsonl`;
  - `results/ref2ct_d4.txt` (the verdict) and `results/report_ref2ct_vs_ref2t.txt`;
  - `results/report_ref2ct_vs_ref2b.txt`, `results/report_ref2ct_vs_ref.r2.txt`;
  - `results/ref2ct_detail.txt` (`tools/ref2ct_detail.py`);
  - `results/day36_runner.log`.

## If NewIdeas wants to separate noise from a real IFEval drop
- A larger IFEval sample can do that: the other 441 IFEval prompts for both REF2c-T and REF2-T (~880 rows), under a new
  PREREG. That costs ~5-6 h of quiet GPU (~300 tokens per IFEval row at ~15 tok/s).
- Weigh it against what REF2c buys: Leap's estimate of ~6 ms per check. NewIdeas' call.
