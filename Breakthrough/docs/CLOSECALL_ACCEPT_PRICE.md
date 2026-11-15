# Lossy acceptance at the 125B's close calls: a price estimate (Kamen 18:2x via the orchestrator; Thu 12 Nov 2026)

> (a quote from the project owner, kept private)

**The rule:** at a drafter's first miss, accept the drafter's token when the 125B rates it almost as highly as its own
top pick. Two forms:
- ratio: p_d / p1 >= rho; RELAX's R1(delta) logit rule is rho = e^-delta, so R1(0.5) = rho 0.61;
- gap: p1 - p_d <= delta.

CPU only, one pass.

## Pre-registration (written before the computation)

**Data: per-check replays of DFlash2 L1-20, teacher-forced on D-02a's captures** (the 125B's own top-32 per position; read-only):
- 16-row checks (TrainLab data/r61n/chatnt_b16.jsonl, R-61N 11-01): IFEval 30, multi 24, agent 12, MGSM 30 and GSM 60
  windows.
- 32-row checks (NewIdeas data/nt/nt_accept_b32.jsonl, NT-1 10-26): code, 24 HumanEval windows. It is the only code
  first-miss data.
- For MTP b5, today's other drafter, only the exact baselines are known: code 9.50 and IFEval 2.74 kept per 16-row check
  (drafterwall a1). Its gains are the DFlash2 replay's relative gains [P].

**Estimators** (nt_analyze.py's, unchanged): LB = +1 token per forgiven check. MOD = the run continues with the exact
replay's conditional survival per depth, and later misses are forgiven at the rule's first-miss rate (an estimate, not a
bound).

**Calibration:** RELAX measured R1(0.5) live on IFEval: kept per check +7.2 % (2.67 vs 2.49). The estimate at rho 0.61
should bracket it, LB <= live <= MOD.

**Thresholds:**
- ratio rho 0.75, 0.61 (= R1(0.5)), 0.5, 0.25, 0.1;
- gap delta 0.05, 0.1, 0.2.

**Self-check** (else the script is wrong): LB reproduces R-61N's chat rows (rho 0.5 / 0.25 / 0.1: 2.32 / 2.41 / 2.49)
and NT-1's code rows (10.34 / 10.41 / 10.47) exactly. MOD lands within 0.05 of theirs (Monte Carlo).

**Speed:**
- tok/s = kept per check / the measured 45 GB cycle: code ~65 ms per 16-row check; chat (IFEval, multi, agent, math)
  ~44 ms. The cycle is taken as fixed per check [P].
- Code at 16 rows uses the exact 16-row baselines (DFlash2's own walk, TrainLab data/d02b_walk_code.npz, and MTP b5's
  9.50), times NT-1's relative gain at 32 rows [P].
- Upper bound: drafterwall's lag-2 ceiling with the lossy oracle gate at the byte floor: HumanEval+ 706 tok/s (964 with
  the perfect pick at 64 rows), MBPP 441 (591) [P].

**Quality, from the only lossy-acceptance measurement:**
- Anchor: RELAX's E arm (R1(0.5) on chat): IFEval 87.0 vs 90.0 (-3.0, p 0.25, n 100), with 3.62 % of IFEval tokens
  deviating and answers +15 % longer.
- Also: R-61N (no quality measured), M-15 EXACT-NT (exact, so no hit; chat +14.0 %), and RELAX's C_E arm (R1 on code:
  HumanEval+ 90.0 vs 95.0, -5.0, p 0.25).
- Expected hit per threshold = RELAX's -3.0 IFEval points scaled linearly by the threshold's deviating-token rate relative
  to rho 0.61 [P]. Code is scaled from C_E's -5.0 in the same way.

**Recommended pre-registration for EvalSuite's same-scores gate:** the loosest threshold whose expected IFEval hit is
<= 1.5 points (half of RELAX's measured -3.0), so the gate has a fair chance.

**Files:** `scripts/m1/closecall_price.py`, `data/m1/closecall_price.out`.

## Result (Thu 18:3x, CPU ~1 min per run)

**Self-check: PASS.** LB reproduces R-61N's chat rows and NT-1's code rows exactly; MOD reproduces them too (same draw
order).

**Calibration: PASS.** At rho 0.61 = R1(0.5) on IFEval:
- kept per check: the estimate is LB +3.3 % to MOD +9.0 %, and RELAX's live +7.2 % sits inside;
- deviating tokens: 3.2-3.3 % estimated vs 3.62 % live.

**Exact baselines:**
- DFlash2 at 16 rows: chat 2.23 kept per check (IFEval 2.44, multi 2.05), agent 3.00, MGSM 3.52, GSM 6.98, code 9.57
  (TrainLab's walk). NT-1's code at 32 rows: 10.23.
- MTP b5 (drafterwall a1): code 9.50, IFEval 2.74.

**Estimate table** [P]. MTP b5 baselines times the DFlash2 replay's relative gains. tok/s at 44 ms (chat) or 65 ms
(code) per 16-row check. The expected hit is RELAX's measured -3.0 IFEval (code: -3.3 HumanEval+, C_E vs ship) scaled
by the rule's deviating-token share (MOD).

| rule | IFEval kept per check | IFEval tok/s | code kept per 16-row check | code tok/s | expected IFEval | expected HumanEval+ |
|---|---|---|---|---|---|---|
| exact | 2.74 | 62 | 9.50 | 146 | 0 | 0 |
| delta 0.05 | 2.77-2.82 | 63-64 | 9.53-9.79 | 147-151 | -1.0 | -1.3 |
| rho 0.75 | 2.80-2.89 | 64-66 | 9.55-9.95 | 147-153 | -1.9 | -2.0 |
| delta 0.1 | 2.80-2.91 | 64-66 | 9.54-9.84 | 147-151 | -2.1 | -1.5 |
| rho 0.61 = R1(0.5) | 2.83-2.99 | 64-68 | 9.58-10.26 | 147-158 | -3.0 (measured, p 0.25) | -3.3 |
| rho 0.5 | 2.87-3.11 | 65-71 | 9.60-10.47 | 148-161 | -4.3 | -4.2 |
| delta 0.2 | 2.89-3.19 | 66-73 | 9.59-10.33 | 147-159 | -5.1 | -3.5 |
| rho 0.25 | 3.00-3.62 | 68-82 | 9.67-11.23 | 149-173 | -8.6 | -7.1 |
| rho 0.1 | 3.13-4.27 | 71-97 | 9.73-11.88 | 150-183 | -12.5 | -9.7 |

**Other domains** (DFlash2, kept per check LB-MOD at rho 0.25):
- chat 2.41-2.74 (+8..+23 %);
- multi 2.18-2.36 (+6..+15 %);
- agent 3.12-3.48 (+4..+16 %);
- GSM 7.17-8.21 (+3..+18 %).

**Upper bound** (drafterwall, byte floor, lossy oracle gate + lag-2 ceiling) [P]: HumanEval+ 706 tok/s (964 with the
perfect pick at 64 rows), MBPP 441 (591).

**Reading:**
- At gate-safe thresholds the rule buys little:
  - delta 0.05, the loosest rule with an expected IFEval hit <= 1.5 points: +1..+3 % kept per check on chat and
    +0..+3 % on code;
  - rho 0.75: +2..+5 % at an expected -2 points.
- Gains of +15-30 % need rho <= 0.25, at an expected 7-12 point hit.
- Live, R1(0.5) also lengthened IFEval answers by 15 %.
- The quality anchors are single small runs: IFEval -3.0 at p 0.25 (n 100), and HumanEval+ from mixed arms (n 60). The
  expected hits are therefore uncertain by about +-100 %.

**Pre-registration recommendation for EvalSuite's same-scores gate:** delta 0.05 (accept when p1 - p_d <= 0.05).
- Expected: +1-3 % kept per check, about -1 IFEval point and -1.3 HumanEval+ points.
- The gate decides.

**Deviation from the pre-registration (noted after the run):**
- The pre-registration scaled code from C_E's -5.0 (C_E vs control). The table uses -3.3 instead (C_E 90.0 vs the ship
  arm's 93.3): the ship arm carries the same other lossy parts as C_E but no R1 on code, so -3.3 isolates the rule.
- With the pre-registered -5.0, every HumanEval+ figure is 1.52x larger: delta 0.05 -2.0, rho 0.75 -3.0, R1(0.5) -5.0,
  rho 0.25 -10.8, rho 0.1 -14.7.
- The IFEval column and the recommendation (delta 0.05) do not change.

## Under the real decode walk (Sun 15 Nov 02:4x; the orchestrator: close-call acceptance as the direct lossy route to GOAL A). Pre-registered before computing

**Data:** t30b's cells (BigMoE data/deepdraft/ceiling/cells_t30b_heplus.npz: 24 HE+ windows; cells_t30b_mbpp100.npz:
100 MBPP files).
- Per anchor and guess depth 0-30 they hold: right (= the 125B's runtime top-1, true prefix fed), m (the 125B's top-1
  minus top-2, nats), rank, and dlp (the guess's log-prob minus the top-1's; <= 0; -inf outside the top-32).
- Walk: EvalSuite's tools/certify98.py functions (kept_walk cap 31 from row 1; strict_counts "walk"), imported read-only.

**Rule:** a guess is accepted if right OR dlp >= -delta, for delta = 0.1, 0.25, 0.5, 1, 2. delta = 0 is the exact walk and
must reproduce t30b's published 13.77 (HE+) / 6.92 (MBPP) kept/32.

**Reported per set and delta:**
- kept/32, with the window-clustered 95 % interval;
- walk-strict over depths 0-14 (P(right at k | right at 0..k-1) at the walk's check starts, pooled);
- exposure = the walk's accepted non-top-1 tokens per 1K kept tokens: in all, by the 125B's margin bin (m < 0.1,
  0.1-0.25, 0.25-0.5, 0.5-1, 1-2, >= 2) and by the target token's class (code / comment / string / prose outside a
  code fence).
- Classes: a character lexer over the decoded window text, from the answer start (the model's tokenizer.json).

**Lines:**
- **PASS** (worth EvalSuite's real-decoding code gate): some delta <= 1 lifts kept/32 by >= +15 % on BOTH sets (HE+
  >= 15.84, MBPP >= 7.96), with <= 10 accepted non-top-1 tokens per 1K kept tokens in code (outside comments, strings
  and prose).
- **KILL:** no delta <= 1 lifts both by >= +15 %.
- **BETWEEN:** otherwise.

**Caveat:** these are teacher-forced UPPER bounds. After a changed token the cells still follow the 125B's own path,
and MoonShots' J(A) found that changed tokens change the code that follows about half the time. EvalSuite's code gate
in real decoding decides.

**Files:** scripts/m1/closecall_walk.py, data/m1/closecall_walk.out.

### Result (Sun 02:46; data/m1/closecall_walk.out)

**Self-checks:** delta 0 reproduces t30b's published kept/32 on both sets (13.77 / 6.92): PASS.

**HE+: NOT PRICEABLE from these cells.** dlp is NaN on all 14,671 wrong guesses (the HE+ capture didn't record the
guess's log-prob gap or rank), so no wrong guess can be judged a close call. The zero lift printed for HE+ is that
artifact, not a result. MBPP's cells have the gaps.

**MBPP** (100 files; kept/32 [95 % window CI]; exposure = the walk's accepted non-top-1 tokens per 1K kept tokens):

| delta | kept/32 | vs exact | walk-strict 0-14 % | exposure | code | comment | string |
|---|---|---|---|---|---|---|---|
| 0 | 6.92 [6.54, 7.39] | - | 85.67 | 0 | 0 | 0 | 0 |
| 0.1 | 7.04 | +1.7 % | 85.94 | 2.7 | 0.5 | 1.0 | 1.3 |
| 0.25 | 7.41 | +7.1 % | 86.70 | 12.5 | 2.5 | 4.2 | 5.9 |
| 0.5 | 7.84 | +13.3 % | 87.45 | 21.4 | 4.3 | 7.9 | 9.3 |
| 1 | 8.52 [8.14, 9.00] | +23.1 % | 88.62 | 36.6 | 8.1 | 12.7 | 15.9 |
| 2 | 10.14 | +46.5 % | 90.83 | 61.9 | 12.8 | 21.0 | 28.1 |

- By the 125B's margin, delta 1's 36.6 per 1K splits 3.1 / 5.2 / 11.7 / 15.4 / 1.2 over m < 0.1 / 0.1-0.25 / 0.25-0.5
  / 0.5-1 / 1-2.
- No MBPP answer has a code fence, so the prose class is 0.
- Most accepted changes land in comments and strings (docstrings): at delta 1, 28.6 of the 36.6 per 1K.

**Verdict (pre-registered): UNDETERMINED**, because HE+ has no miss gaps.
- MBPP alone meets both lines first at delta 1: +23.1 % kept/32 with 8.1 code-token changes per 1K. delta 0.5 misses
  the lift (+13.3 %).
- Why the lift is modest: MBPP's misses are mostly far from the top-1 (median gap -4.5 nats; only ~10 % within 0.625),
  so close-call acceptance recovers a small part of them. NewIdeas' margin ceilings (e.g. >= 1 nat right -> MBPP 13.64)
  assume the drafter is right on every confident row, which is a different lever.

**Needed for the verdict:** HE+ cells with rank / dlp on the misses (BigMoE's dd_ceiling misses path, as for MBPP).
The caveat stands: these are teacher-forced upper bounds.

### Final result (Sun 02:52): HE+ priced on BigMoE's re-emitted cells (cells_t30b_heplus24.npz, b316365); verdict PASS (delta 1)

The re-emitted cells are identical to the old file (right and m equal on all 186,465 cells), with rank / dlp now on
every miss. The delta 0 self-check reproduces 13.77: PASS.

**HE+** (24 windows):

| delta | kept/32 [95 % window CI] | vs exact | walk-strict 0-14 % | exposure per 1K kept | code | comment | string |
|---|---|---|---|---|---|---|---|
| 0 | 13.77 [11.08, 17.26] | - | 92.98 | 0 | 0 | 0 | 0 |
| 0.1 | 14.26 | +3.6 % | 93.39 | 2.7 | 1.1 | 1.4 | 0.2 |
| 0.25 | 14.52 | +5.5 % | 93.64 | 4.4 | 1.4 | 2.8 | 0.2 |
| 0.5 | 15.11 | +9.7 % | 94.06 | 8.0 | 2.4 | 5.5 | 0.2 |
| **1** | **16.78 [14.38, 19.44]** | **+21.8 %** | 95.14 | 15.4 | **4.1** | 11.0 | 0.3 |
| 2 | 18.21 | +32.3 % | 95.92 | 22.3 | 5.8 | 16.0 | 0.5 |

At delta 1, the margin bins (m < 0.1 / 0.1-0.25 / 0.25-0.5 / 0.5-1 / 1-2) hold 1.4 / 2.5 / 4.1 / 6.6 / 0.8 per 1K.

**VERDICT (pre-registered): PASS at delta 1** (the earlier UNDETERMINED is superseded).
- kept/32 rises +21.8 % on HE+ (13.77 -> 16.78) and +23.1 % on MBPP (6.92 -> 8.52).
- Code-token changes stay at 4.1 (HE+) and 8.1 (MBPP) per 1K kept tokens, under the line of 10. Most changes land in
  comments (HE+) and comments + docstrings (MBPP).
- delta 0.5 misses the lift on both sets (+9.7 / +13.3 %).
- Next, per the pre-registration: EvalSuite's real-decoding code gate at delta 1 (a lossy variant against the lossy-45
  base).
- Caveat: these are teacher-forced UPPER bounds (J(A): about half of changed tokens alter the code that follows), so
  the real-decoding lift is lower.
