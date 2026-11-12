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
