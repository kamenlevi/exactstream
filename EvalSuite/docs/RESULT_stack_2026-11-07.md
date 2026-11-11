# RESULT: the FAST stack's code verdicts, the Fri -> Sat night (runner 49; Sat 2026-11-07 04:0x, long code 04:34)

Pre-registered in docs/PREREG_stack_2026-11-06.md (+ amendments 1-3), PREREG_relax_drift amendment 2 (the top-up) and
PREREG_mbpp (+ amendment 3).

**The arm:** stackb, BigMoE's CODE profile in data/stack/code_env.txt, at C 224. The screen is BigMoE's committed snapshot
data/r57/snap/r57_screen_sat.py (a4d659190145). Provenance:
- BigMoE's snapshot is commit 5fecd8f; every snapshot module a process loaded equals it.
- The live bigmoe/ package is unchanged between 5e41177 and 3c1790a.
- Each order's processes loaded the same 35 outside modules, with the same sha256s as the first process.

## Verdicts
| Line | Result | Verdict |
|---|---|---|
| Same scores vs REF2-T, forward | math 88.0 vs 88.0 % (PASS); IFEval 87.0 vs 90.0 (PASS; NAX6's own rows, chat stays EXACT); **HumanEval+ 90.0 vs 95.0 %, delta -5.00 pt, McNemar p 0.25 (0 arm-only / 3 reference-only): FAIL** | FAIL |
| Same scores vs REF2-T, reverse | math 84.0 vs 88.0 (-4.00, p 0.50: PASS); IFEval PASS; **HumanEval+ 90.0 vs 95.0, delta -5.00, p 0.25: FAIL** | FAIL |
| **STACK** (both orders must pass) | | **FAIL** (results/STACK_DONE) |
| B top-up (code-only, 51 rows >= 512 tokens) | blunders 2 = 0.30 per 10K (line <= 1); pooled q4 KLD 0.00671 (<= 0.02); largest row 0.02761 (<= 0.05) | **PASS** (results/relax_topup_report.txt) |
| MBPP (the stack without CD16) vs NAX6-N's AR | 76.0 vs 79.0 % in both orders, delta -3.00, p 0.4531 (2 / 5) | **PASS** (results/MBPP_DONE) |
| Long code: the stack's CODE env on the 41 humanevalt rows (tfwin vs NAX6-N; 04:34) | 54,561 tokens, blunders 1 = 0.18 per 10K (line <= 1), deviations 1.76 %; QOM engaged on 41 / 41 runs (5 processes, all typed code) | **PASS** (results/stack_ht_report.txt) |

The same-scores FAIL is decided by the strict bar: the delta must be above -5 pt, and it is exactly -5.00 (3 tasks of
60). It is not significant (p 0.25).

## Information (not a line)
- **vs NAX6, the stack's own exact base** (results/stack_{fwd,rev}_vs_nax6_d4.txt), both orders PASS:
  - HumanEval+ 90.0 vs 91.7 % (-1.67 pt, p 1.0: 2/3 forward, 1/2 reverse);
  - math +0.00 forward, -4.00 reverse.
- NAX6 itself is -3.33 pt on HumanEval+ vs REF2-T (RESULT_nax6), so most of the -5.00 is the exact base's own gap.
- The two orders score HumanEval+ the same (90.0 %), but on different rows: the discordant pairs vs NAX6 differ by order.
- **The open decision** (sent to the orchestrator at 04:0x): S4's pre-registration uses this same line vs REF2-T; CERT
  uses NAX6-N. Option (b) would amend S4, before any of its rows, to decision 4 vs NAX6-N, with REF2-T as information.
  That choice is made after seeing this result, so it is for the orchestrator / Kamen. This FAIL stays on record as
  pre-registered.

## Process notes
- Runner 49 stopped twice on its own checks, both fixed and restarted with finished rows kept:
  - 00:54: BigMoE's untracked snapshot (58d1ad9);
  - 01:17: BigMoE's uncommitted live screen, which the run does not execute (e94b9e8).
- Its rows carried no text, so the scorers read text filled in by tools/rows_text.py. The fill equals the decode
  stage's text on 520/520 reference rows, and refilled NAX6 rows give identical d4x output.
