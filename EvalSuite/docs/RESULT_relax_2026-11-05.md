# RESULT: RELAX, the 45 GB quality-defined modes on the 16-row live loop (night Wed 22:54 - Thu 07:37)

Pre-registration: docs/PREREG_relax_2026-11-02.md (revision 3 + amendments: per-task line, C 160, start 22:54).
Report: results/relax_report.txt (tools/relax_report.py --score).

## Verdicts (decision 4 vs REF2-T on each mode's 210 rows, then mission 7's bars)
| mode | arms | math (50) | IFEval (100) | HumanEval+ (60) | decision 4 | bars | verdict |
|---|---|---|---|---|---|---|---|
| **ship** | B code + E chat | 90.0 vs 88.0 (+2.0) | 87.0 vs 90.0 (-3.0, p 0.25) | 93.3 vs 95.0 (-1.7) | PASS | agent 1 blunder; HumanEval+ drift 6.0 | **BETWEEN** |
| A_E | A + E | **82.0** vs 88.0 (-6.0, p 0.25) | 87.0 vs 90.0 | 91.7 vs 95.0 (-3.3) | FAIL | - | KILL |
| C_E | C + E | 86.0 vs 88.0 (-2.0) | 87.0 vs 90.0 | **90.0** vs 95.0 (-5.0, p 0.25) | FAIL | - | KILL |
| B_F | B + F | 90.0 vs 88.0 | **85.0** vs 90.0 (-5.0, p 0.12) | 93.3 vs 95.0 | FAIL | - | KILL |
| B_G | B + G | 90.0 vs 88.0 | **84.0** vs 90.0 (-6.0, p 0.07) | 93.3 vs 95.0 | FAIL | - | KILL |
| T3ALL | all ranks at 3 bits | 86.0 vs 88.0 | 89.0 vs 90.0 (-1.0) | **88.3** vs 95.0 (-6.7, p 0.12) | FAIL | - | KILL |

- **The rule:** per set, paired delta > -5 points AND exact McNemar p >= 0.05. Five modes fail on a set's delta.
- **All six verdicts are final.**
  - The REF2-T replays that RELAX did not finish (C, G, the last 32 of T3ALL) feed only the bars.
  - Every mode that needs them already fails decision 4.
  - So Friday's leftover replays are information (deviation causes, KLD), not a verdict input.
- **T3ALL's code-only fallback** (chat at 4433333333) reads T3ALL's math and HumanEval+ lines. HumanEval+ fails, so it
  is KILL too.

## Ship = BETWEEN: what fails, in detail
- **The agent bar** (no blunder in the agent rows of A / B): one blunder.
  - Where: agent/bash22, token 86 (" from", p 0.007 under REF2-T along B's own prefix).
  - B leaves the control at token 0, because HC8 changes the prefill. It writes a different, valid tool call:
    `sed -n '11334,11700p' <file>` ("Read transcript tail from 11334") instead of the control's `wc -l <file>`.
  - The flagged token is in the call's description text, not in the command.
- **The drift bar** (HumanEval+, rows >= 512 tokens: last-quarter KLD <= 1.5x the first): 6.04.
  - KLD q1 0.00157, q4 0.00948, on 4 long rows only.
  - Math, whose long rows are 23, passes at 0.89.
- **What passes:** blunders on math + code are 0.21 per 10K (bar 1.0). HumanEval/92 has 1 blunder, math none.
- **Deviated tokens along B's prefix:** code 1.30 %, math 1.43 %, agent 2.69 %. HC8 changes every row, so "other" is
  most of them. E deviates on 3.62 % of IFEval tokens, all from R1.
- **The verdict:** by the PREREG's rule, BETWEEN means decision 4 passes and a bar fails. Adopting it for Friday is
  Kamen's call, routed by NewIdeas.

## The per-task line (amendment, the orchestrator Wed 21:5x): no relaxed arm is killed
| arm | shared keys | tokens / task | s / task | s / token |
|---|---|---|---|---|
| B | 82 | 1.019x | 0.850x | 0.699x |
| C | 69 | 1.037x | 0.830x | 0.709x |
| E | 20 | 1.149x | 0.820x | 0.687x |
| F | 20 | 1.170x | 0.803x | 0.654x |
| G | 20 | **1.433x** | 0.834x | 0.562x |
| A (gate only) | 73 | 1.029x | 0.858x | 0.812x |
| T3ALL (gate only) | 88 | 1.055x | 0.982x | 0.924x |

- All ratios are against the control on shared keys.
- Every arm is faster per task, not only per token. The chat rules lengthen answers (E +15 %, F +17 %, G +43 %, the
  effect arXiv 2610.23818 reports), but they speed decode up more.

## Speed and bytes (information: EvalSuite's harness at C 160, quiet lock, paired with the control on shared keys)
- **B (the ship code arm):**
  - tok/s x1.49 HumanEval+, x1.32 math, x1.54 agent;
  - expert union per check -41 % on code (4.79 vs 8.15 GB), -28 % on math;
  - kept per check unchanged (the gate does not change acceptance; the comment rule adds 1.8 % on HumanEval+).
- **E (the ship chat arm):** tok/s x1.46 IFEval; kept per check +7.2 % (2.67 vs 2.49).
- **AT14** (14 agent turns at 9-12K prompts):
  - turn tok/s 1.27 (B) vs 1.16 (control);
  - prefill 92-94 % of a turn: every turn re-prefills its whole context.
- **C:** 160, not the pre-registered 256 (NO-SWAP 36 GB at 22:5x). Ratios compare within the run; absolute tok/s are
  not C 256's.

## Run notes
- **The run:** runner 40, 22:54-07:37, all 8 arms complete (control, A, B, E, T3ALL, F, C, G).
- **Replays:** A 116/116 and B 124/124 are done; T3ALL 178/210, C and G not started.
  - The run stopped at its deadline: "no replay chunk after 07:10" capped starts, not ends.
  - The last chunk ran to 07:37 and held BigMoE's 07:05 ticket ~32 min. Cutoffs now bound a chunk's end (the
    orchestrator 07:5x).
- **Step 0** passed in every process: HumanEval/27 equal to its REF2-T (or REF2-T + HC8) reference.
- The control arm is token-equal to REF2-T on all 90 rows that have a stored REF2-T row (HumanEval+ 60, math 10,
  IFEval 20). Its 14 AT14 rows have no stored REF2-T row; they are the agent-turn baseline.
