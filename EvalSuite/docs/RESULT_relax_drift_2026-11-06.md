# RELAX ship mode, Friday's checks: the long-code drift re-check and E8 (Fri 2026-11-06; rows 03:54 – 08:00)

Pre-registration: `docs/PREREG_relax_drift_2026-11-05.md` (a692f0f, Thu 08:29) and its amendment 1 (7fed202, Thu 09:03).
> (a quote from the project owner, kept private)
- **The night (RESULT_relax, 8797f54):** ship (B on code + E on chat) was BETWEEN. Decision 4 passed, but two of
  mission 7's bars failed thinly:
  - 1 agent blunder, in a description;
  - HumanEval+ drift 6.04 on 4 long rows.
- **These checks ask two questions:**
  1. Does B's drift really grow along long code answers?
  2. Does the shipped build, with HC8 process-global, still pass on chat? That is E8 = E + HC8.

## Verdicts
| check | verdict |
|---|---|
| drift re-check (B on code rows of >= 512 tokens, pooled) | **PASS** on the registered line: q4 / q1 = 0.82 (<= 1.5). It rests on 16 long rows, not the >= 20 the PREREG wanted, so whether that is enough is the orchestrator's call, as registered. |
| ship8 (B on code + E8 on chat) | **KILL** (decision 4): IFEval 84.0 vs 90.0, −6.0 pt, 0/6, McNemar p 0.03 |

Two findings registered as information change what the verdicts mean for shipping. See the next two sections.

## 1. The drift: PASS pooled, but the HumanEval-type rows repeat the growth
| source | long rows (>= 512 tokens) | KLD q1 | KLD q4 | q4 / q1 |
|---|---|---|---|---|
| humaneval (the night's) | 4 | 0.00157 | 0.00948 | 6.04 |
| humanevalx (new, HumanEval+'s other 104) | 6 | 0.00298 | 0.01215 | 4.07 |
| agentlong (new, 17 agent write / edit turns at 1,024 tokens) | 6 | 0.12650 | 0.09350 | 0.74 |
| **pooled (the registered line)** | **16** | **0.06067** | **0.04976** | **0.82** |

- **The pooled ratio is dominated by the agentlong rows.** Their KLD is far larger:
  - agentlong: q1 0.06-0.19 (median 0.11), q4 0.04-0.16;
  - the 10 HumanEval-type rows: q1 0.0003-0.014 (median 0.0006), q4 0.0006-0.028 (median 0.010).
- **On code answers the growth is borne out:** ×4.07 on the 6 new rows, beside the night's ×6.04. It starts from tiny
  values.
- **On agent writes, KLD is flat (×0.74) but high from the first quarter.**
- The registered line is the pooled ratio, so the verdict is PASS. The per-source reading is the one to look at.
- Per row (results/relax_drift_report.txt):
  - agentlong write11-16: q1 0.064-0.189, q4 0.044-0.162.
  - HumanEval/113: q1 0.00077 -> q4 0.02761. HumanEval/129: 0.00037 -> 0.01866. HumanEval/147: 0.00249 -> 0.01647.

## 2. B on long agent writes: 20 blunders (mission 7's agent bar is "none")
| set | rows | tokens | deviated from REF2-T's argmax | blunders (E's p < 0.01) | per 10K |
|---|---|---|---|---|---|
| agentlong | 17 | 8,842 | 6.67 % | **20** (16 inside the written code, 4 in descriptions) | **22.6** |
| humanevalx | 104 | 25,656 | 1.20 % | 0 | 0.00 |
| humaneval (night) | 60 | 13,975 | 1.30 % | 1 | 0.72 |
| math (night) | 50 | 33,742 | 1.43 % | 0 | 0.00 |
| agent, 128-token turns (night) | 14 | 1,596 | 2.69 % | 1 (description) | 6.27 |

- **The 16 code blunders** are in the first tool-call parameter, i.e. the file content:
  - write12: ' Post', '.y' and indentation runs;
  - write15: a '"' at p 1e-05;
  - write16: ' let', 'Options';
  - and more (results/relax_drift_report.txt, "blunders").
- **They come from the gate, not HC8.** The night's arm A (Q1-UNION + copy, no HC8) had 2 blunders in 732 agent tokens
  (27.3 per 10K). The exact control had 0.
- **The night's 128-token agent turns undersampled this.** The 1,024-token turns show it clearly.
- **What it means:**
  - B looks clean on code answers: 0 blunders in 25,656 new HumanEval+ tokens.
  - B is not fit for agent code edits: it writes low-probability tokens into files at ~23 per 10K.
  - Registered as information (the drift PREREG), so no verdict changes. It is the main input for "B on code ships".

## 3. ship8 = B on code + E8 on chat: KILL
| set | ship8 | REF2-T | delta [90% CI] | McNemar p (only ship8 right / only REF2-T right) | rule |
|---|---|---|---|---|---|
| math (n 50) | 90.0 % | 88.0 % | +2.0 [−3.7, +7.7] | 1.00 (2/1) | PASS |
| IFEval (n 100) | 84.0 % | 90.0 % | −6.0 [−9.9, −2.1] | 0.03 (0/6) | **FAIL** |
| HumanEval+ (n 60) | 93.3 % | 95.0 % | −1.7 [−4.4, +1.1] | 1.00 (0/1) | PASS |

- **E8's IFEval losses vs REF2-T:**

  | item | missed | E alone | HC8 alone (REF2-T + HC8) |
  |---|---|---|---|
  | ifeval/2497 | a highlighted section | fails too | passes |
  | ifeval/2616 | keyword frequency | passes | passes |
  | ifeval/2853 | capital-word frequency | fails too | fails too |
  | ifeval/2870 | capital-word frequency | passes | passes |
  | ifeval/3326 | letter frequency | passes | passes |
  | ifeval/3409 | highlighted sections | passes | passes |

  - E alone also fails ifeval/1219, which E8 passes.
  - IFEval scores: E 87, HC8 alone 89, E8 84. R1's relaxed acceptance and HC8's 8-bit mixers compound on chat.
- **What it means:** with HC8 process-global as built, the chat part (R1) cannot ship as is. The options are HC8
  code-only, or chat without R1. That is NewIdeas' and Kamen's call.
- **E8's per-task line:** no kill. E8's IFEval drift is 0.67 (information).

## The bridge and the gates (all PASS)
- **Converted store** (the code by store, amendment 1): snap/bigmoe-da8e7a6 on ext/build, REF2-T as `ref2tf0`.
- **The bridge first** (03:58): B's replay step 0 replays the night control's gate-off rows bitwise: 57/57 rows, first
  token equal. Every later replay process repeated it, 57/57 each time.
- **Every G process's step 0:** HumanEval/27 with the gate and rule off == results/ref2th8 token for token (B 03:58; E8
  05:45, 06:47).
- **Rows:** B 121/121 drift rows + 121 replays; E8 124/124 rows + 124 replays.

## Runs (runner 45, attached; results/day45_runner.log)
- GO 03:54 (Breakthrough's night driver).
- Bridge 03:54-03:58.
- B drift rows 03:58-04:50 (51 min).
- B replay 04:50-05:44 (54 min).
- E8 rows 05:45-06:49: the first chunk hit the cli's 60-min timeout; the rows resumed.
- E8 replay 06:49-07:59:
  - 07:40: a quiet ticket asked the [low] stage to stop at its next row;
  - 07:50: the timeout;
  - 07:56: the stage's own swap guard stopped one start (swap 12.5 GB > 9.9 + 2);
  - then the last rows.
- Report 08:00, then the runner ended.
- **Swap** is 12.6 GB now, up ~2.7 GB since Thursday. Nothing of EvalSuite runs.
- **Time:** ~4 h of GPU, against amendment 1's ~3.4 h.
- **Data:**
  - results/relax_B/ (+ the drift rows) and results/relax_E8/;
  - results/relax_drift_report.txt and results/relax_e8_report.txt;
  - results/relax_mode_ship8/.
