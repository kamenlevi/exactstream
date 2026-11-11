# L5 result (QUEUE L1-04 stage 2): DFlash2 at block 16, fine-tuned on code / math. Leap, Sat 2026-10-24 05:47
Pre-registration: DFL2E_PREREG.md (02:42; amendment 2 at 04:41 moved the kill line to 10.3 before any fine-tune
number). One run, as registered. All numbers are the L2 replay of the official loop, fixed counting, on held-out
prompts (24 HumanEval+ / 120 GSM8K items that the training never saw).

## Verdict: KILLED for exact decoding (code tau16 9.88 < 10.3). A x1.44 lever where rows are cheap.

| drafter | code tau | GSM8K tau |
|---|---|---|
| original, block 8 (L4 batch) | 6.87 | 5.70 |
| original, block 16 (depths 8-15 never trained) | 8.88 | 6.56 |
| **fine-tuned, block 16** | **9.88** (x1.44 vs block 8) | 6.98 (x1.22) |
| fine-tuned, block 8 (guard: < 6.6 = it hurt block 8) | 6.76 (guard holds) | - |

P(first j drafts right), code:
- original at block 16: 0.96 0.91 0.86 0.80 0.76 0.73 0.67 | 0.61 0.53 0.44 0.32 0.21 0.12 0.08 0.04
- fine-tuned at block 16: 0.94 0.88 0.81 0.75 0.72 0.68 0.64 | 0.60 0.57 0.52 0.48 0.44 0.39 0.35 0.30
- original at block 8, for reference: 0.97 0.92 0.88 0.85 0.80 0.77 0.74

## Lines and predictions
- Amended kill line (exact frame, from Fit12's T16/T8 ~1.5-1.65): code tau16 < 10.3. **9.88: KILLED.**
- Original lines: kill < 9.0 (passed), interesting >= 10 (missed by 0.12).
- Predictions:
  - code tau16 ~9.6 (8.8-10.4): in range;
  - deep conditional acceptance ~0.82 (0.75-0.89): measured 0.91, above the range;
  - GSM8K ~7.0: 6.98;
  - original at block 16 ~7.0-7.3: measured 8.88, a clear miss;
  - fine-tuned at block 8 ~6.8: 6.76.

## What 9.88 is worth, by how a pass's cost grows with its rows
- **Exact, today's kernels:** 3c in RAM costs ~6 us per (row, rank) pair. SSD-streamed at C192, Fit12 measured
  4 -> 8 rows and extrapolated. Either way T16/T8 ~1.5-1.65, so block 16 is 4-13 % SLOWER than block 8. Also, the
  engine routes <= 9 rows today. Dead.
- **Non-exact per-pass expert budget** (UGB / Chat1000's U16 = 32; TrainLab L1-08): the budget caps the union, so a
  16-row pass costs about an 8-row one (31 ms). Block 16 is then a free x1.44 in tokens per pass; the untrained original
  already gives x1.29.
- **350 on code** (Chat1000 REPORT: MMA kernel + budget routing + a drafter at tau16 ~10.9): 9.88 is ~9 % short.
  - The loss is in the early depths. Block 16's extra mask tokens lower P(first 7 right) from 0.74 (block 8) to 0.64.
  - Depths 8-15 are already at 0.91 conditional.
  - If block 16 kept block 8's early curve, the same deep curve would give ~10.7. That is a projection, not measured.

## Side finding (Sat 06:31, L4b batch): verify fewer rows when the drafter is unsure
scripts/dfl2_trunc.py; data/trunc_*.txt. Same replay, same 24 code prompts.
- It records the selector's confidence per depth and verifies only the rows whose running confidence product is
  >= t. This is dynamic draft length, published (SpecDec++, AdaEDL).
- Costs: in RAM, 15 + 2.88 ms per row (today's 3c kernel); SSD-streamed C192, the union model.

| drafter | rows used | tok/pass | ms/token, RAM | ms/token, SSD |
|---|---|---|---|---|
| original, block 8, all rows | 7.95 | 6.87 | 5.51 | 21.6 |
| original, block 8, conf >= 0.5 | 7.03 | 6.71 | 5.25 | 20.1 |
| original, block 16, all rows | 15.67 | 8.88 | 6.77 | 26.5 |
| original, block 16, conf >= 0.5 | 9.95 | 8.49 | 5.14 | 19.9 |
| fine-tuned, block 16, all rows | 15.65 | 9.88 | 6.08 | 23.9 |
| **fine-tuned, block 16, conf >= 0.5** | 10.34 | 9.38 | **4.77** | **18.3** |
| fine-tuned, block 16, oracle rows | 9.91 | 9.88 | 4.41 | 16.8 |

- With truncation, the fine-tuned block-16 drafter is x1.10 faster than block 8 truncated the same way (x1.16 vs
  plain block 8) in both cost models. So exact block 16 is not dead when the rows adapt.
- But passes then average ~10.3 rows, above today's 9-row routing limit. The gain is a cost model, not a decode.
- t = 0.5 was the highest threshold tried, and RAM ms/token was still falling, so the optimum may be higher.
- Worth one engine test once the engine verifies variable rows above 9 (BigMoE / Fit12). Not an L5 line; the kill above
  stands as registered.

## Caveats
- One short run: 23.8K tokens of the 125B's own HumanEval+ / GSM8K outputs, 2 epochs = 189 steps, 12.8 min.
- In-domain prompts (HumanEval-style code). General coding may gain less.
- Replay, not a real decode. On token-identical spans the real loop gives 0.93-0.98x the replay (Fit12 x1 data,
  04:50).

## Data
- Features data/dfl2e (158 MB): deleted after this result.
- Weights data/dfl2e_ft.safetensors (1.1 GB): kept for L4b's side evals (agentic at block 16; the row-truncation
  analysis), then deleted unless NewIdeas wants them for the budget-routing route.
- All of it is public benchmark text; nothing from Kamen's sessions.
