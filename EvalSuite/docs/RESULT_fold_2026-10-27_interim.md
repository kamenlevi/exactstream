# B12_FOLD on vs off: interim result, 45 of 239 rows per arm (held 2026-10-28 ~16:00)

Pre-registration: `docs/PREREG_fold_2026-10-27.md` (frozen 22:25 Tue, commit 35bcfe7).

## Status
- **On hold (NewIdeas, Wed ~16:00).** Breakthrough's bisect found the one difference's cause, and the fold is being rebuilt. Runner 24 is not resumed. NewIdeas re-plans the verdict with EvalSuite once the rebuilt fold passes, likely as a shorter confirmation on the 29 agent rows.
- **Why it stopped at 45 rows.** The night's quiet window froze chunk 4 (fold on) from 23:53. It timed out at 08:16 without writing a row, and the runner stopped at 08:34, past its 07:00 deadline.
- **What these rows cover.** Chunks 1–3 (ABBA): agent 29 and math 16.
- **The rows are valid data on the old fold.** Every chunk ran the same code: bt_b12 664b010b0d, v3l build d50006dffc, v3m build 5d7c3d06a7, SpecDecode 2d4fa7b. Breakthrough's HEAD moved between chunks only through HANDOVER commits.
- No row was paused or frozen.

## Answers (PREREG numbering)
1. **Tokens: 44 of 45 rows identical, so not EXACT.**
   - **The one difference:** agent/write16 (8,277-token prompt, 128 tokens) differs from token 32: fold on 321, fold off 466.
   - **Against the saved B12LL run** (`results/b12ll`, v3f): fold off matches it on all 45 rows, fold on on 44.
   - **Cause** (Breakthrough's bisect, 9 fresh-process arms on write16's prompt): PFR on the MTP path ran the prompt's MoE as one 8,276-row chunk instead of the shipped 4096 + 4096 + 84 grid. The 84-row tail goes through another gather kernel, so the last prompt rows' bits differ.
   - ALLWAIT, MOV and the rows target, each applied alone, give fold off's tokens.
   - The rule-1 re-run was not run: the bisect superseded it.
2. **Task scores: no scored row differs.**
   - The 16 math rows are token-identical. write16 is an agent row (no gold).
   - The 5-rule tables were not made: only 16 of the 210 core rows were run.
3. **Speed.** Paired rows, all 45 timed; on/off > 1 = fold on faster.

   | set | rows | decode tok/s off → on | gmean on/off | prefill s off → on | gmean prefill on/off |
   |---|---|---|---|---|---|
   | agent (1.6K–12K prompts) | 29 | 16.60 → 17.19 | 1.036 | 941.6 → 666.7 | 0.722 |
   | math (≤ 520 prompts) | 16 | 18.93 → 20.61 | 1.089 | 52.7 → 56.4 | 1.072 |
   | all | 45 | 18.46 → 19.90 | 1.055 | 994.4 → 723.1 | 0.831 |

   - On short prompts the fold's prefill is about 7 % slower (3.30 → 3.52 s per math row). It releases and re-creates the 6 GB tier pools on every MTP prefill, even at 62 tokens. Breakthrough measured that at about 0.2 s per prompt (release 0.03 s, re-create 0.16–0.19 s). The rebuilt fold uses PFR only above 4096 tokens, as the AR path does.
   - The decode gain is larger there: C_m 64 against 51.
4. **Memory: the 11.95 GB line is not met, by 0.05 GB on one row.**
   - **Fold on:** max row peak 12.00 GB, on agent/edit10 (1,565 tokens, C_m 64). That row was the 17th prompt in its process, and its footprint at row start was 11.60 GB. Every other fold-on row is ≤ 11.95.
   - **Fold off:** 28 rows over the line, max 13.15 GB.
   - **Growth within a process:** fold on's footprint at row start rose to 11.75 GB (agent) and 11.79 GB (math). This is the per-prompt growth Breakthrough expected.
   - Caveat: each process was one chunk of 12–17 prompts, not one long process.
5. **SSD, decode phase (MB / token; reads / token).**

   | set | fold off | fold on |
   |---|---|---|
   | agent | 482.3; 253.2 | 458.1; 242.7 |
   | math | 441.1; 226.0 | 393.3; 204.0 |
   | all | 448.5; 230.9 | 405.0; 211.0 |

   That is −10 % MB per token overall.

## Files
- Rows: `results/b12llfoff/decode.jsonl`, `results/b12llfon/decode.jsonl`, with a `chunks.jsonl` per arm (provenance, chunk peaks).
- Full comparison: `results/fold_compare_45.txt`, from `tools/fold_compare.py`.
- Runner log: `results/day24_runner.log`.
