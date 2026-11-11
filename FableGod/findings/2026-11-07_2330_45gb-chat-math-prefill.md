# 45 GB chat, math and prefill speeds: chat decodes 40-45 tok/s, math 79-88, prefill ~100-124 tok/s; the lossy config is NOT exact on chat / math, and peaks at 45.1 GB on agent prompts
- Date: 2026-11-07 23:26 EET. BigMoE N45 (data/n45/n45_report.txt; S5' snapshot 6db4c24; C 256; quiet). Orchestrator:
  this note, decode speeds recomputed from data/r57/n{ie,il,me,ml}{1,2}_256_runs.json.
- Decode, whole decode, 12 prompts each, two passes, cap 256 tokens (EvalSuite's sets):
  - IFEval (chat): exact 39.9 tok/s (39.9 / 39.8), lossy 45.0 (45.2 / 44.8). Lifetime peak 41.7 / 42.4 GB.
  - Math: exact 78.8 (79.4 / 78.3), lossy 87.9 (85.3 / 90.4). Lifetime peak 42.5 / 43.1 GB.
  - For comparison, code: exact 102.7 / 73.3, lossy 181.4 / 123.6 (HumanEval / MBPP). HumanEval flatters; chat is the
    slowest workload.
- Prefill (time to first token, then prefill tok/s), exact / lossy:
  - 1,565 tokens: 15.9 s, 98 / 16.3 s, 96.
  - 4,730 tokens: 40.1 s, 118 / 41.2 s, 115.
  - 7,932 tokens: 64.1 s, 124 / 66.6 s, 119.
  - The prefill path streams experts from the SSD per 1,024-row chunk. This is the engine's weakest part. A user report on
    Strata (2026-11-07_1730) gives ~226 tok/s at 4K on an M1 Max with the whole (2-bit) model in RAM.
- FLAG 1: exactness. The exact config equals NAX6-N's one-token AR on 24 / 24 runs for both chat and math. The lossy
  config does not: 6 / 24 (IFEval) and 7 / 24 (math), although those sets use exact acceptance (QOM runs on code only).
  - A non-QOM lever changes their bits: R57_FASTTAX, REF2_HC8_FAST or S2's SVH / DSUB; untested which.
  - So the lossy config is "lossy, same scores" on chat and math, not exact.
  - Math is gated (S4 / CERT). IFEval joins CERT tonight as a scored set (orchestrator's amendment, before rows).
  - Earlier notes that said "chat runs the exact profile" in the lossy config are corrected.
- FLAG 2: memory. The lossy config's lifetime peak is 45.1 GB on the agent prompts (over the 45.0 line). Exact peaks at
  44.5 GB there. Code / chat / math prompts stay at 42-43 GB. A fix (C 248, prefill chunk or buffer cap) is queued after CERT.
