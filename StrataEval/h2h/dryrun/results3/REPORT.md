# Strata head-to-head (2026-11-07 23:53)

Strata 0.1.58-metal (commit 146df3c), Qwen3.8-Flash-Next ISTA-DASLab IQ3_XXS, experts streamed from the SSD (64 GB Mac), greedy, 256 new tokens, EOS stop, one warm process per arm and pass (palindrome A B C D D C B A), the exact prompt ids S5 / N45 used. Arms: **A** default (HC Q8_0, lossy), no MTP; **B** default + MTP; **C** exact (STRATA_HC_Q8=0), no MTP; **D** exact + MTP.

Records: 24 measured requests used; dropped: {'error': 0, 'freeze': 0, 'incomplete_process': 0}. Freeze windows in fanwatch.log: 516.

## Speed and identity

| arm | set | WHOLE DECODE tok/s (passes) | pooled | END TO END | prefill tok/s (engine / P÷TTFT) | TTFT s (median) | tokens/task | stops | MTP accepted / offered | == NAX6-N AR |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 12 HumanEval (S5) | **666.7** (666.7) | 666.7 | 367.3 | 2500 / 2161 | 0.07 | 256 | 1/3 | - | 0 / 3 |
| A | 12 MBPP (S5) | **666.7** (666.7) | 666.7 | 410.9 | 2500 / 2035 | 0.05 | 256 | 0/3 | - | n/a (no AR ref) |
| A | 12 IFEval (N45) | **666.7** (666.7) | 666.7 | 396.0 | 2500 / 1976 | 0.03 | 153 | 2/3 | - | 0 / 3 |
| A | 12 math (N45) | **666.7** (666.7) | 666.7 | 417.9 | 2500 / 2115 | 0.03 | 256 | 0/3 | - | 0 / 3 |
| A | all | **666.7** (666.7) | 666.7 | 397.6 | 2500 / 2071 | 0.04 | 256 | 3/12 | - | - |
| B | 12 HumanEval (S5) | **1000.0** (1000.0) | 1000.0 | 507.3 | 2500 / 2308 | 0.06 | 256 | 1/3 | 296 / 593 | 0 / 3 |
| B | 12 MBPP (S5) | **1000.0** (1000.0) | 1000.0 | 593.3 | 2500 / 2012 | 0.05 | 256 | 0/3 | 384 / 768 | n/a (no AR ref) |
| B | 12 IFEval (N45) | **1000.0** (1000.0) | 1000.0 | 564.3 | 2500 / 1945 | 0.03 | 153 | 2/3 | 231 / 464 | 0 / 3 |
| B | 12 math (N45) | **1000.0** (1000.0) | 1000.0 | 604.3 | 2500 / 2018 | 0.04 | 256 | 0/3 | 384 / 768 | 0 / 3 |
| B | all | **1000.0** (1000.0) | 1000.0 | 566.0 | 2500 / 2066 | 0.04 | 256 | 3/12 | 1295 / 2593 | - |

## Processes (peak = /usr/bin/time -l peak memory footprint)

| run | done | peak GB | max RSS GB | arena GB (slots) | wired GB | footprint after fill / cap GB | HC Q8 | ready s | wall s | note |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 | True | 0.01 | 0.02 | 30.90 (18432) | 30.9 | 40.6 / 41.2 | True | 0.3 | 8 |  |
| B1 | True | 0.01 | 0.02 | 30.90 (18432) | 30.9 | 40.6 / 41.2 | True | 0.3 | 6 |  |

Peak footprint per arm (max over its processes): A 0.01 GB, B 0.01 GB

## Token identity across arms (same prompt and pass; information)

- B vs A: 12 / 12 identical

## Our engine on the same prompt sets (read-only from BigMoE's reports; information)

| set | config | WHOLE DECODE tok/s | END TO END | tokens/task | lifetime peak GB | == NAX6-N AR | source |
|---|---|---|---|---|---|---|---|
| 12 HumanEval (S5) | S5' B (cert config, lossy FAST) | 174.4 | 34.8 | 246 | 42.9 | - | BigMoE data/s5p/s5p_report.txt |
| 12 HumanEval (S5) | S5' X (lossy FAST) | 181.4 | 35.4 | 243 | 42.9 | - | BigMoE data/s5p/s5p_report.txt |
| 12 MBPP (S5) | S5' B (cert config, lossy FAST) | 120.2 | 23.6 | 120 | 42.6 | - | BigMoE data/s5p/s5p_report.txt |
| 12 MBPP (S5) | S5' X (lossy FAST) | 123.6 | 23.4 | 116 | 42.6 | - | BigMoE data/s5p/s5p_report.txt |
| 12 IFEval (N45) | N45 EB (exact) | 39.9 | 17.1 | 142 | 41.7 | 24 / 24 | BigMoE data/n45/n45_report.txt |
| 12 IFEval (N45) | N45 L (lossy) | 45.0 | 17.3 | 144 | 42.4 | 6 / 24 | BigMoE data/n45/n45_report.txt |
| 12 math (N45) | N45 EB (exact) | 78.8 | 39.5 | 256 | 42.5 | 24 / 24 | BigMoE data/n45/n45_report.txt |
| 12 math (N45) | N45 L (lossy) | 87.9 | 40.3 | 256 | 43.1 | 7 / 24 | BigMoE data/n45/n45_report.txt |

Note: our END TO END includes the screen's prefill + cache / drafter setup; Strata's is the request's wall time.
