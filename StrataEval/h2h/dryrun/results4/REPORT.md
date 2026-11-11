# Strata head-to-head (2026-11-08 11:57)

Strata 0.1.58-metal (commit 146df3c), Qwen3.8-Flash-Next ISTA-DASLab IQ3_XXS, experts streamed from the SSD (64 GB Mac), greedy, 256 new tokens, EOS stop, one warm process per arm and pass (palindrome A B C D D C B A), the exact prompt ids S5 / N45 used. Arms: **A** default (HC Q8_0, lossy), no MTP; **B** default + MTP; **C** exact (STRATA_HC_Q8=0), no MTP; **D** exact + MTP.

Records: 32 measured requests used; dropped: {'error': 0, 'freeze': 0, 'incomplete_process': 0}. Freeze windows in fanwatch.log: 527.

## Speed and identity

| arm | set | WHOLE DECODE tok/s (passes) | pooled | END TO END | prefill tok/s (engine / P÷TTFT) | TTFT s (median) | tokens/task | stops | MTP accepted / offered | == NAX6-N AR |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 12 HumanEval (S5) | **666.7** (666.7, 666.7) | 666.7 | 352.5 | 2500 / 2387 | 0.15 | 256 | 0/2 | - | 0 / 2 |
| A | 12 MBPP (S5) | **666.7** (666.7, 666.7) | 666.7 | 414.1 | 2500 / 2287 | 0.05 | 256 | 0/2 | - | n/a (no AR ref) |
| A | 12 IFEval (N45) | **666.7** (666.7, 666.7) | 666.7 | 418.9 | 2500 / 1856 | 0.02 | 153 | 2/2 | - | 0 / 2 |
| A | 12 math (N45) | **666.7** (666.7, 666.7) | 666.7 | 422.3 | 2500 / 1962 | 0.03 | 256 | 0/2 | - | 0 / 2 |
| A | all | **666.7** (666.7, 666.7) | 666.7 | 400.9 | 2500 / 2112 | 0.04 | 256 | 2/8 | - | - |
| B | 12 HumanEval (S5) | **1000.0** (1000.0, 1000.0) | 1000.0 | 476.3 | 2500 / 2413 | 0.15 | 256 | 0/2 | 256 / 512 | 0 / 2 |
| B | 12 MBPP (S5) | **1000.0** (1000.0, 1000.0) | 1000.0 | 582.1 | 2500 / 2046 | 0.05 | 256 | 0/2 | 256 / 512 | n/a (no AR ref) |
| B | 12 IFEval (N45) | **1000.0** (1000.0, 1000.0) | 1000.0 | 605.9 | 2500 / 2099 | 0.02 | 153 | 2/2 | 152 / 306 | 0 / 2 |
| B | 12 math (N45) | **1000.0** (1000.0, 1000.0) | 1000.0 | 614.6 | 2500 / 2087 | 0.03 | 256 | 0/2 | 256 / 512 | 0 / 2 |
| B | all | **1000.0** (1000.0, 1000.0) | 1000.0 | 566.9 | 2500 / 2157 | 0.04 | 256 | 2/8 | 920 / 1842 | - |
| C | 12 HumanEval (S5) | **666.7** (666.7, 666.7) | 666.7 | 353.7 | 2500 / 2410 | 0.15 | 256 | 0/2 | - | 2 / 2 |
| C | 12 MBPP (S5) | **666.7** (666.7, 666.7) | 666.7 | 413.9 | 2500 / 2307 | 0.05 | 256 | 0/2 | - | n/a (no AR ref) |
| C | 12 IFEval (N45) | **666.7** (666.7, 666.7) | 666.7 | 418.2 | 2500 / 2004 | 0.02 | 153 | 2/2 | - | 2 / 2 |
| C | 12 math (N45) | **666.7** (666.7, 666.7) | 666.7 | 426.8 | 2500 / 2025 | 0.03 | 256 | 0/2 | - | 2 / 2 |
| C | all | **666.7** (666.7, 666.7) | 666.7 | 402.1 | 2500 / 2179 | 0.04 | 256 | 2/8 | - | - |
| D | 12 HumanEval (S5) | **1000.0** (1000.0, 1000.0) | 1000.0 | 475.4 | 2500 / 2412 | 0.15 | 256 | 0/2 | 256 / 512 | 2 / 2 |
| D | 12 MBPP (S5) | **1000.0** (1000.0, 1000.0) | 1000.0 | 590.2 | 2500 / 2192 | 0.05 | 256 | 0/2 | 256 / 512 | n/a (no AR ref) |
| D | 12 IFEval (N45) | **1000.0** (1000.0, 1000.0) | 1000.0 | 603.3 | 2500 / 1994 | 0.02 | 153 | 2/2 | 152 / 306 | 2 / 2 |
| D | 12 math (N45) | **1000.0** (1000.0, 1000.0) | 1000.0 | 610.8 | 2500 / 1835 | 0.03 | 256 | 0/2 | 256 / 512 | 2 / 2 |
| D | all | **1000.0** (1000.0, 1000.0) | 1000.0 | 567.1 | 2500 / 2097 | 0.04 | 256 | 2/8 | 920 / 1842 | - |

## Processes (peak = /usr/bin/time -l peak memory footprint)

| run | done | peak GB | max RSS GB | arena GB (slots) | wired GB | footprint after fill / cap GB | HC Q8 | ready s | wall s | note |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 | True | 0.01 | 0.02 | 30.90 (18432) | 30.9 | 40.6 / 41.2 | True | 0.3 | 3 |  |
| A2 | True | 0.01 | 0.02 | 30.90 (18432) | 30.9 | 40.6 / 41.2 | True | 0.3 | 3 |  |
| B1 | True | 0.01 | 0.02 | 30.90 (18432) | 30.9 | 40.6 / 41.2 | True | 0.3 | 2 |  |
| B2 | True | 0.01 | 0.02 | 30.90 (18432) | 30.9 | 40.6 / 41.2 | True | 0.3 | 2 |  |
| C1 | True | 0.01 | 0.02 | 30.90 (18432) | 30.9 | 40.6 / 41.2 | False | 0.3 | 3 |  |
| C2 | True | 0.01 | 0.02 | 30.90 (18432) | 30.9 | 40.6 / 41.2 | False | 0.3 | 3 |  |
| D1 | True | 0.01 | 0.02 | 30.90 (18432) | 30.9 | 40.6 / 41.2 | False | 0.3 | 2 |  |
| D2 | True | 0.01 | 0.02 | 30.90 (18432) | 30.9 | 40.6 / 41.2 | False | 0.3 | 2 |  |

Peak footprint per arm (max over its processes): A 0.01 GB, B 0.01 GB, C 0.01 GB, D 0.01 GB

## Token identity across arms (same prompt and pass; information)

- B vs A: 8 / 8 identical
- D vs C: 8 / 8 identical
- C vs A: 0 / 8 identical
- D vs B: 0 / 8 identical

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
