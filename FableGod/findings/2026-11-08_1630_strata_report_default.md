# Strata head-to-head (2026-11-08 16:11)

Strata 0.1.58-metal (commit 146df3c), Qwen3.8-Flash-Next ISTA-DASLab IQ3_XXS, experts streamed from the SSD (64 GB Mac), greedy, 256 new tokens, EOS stop, one warm process per arm and pass (palindrome A B C D D C B A), the exact prompt ids S5 / N45 used. Arms: **A** default (HC Q8_0, lossy), no MTP; **B** default + MTP; **C** exact (STRATA_HC_Q8=0), no MTP; **D** exact + MTP.

Records: 384 measured requests used; dropped: {'error': 0, 'freeze': 0, 'incomplete_process': 0}. Freeze windows in fanwatch.log: 541.

## Speed and identity

| arm | set | WHOLE DECODE tok/s (passes) | pooled | END TO END | prefill tok/s (engine / P÷TTFT) | TTFT s (median) | tokens/task | stops | MTP accepted / offered | == NAX6-N AR |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 12 HumanEval (S5) | **24.6** (24.9, 24.3) | 24.5 | 22.0 | 185 / 176 | 0.94 | 232 | 18/24 | - | 4 / 24 |
| A | 12 MBPP (S5) | **24.9** (25.4, 24.5) | 24.8 | 19.2 | 154 / 146 | 0.79 | 63 | 22/24 | - | n/a (no AR ref) |
| A | 12 IFEval (N45) | **26.3** (27.5, 25.2) | 26.7 | 20.2 | 66 / 64 | 0.91 | 118 | 18/24 | - | 2 / 24 |
| A | 12 math (N45) | **25.7** (26.6, 24.7) | 25.6 | 23.0 | 77 / 74 | 1.17 | 256 | 0/24 | - | 0 / 24 |
| A | all | **25.4** (26.1, 24.7) | 25.3 | 21.1 | 110 / 105 | 0.88 | 223 | 58/96 | - | - |
| B | 12 HumanEval (S5) | **38.5** (38.6, 38.3) | 38.7 | 32.3 | 173 / 165 | 0.99 | 232 | 18/24 | 2474 / 2602 | 4 / 24 |
| B | 12 MBPP (S5) | **38.6** (38.4, 38.7) | 38.4 | 26.1 | 143 / 136 | 0.83 | 63 | 22/24 | 1168 / 1304 | n/a (no AR ref) |
| B | 12 IFEval (N45) | **36.3** (36.4, 36.2) | 37.3 | 25.6 | 62 / 59 | 0.98 | 118 | 18/24 | 1296 / 1918 | 2 / 24 |
| B | 12 math (N45) | **39.8** (40.1, 39.5) | 39.7 | 33.4 | 70 / 68 | 1.34 | 256 | 0/24 | 2914 / 3222 | 0 / 24 |
| B | all | **38.3** (38.4, 38.2) | 38.7 | 29.1 | 102 / 98 | 0.93 | 223 | 58/96 | 7852 / 9046 | - |
| C | 12 HumanEval (S5) | **23.9** (23.8, 24.1) | 24.0 | 21.5 | 179 / 170 | 0.96 | 232 | 18/24 | - | 4 / 24 |
| C | 12 MBPP (S5) | **24.1** (23.9, 24.2) | 24.2 | 18.6 | 151 / 142 | 0.79 | 64 | 20/24 | - | n/a (no AR ref) |
| C | 12 IFEval (N45) | **25.2** (25.4, 25.0) | 25.4 | 19.4 | 64 / 61 | 1.01 | 126 | 20/24 | - | 2 / 24 |
| C | 12 math (N45) | **25.3** (25.8, 24.8) | 25.3 | 22.6 | 73 / 71 | 1.32 | 256 | 0/24 | - | 0 / 24 |
| C | all | **24.6** (24.7, 24.5) | 24.7 | 20.5 | 106 / 101 | 0.91 | 223 | 58/96 | - | - |
| D | 12 HumanEval (S5) | **37.6** (38.0, 37.2) | 37.8 | 31.6 | 170 / 162 | 1.00 | 232 | 18/24 | 2488 / 2618 | 4 / 24 |
| D | 12 MBPP (S5) | **36.5** (36.3, 36.7) | 36.6 | 25.1 | 141 / 134 | 0.85 | 64 | 20/24 | 1174 / 1324 | n/a (no AR ref) |
| D | 12 IFEval (N45) | **34.7** (35.0, 34.5) | 35.8 | 24.5 | 58 / 56 | 1.06 | 126 | 20/24 | 1274 / 1938 | 2 / 24 |
| D | 12 math (N45) | **39.2** (40.2, 38.3) | 39.2 | 33.1 | 71 / 68 | 1.38 | 256 | 0/24 | 2894 / 3242 | 0 / 24 |
| D | all | **37.0** (37.3, 36.6) | 37.7 | 28.3 | 100 / 95 | 0.95 | 223 | 58/96 | 7830 / 9122 | - |

## Processes (peak = /usr/bin/time -l peak memory footprint)

| run | done | peak GB | max RSS GB | arena GB (slots) | wired GB | footprint after fill / cap GB | HC Q8 | ready s | wall s | note |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 | True | 34.90 | 35.74 | 27.70 (14909) | 27.7 | 34.9 / 41.2 | True | 1.1 | 378 |  |
| A2 | True | 33.56 | 34.40 | 26.36 (14195) | 26.4 | 33.5 / 41.2 | True | 1.1 | 401 |  |
| B1 | True | 33.64 | 35.38 | 25.89 (13956) | 25.9 | 33.6 / 41.2 | True | 1.2 | 274 |  |
| B2 | True | 33.47 | 35.19 | 25.78 (13893) | - | 32.0 / 41.2 | True | 1.5 | 277 |  |
| C1 | True | 33.94 | 34.78 | 26.46 (14239) | 26.5 | 33.9 / 41.2 | False | 1.2 | 398 |  |
| C2 | True | 34.05 | 34.89 | 26.56 (14288) | 26.6 | 34.0 / 41.2 | False | 1.2 | 401 |  |
| D1 | True | 34.03 | 35.76 | 25.88 (13950) | 25.9 | 34.0 / 41.2 | False | 1.4 | 281 |  |
| D2 | True | 33.45 | 35.19 | 25.31 (13622) | 25.3 | 33.4 / 41.2 | False | 1.2 | 286 |  |

Peak footprint per arm (max over its processes): A 34.90 GB, B 33.64 GB, C 34.05 GB, D 34.03 GB

## Token identity across arms (same prompt and pass; information)

- B vs A: 96 / 96 identical
- D vs C: 96 / 96 identical
- C vs A: 50 / 96 identical
- D vs B: 50 / 96 identical

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
