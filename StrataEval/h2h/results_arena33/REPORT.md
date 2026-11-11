# Strata head-to-head (2026-11-08 16:25)

Strata 0.1.58-metal (commit 146df3c), Qwen3.8-Flash-Next ISTA-DASLab IQ3_XXS, experts streamed from the SSD (64 GB Mac), greedy, 256 new tokens, EOS stop, one warm process per arm and pass (palindrome A B C D D C B A), the exact prompt ids S5 / N45 used. Arms: **A** default (HC Q8_0, lossy), no MTP; **B** default + MTP; **C** exact (STRATA_HC_Q8=0), no MTP; **D** exact + MTP.

Records: 96 measured requests used; dropped: {'error': 0, 'freeze': 0, 'incomplete_process': 0}. Freeze windows in fanwatch.log: 542.

## Speed and identity

| arm | set | WHOLE DECODE tok/s (passes) | pooled | END TO END | prefill tok/s (engine / P÷TTFT) | TTFT s (median) | tokens/task | stops | MTP accepted / offered | == NAX6-N AR |
|---|---|---|---|---|---|---|---|---|---|---|
| B | 12 HumanEval (S5) | **46.9** (46.9) | 47.0 | 40.5 | 259 / 247 | 0.67 | 232 | 9/12 | 1237 / 1301 | 2 / 12 |
| B | 12 MBPP (S5) | **45.8** (45.8) | 45.6 | 33.1 | 214 / 202 | 0.54 | 63 | 11/12 | 584 / 652 | n/a (no AR ref) |
| B | 12 IFEval (N45) | **41.4** (41.4) | 41.8 | 29.7 | 76 / 73 | 0.83 | 118 | 9/12 | 648 / 959 | 1 / 12 |
| B | 12 math (N45) | **47.2** (47.2) | 47.1 | 40.6 | 100 / 96 | 1.01 | 256 | 0/12 | 1457 / 1611 | 0 / 12 |
| B | all | **45.2** (45.2) | 45.8 | 35.7 | 143 / 137 | 0.64 | 223 | 29/48 | 3926 / 4523 | - |
| D | 12 HumanEval (S5) | **46.2** (46.2) | 46.4 | 40.1 | 263 / 250 | 0.66 | 232 | 9/12 | 1244 / 1309 | 2 / 12 |
| D | 12 MBPP (S5) | **43.8** (43.8) | 43.6 | 32.2 | 219 / 205 | 0.54 | 64 | 10/12 | 587 / 662 | n/a (no AR ref) |
| D | 12 IFEval (N45) | **40.1** (40.1) | 40.6 | 29.2 | 77 / 74 | 0.83 | 126 | 10/12 | 637 / 969 | 1 / 12 |
| D | 12 math (N45) | **46.3** (46.3) | 46.3 | 39.9 | 100 / 96 | 1.01 | 256 | 0/12 | 1447 / 1621 | 0 / 12 |
| D | all | **44.0** (44.0) | 44.7 | 35.0 | 145 / 138 | 0.63 | 223 | 29/48 | 3915 / 4561 | - |

## Processes (peak = /usr/bin/time -l peak memory footprint)

| run | done | peak GB | max RSS GB | arena GB (slots) | wired GB | footprint after fill / cap GB | HC Q8 | ready s | wall s | note |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 | True | 40.76 | 38.49 | 33.00 (17723) | 33.0 | 40.7 / 41.2 | True | 1.3 | 238 |  |
| D1 | True | 41.15 | 41.54 | 33.00 (17723) | 33.0 | 41.1 / 41.2 | False | 3.4 | 237 |  |

Peak footprint per arm (max over its processes): B 40.76 GB, D 41.15 GB

## Token identity across arms (same prompt and pass; information)

- D vs B: 25 / 48 identical

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
