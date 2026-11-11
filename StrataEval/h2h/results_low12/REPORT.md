# Strata head-to-head (2026-11-08 18:57)

Strata 0.1.58-metal (commit 146df3c), Qwen3.8-Flash-Next ISTA-DASLab IQ3_XXS, experts streamed from the SSD (64 GB Mac), greedy, 256 new tokens, EOS stop, one warm process per arm and pass (palindrome A B C D D C B A), the exact prompt ids S5 / N45 used. Arms: **A** default (HC Q8_0, lossy), no MTP; **B** default + MTP; **C** exact (STRATA_HC_Q8=0), no MTP; **D** exact + MTP.

Records: 48 measured requests used; dropped: {'error': 0, 'freeze': 0, 'incomplete_process': 7}. Freeze windows in fanwatch.log: 543.

## Speed and identity

| arm | set | WHOLE DECODE tok/s (passes) | pooled | END TO END | prefill tok/s (engine / P÷TTFT) | TTFT s (median) | tokens/task | stops | MTP accepted / offered | == NAX6-N AR |
|---|---|---|---|---|---|---|---|---|---|---|
| B | 12 HumanEval (S5) | **14.9** (14.9) | 15.0 | 12.8 | 77 / 74 | 2.21 | 232 | 9/12 | 1237 / 1301 | 2 / 12 |
| B | 12 MBPP (S5) | **16.4** (16.4) | 16.5 | 11.2 | 62 / 60 | 1.87 | 63 | 11/12 | 584 / 652 | n/a (no AR ref) |
| B | 12 IFEval (N45) | **15.3** (15.3) | 15.5 | 10.3 | 23 / 22 | 2.68 | 118 | 9/12 | 648 / 959 | 1 / 12 |
| B | 12 math (N45) | **15.5** (15.5) | 15.4 | 13.2 | 31 / 30 | 3.23 | 256 | 0/12 | 1457 / 1611 | 0 / 12 |
| B | all | **15.5** (15.5) | 15.5 | 11.8 | 43 / 41 | 2.11 | 223 | 29/48 | 3926 / 4523 | - |

## Processes (peak = /usr/bin/time -l peak memory footprint)

| run | done | peak GB | max RSS GB | arena GB (slots) | wired GB | footprint after fill / cap GB | HC Q8 | ready s | wall s | note |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 | True | 14.82 | 13.52 | 4.00 (2085) | 4.0 | 14.8 / 41.2 | True | 1.0 | 672 |  |

Peak footprint per arm (max over its processes): B 14.82 GB

## Token identity across arms (same prompt and pass; information)


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
