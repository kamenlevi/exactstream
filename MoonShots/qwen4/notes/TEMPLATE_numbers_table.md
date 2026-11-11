# Qwen 4 day-0 numbers (<model id>), live table: update every row the moment it is measured (T = hours since public)

| T | line | mode | workload | tok/s whole decode | steady | exact check | baseline (same Mac) | commit / log |
|---|---|---|---|---|---|---|---|---|
| | 45 GB | exact | 12 HumanEval | | | n / 12 == AR | | |
| | 45 GB | lossy (FAST) | 12 HumanEval | | | quality gate: | | |
| | 45 GB | exact | 12 IFEval (chat) | | | | | |
| | 45 GB | exact | 12 math | | | | | |
| | 45 GB | exact | prefill 1K / 4K / 8K tok/s + TTFT | | | | | |
| | 24 GB | exact | 12 HumanEval | | | | | |
| | 24 GB | lossy | 12 HumanEval | | | | | |
| | 12 GB | exact / lossy | 12 HumanEval | | | | | |

Rules: whole-decode tok/s = generated tokens / decode wall time incl. stalls (state it); every exact row names its reference
AR; every lossy row names its quality gate; numbers are [K] only when measured on this machine in this table's commit.
