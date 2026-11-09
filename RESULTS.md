# Results

Measured on the test MacBook Pro, pooled over 12 prompts per set, 256 new tokens, greedy: whole decode / whole request (tok/s) /
first token (median, s) / lifetime peak (GB). From the RELEASE paper, section 2, which gives the sources.

| config | line | HumanEval | MBPP | IFEval (chat) | math |
|---|---|---|---|---|---|
| `exact` | 45 | 101.4 / 32.6 / 4.41 / 42.24 | 70.4 / 21.7 / 3.80 / 41.92 | 37.1 / 20.8 / 2.89 / 41.70 | 77.9 / 38.9 / 2.73 / 42.50 |
| `exact` | 32 | not measured | not measured | 33.8 / 19.8 / 2.83 / 30.62 | not measured |
| `lossy` | 45 | 164.0 / 36.2 / 4.60 / 42.94 | 112.7 / 25.3 / 3.72 / 42.60 | 45.2 / 22.4 / 3.01 / 42.42 (two passes) | 78.9 / 37.9 / 3.20 / 43.17 |
| `lossy` | 32 | not measured (1) | not measured (1) | 39.0 / 21.1 / 2.90 / 31.30 (one pass) | not measured |
| `chat` | 45 | not run | not run | 50.7 / 28.8 / 2.26 / 43.69 (one pass) | not run |
| `chat` | 32 | not run | not run | 51.3 / 22.5 / 2.99 / 31.22 | not run |

(1) Only before coherent reads (config L at C 160): HumanEval 150.7 / 36.0 / 4.43 / 31.78, MBPP 111.0 / 26.8 / 3.69 / 31.44.

The configs and their labels are in [CONFIGS.md](CONFIGS.md); the limits, including where Strata is ahead, in
[LIMITS.md](LIMITS.md).
