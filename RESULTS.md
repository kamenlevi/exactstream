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
| `lossy`, `chat` | 24, without P2 (2) | 130.3 / 40.8 / 3.54 / 23.82 (two passes) | 91.2 / 26.1 / 3.34 / 23.74 (two passes) | 30.6 / 18.1 / 3.09 / 23.66 (two passes) | 40.8 / 27.4 / 2.86 / 24.01 (two passes) |
| `lossy`, `chat` | 24, with P2 (1.2.0) (3) | not timed | not timed | 31.4 / 24.4 / 1.12 / 23.00 (two passes) | not timed |

(1) Only before coherent reads (config L at C 160): HumanEval 150.7 / 36.0 / 4.43 / 31.78, MBPP 111.0 / 26.8 / 3.69 / 31.44.

(2) One configuration for both names (CONFIGS.md). These runs are in the 12 GB and 24 GB paper and in the RELEASE paper,
section 3. HumanEval comes from the research
run W24d, MBPP and IFEval from W24f, math (MATH-500) from W24g: the gated configuration, two passes each (data/w24 in BigMoE's
research tree). The peaks on these benchmark prompts reach 24.01 GB (a 520-token math prompt); longer prompts take more, up to
28.2 GB at 30K tokens (LIMITS.md).

(3) BigMoE's W24T run (Thu 12 Nov, an ABBA under a quiet lock: the 24 GB configuration without P2, with P2, with P2
again, without it again, on the same 12 IFEval prompts). Without P2 in that run: 30.8 / 17.9 / 3.11 / 23.70. The two P2
passes gave the same tokens on 12 of 12 prompts and the same logits on 639 of 639 checks. Only prompts of 256 tokens or
fewer take P2; on longer ones the 24 GB configuration runs as before. P2 reads 12.75 GB from the SSD per prompt for its
first token instead of 20.4. Strata on the same prompts: first token 1.46 s (median).

## Coding agents (serve.py, Claude Code)

EvalSuite's Claude Code agent test: a probe of 30 tool-call prompts (the line: 28 or more right, no invented tool) and 12
coding tasks, each in a fresh folder in a sandbox, graded by a check script (the line: 10 or more). Claude Code 2.1.321,
the `chat` configuration at the memory line given, through `serve.py`, greedy.

| run | conversation cache | probe | tasks | wall time | first token, median | lifetime peak | verdict |
|---|---|---|---|---|---|---|---|
| 32 GB (11 Nov) | off | 27 / 30 | 11 / 12 | 2,595 s | 31.9 s | not recorded | FAIL (probe) |
| 32 GB, 32c (11 Nov) | on | 29 / 30 | 11 / 12 | 915.5 s | 6.8 s | not recorded | PASS |
| 45 GB, 45c (11 Nov) | on | 29 / 30 | 11 / 12 (4) | 1,097 s | 6.8 s | 49.43 GB | PASS |
| 32 GB, 32c2 (11 Nov, a rerun with a stricter sandbox) | on | 29 / 30 | 11 / 12 | 875 s (tasks) | 6.3 s | 36.17 GB | PASS |

(4) All 12 checks passed, but one task read two other tasks' check scripts, which fails it under the test's rules.

In 32c2 the cache answered 122 of 123 requests from a stored state (the miss was the first request), so the server
prefilled 8.2 % of the prompt tokens: a median 636 new tokens per request. Its first token: median 6.3 s, 90th percentile
12.4 s, at most 21.0 s; decode median 72.7 tok/s on answers of a median 91 tokens; the cache held at most 2.50 GB.

## Conversation cache: exactness

A cache hit restores a stored state and prefills only the new tokens. To check that this gives a full prefill's results,
we replayed 45 recorded Claude Code and Codex requests with `CONV_CACHE_CHECK=1`, which also runs a cold prefill on every
hit and compares them: 12 of 12 hits bitwise equal at 32 GB and 12 of 12 at 45 GB (branches on). At 24 GB the cache is
unchecked and off by default.

The configs and their labels are in [CONFIGS.md](CONFIGS.md); the limits, including where Strata is ahead, in
[LIMITS.md](LIMITS.md).
