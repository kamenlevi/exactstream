# Configs

ExactStream streams the model's experts from the internal SSD into fixed memory pools, with speculative decoding checked
on the matrix units. Exact configs produce tokens bitwise equal to the exact reference; lossy configs are labelled by
gates on the model's known benchmarks. This is research code, released as measured.

| config | memory line | what it is | label |
|---|---|---|---|
| `exact` | 45 (C 256) / 32 (C 160) | config E: exact, tokens bitwise equal to the exact reference (NAX6-N) | exact |
| `lossy` | 45 / 32 | config L + COH | 45: lossy, same scores as `exact` on IFBench, GPQA Diamond and LiveCodeBench v6 (a card gate, both orders); 32: lossy, not gated (no gate ran at C 160; on IFEval it gives the 45 GB line's tokens on 12 / 12 prompts) |
| `chat` | 45 / 32 | config zs = M1-B + COH + SB1 | 45: lossy, same scores as `exact` on IFEval, IFBench, GPQA Diamond and LiveCodeBench v6 (gates on M1-B + COH, both orders; the shipped config gives its tokens on all 12 IFEval prompts); 32: lossy, not gated (its card gate did not run) |

Each config is one environment line in `configs/<config>-<mem>.env`, as the research runs used it. `--kind
chat|code|math` picks which benchmark family's switches apply to a prompt (default chat). The memory line (45 or 32 GB)
is the engine's lifetime peak footprint (phys_footprint, GB = 10^9 bytes). Besides it, every config holds a 7.45 GB
volatile expert tier as purgeable memory, which macOS reclaims before it compresses or swaps and which the footprint does
not count.

The measured speeds are in [RESULTS.md](RESULTS.md); the limits in [LIMITS.md](LIMITS.md).
