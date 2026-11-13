# Configs

ExactStream streams the model's experts from the internal SSD into fixed memory pools, with speculative decoding checked
on the matrix units. Exact configs produce tokens bitwise equal to the exact reference; lossy configs are labelled by
gates on the model's known benchmarks. This is research code, released as measured.

| config | memory line | what it is | label |
|---|---|---|---|
| `exact` | 45 (C 256) / 32 (C 160) | config E: exact, tokens bitwise equal to the exact reference (NAX6-N) | exact |
| `lossy` | 45 / 32 | config L + COH | 45: lossy, same scores as `exact` on IFBench, GPQA Diamond and LiveCodeBench v6 (a card gate, both orders); 32: lossy, not gated (no gate ran at C 160; on IFEval it gives the 45 GB line's tokens on 12 / 12 prompts) |
| `chat` | 45 / 32 | config zs = M1-B + COH + SB1; since 1.2.1 at 45 GB a prompt of 256 tokens or fewer skips the prefill and runs as 16-row forwards on the decode's check path (P2); decode loop `engine/loop/r57_screen_c45p2.py` | 45: lossy, same scores as `exact` on IFEval, IFBench, GPQA Diamond and LiveCodeBench v6 (gates on M1-B + COH, both orders; without P2 the shipped config gives its tokens on all 12 IFEval prompts); P2 lossy, same scores as the configuration without it on IFBench and GPQA Diamond's prompts of 256 tokens or fewer (the 45 GB P2 gate, Fri 13 Nov 2026); with P2 the first token on IFEval takes 0.47 s instead of 2.44 s ([RESULTS.md](RESULTS.md)); 32: lossy, not gated (its card gate did not run) |
| `lossy`, `chat` | 24 (C 100) | one config, w13b + P2: L + COH below 24 GB (pools of 141 base + 26 sidecar slots, at most 13 rows per check, only the 4-bit block drafter); since 1.2.0 a prompt of 256 tokens or fewer skips the prefill and runs as 13-row forwards on the decode's check path (P2); decode loop `engine/loop/r57_screen_w24p2.py` | lossy, same scores as `exact` on IFBench, GPQA Diamond and LiveCodeBench v6 (the 24 GB card gate, Tue 10 Nov 2026); P2 lossy, same scores as the configuration without it on IFBench and GPQA Diamond's prompts of 256 tokens or fewer (the P2 gate, Wed 11 Nov 2026); under 24 GB on prompts up to about 500 tokens and more on longer ones: about 25.0 GB at 3K tokens, 25.7 GB at 7K, 25.8 GB at 10K and 28.2 GB at 30K; about 130 tok/s on HumanEval, 91 on MBPP, 41 on math and 31 on chat (IFEval); with P2 the first token on IFEval takes 1.12 s instead of 3.11 s ([RESULTS.md](RESULTS.md)) |

Each config is one environment line in `configs/<config>-<mem>.env`, as the research runs used it. The 24 GB configs add
a `SNAPSHOT=w24p2` line (their decode loop; `chat` at 45 GB has `SNAPSHOT=c45p2` since 1.2.1) and keep the prefill chunk, `R57_PFCHUNK`, on a line of its own. The exact
configs pin `MTPLX_QSA_PREFILL=0` (1.2.0): MTPLX's sparse prefill attention, which is not exact, can only switch on in a
prompt phase this engine never sets, so the pin changes nothing today and keeps the exact configs dense if that changes. `--kind
chat|code|math` picks which benchmark family's switches apply to a prompt (default chat). The memory line (45, 32 or 24
GB) is the engine's lifetime peak footprint (phys_footprint, GB = 10^9 bytes). Besides it, every 45 and 32 GB config holds
a 7.45 GB volatile expert tier as purgeable memory, which macOS reclaims before it compresses or swaps and which the
footprint does not count; the 24 GB configs hold none.

The measured speeds are in [RESULTS.md](RESULTS.md); the limits in [LIMITS.md](LIMITS.md).
