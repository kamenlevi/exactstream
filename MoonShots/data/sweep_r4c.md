R4 (c), prefill/decode overlap: summary
- **BaseRT (arXiv 2607.19438, native Metal) claims 3.9x MLX's prefill on an M5 Pro** using the neural accelerators. Check whether that holds for MoE gather (our 8 vs 20 TFLOPS gap). If it does, prefill could reach ~2x.
- Layered prefill (2510.08055, CUDA): also removes the up to 39% extra expert traffic that chunked prefill causes, and lets decode run during long prefills. Our #18 tests the per-token gather cost.
- vllm-metal v0.28 ports Sarathi-style chunked prefill to Apple: helps parallel subagents (single stream ~0).
- FusionML CPU+GPU prefill: 1.15–1.38x on dense models; ANE ruled out.
- Speculative prefill of predictable prompts during tool/idle time: nobody does it. Est. hides 5–20 s at session starts/compactions, but SD's persisted prefixes already cover most of the fixed part.
