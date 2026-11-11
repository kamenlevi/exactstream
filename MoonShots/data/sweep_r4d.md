R4 (d), 2026 community: summary
- ZMLX gather_qmm_swiglu (fused gate+up+SwiGLU per expert, needs a custom MLX build): +6% on GLM-4.7-Flash, ~0 on Qwen3-30B. Fused gating/combine: +2% decode on Qwen3.5-35B.
- 1-bit affine quantization in mlx 0.32.4 (new): a possible cold/substitute tier for MoEPrefetch, but low-bit cold tiers lost before (MP step 6).
- MLX_SDPA_BLOCKS tuning, TurboQuant KV (SwiftLM), mlx-mfa: long-context only.
- vllm-mlx continuous batching: parallel subagents only. Exo RDMA: multiple Macs. ReDrafter: covered by MTP.
