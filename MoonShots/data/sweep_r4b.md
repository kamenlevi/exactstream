R4 (b), router/expert set fixed: summary
- Lossless options all measured or covered already: expert-major segmented GEMM (mlx-kquant gather_qmm_seg, MLX prefill) ≈ our dedup kernel (0–5%); EcoSpec/EVICT ≈ SD's simulation (no gain); ELDR batching and MoE-Infinity caching don't apply to one user with unified memory.
- Lossy: MoE-Spec expert cap, SERE remap of minor experts (vLLM, 2602.07616) could take verify from 1.75x to ~1.45x.
- Conclusion: the lossless verify headroom on this model is ~5–10%. The expert union is a hard floor without changing outputs.
