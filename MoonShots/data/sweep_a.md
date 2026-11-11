Sweep (a), weights read per token: summary
- Nothing lossless: SiLU activations are almost never exactly zero.
- TEAL (2408.14690, CUDA): 40% sparsity on linear-attn + full-attn + shared ≈ −0.60 GB, ~+21%, lossy.
- Apple Metal dynamic sparsity (2511.04477, not MLX): 1.3–1.55x on M2/A19 Pro, Llama-2-7B ppl 5.28 → 5.65 at 50%.
- R-Sparse (2504.19449, CUDA): +43% end to end, lossy.
- Expert-neuron skipping (2605.08575, vLLM): down_proj 80% skipped, +9%, keeps 95% of benchmark score.
- DeltaLLM temporal delta (2507.19608): attention only; consecutive hidden states are far apart.
- SeedLM (2410.10714, Apple): same bytes as Q4; T-MAC saves compute, not bytes.
- Gaps: TEAL-style sparsity on GatedDeltaNet projections (+14%, lossy, not on MLX); neuron-level union in the verify pass (+10–15% on spec, lossy).
