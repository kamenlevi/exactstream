Sweep (c), tokens one at a time: summary
- **DFlash block-diffusion drafter exists for Qwen3.6-35B-A3B, with an MLX backend** (huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash; z-lab/dflash; dflash-mlx). Reported 2.9x on B200, 2.2x HumanEval on an Intel laptop. Check whether SpecDecode knows it: it may beat MTP on fresh code.
- MoE-aware verify (MoE-Spec 2602.16052, EVICT, ECOSPEC): CUDA papers. SD's simulation (§5.3) found expert-aware truncation never beats adaptive lookup on this model, so likely ≤5% here.
- DDTree/CaDDTree: wide trees cost expert bytes; only the cost-aware variant could pay.
- Lookahead/Jacobi, layer-skip self-spec: ≤5% / ~0 on a 3B-active MoE with MTP.
- PASTE (2603.18897): speculative execution of read-only tools in agents, −40–48% task wall-clock; nobody has it locally. Outside token speed, but a Claude Code proxy could do it.
