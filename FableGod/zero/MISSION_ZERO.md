# MISSION ZERO: the fastest way to run this model on this Mac, from first principles

You know nothing about anyone's previous attempts, and that is the point. Don't read any file outside ~/FableGod/zero/.
Think from physics and first principles; use the web only for published facts, never for this project.

## The problem
- **Model:** a 125B-parameter Mixture-of-Experts language model. 48 layers: 36 linear-attention (gated delta-net) layers
  and 12 full-attention layers (every 4th; 24 query heads, 2 KV heads). Hidden size 2,560.
  - Each layer: 512 routed experts, the top 10 used per token (expert MLP width 640, SwiGLU), plus 1 shared expert
    (width 640).
  - Vocabulary 248,320. One built-in multi-token-prediction (MTP) layer.
  - Weights are quantized to ~3-4 bits: the whole model is ~45-70 GB, the experts are most of it, each expert ~2-3 MB.
- **Machine:** one MacBook Pro, Apple M5 Pro, 64 GB unified memory.
  - The GPU (20 cores, with matrix units) streams memory at ~286 GB/s.
  - The internal SSD reads at ~12-13 GB/s.
  - A CPU and a Neural Engine are also there.
- **Budget:** the model's process may use at most 24 GB of RAM (stretch goal: 12 GB). The rest of the weights live on the
  SSD and must be read on demand.
- **Use:** one user, interactive generation (code, chat, math, agent tasks), batch size 1.
- **Quality rule, two modes:**
  - EXACT: output identical to running the model token by token.
  - FAST: may differ, but must score the same on standard benchmarks (code, math, instruction following).
- **The target:** as many tokens per second as possible. Goals: >= 300 tok/s on code in FAST mode, and <= 10 MB of SSD
  reads per generated token.

## Deliver (by ~06:30; one file: ~/FableGod/zero/ZERO.md)
1. The physics: what bounds tok/s here (bytes, compute, latency, sequential dependence), with numbers.
2. Your top 3 designs to approach the target, each with:
   - the mechanism;
   - expected tok/s and SSD MB per token, as arithmetic;
   - EXACT or FAST;
   - the biggest risk;
   - the cheapest test (<= 10 min of GPU, or CPU only) with a PASS / KILL line written before the run.
3. Which one you would bet on, and why.

## Rules
- Thinking and writing only: no GPU jobs, no downloads, no files outside your folder.
- Mark claims [K] known/published (with a link), [P] arithmetic, [G] judgement.
- Stop when the file is complete or at 06:30, whichever comes first. No waiting loops.
- When done, send the orchestrator one message: your best idea in one line, then the file path.
