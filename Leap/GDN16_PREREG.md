# L1-15: multi-row GDN kernel for 16-row verify passes (Leap, pre-registered Sat 2026-10-24 10:33)
Assigned by NewIdeas (10:13). The GDN mixers are the biggest kernel gap on both roads. On the 35B at 16 rows they take
12.2 ms against a 2.04 ms bytes floor (L1-13). The 125B has 36 GDN layers.

## Bet
A kernel program for the GDN mixer (projections, conv, norms, gates, recurrence, gated norm) that runs 16 verify rows
fast. It must be batch-invariant: the same kernels and the same fixed order for 1-row decode and 16-row verify, so that
spec == plain bit for bit under its own numerics. Prototype on the 35B (fits in RAM); the 125B port comes after.

## Pre-registered (NewIdeas' line, adopted as is)
- **Pass:** 30 GDN mixers at 16 rows <= 4 ms on the 35B, with spec == plain bitwise (each row of a 16-row call equal to
  plain 1-row decode).
- **Kill:** > 6 ms, or not batch-invariant.
- **My prediction (before any data):** alive but borderline, 4-5 ms. At 16 rows MLX runs the projections as qmm and the
  rest as ~20 small dependent kernels per layer. My guess is about half each. Fusing the chain should leave ~1-1.5 ms.
  4 ms then needs the projections at >= ~70 % of their bytes floor in a fixed-tile kernel that decode also runs.

## Prior art (checked 10:20-10:33)
- **Chunkwise forms.** The delta rule's WY / UT form is from Yang et al., NeurIPS 2024, and Gated DeltaNet from Yang,
  Kautz, Hatamizadeh, ICLR 2025. flash-linear-attention's chunk_gated_delta_rule is for training and prefill; its
  fused_recurrent_gated_delta_rule (vLLM: num_accepted_tokens, ssm_state_indices) is for decode and verify.
  - MLX 0.32.4 itself now ships mx.fast.gated_delta_update. It has a chunked WY kernel on simdgroup MMA
    (gated_delta_fused_chunk), a NAX variant for the M5's neural accelerators, and a sequential kernel (gated_delta_seq).
  - A chunk form changes the arithmetic, so it is not bitwise to 1-row decode.
- **Multi-token GDN verify.**
  - FlashInfer gated_delta_rule_mtp: multi-token verify with intermediate states cached for rollback.
  - TreeWY (arXiv 2608.20961): a tree WY transform verifies all draft nodes with one triangular solve, then rebuilds only
    the accepted state. Evaluated on Qwen3.5 35B / 397B. No exactness claim.
  - ReplaySSM (NVIDIA + Dao AI Lab, 2026): keep a state checkpoint plus the recent inputs and replay on commit (z-lab's
    MLX loop already rolls GDN back this way). It is slower at batch 1.
- **Batch-invariant kernels.** Thinking Machines, "Defeating Nondeterminism in LLM Inference" (2025): fixed reduction
  order regardless of batch size, for matmul, RMSNorm and attention. That costs time on GPUs.
- **Here.**
  - MoonShots' exact fused GDN prework and gated norm (bitwise to MLX's eager chain, 125B shapes, S <= 8).
  - MoonShots L1-01: a multi-row qmv that is bitwise per row is ALU-bound: 1.6x MLX at M = 8, 4x at M = 16. Killed.
  - MoonShots L1-01b: 16-row padded qmm for every M, exact under its own numerics.
  - mlx_lm's recurrence kernel keeps the state in fp32 and takes T at run time, so per-token arithmetic does not depend
    on T. It should be batch-invariant as is; step 1 checks it.
- **Novelty: none claimed.** Every piece is published. New here: a bitwise spec == plain 16-row GDN on Apple's GPU with
  the projections included.

## Steps
1. GPU ~5 min (scripts/gdn16_profile.py): per-op time of the 30 mixers at S = 1-16, and which ops are batch-invariant
   (S rows vs plain 1-row decode, bitwise).
2. Kernels, in order of measured time: projections in one fixed-tile kernel used for decode too, then the fused chain.
3. Decisive test: 30 mixers at 16 rows, ms, plus a bitwise check of every row against plain decode (same kernels at
   S = 1) on real verify rows. Then the whole-model spec == plain check on real decodes.
