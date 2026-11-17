# S1: chunked TREE-form GDN for 64-128-row lossy checks (design notes; DEFERRED by the orchestrator 08:3x, Tue 17 Nov)

Source: ZERO's IDEAS_1117b_cheapest_row.md, idea 1 (a). Deferred because, on Lunatic's limits (IDEAS_1117b_limits.md), a
marginal row stays >= 0.25-0.37 ms and own-information trees peak near 48 rows (+3-10 %), so the 32-row check is the line.
S2 (s2_bench.txt, 08:30) measured the expert side: 64 rows = 1.15-1.19x the 32-row call; 96 / 128 project 1.5x / 1.7-1.9x.

## Why gdn32c does not extend
- gdn32c.py (Mon 13:0x) is the CHAIN chunk form for T <= 32: K and Q tiles (bf16) + two C x C fp32 matrices in threadgroup
  memory (32 KB at C 32), and a per-dv-column triangular solve in registers. That solve made it 3.1x slower than the
  sequential kernel at T 32. At C 64 / 128 the tiles alone exceed 32 KB.

## The tree form (what to build)
- Inputs per layer and head: block-start state S0 [Dv 128, Dk 128] fp32, rows in topological order (parents first), the
  ancestor lists (depth <= ~30), q / k / v / g / beta per row.
- Lambda_j = sum of log g over j's root path (a prefix sum over the tree); Gam_ji = exp(Lambda_j - Lambda_i) for i an
  ancestor of j.
- Solve (I + diag(beta) L) Delta = diag(beta) (V - diag(gamma) K S0^T), L_ji = Gam_ji (k_j . k_i) for i a STRICT ancestor.
  Each row sums only over its ancestors (<= depth terms, not T), and rows at the same depth are independent: ~30 serial
  LEVELS of a parallel step, not 128 serial rows.
- Y = diag(gamma) Q S0^T + M Delta, M_ji = Gam_ji (q_j . k_i) over ancestors incl. itself (sparse: ~T x depth).
- The next state only along the ACCEPTED path (<= ~32 rows), after the commit: S_C = gamma_C S0 + Delta_path^T (...) K_path.
- MACs per head-layer: Q S0^T and K S0^T 2 x 128^3 = 4.2 M (matrix-unit work, 32x32x32 tiles); ancestor dots and the solve
  ~0.5 M. 36 layers x 48 heads: ~8.6 G MAC. Bytes: the 226 MB state read + write (0.81 ms [K]). Estimate ~1.5-3 ms at 128
  rows, less at 64 [P].
- Kernel shape: one threadgroup per (head, Dv quarter); S0 quarter in registers; the dense Q S0^T / K S0^T as matmul2d
  32x32x32 tiles; the level loop with a threadgroup barrier per level; Gam via log-space prefix sums along the tree.
- Lossy (fp reassociation); exact mode keeps the sequential cores. Bit target: max |dy| / max |y| vs the sequential form
  on real blocks ~1e-3.
- Effort: ~2-4 h build + check. Revisit only if trees > 32 rows come back after Saturday.
