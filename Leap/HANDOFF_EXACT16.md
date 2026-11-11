# Hand-off: three bit-identical speedups for the 16-row verify check (Leap -> BigMoE, Tue 2026-10-27 22:25)

Owner of the engine: BigMoE (MTPLX). Nothing here touches BigMoE's repo. Everything is in ~/Leap, as monkeypatches on
MTPLX read-only. NewIdeas routes the priority (it matters mainly for the 45 GB / resident engine; at 24 GB the SSD
caps it).

## What it buys (measured in CA's real 12-layer check of the 125B, bitwise vs the shipped engine)

| lever | 12-layer check, 1K | 12-layer check, 8K | x4 for 48 layers | bits |
|---|---|---|---|---|
| S4 + D2 together (R-36) | -4.60 ms (36.09 -> 31.51) | -4.55 ms (38.30 -> 33.79) | ~-18.3 ms per check | logits of 43 checks + every cache state equal |
| SB (R-11c, measured per draft + check cycle) | -4.04 ms | -4.21 ms | ~-11 to -16 ms | drafts, tokens and states equal |
| **all three stacked (R-39, checks only)** | **-6.63 ms (36.46 -> 29.98)** | **-7.73 ms (39.65 -> 31.76)** | **~-26.5 / -30.9 ms per check** | logits of 43 checks + every cache state equal |

Standalone kernel numbers (kern23's cold shapes / counts): S4 -14.68 ms per 96 HC reads (R-09b), D2 -4.52 ms per check
(R-13d). The three stack at 0.97-1.0 of the sum of their parts. SB is checks-only in R-39 (2.1-3.2 ms per 12
layers); with the drafter in the cycle (R-11c) it was 4.0-4.2, so the real loop may gain ~-30 to -34 ms per check.

## The three changes

1. **S4: the 16-row hyper-connection read** (R-09 -> R-09b -> R-36).
   - Where: `GatedResidual.__call__`, when the read has 16 rows, the module has `block_inject_weight`, and
     `current_attention_phase() == "decode_verify"`. That is today's eager chain (~14 kernels), because
     `_verify_rows_read_applies` only covers 2-8 rows.
   - What: `leapk.hc_read(module, wcat, hyper_input, pending, rows=16)`:
     - MTPLX's own norm kernel (pending write folded in, or not);
     - one GEMM for down + inject (wcat = concat(down.weight, block_inject.weight), cached per module) that
       replicates MLX's NAX split-K order: 4,096-wide K partitions, 33 one-simdgroup tiles, loads 3 steps ahead;
     - the gate from tables of the stock functions, MLX's own up GEMM, and the tail (MTPLX's arithmetic).
   - Bits: equal to the eager chain (96-read chain test, R-09b; in-check, R-36).
2. **D2: the split-K quantized matmuls at 16 rows** (R-13d -> R-36).
   - Where: any `mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=64, bits=4, mode="affine")`
     with bf16 x of 16 rows (batch 1) that MLX 0.32.2 would serve with qmm_t_splitk. `leapk.routes_splitk(16, N, K)`
     mirrors that dispatch. In the 125B: in_proj_z (36 per check), out_proj (36), QSA o_proj (12).
   - What: `leapk.splitk_d2(x2d, (w, scales, biases))`. It is MLX's own `qmm_t_impl` at 16-row tiles instead of 32,
     then `mx.sum(partials, axis=0)`, MLX's own reduction.
   - Bits: equal to MLX's call (R-13d standalone, R-36 in-check).
3. **SB: two scheduling lines** (R-11c).
   - In the 16-row verify forward: `mx.async_eval(h, *Q._midloop_state_arrays(cache[i + 1 - 4:i + 1]))` every 4
     layers (CA's `layers_mid`).
   - Right after the acceptance read: `mx.async_eval(*cache states)`, so the commit replay goes out on its own while
     the CPU builds the next draft.
   - Bits: unchanged (only scheduling).

## Optional, for the drafter (R-15 step 2, 4-bit DFlash2 drafter, eager path)
4. **D2 for the drafter's MLP** (bit-identical): the same `mx.quantized_matmul` shim. At block 16 the drafter's
   gate / up (7680 x 2560, split 2) and down (2560 x 7680, split 6) take split-K. Draft 9.93 -> 8.47 ms at b16, tokens
   x1.0000. At b32 nothing changes (32-row calls leave split-K).
5. **Gate | up merged** (drafter-only rounding, not bit-identical): one 15,360-wide quantized matmul (480 tiles, so
   MLX's NAX qmm, no split), split after. Alone: 8.29 ms at b16, tokens x0.9955 (b32 10.06, x0.9984). With D2 on down:
   7.93 ms at b16, x0.9955. The drafter only proposes tokens, so this needs no reference change, but it does change
   which tokens are drafted (-0.45 % accepted at b16).

## Files
- `~/Leap/scripts/leapk.py`: the kernels (hc_read, s4_gemm, splitk_d2, routes_splitk, split_k_of). MLX's headers are
  inlined from the installed package at first use.
- `~/Leap/scripts/r36_screen.py`: the two patch points as monkeypatches (gr_call, qmm). The SB lines are in
  `r11c_screen.py`'s `cycle()`. The drafter patches (4, 5) are in `r15b_screen.py` (qmm shim, mlp_call).
- Bitwise / timing tests: r09c_screen.py (S4 standalone), r13d_screen.py (D2 standalone), r36_screen.py (in-check),
  r11c_screen.py (SB), r36b_screen.py (R-39, all three).

## Conditions and caveats
- **Tied to MLX 0.32.2 on gen-17 (M5), non-'d' GPUs.** The replicas mirror MLX's dispatch: the NAX split-K partitions
  of 4,096, qmm_splitk's split rule, MLX's 32-row padding at M = 16. After any MLX upgrade, re-run the bitwise tests
  before enabling. Better, add a load-time self-test (a few hc_read / splitk_d2 outputs vs the stock path, bitwise)
  that falls back to the stock path on any difference.
- Integer constants live in each kernel's header (`constant constexpr int`). Integer template arguments next to a
  header did not build in mx.fast.metal_kernel; only the dtype stays a template argument.
- Memory: wcat costs 324 x 10,240 bf16 = 6.6 MB per HC module, ~0.64 GB for all 96. That is fine at 45 GB. For 24 GB,
  the S4 kernel can take the down and inject matrices as two inputs (its B rows by column index) to avoid the copy.
- First use compiles each kernel shape (~1 s each; a handful of shapes).
- Measured on a 12-layer slice (layers 0-11) and scaled x4. A 48-layer engine A/B is the final confirmation.
