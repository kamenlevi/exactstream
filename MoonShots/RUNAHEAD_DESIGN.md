# RUNAHEAD (M-16): engine sketch for the 45 GB default

MoonShots, Sun 2026-11-01. Status: step 1 PASS (CPU), step 2 (live dual pass) tonight. This note is what BigMoE / Leap would
build if step 2 passes; nothing here is built.

## Idea

Runahead execution (Mutlu et al., HPCA 2003) applied to the MoE verify pass.

Today each check waits on its SSD misses layer by layer. R-69b (C 160, OOO) measured cycle ~ 50 ms + 1.1 x demand SSD, and
Breakthrough's FIXED-PART fit on D-23 (C 224) measured check + draft = 42.2 + 0.559 x io + 7.82 x rows. Either way, the misses
sit on the critical path.

RUNAHEAD takes them off. Pass k computes two row groups:
- **A**: the rows that pass k-1 accepted provisionally. By now their experts are resident, so this is the **exact** verdict.
  Tokens are emitted only from A.
- **B**: the next drafted block. Non-resident experts are **skipped**, with no wait. Their reads are queued at deep queue
  and land during the rest of pass k. B's provisional verdict picks the bonus token and the next block.

Pass k+1 re-checks B as its A. The output is bit-exact: every emitted row comes from an all-resident exact pass, provided the
kernels are batch-invariant (REF2b).

## Pass structure

1. **Rows.** Each pass runs `[A rows | B rows]` in one forward. A comes first, so B never influences A (causal attention, GDN
   order, per-row MoE).
2. **Gate.**
   - A rows post as today, expect ~all hits, and spin for any residual miss.
   - B rows post with **no spin**. The loader reads their misses into slots with the usual eviction, so experts used by the
     current pass stay protected. On the GPU, a B row's non-resident pairs are phase 2. They are never computed and give 0 in
     `combine_phase`.
3. **Hybrid option.** B also waits for misses that some row needs at a 4-bit rank (rank 0-1, 9.6 % of misses at C 224). In
   step 1 this halved the runahead error (0.6x -> 0.3x REF2-T's own tier error).
4. **Commit.**
   - A: commit up to its exact accepted length (+ bonus).
   - B: roll back B's KV / GDN rows entirely (`rollback_after_verify`). They are recomputed as A next pass.
   - If A's exact verdict differs from the provisional one (a **bubble**), B was drafted from the wrong bonus. Discard B (its
     queued reads are wasted). The next pass has no A and a fresh B drafted from the exact bonus.
5. **Draft.** After pass k, draft B' from B's provisional bonus. The drafter context uses B's runahead taps for the
   provisionally accepted rows. Optionally repair the drafter cache with the exact taps when A is confirmed.
6. **SSD.**
   - B's reads have one pass of lead time, so they can be batched across layers (>= 13 in flight). That gives ~13.8 GB/s
     instead of 7.8-10.1 GB/s for per-layer bursts.
   - The binding condition is SSD busy time per pass <= compute(A + B).
7. **Rows per pass.** A + B can reach ~32 rows on code.
   - **Capped at 16 (buildable now).** REF2 keeps bits only up to 16 rows per call, so B gets 16 - |A| rows. Breakthrough's
     RUNAHEAD-BUILD review: code tokens per pass fall 10.54 -> 6.85. A +15 % whole-set gain then needs beta <= 3.45 ms per row,
     and beta 4.0 gives -7 .. +10 %.
   - **Uncapped (the build if the cap is lifted).** The contiguous [A | B] sequence of up to 32 rows, dispatched as
     independent 16-row tiles. qmm16c / GEMM / 3c pairs are per-row within a fixed tile; the R = 16 guards and the GDN prework
     extend to 32. The gate needs 320 ids per layer, above today's 185-id routing-row ceiling. B stays on A's exact states.
   - **Ruled out.** A batch-2 "shadow cache" (B on its own cache) is the same 32-row call to REF2's guards, plus drift. As two
     back-to-back passes it loses (-11.5 .. -19.4 %).

## Cost model (what step 2 measures)

pass = draft + max(compute(A + B), SSD busy for B's reads) + residual stalls + bubbles x (draft + compute(B))

- compute(rows) = constant + beta x rows, where beta = the pure compute per re-checked row. D-23b's 7.82 ms per row includes
  that row's SSD misses.
- Leap's 3-bit expert kernel runs 11.1 us per expert at 16 rows, so beta is about 2.4 ms plus engine overheads.
- Sensitivity on D-23's logged checks (`data/runahead_sens.txt`):

| beta (ms/row) | code | math | chat |
|---|---|---|---|
| 2.5 | +70-81 % | +65-78 % | +42-54 % |
| 3.5 | +46-55 % | +40-50 % | +30-41 % |
| 5.0 | +19-27 % | +13-21 % | +16-25 % |
| 7.82 | loses | loses | ~0 % |

Step 2 reads beta from an all-resident exact re-run of each check, with ms vs rows.

## Where it does not help

- **24 GB.** The SSD is 90-98 % busy there, so cycle ~ SSD time already. RUNAHEAD only adds overlap of compute (~+5-10 %).
- **Trees on top.** A tree in the runahead only, re-checking the chosen branch, costs about the rows that Breakthrough's
  dynamic chat trees need (+19 % kept for +4.9 rows), so it nets ~0.

## Risks to check in step 2

- **U2** (provisional == exact verdict): chat near-ties may flip more often.
- **U1** (miss recall): B's runahead routing vs the exact routing.
- **U3** (next block from runahead taps vs exact taps).
- **beta**.
