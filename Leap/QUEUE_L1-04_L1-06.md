# QUEUE L1-04 (block-16 DFlash, CPU step) and L1-06 (joint verify, CPU): Leap, Sat 2026-10-24 02:42
NewIdeas' QUEUE.md set the methods and kill lines (taken 02:35). All tau use the fixed counting (bonus counted;
DFL2D_PREREG.md).

## L1-04: block-16 DFlash, the CPU projection. NOT KILLED (projected tau16 >= 9)
The method is NewIdeas': extrapolate DFlash2's per-depth acceptance curve to depth 15.

- Code (HumanEval+, L2): P(first j right) = .97 .92 .88 .85 .80 .77 .74.
  - The conditional acceptance at depths 2-7 is 0.948 0.957 0.966 0.941 0.963 0.961: mean 0.956, trend +0.002 per
    depth. It is flat, with no decay visible.
- Projected tau16, tokens per 16-row pass:

| depths 8-15 assumed | code | GSM8K |
|---|---|---|
| flat at the depth 2-7 mean | 11.79 | 7.93 |
| the depth 2-7 trend continued | 11.90 | 7.70 |
| -1 point per depth | 11.15 | 7.66 |
| -2 points per depth | 10.63 | 7.44 |
| -3 points per depth | 10.20 | 7.26 |
| (tau8 today, block 8) | 6.93 | 5.74 |

- Code stays above 9 unless the deep positions collapse to ~0.75 conditional acceptance. So the bet goes on to the
  real test: a fine-tune of the block-8 drafter to block 16 on code, local, < 1 h.
- Caveat: depths 8-15 are untrained, so this projection is optimistic by construction. Only the fine-tune measures it.
- What it is worth. Expert union per layer, from the 12 agentic turns' recorded routes: 46.2 experts at 8 rows,
  73.3 at 16 (x1.59).
  - With experts ~60 % of a pass, a 16-row pass ~= 1.35x an 8-row one.
  - Block 16 wins per unit time if tau16 / tau8 > T16 / T8: projected 11.2 / 6.93 = 1.62 > ~1.35, so about +20 %.
  - Code-350 (A / T >= 257/s): today's chain-8 needs an 8-row pass <= 27 ms. Block 16 at 11.2 needs a 16-row pass
    <= 43.6 ms, i.e. an 8-row-equivalent of ~32 ms.

## L1-06: joint verify (DFlash2 block + MTP chain in one pass, accept the longer). KILLED on CPU, unless the extra rows are almost free
- The gain is per pass. Oracle best-of on the 12 agentic turns: 4.04 vs MTP alone 3.70, i.e. +9.2 %. Verifying both
  branches gets exactly this.
- The cost: up to 7 more rows (8 -> 15).
- NewIdeas' line is net >= +5 %, so 15 rows must cost <= 1.04x 8 rows.
- Expert union, from the routes of 15 distinct positions: 70.3 vs 46.2 experts per layer (x1.52).
  - The second branch's rows sit at the same positions but hold different tokens after the branches split, so the
    real growth lies between x1.0 and x1.52.
  - Staying within 1.04x needs the second branch to reuse >= ~85 % of the first branch's experts, with zero growth in
    dense cost.
- Wide passes also pay more in dense cost today: 5-row chunking costs 3.5-6.4x on the GDN shapes (QUEUE L1-01).
  MoonShots measures the expert half at 0.11-0.15 ms per layer at 1 row and 0.57-0.91 ms at 8 rows.
- Realistic net: -3 % to -12 %. **KILLED.**
- It would come back only if a real 15-row pass measures <= 1.04x an 8-row one. Fit12's L1-02 pass-time batch could
  add a 16-row point cheaply; the same point also prices block 16.
