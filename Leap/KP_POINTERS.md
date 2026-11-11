# KP pointers for BigMoE (Leap, Mon 2026-10-26 19:10 EET)
Per-kernel data, all in ~/Leap/data (sections of the same name in ~/Leap/TABLE_350.md):
- kern23.txt (L1-23): every dense shape of a 16-row check, ms per check at MLX 1 row / MLX 16 rows / today's chunked exact /
  qmm16c, plus padding-independence and max error vs fp32. MLX's own 16-row call is padding-independent for every shape.
  qmm16c is batch-invariant but NOT bitwise vs MLX (a different K split).
- d32.txt (D32): dense at 16 vs 32 rows (mr2 / mg2 bitwise per row vs the 16-row path; MLX's 32-row call bitwise to
  2 x 16 except 4-bit small-N shapes and the 8-bit router / indexer).
- gmr24.txt (L1-24): experts read once per distinct expert, 4 / 8 / 16 rows: BITWISE equal to the engine's path, but only 6 %
  faster at 16 rows (1.132 vs 1.209 ms per layer); 4 rows slightly slower.
- qmmg25.txt (L1-25): experts on the tensor units (NAX), 4 / 8 / 16 rows: batch-invariant, error <= MLX's, NOT bitwise vs MLX
  (0.776 ms per layer at 16 rows vs 1.21-1.26 sorted).
- qmm26.txt / qmm27.txt (L1-26 / L1-27): native 4-bit layout, bitwise vs qmm16c / experts_nax (not vs MLX).
- ca_diff_report.txt (CA run 3): the 45 GB engine's own 16-row check split in situ, 48 layers: experts 58.7, HC 31,
  GDN projections 17, QSA 7 (1K) / 19 (8K), commit replay 9.7, eager glue 5.8 ms; 8 and 32 rows too.
Bottom line for KP: none of Leap's faster kernels is bitwise vs MLX's shipped path. The only bitwise-equal one (gmr24) gains 6 %.
The rest are batch-invariant reference changes (Kamen's call). Rows 3-7 are covered by gmr24 / qmmg25 (4 and 8 rows) and
kern23's MLX 1-row column.
