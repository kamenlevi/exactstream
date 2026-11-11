# Reference v2 batch for BigMoE (125B Q4, 64 GB M5 Pro)

Changes that are blocked **only** because they aren't bit-identical to the current reference (compiled eager MLX + dup-row).
Under Kamen's option 2 they can all be adopted together as reference v2. That takes one validation (near-tie check +
GSM8K-150 identical to eager) instead of one per change. Each one is deterministic and batch-invariant on its own; none changes
the model's math beyond summation order and bf16 rounding points. Gains are against BigMoE's split at 55 % resident (~44 ms/token).

| # | change | why it isn't bit-identical today | est. gain | status / evidence |
|---|---|---|---|---|
| v2-1 | **Packed HC read with its own schedule** (`~/MoonShots/patches/hc_v2/hc_read_v2.py`): exponent-offset pack (−22 % bytes, lossless) + 3 dispatches, split-K down with a fixed-order reduction | different lane schedule and reduction order from MLX's GemvWide | measured: −8–10 % per read at S=1 but slower at S=3 (97 vs 87 µs) → **no net gain, dropped** | data/test_hc_v2.txt, test_hc_v3.txt |
| v2-2 | **Fused routing kernel** (MoEPrefetch's: softmax + top-K + weight normalisation in one Metal kernel, 15.7 vs 88 µs per call) | fp32 softmax and expert order differ in the last bits (LEDGER row 4) | up to ~70 µs × 48 ≈ **−3 ms/token** if the router chain really costs ~88 µs on the 125B | measured on the 35B by MoEPrefetch |
| v2-3 | **Split-phase MoE**: compute resident experts into per-expert slots while misses load, then reduce in a fixed order | expert accumulation order changes | 25–40 % of the 9.5 ms SSD waits → **−2.5 to −4 ms/token** | design only (IDEAS round 6) |
| v2-4 | MTPLX's fused GDN decode kernels (`MTPLX_FUSED_GDN_STEP`, `_CONVNORM`, `_OUT`, `_INPROJ`), MTPLX "family defaults" in its server | reduction order (LEDGER A2: "output differs from token 1") | unknown: A2 measured *slower* (19.8 vs 22.1 tok/s) on BigMoE's path, so re-measure inside the batch, not in front of it | measured once, negative |
| v2-5 | MLX 0.32.4 instead of 0.32.2 (SpecDecode: +17 % decode, +25 % prefill on the 35B) | kernel changes between MLX versions change bits | **+5–15 %** | needs a separate venv (MTPLX pins 0.32.2) |
| v2-6 | Expert scales/biases exponent pack (15b): bf16 scales/biases 7.5 GB → 5.2 GB, lossless | needs a custom gather_qmm (different reduction order) | −2.3 GB RAM → ~5 % more experts resident → fewer SSD misses (~−0.5 to −1 ms/token) | pack measured; kernel not built |
| v2-7 | Inject rows folded into the HC down matmul (324 rows) | identical at M ≤ 6 (GemvWide lanes match), differs at M = 12 | ~−0.6 ms/token | only matters for S > 6; subsumed by v2-1 |

Not in the batch (already exact or dead): `hc_verify_read` (bit-identical, −1.1 ms/token, adopt now), pipelined decode (+15 %, exact),
megakernel between gates (+3 % measured), split-K matvecs (slower than MLX), GPU→CPU mid-kernel publish (impossible on Metal).

Suggested: one extra column in LEDGER.md, "v2 batch", marking each row that waits for this validation.
